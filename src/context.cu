#include <algorithm>
#include <stdexcept>
#include <utility>
#include "context.cuh"

#include "util/common.h"
#include "util/uintarith.h"
#include "util/uintarithsmallmod.h"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

namespace phantom {
    namespace util::global_variables {
        std::unique_ptr<util::cuda_stream_wrapper> default_stream;
    }

    ContextData::ContextData(const EncryptionParameters &params, const cudaStream_t &stream) {
        parms_ = params;
        const auto &key_modulus = params.key_modulus();
        const auto &coeff_modulus = params.coeff_modulus();
        const auto &plain_modulus = params.plain_modulus();
        const size_t special_modulus_size = params.special_modulus_size();

        const size_t coeff_modulus_size = coeff_modulus.size();

        // Compute the product of all coeff modulus
        total_coeff_modulus_ = std::vector<uint64_t>(coeff_modulus_size);
        auto coeff_modulus_values = std::vector<uint64_t>(coeff_modulus_size);
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            coeff_modulus_values[i] = coeff_modulus[i].value();
        }
        multiply_many_uint64(coeff_modulus_values.data(), coeff_modulus_size, total_coeff_modulus_.data());
        total_coeff_modulus_bit_count_ =
                get_significant_bit_count_uint(total_coeff_modulus_.data(), coeff_modulus_size);

        size_t poly_modulus_degree = params.poly_modulus_degree();

        int coeff_count_power = get_power_of_two(poly_modulus_degree);

        const auto coeff_modulus_base = make_shared<RNSBase>(RNSBase(coeff_modulus));

        small_ntt_tables_ = make_shared<RNSNTT>(coeff_count_power, coeff_modulus);

        if (params.scheme() == scheme_type::bfv || params.scheme() == scheme_type::bgv) {
            plain_ntt_tables_ = make_shared<NTT>(coeff_count_power, plain_modulus);

            // Calculate coeff_div_plain_modulus (BFV-"Delta") and the remainder upper_half_increment
            auto temp_coeff_div_plain_modulus = std::vector<uint64_t>(coeff_modulus_size);
            auto wide_plain_modulus = std::vector<uint64_t>(coeff_modulus_size);
            coeff_div_plain_modulus_.resize(coeff_modulus_size);
            coeff_div_plain_modulus_shoup_.resize(coeff_modulus_size);
            plain_modulus_.resize(coeff_modulus_size);
            plain_modulus_shoup_.resize(coeff_modulus_size);
            upper_half_increment_.resize(coeff_modulus_size);
            wide_plain_modulus[0] = plain_modulus.value();
            // temp_coeff_div_plain_modulus = total_coeff_modulus_ / wide_plain_modulus,
            // upper_half_increment_ is the remainder
            divide_uint(total_coeff_modulus_.data(), wide_plain_modulus.data(), coeff_modulus_size,
                        temp_coeff_div_plain_modulus.data(), upper_half_increment_.data());

            // Store the non-RNS form of upper_half_increment for BFV encryption
            coeff_modulus_mod_plain_modulus_ = upper_half_increment_[0];

            // Decompose coeff_div_plain_modulus into RNS factors
            coeff_modulus_base->decompose(temp_coeff_div_plain_modulus.data());

            for (size_t i = 0; i < coeff_modulus_size; i++) {
                coeff_div_plain_modulus_[i] = temp_coeff_div_plain_modulus[i];
                coeff_div_plain_modulus_shoup_[i] =
                        compute_shoup(temp_coeff_div_plain_modulus[i], coeff_modulus_base->base()[i].value());

                plain_modulus_[i] = plain_modulus.value();
                plain_modulus_shoup_[i] = compute_shoup(plain_modulus.value(), coeff_modulus_base->base()[i].value());
            }

            // Decompose upper_half_increment into RNS factors
            coeff_modulus_base->decompose(upper_half_increment_.data());

            // Calculate (plain_modulus + 1) / 2.
            plain_upper_half_threshold_ = (plain_modulus.value() + 1) >> 1;

            // Calculate coeff_modulus - plain_modulus.
            plain_upper_half_increment_.resize(coeff_modulus_size);
            // Calculate coeff_modulus[i] - plain_modulus if using_fast_plain_lift
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                plain_upper_half_increment_[i] = coeff_modulus[i].value() - plain_modulus.value();
            }
        } else if (params.scheme() == scheme_type::ckks) {
            // plain_modulus should be zero
            if (!plain_modulus.is_zero()) {
                throw std::invalid_argument("plain_modulus must be zero for CKKS");
            }

            // Calculate 2^64 / 2 (most negative plaintext coefficient value)
            plain_upper_half_threshold_ = uint64_t(1) << 63;

            // Calculate plain_upper_half_increment = 2^64 mod coeff_modulus for CKKS plaintexts
            plain_upper_half_increment_.resize(coeff_modulus_size);
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                // tmp = (1 << 63) % coeff_modulus[i]
                uint64_t tmp = barrett_reduce_64(uint64_t(1) << 63, coeff_modulus[i]);
                // plain_upper_half_increment_[i] = tmp * (coeff_modulus[i] - 2) % coeff_modulus[i]
                plain_upper_half_increment_[i] =
                        multiply_uint_mod(tmp, sub_safe(coeff_modulus[i].value(), uint64_t(2)), coeff_modulus[i]);
            }

            // Compute the upper_half_threshold for this modulus.
            upper_half_threshold_.resize(coeff_modulus_size);
            // upper_half_threshold_ = (total_coeff_modulus_ + 1) /2
            increment_uint(total_coeff_modulus_.data(), coeff_modulus_size, upper_half_threshold_.data());
            right_shift_uint(upper_half_threshold_.data(), 1, coeff_modulus_size, upper_half_threshold_.data());
        } else {
            throw std::invalid_argument("unsupported scheme");
        }

        // Create RNSTool
        gpu_rns_tool_ = std::make_shared<DRNSTool>(poly_modulus_degree, special_modulus_size, *coeff_modulus_base,
                                                   key_modulus, plain_modulus, params.mul_tech(), stream);
    }
}

PhantomContext::PhantomContext(const phantom::EncryptionParameters &params) {
    if (params.coeff_modulus().size() == 1)
        throw std::invalid_argument("The coefficient modulus must be a vector of at least two primes");

    // GPU setup
    int device;
    cudaGetDevice(&device);
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, device);
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

    phantom::util::global_variables::default_stream = std::make_unique<phantom::util::cuda_stream_wrapper>();

    const auto &s = phantom::util::global_variables::default_stream->get_stream();

    using_keyswitching_ = false;
    mul_tech_ = params.mul_tech();
    size_t size_P = params.special_modulus_size();
    size_t size_QP = params.coeff_modulus().size();
    size_t size_Q = size_QP - size_P;
    poly_degree_ = params.poly_modulus_degree();

    auto temp_parms = params;
    auto &coeff_modulus = temp_parms.coeff_modulus();

    context_data_.emplace_back(temp_parms, s);

    if (size_P != 0) {
        using_keyswitching_ = true;
    }

    // Drop all special modulus at first data level
    for (size_t i = 0; i < size_P; i++)
        coeff_modulus.pop_back();

    for (size_t i = 0; i < size_Q; i++) {
        context_data_.emplace_back(temp_parms, s);
        // Drop one modulus after each data level
        coeff_modulus.pop_back();
    }

    first_parm_index_ = 0;
    if (context_data_.size() > 1)
        first_parm_index_ = 1;

    // set chain index
    for (size_t idx = 0; idx < context_data_.size(); idx++) {
        context_data_[idx].set_chain_index(idx);
    }

    auto &coeff_modulus_cpu = params.coeff_modulus();
    coeff_mod_size_ = coeff_modulus_cpu.size();
    auto &small_ntt_tables = get_context_data(0).small_ntt_tables();
    gpu_rns_tables().init(poly_degree_, coeff_mod_size_, s);
    for (size_t i = 0; i < coeff_mod_size_; i++) {
        DModulus temp = DModulus(coeff_modulus_cpu[i].value(), coeff_modulus_cpu[i].const_ratio()[0],
                                 coeff_modulus_cpu[i].const_ratio()[1]);
        gpu_rns_tables().set(&temp, small_ntt_tables->get_ntt_at(i).get_from_root_powers().data(),
                             small_ntt_tables->get_ntt_at(i).get_from_root_powers_shoup().data(),
                             small_ntt_tables->get_ntt_at(i).get_from_inv_root_powers().data(),
                             small_ntt_tables->get_ntt_at(i).get_from_inv_root_powers_shoup().data(),
                             small_ntt_tables->get_ntt_at(i).inv_degree_modulo(),
                             small_ntt_tables->get_ntt_at(i).inv_degree_modulo_shoup(), i, s);
    }

    if (params.scheme() == phantom::scheme_type::bfv || params.scheme() == phantom::scheme_type::bgv) {
        auto &plain_ntt_tables = get_context_data(0).plain_ntt_tables();
        auto &plain_modulus_cpu = params.plain_modulus();
        gpu_plain_tables().init(poly_degree_, 1, s);
        const auto temp = DModulus(plain_modulus_cpu.value(), plain_modulus_cpu.const_ratio()[0],
                                   plain_modulus_cpu.const_ratio()[1]);
        gpu_plain_tables().set(&temp, plain_ntt_tables->get_from_root_powers().data(),
                               plain_ntt_tables->get_from_root_powers_shoup().data(),
                               plain_ntt_tables->get_from_inv_root_powers().data(),
                               plain_ntt_tables->get_from_inv_root_powers_shoup().data(),
                               plain_ntt_tables->inv_degree_modulo(), plain_ntt_tables->inv_degree_modulo_shoup(), 0,
                               s);

        plain_modulus_ = make_cuda_auto_ptr<uint64_t>(coeff_mod_size_, s);
        plain_modulus_shoup_ = make_cuda_auto_ptr<uint64_t>(coeff_mod_size_, s);
        cudaMemcpyAsync(plain_modulus_.get(), get_context_data(0).plain_modulus().data(),
                        coeff_mod_size_ * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(plain_modulus_shoup_.get(), get_context_data(0).plain_modulus_shoup().data(),
                        coeff_mod_size_ * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
    }

    if (params.scheme() == phantom::scheme_type::bfv) {
        const auto coeff_div_plain_size = (coeff_mod_size_ * 2 - total_parm_size() + 1) * total_parm_size() / 2;
        coeff_div_plain_ = make_cuda_auto_ptr<uint64_t>(coeff_div_plain_size, s);
        coeff_div_plain_shoup_ = make_cuda_auto_ptr<uint64_t>(coeff_div_plain_size, s);
        auto cdp_pos = 0;
        for (size_t i = 0; i < total_parm_size(); i++) {
            const auto size = get_context_data(i).coeff_div_plain_modulus().size();
            // force to memcpy, as the type is different but the values are consistent
            cudaMemcpyAsync(coeff_div_plain_.get() + cdp_pos,
                            get_context_data(i).coeff_div_plain_modulus().data(), size * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, s);
            cudaMemcpyAsync(coeff_div_plain_shoup_.get() + cdp_pos,
                            get_context_data(i).coeff_div_plain_modulus_shoup().data(), size * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, s);
            cdp_pos += size;
        }

        plain_upper_half_increment_ = make_cuda_auto_ptr<uint64_t>(coeff_mod_size_, s);
        cudaMemcpyAsync(plain_upper_half_increment_.get(),
                        get_context_data(0).plain_upper_half_increment().data(),
                        coeff_mod_size_ * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
    }

    int log_n = phantom::arith::get_power_of_two(poly_degree_);
    bool is_bfv = (params.scheme() == phantom::scheme_type::bfv);
    key_galois_tool_ = std::make_unique<PhantomGaloisTool>(params.galois_elts(), log_n, s, is_bfv);
}

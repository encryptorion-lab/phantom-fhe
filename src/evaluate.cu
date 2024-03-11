#include "evaluate.h"

#include "mempool.cuh"
#include "rns_bconv.cuh"
#include "scalingvariant.cuh"
#include "util.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;

static void negate_internal(const PhantomContext& context, PhantomCiphertext& encrypted, const cudaStream_t stream) {
    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();
    auto& coeff_modulus = parms.coeff_modulus();
    const auto coeff_mod_size = coeff_modulus.size();
    const auto poly_degree = parms.poly_modulus_degree();
    const auto base_rns = context.gpu_rns_tables().modulus();
    const auto rns_coeff_count = poly_degree * coeff_mod_size;

    uint64_t gridDimGlb = rns_coeff_count / blockDimGlb.x;
    for (size_t i = 0; i < encrypted.size(); i++) {
        negate_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(encrypted.data() + i * rns_coeff_count, base_rns,
                                                                encrypted.data() + i * rns_coeff_count,
                                                                poly_degree,
                                                                coeff_mod_size);
    }
}

void negate_inplace(const PhantomContext& context, PhantomCiphertext& encrypted) {
    negate_internal(context, encrypted, cudaStreamLegacy);
}

void negate_inplace_async(const PhantomContext& context, PhantomCiphertext& encrypted, const cudaStream_t stream) {
    negate_internal(context, encrypted, stream);
}

/**
 * Adds two ciphertexts. This function adds together encrypted1 and encrypted2 and stores the result in encrypted1.
 * @param[in] encrypted1 The first ciphertext to add
 * @param[in] encrypted2 The second ciphertext to add
 */
void add_inplace(const PhantomContext& context, PhantomCiphertext& encrypted1, const PhantomCiphertext& encrypted2) {
    if (encrypted1.chain_index() != encrypted2.chain_index()) {
        throw std::invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    }
    if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form()) {
        throw std::invalid_argument("NTT form mismatch");
    }
    if (encrypted1.scale() != encrypted2.scale()) {
        throw std::invalid_argument("scale mismatch");
    }
    if (encrypted1.size_ != encrypted2.size_) {
        throw std::invalid_argument("poly number mismatch");
    }

    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted1.chain_index());
    auto& parms = context_data.parms();
    auto& coeff_modulus = parms.coeff_modulus();
    auto& plain_modulus = parms.plain_modulus();
    auto coeff_modulus_size = coeff_modulus.size();
    auto poly_degree = context.gpu_rns_tables().n();
    auto base_rns = context.gpu_rns_tables().modulus();
    auto rns_coeff_count = poly_degree * coeff_modulus_size;
    size_t encrypted1_size = encrypted1.size();
    size_t encrypted2_size = encrypted2.size();
    size_t max_size = max(encrypted1_size, encrypted2_size);
    size_t min_size = min(encrypted1_size, encrypted2_size);

    uint64_t gridDimGlb = rns_coeff_count / blockDimGlb.x;

    if (encrypted1.correction_factor() != encrypted2.correction_factor()) {
        // Balance correction factors and multiply by scalars before addition in BGV
        auto factors = balance_correction_factors(encrypted1.correction_factor(), encrypted2.correction_factor(),
                                                  plain_modulus);
        for (size_t i = 0; i < encrypted1.size(); i++) {
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(
                encrypted1.data() + i * rns_coeff_count, get<1>(factors), base_rns,
                encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }

        PhantomCiphertext encrypted2_copy(context);
        encrypted2_copy = encrypted2;
        for (size_t i = 0; i < encrypted2.size(); i++) {
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(
                encrypted2_copy.data() + i * rns_coeff_count, get<2>(factors), base_rns,
                encrypted2_copy.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }

        // Set new correction factor
        encrypted1.correction_factor() = get<0>(factors);
        encrypted2_copy.correction_factor() = get<0>(factors);

        add_inplace(context, encrypted1, encrypted2_copy);
    } else {
        // Prepare destination
        encrypted1.resize(context, context_data.chain_index(), max_size);
        for (size_t i = 0; i < min_size; i++) {
            add_rns_poly<<<gridDimGlb, blockDimGlb>>>(
                encrypted1.data() + i * rns_coeff_count, encrypted2.data() + i * rns_coeff_count, base_rns,
                encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }
        if (encrypted1_size < encrypted2_size) {
            CUDA_CHECK(cudaMemcpy(encrypted1.data() + min_size * rns_coeff_count,
                encrypted2.data() + min_size * rns_coeff_count,
                (encrypted2_size - encrypted1_size) * rns_coeff_count * sizeof(uint64_t),
                cudaMemcpyDeviceToDevice));
        }
    }
}

// TODO: fixme
void add_many(const PhantomContext& context, const vector<PhantomCiphertext>& encrypteds,
              PhantomCiphertext& destination) {
    if (encrypteds.empty()) {
        throw std::invalid_argument("encrypteds cannot be empty");
    }
    for (size_t i = 0; i < encrypteds.size(); i++) {
        if (&encrypteds[i] == &destination) {
            throw std::invalid_argument("encrypteds must be different from destination");
        }
        if (encrypteds[0].chain_index() != encrypteds[i].chain_index()) {
            throw invalid_argument("encrypteds parameter mismatch");
        }
        if (encrypteds[0].is_ntt_form() != encrypteds[i].is_ntt_form()) {
            throw std::invalid_argument("NTT form mismatch");
        }
        if (encrypteds[0].scale() != encrypteds[i].scale()) {
            throw std::invalid_argument("scale mismatch");
        }
        if (encrypteds[0].size() != encrypteds[i].size()) {
            throw std::invalid_argument("poly number mismatch");
        }
    }

    auto& context_data = context.get_context_data(encrypteds[0].chain_index());
    auto& parms = context_data.parms();
    auto& coeff_modulus = parms.coeff_modulus();
    auto coeff_mod_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto poly_num = encrypteds[0].size();
    auto base_rns = context.gpu_rns_tables().modulus();
    // reduction_threshold = 2 ^ (64 - max modulus bits)
    // max modulus bits = static_cast<uint64_t>(log2(coeff_modulus.front().value())) + 1
    auto reduction_threshold =
            (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(coeff_modulus.front().value())) - 1)) - 1;

    destination.resize(context, encrypteds[0].chain_index(), encrypteds[0].size());
    destination.is_ntt_form() = encrypteds[0].is_ntt_form();
    destination.scale() = encrypteds[0].scale();

    if (parms.scheme() == phantom::scheme_type::bgv) // TODO: any optimizations?
    {
        CUDA_CHECK(cudaMemcpy(destination.data(), encrypteds[0].data(),
            poly_degree * coeff_mod_size * encrypteds[0].size() * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice));
        for (size_t i = 1; i < encrypteds.size(); i++) {
            add_inplace(context, destination, encrypteds[i]);
        }
    } else {
        Pointer<uint64_t *> enc_device_ptr;
        enc_device_ptr.acquire(allocate<uint64_t *>(global_pool(), encrypteds.size()));
        uint64_t* enc_host_ptr[encrypteds.size()];

        for (size_t i = 0; i < encrypteds.size(); i++) {
            enc_host_ptr[i] = encrypteds[i].data();
        }
        CUDA_CHECK(cudaMemcpy(enc_device_ptr.get(), enc_host_ptr, sizeof(uint64_t *) * encrypteds.size(),
            cudaMemcpyHostToDevice));

        uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
        for (size_t i = 0; i < poly_num; i++) {
            add_many_rns_poly<<<gridDimGlb, blockDimGlb>>>(enc_device_ptr.get(), encrypteds.size(), base_rns,
                                                           destination.data(), i, poly_degree, coeff_mod_size,
                                                           reduction_threshold);
        }
    }
}

void sub_inplace(const PhantomContext& context, PhantomCiphertext& encrypted1, const PhantomCiphertext& encrypted2,
                 const bool& negate) {
    if (encrypted1.parms_id() != encrypted2.parms_id()) {
        throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    }
    if (encrypted1.chain_index() != encrypted2.chain_index())
        throw std::invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        throw std::invalid_argument("NTT form mismatch");
    if (encrypted1.scale() != encrypted2.scale())
        throw std::invalid_argument("scale mismatch");
    if (encrypted1.size_ != encrypted2.size_)
        throw std::invalid_argument("poly number mismatch");

    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted1.chain_index());
    auto& parms = context_data.parms();
    auto& coeff_modulus = parms.coeff_modulus();
    auto& plain_modulus = parms.plain_modulus();
    auto coeff_modulus_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto base_rns = context.gpu_rns_tables().modulus();
    auto rns_coeff_count = poly_degree * coeff_modulus_size;
    size_t encrypted1_size = encrypted1.size();
    size_t encrypted2_size = encrypted2.size();
    size_t max_count = max(encrypted1_size, encrypted2_size);
    size_t min_count = min(encrypted1_size, encrypted2_size);

    uint64_t gridDimGlb = rns_coeff_count / blockDimGlb.x;

    if (encrypted1.correction_factor() != encrypted2.correction_factor()) {
        // Balance correction factors and multiply by scalars before addition in BGV
        auto factors = balance_correction_factors(encrypted1.correction_factor(), encrypted2.correction_factor(),
                                                  plain_modulus);
        for (size_t i = 0; i < encrypted1.size(); i++) {
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(
                encrypted1.data() + i * rns_coeff_count, get<1>(factors), base_rns,
                encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }

        PhantomCiphertext encrypted2_copy(context);
        encrypted2_copy = encrypted2;
        for (size_t i = 0; i < encrypted2.size(); i++) {
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(
                encrypted2_copy.data() + i * rns_coeff_count, get<2>(factors), base_rns,
                encrypted2_copy.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }

        // Set new correction factor
        encrypted1.correction_factor() = get<0>(factors);
        encrypted2_copy.correction_factor() = get<0>(factors);

        sub_inplace(context, encrypted1, encrypted2_copy, negate);
    } else {
        if (negate) {
            for (size_t i = 0; i < encrypted1.size(); i++) {
                sub_rns_poly<<<gridDimGlb, blockDimGlb>>>(
                    encrypted2.data() + i * rns_coeff_count, encrypted1.data() + i * rns_coeff_count, base_rns,
                    encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
            }
        } else {
            for (size_t i = 0; i < encrypted1.size(); i++) {
                sub_rns_poly<<<gridDimGlb, blockDimGlb>>>(
                    encrypted1.data() + i * rns_coeff_count, encrypted2.data() + i * rns_coeff_count, base_rns,
                    encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
            }
        }
    }
}

void bgv_ckks_multiply(const PhantomContext& context, PhantomCiphertext& encrypted1,
                       const PhantomCiphertext& encrypted2) {
    if (!(encrypted1.is_ntt_form() && encrypted2.is_ntt_form()))
        throw invalid_argument("encrypted1 and encrypted2 must be in NTT form");

    if (encrypted1.chain_index() != encrypted2.chain_index())
        throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");

    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted1.chain_index());
    auto& parms = context_data.parms();
    auto base_rns = context.gpu_rns_tables().modulus();
    size_t coeff_mod_size = parms.coeff_modulus().size();
    size_t poly_degree = parms.poly_modulus_degree();
    uint32_t encrypted1_size = encrypted1.size();
    uint32_t encrypted2_size = encrypted2.size();

    // Determine destination.size()
    // Default is 3 (c_0, c_1, c_2)
    uint32_t dest_size = encrypted1_size + encrypted2_size - 1;

    // Size check
    // Prepare destination
    encrypted1.resize(context, encrypted1.chain_index(), dest_size);

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;

    if (dest_size == 3) {
        if (&encrypted1 == &encrypted2) {
            // square
            tensor_square_2x2_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted1.data(), base_rns, encrypted1.data(),
                                                                    poly_degree, coeff_mod_size);
        } else {
            // standard multiply
            tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted1.data(), encrypted2.data(), base_rns,
                                                                  encrypted1.data(), poly_degree, coeff_mod_size);
        }
    } else {
        tensor_prod_mxn_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted1.data(), encrypted1_size, encrypted2.data(),
                                                              encrypted2_size, base_rns, encrypted1.data(), dest_size,
                                                              poly_degree, coeff_mod_size);
    }

    // CKKS needs to do scaling
    if (parms.scheme() == scheme_type::ckks)
        encrypted1.scale() *= encrypted2.scale();

    // BGV needs to update correction factor
    if (parms.scheme() == scheme_type::bgv)
        encrypted1.correction_factor() = multiply_uint_mod(encrypted1.correction_factor(),
                                                           encrypted2.correction_factor(), parms.plain_modulus());
}

// Perform BEHZ steps (1)-(3) for PhantomCiphertext
// (1) Lift encrypted (initially in base q) to an extended base q U Bsk U {m_tilde}
// (2) Remove extra multiples of q from the results with Montgomery reduction, switching base to q U Bsk
// (3) Transform the data to NTT form
// @notice: temp is used to avoid memory malloc in sm_mrq
void BEHZ_mul_1(const PhantomContext& context, const PhantomCiphertext& encrypted, uint64_t* encrypted_q,
                uint64_t* encrypted_Bsk) {
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();
    auto poly_degree = parms.poly_modulus_degree();
    auto& rns_tool = context.get_context_data(encrypted.chain_index()).gpu_rns_tool();

    size_t base_q_size = rns_tool.base_Ql().size();
    size_t base_Bsk_size = rns_tool.base_Bsk().size();
    size_t base_Bsk_m_tilde_size = rns_tool.base_Bsk_m_tilde().size();

    size_t q_coeff_count = poly_degree * base_q_size;
    size_t bsk_coeff_count = poly_degree * base_Bsk_size;

    Pointer<uint64_t> temp_base_Bsk_m_tilde;
    temp_base_Bsk_m_tilde.acquire(allocate<uint64_t>(global_pool(), poly_degree * base_Bsk_m_tilde_size));

    CUDA_CHECK(cudaMemcpy(encrypted_q, encrypted.data(), encrypted.size() * q_coeff_count * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice));

    for (size_t i = 0; i < encrypted.size(); i++) {
        uint64_t* encrypted_ptr = encrypted.data() + i * q_coeff_count;
        uint64_t* encrypted_q_ptr = encrypted_q + i * q_coeff_count;
        uint64_t* encrypted_bsk_ptr = encrypted_Bsk + i * bsk_coeff_count;
        // NTT forward
        nwt_2d_radix8_forward_inplace(encrypted_q_ptr, context.gpu_rns_tables(), base_q_size, 0);
        // (1) Convert from base q to base Bsk U {m_tilde}
        rns_tool.fastbconv_m_tilde(temp_base_Bsk_m_tilde.get(), encrypted_ptr);
        // (2) Reduce q-overflows in with Montgomery reduction, switching base to Bsk
        rns_tool.sm_mrq(encrypted_bsk_ptr, temp_base_Bsk_m_tilde.get());
        // NTT forward
        nwt_2d_radix8_forward_inplace_include_temp_mod(encrypted_bsk_ptr, rns_tool.gpu_Bsk_tables(), base_Bsk_size, 0,
                                                       rns_tool.gpu_Bsk_tables().size());
    }
    temp_base_Bsk_m_tilde.release();
}

// encrypted1 = encrypted1 * encrypted2
// (c0, c1) * (c0', c1') = (c0*c0', c0'c1+c0c1', c1c1')
// BEHZ RNS multiplication, which completes the multiplication in RNS form.
// (1) Lift encrypted1 and encrypted2 (initially in base q) to an extended base q U Bsk U {m_tilde}
// (2) Remove extra multiples of q from the results with Montgomery reduction, switching base to q U Bsk
// (3) Transform the data to NTT form
// (4) Compute the ciphertext polynomial product using dyadic multiplication
// (5) Transform the data back from NTT form
// (6) Multiply the result by t (plain_modulus)
// (7) Scale the result by q using a divide-and-floor algorithm, switching base to Bsk
// (8) Use Shenoy-Kumaresan method to convert the result to base q
void bfv_multiply_behz(const PhantomContext& context, PhantomCiphertext& encrypted1,
                       const PhantomCiphertext& encrypted2) {
    if (encrypted1.is_ntt_form() || encrypted2.is_ntt_form()) {
        throw std::invalid_argument("encrypted1 or encrypted2 cannot be in NTT form");
    }

    // Extract encryption parameters.
    const auto& context_data = context.get_context_data(encrypted1.chain_index());
    const auto& parms = context_data.parms();
    const auto& rns_tool = context.get_context_data(encrypted1.chain_index()).gpu_rns_tool();

    const size_t poly_degree = parms.poly_modulus_degree();
    const size_t encrypted1_size = encrypted1.size();
    const size_t encrypted2_size = encrypted2.size();
    const size_t base_q_size = rns_tool.base_Ql().size();
    const size_t base_Bsk_size = rns_tool.base_Bsk().size();
    const size_t dest_size = encrypted1_size + encrypted2_size - 1;

    const DModulus* base_rns = context.gpu_rns_tables().modulus();
    const DModulus* base_Bsk = rns_tool.base_Bsk().base();

    // malloc memory, which needs to be freed at the end of the function.
    Pointer<uint64_t> encrypted1_q, encrypted1_Bsk;
    Pointer<uint64_t> encrypted2_q, encrypted2_Bsk;

    encrypted1_q.acquire(allocate<uint64_t>(global_pool(), dest_size * poly_degree * base_q_size));
    encrypted1_Bsk.acquire(allocate<uint64_t>(global_pool(), dest_size * poly_degree * base_Bsk_size));
    encrypted2_q.acquire(allocate<uint64_t>(global_pool(), encrypted2_size * poly_degree * base_q_size));
    encrypted2_Bsk.acquire(allocate<uint64_t>(global_pool(), encrypted2_size * poly_degree * base_Bsk_size));

    // BEHZ, step 1-3
    BEHZ_mul_1(context, encrypted1, encrypted1_q.get(), encrypted1_Bsk.get());
    if (dest_size != 3 || &encrypted1 != &encrypted2)
        BEHZ_mul_1(context, encrypted2, encrypted2_q.get(), encrypted2_Bsk.get());

    uint64_t gridDimGlb;
    // BEHZ, step 4 Compute the ciphertext polynomial product using dyadic multiplication
    // (c0, c1, c2, ...) * (c0', c1', c2', ...)
    //    = (c0 * c0', c0*c1' + c1*c0', c0*c2'+c1*c1'+c2*c0', ...)
    if (dest_size == 3) {
        gridDimGlb = poly_degree * base_q_size / blockDimGlb.x;
        if (&encrypted1 == &encrypted2)
            tensor_square_2x2_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted1_q.get(), base_rns, encrypted1_q.get(),
                                                                    poly_degree, base_q_size);

        else
            tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted1_q.get(), encrypted2_q.get(), base_rns,
                                                                  encrypted1_q.get(), poly_degree, base_q_size);

        gridDimGlb = poly_degree * base_Bsk_size / blockDimGlb.x;
        if (&encrypted1 == &encrypted2)
            tensor_square_2x2_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted1_Bsk.get(), base_Bsk,
                                                                    encrypted1_Bsk.get(), poly_degree, base_Bsk_size);
        else
            tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted1_Bsk.get(), encrypted2_Bsk.get(), base_Bsk,
                                                                  encrypted1_Bsk.get(), poly_degree, base_Bsk_size);
    } else {
        gridDimGlb = poly_degree * base_q_size / blockDimGlb.x;
        tensor_prod_mxn_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted1_q.get(), encrypted1_size, encrypted2_q.get(),
                                                              encrypted2_size, base_rns, encrypted1_q.get(), dest_size,
                                                              poly_degree, base_q_size);

        gridDimGlb = poly_degree * base_Bsk_size / blockDimGlb.x;
        tensor_prod_mxn_rns_poly<<<gridDimGlb, blockDimGlb>>>(
            encrypted1_Bsk.get(), encrypted1_size, encrypted2_Bsk.get(), encrypted2_size, base_Bsk,
            encrypted1_Bsk.get(), dest_size, poly_degree, base_Bsk_size);
    }

    // BEHZ, step 5: NTT backward
    // Step (6): multiply base q components by t (plain_modulus)
    for (size_t i = 0; i < dest_size; i++) {
        nwt_2d_radix8_backward_inplace_scale(encrypted1_q.get() + i * poly_degree * base_q_size,
                                             context.gpu_rns_tables(), base_q_size, 0, context.plain_modulus(),
                                             context.plain_modulus_shoup());
    }
    for (size_t i = 0; i < dest_size; i++) {
        nwt_2d_radix8_backward_inplace_include_temp_mod_scale(
            encrypted1_Bsk.get() + i * poly_degree * base_Bsk_size, rns_tool.gpu_Bsk_tables(), base_Bsk_size, 0,
            rns_tool.gpu_Bsk_tables().size(), rns_tool.tModBsk(), rns_tool.tModBsk_shoup());
    }

    // Resize encrypted1 to destination size
    encrypted1.resize(context, encrypted1.chain_index(), dest_size);

    Pointer<uint64_t> temp;
    temp.acquire(allocate<uint64_t>(global_pool(), poly_degree * base_Bsk_size));
    for (size_t i = 0; i < dest_size; i++) {
        uint64_t* encrypted1_q_iter = encrypted1_q.get() + i * base_q_size * poly_degree;
        uint64_t* encrypted1_Bsk_iter = encrypted1_Bsk.get() + i * base_Bsk_size * poly_degree;
        uint64_t* encrypted1_iter = encrypted1.data() + i * base_q_size * poly_degree;
        // Step (7): divide by q and floor, producing a result(stored in encrypted2_Bsk) in base Bsk
        rns_tool.fast_floor(encrypted1_q_iter, encrypted1_Bsk_iter, temp.get());
        // encrypted1_q is used to avoid malloc in fastbconv_sk
        // Step (8): use Shenoy-Kumaresan method to convert the result to base q and write to encrypted1
        rns_tool.fastbconv_sk(temp.get(), encrypted1_iter);
        // encrypted1_q is used to avoid malloc in fastbconv_sk
    }
    temp.release();
}

size_t FindLevelsToDrop(const PhantomContext& context, size_t multiplicativeDepth, double dcrtBits, bool isKeySwitch,
                        bool isAsymmetric) {
    // Extract encryption parameters.
    auto& context_data = context.get_context_data(0);
    auto& parms = context_data.parms();
    auto n = parms.poly_modulus_degree();

    // handle no relin scenario
    size_t gpu_rns_tool_index = 0;
    if (context.using_keyswitching() == true) {
        gpu_rns_tool_index = 1;
    }

    auto& rns_tool = context.get_context_data(gpu_rns_tool_index).gpu_rns_tool(); // BFV does not drop modulus
    auto mul_tech = rns_tool.mul_tech();

    if (mul_tech != mul_tech_type::hps_overq_leveled)
        throw invalid_argument("FindLevelsToDrop is only used in HPS over Q Leveled");

    double sigma = distributionParameter;
    double alpha = assuranceMeasure;

    double p = parms.plain_modulus().value();

    uint32_t k = rns_tool.size_P();
    uint32_t numPartQ = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
    uint32_t thresholdParties = 1;
    // Bkey set to thresholdParties * 1 for ternary distribution
    const double Bkey = thresholdParties;

    double w = pow(2, dcrtBits);

    // Bound of the Gaussian error polynomial
    double Berr = sigma * sqrt(alpha);

    // expansion factor delta
    auto delta = [](uint32_t n) -> double { return (2. * sqrt(n)); };

    // norm of fresh ciphertext polynomial (for EXTENDED the noise is reduced to modulus switching noise)
    auto Vnorm = [&](uint32_t n) -> double {
        if (isAsymmetric)
            return (1. + delta(n) * Bkey) / 2.;
        return Berr * (1. + 2. * delta(n) * Bkey);
    };

    auto noiseKS = [&](uint32_t n, double logqPrev, double w) -> double {
        return k * (numPartQ * delta(n) * Berr + delta(n) * Bkey + 1.0) / 2;
    };

    // function used in the EvalMult constraint
    auto C1 = [&](uint32_t n) -> double { return delta(n) * delta(n) * p * Bkey; };

    // function used in the EvalMult constraint
    auto C2 = [&](uint32_t n, double logqPrev) -> double {
        return delta(n) * delta(n) * Bkey * Bkey / 2.0 + noiseKS(n, logqPrev, w);
    };

    // main correctness constraint
    auto logqBFV = [&](uint32_t n, double logqPrev) -> double {
        if (multiplicativeDepth > 0) {
            return log(4 * p) + (multiplicativeDepth - 1) * log(C1(n)) +
                   log(C1(n) * Vnorm(n) + multiplicativeDepth * C2(n, logqPrev));
        }
        return log(p * (4 * (Vnorm(n))));
    };

    // initial values
    double logqPrev = 6. * log(10);
    double logq = logqBFV(n, logqPrev);

    while (fabs(logq - logqPrev) > log(1.001)) {
        logqPrev = logq;
        logq = logqBFV(n, logqPrev);
    }

    // get an estimate of the error q / (4t)
    double loge = logq / log(2) - 2 - log2(p);

    double logExtra = isKeySwitch ? log2(noiseKS(n, logq, w)) : log2(delta(n));

    // adding the cushon to the error (see Appendix D of https://eprint.iacr.org/2021/204.pdf for details)
    // adjusted empirical parameter to 16 from 4 for threshold scenarios to work correctly, this might need to
    // be further refined
    int32_t levels = std::floor((loge - 2 * multiplicativeDepth - 16 - logExtra) / dcrtBits);
    auto sizeQ = static_cast<int32_t>(rns_tool.base_Q().size());

    if (levels < 0)
        levels = 0;
    else if (levels > sizeQ - 1)
        levels = sizeQ - 1;

    return levels;
}

// encrypted1 = encrypted1 * encrypted2
// (c0, c1) * (c0', c1') = (c0*c0', c0'c1+c0c1', c1c1')
// HPS
void bfv_multiply_hps(const PhantomContext& context, PhantomCiphertext& encrypted1,
                      const PhantomCiphertext& encrypted2) {
    if (encrypted1.is_ntt_form() || encrypted2.is_ntt_form()) {
        throw std::invalid_argument("encrypted1 or encrypted2 cannot be in NTT form");
    }

    // Extract encryption parameters.
    const auto& context_data = context.get_context_data(encrypted1.chain_index());
    const auto& parms = context_data.parms();
    const auto n = parms.poly_modulus_degree();
    const auto mul_tech = parms.mul_tech();

    const size_t ct1_size = encrypted1.size_;
    const size_t ct2_size = encrypted2.size_;
    const size_t dest_size = ct1_size + ct2_size - 1;
    if (dest_size != 3)
        throw std::logic_error("dest_size must be 3 when computing BFV multiplication using HPS");

    // Resize encrypted1 to destination size
    encrypted1.resize(context, encrypted1.chain_index(), dest_size);

    // HPS and HPSOverQ does not drop modulus
    uint32_t levelsDropped = 0;

    // handle no relin scenario
    size_t gpu_rns_tool_index = 0;
    if (context.using_keyswitching() == true) {
        gpu_rns_tool_index = 1;
    }

    if (mul_tech == mul_tech_type::hps_overq_leveled) {
        const size_t c1depth = encrypted1.GetNoiseScaleDeg();
        const size_t c2depth = encrypted2.GetNoiseScaleDeg();

        const bool is_Asymmetric = encrypted1.is_asymmetric();
        const size_t levels = std::max(c1depth, c2depth) - 1;
        const auto dcrtBits = static_cast<double>(context.get_context_data(gpu_rns_tool_index).gpu_rns_tool().qMSB());

        // how many levels to drop
        levelsDropped = FindLevelsToDrop(context, levels, dcrtBits, false, is_Asymmetric);
    }

    const auto& rns_tool = context.get_context_data(gpu_rns_tool_index + levelsDropped).gpu_rns_tool();
    const DModulus* base_QlRl = rns_tool.base_QlRl().base();
    const auto& gpu_QlRl_tables = rns_tool.gpu_QlRl_tables();
    const size_t size_Q = rns_tool.base_Q().size();
    const size_t size_Ql = rns_tool.base_Ql().size();
    const size_t size_Rl = rns_tool.base_Rl().size();
    const size_t size_QlRl = size_Ql + size_Rl;

    /* --------------------------------- ct1 BConv -------------------------------- */
    Pointer<uint64_t> ct1;
    ct1.acquire(allocate<uint64_t>(global_pool(), dest_size * size_QlRl * n));
    for (size_t i = 0; i < ct1_size; i++) {
        const uint64_t* encrypted1_ptr = encrypted1.data() + i * size_Q * n;
        uint64_t* ct1_ptr = ct1.get() + i * size_QlRl * n;
        uint64_t* ct1_Ql_ptr = ct1_ptr;
        uint64_t* ct1_Rl_ptr = ct1_Ql_ptr + size_Ql * n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped)
            rns_tool.scaleAndRound_HPS_Q_Ql(ct1_Ql_ptr, encrypted1_ptr);
        else
            CUDA_CHECK(
            cudaMemcpy(ct1_Ql_ptr, encrypted1_ptr, size_Ql * n * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

        rns_tool.base_Ql_to_Rl_conv().bConv_HPS(ct1_Rl_ptr, ct1_Ql_ptr, n);
    }

    if (&encrypted1 == &encrypted2) {
        // if square, no need to compute ct2
        /* --------------------------------- ct1 *= ct1 -------------------------------- */
        // forward NTT
        for (size_t i = 0; i < ct1_size; i++) {
            uint64_t* ct1_ptr = ct1.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0);
        }

        // (c0, c1, c2, ...) * (c0', c1', c2', ...)
        //    = (c0 * c0', c0*c1' + c1*c0', c0*c2'+c1*c1'+c2*c0', ...)
        uint64_t gridDimGlb = n * size_QlRl / blockDimGlb.x;
        tensor_square_2x2_rns_poly<<<gridDimGlb, blockDimGlb>>>(ct1.get(), base_QlRl, ct1.get(), n, size_QlRl);
    } else {
        /* --------------------------------- ct2 BConv -------------------------------- */
        Pointer<uint64_t> ct2;
        // allocate enough space
        ct2.acquire(allocate<uint64_t>(global_pool(), ct2_size * size_QlRl * n));
        for (size_t i = 0; i < ct2_size; i++) {
            const uint64_t* encrypted2_ptr = encrypted2.data() + i * size_Q * n;
            uint64_t* ct2_ptr = ct2.get() + i * size_QlRl * n;
            uint64_t* ct2_Ql_ptr = ct2_ptr;
            uint64_t* ct2_Rl_ptr = ct2_Ql_ptr + size_Ql * n;

            if (mul_tech == mul_tech_type::hps) {
                CUDA_CHECK(cudaMemcpy(ct2_Ql_ptr, encrypted2_ptr, size_Ql * n * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice));
                rns_tool.base_Ql_to_Rl_conv().bConv_HPS(ct2_Rl_ptr, ct2_Ql_ptr, n);
            } else if (mul_tech == mul_tech_type::hps_overq || mul_tech == mul_tech_type::hps_overq_leveled) {
                if (levelsDropped)
                    rns_tool.base_Q_to_Rl_conv().bConv_BEHZ_var1(ct2_Rl_ptr, encrypted2_ptr, n);
                else
                    rns_tool.base_Ql_to_Rl_conv().bConv_BEHZ_var1(ct2_Rl_ptr, encrypted2_ptr, n);
                rns_tool.base_Rl_to_Ql_conv().bConv_HPS(ct2_Ql_ptr, ct2_Rl_ptr, n);
            }
        }

        /* --------------------------------- ct1 *= ct2 -------------------------------- */
        // forward NTT
        for (size_t i = 0; i < ct1_size; i++) {
            uint64_t* ct1_ptr = ct1.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0);
        }

        for (size_t i = 0; i < ct2_size; i++) {
            uint64_t* ct2_ptr = ct2.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct2_ptr, gpu_QlRl_tables, size_QlRl, 0);
        }

        // (c0, c1, c2, ...) * (c0', c1', c2', ...)
        //    = (c0 * c0', c0*c1' + c1*c0', c0*c2'+c1*c1'+c2*c0', ...)
        uint64_t gridDimGlb = n * size_QlRl / blockDimGlb.x;
        tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb>>>(ct1.get(), ct2.get(), base_QlRl, ct1.get(), n, size_QlRl);
    }

    // inverse NTT
    for (size_t i = 0; i < dest_size; i++) {
        uint64_t* ct1_ptr = ct1.get() + i * size_QlRl * n;
        nwt_2d_radix8_backward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0);
    }

    /* --------------------------------- ct1 BConv -------------------------------- */
    // scale and round
    for (size_t i = 0; i < dest_size; i++) {
        uint64_t* encrypted1_ptr = encrypted1.data() + i * size_Q * n;
        const uint64_t* ct1_ptr = ct1.get() + i * size_QlRl * n;
        if (mul_tech == mul_tech_type::hps) {
            Pointer<uint64_t> temp;
            temp.acquire(allocate<uint64_t>(global_pool(), size_Rl * n));
            // scale and round QlRl to Rl
            rns_tool.scaleAndRound_HPS_QR_R(temp.get(), ct1_ptr);
            // Rl -> Ql
            rns_tool.base_Rl_to_Ql_conv().bConv_HPS(encrypted1_ptr, temp.get(), n);
            temp.release();
        } else if (mul_tech == mul_tech_type::hps_overq || mul_tech == mul_tech_type::hps_overq_leveled) {
            // scale and round QlRl to Ql
            rns_tool.scaleAndRound_HPS_QlRl_Ql(encrypted1_ptr, ct1_ptr);

            if (levelsDropped) {
                rns_tool.ExpandCRTBasis_Ql_Q(encrypted1_ptr, encrypted1_ptr);
            }
        }
    }

    if (mul_tech == mul_tech_type::hps_overq_leveled) {
        encrypted1.SetNoiseScaleDeg(std::max(encrypted1.GetNoiseScaleDeg(), encrypted2.GetNoiseScaleDeg()) + 1);
    }
}

// encrypted1 = encrypted1 * encrypted2
// (c0, c1) * (c0', c1') = (c0*c0', c0'c1+c0c1', c1c1')
void bfv_multiply(const PhantomContext& context, PhantomCiphertext& encrypted1, const PhantomCiphertext& encrypted2) {
    auto mul_tech = context.mul_tech();
    if (mul_tech == mul_tech_type::behz) {
        bfv_multiply_behz(context, encrypted1, encrypted2);
    } else if (mul_tech == mul_tech_type::hps || mul_tech == mul_tech_type::hps_overq ||
               mul_tech == mul_tech_type::hps_overq_leveled) {
        bfv_multiply_hps(context, encrypted1, encrypted2);
    } else {
        throw invalid_argument("mul_tech not supported for bfv_multiply");
    }
}

void bfv_mul_relin_hps(const PhantomContext& context, PhantomCiphertext& encrypted1,
                       const PhantomCiphertext& encrypted2, const PhantomRelinKey& relin_keys) {
    if (encrypted1.is_ntt_form() || encrypted2.is_ntt_form()) {
        throw std::invalid_argument("encrypted1 or encrypted2 cannot be in NTT form");
    }

    // Extract encryption parameters.
    const auto& context_data = context.get_context_data(encrypted1.chain_index());
    const auto& parms = context_data.parms();
    const auto n = parms.poly_modulus_degree();
    const auto mul_tech = parms.mul_tech();

    const size_t ct1_size = encrypted1.size_;
    const size_t ct2_size = encrypted2.size_;
    const size_t dest_size = ct1_size + ct2_size - 1;
    if (dest_size != 3)
        throw std::logic_error("dest_size must be 3 when computing BFV multiplication using HPS");

    // Resize encrypted1 to destination size
    encrypted1.resize(context, encrypted1.chain_index(), dest_size);

    // HPS and HPSOverQ does not drop modulus
    uint32_t levelsDropped = 0;

    if (mul_tech == mul_tech_type::hps_overq_leveled) {
        const size_t c1depth = encrypted1.GetNoiseScaleDeg();
        const size_t c2depth = encrypted2.GetNoiseScaleDeg();

        const bool is_Asymmetric = encrypted1.is_asymmetric();
        const size_t levels = std::max(c1depth, c2depth) - 1;
        const auto dcrtBits = static_cast<double>(context.get_context_data(1).gpu_rns_tool().qMSB());

        // how many levels to drop
        levelsDropped = FindLevelsToDrop(context, levels, dcrtBits, false, is_Asymmetric);
    }

    const auto& rns_tool = context.get_context_data(1 + levelsDropped).gpu_rns_tool();
    const DModulus* base_QlRl = rns_tool.base_QlRl().base();
    const auto& gpu_QlRl_tables = rns_tool.gpu_QlRl_tables();
    const size_t size_Q = rns_tool.base_Q().size();
    const size_t size_Ql = rns_tool.base_Ql().size();
    const size_t size_Rl = rns_tool.base_Rl().size();
    const size_t size_QlRl = size_Ql + size_Rl;

    /* --------------------------------- ct1 BConv -------------------------------- */
    Pointer<uint64_t> ct1;
    ct1.acquire(allocate<uint64_t>(global_pool(), dest_size * size_QlRl * n));
    for (size_t i = 0; i < ct1_size; i++) {
        const uint64_t* encrypted1_ptr = encrypted1.data() + i * size_Q * n;
        uint64_t* ct1_ptr = ct1.get() + i * size_QlRl * n;
        uint64_t* ct1_Ql_ptr = ct1_ptr;
        uint64_t* ct1_Rl_ptr = ct1_Ql_ptr + size_Ql * n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped)
            rns_tool.scaleAndRound_HPS_Q_Ql(ct1_Ql_ptr, encrypted1_ptr);
        else
            CUDA_CHECK(
            cudaMemcpy(ct1_Ql_ptr, encrypted1_ptr, size_Ql * n * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

        rns_tool.base_Ql_to_Rl_conv().bConv_HPS(ct1_Rl_ptr, ct1_Ql_ptr, n);
    }

    if (&encrypted1 == &encrypted2) {
        // square
        /* --------------------------------- ct1 *= ct1 -------------------------------- */
        // forward NTT
        for (size_t i = 0; i < ct1_size; i++) {
            uint64_t* ct1_ptr = ct1.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0);
        }

        // (c0, c1, c2, ...) * (c0', c1', c2', ...)
        //    = (c0 * c0', c0*c1' + c1*c0', c0*c2'+c1*c1'+c2*c0', ...)
        uint64_t gridDimGlb = n * size_QlRl / blockDimGlb.x;
        tensor_square_2x2_rns_poly<<<gridDimGlb, blockDimGlb>>>(ct1.get(), base_QlRl, ct1.get(), n, size_QlRl);
    } else {
        /* --------------------------------- ct2 BConv -------------------------------- */
        Pointer<uint64_t> ct2;
        // allocate enough space
        ct2.acquire(allocate<uint64_t>(global_pool(), ct2_size * size_QlRl * n));
        for (size_t i = 0; i < ct2_size; i++) {
            const uint64_t* encrypted2_ptr = encrypted2.data() + i * size_Q * n;
            uint64_t* ct2_ptr = ct2.get() + i * size_QlRl * n;
            uint64_t* ct2_Ql_ptr = ct2_ptr;
            uint64_t* ct2_Rl_ptr = ct2_Ql_ptr + size_Ql * n;

            if (mul_tech == mul_tech_type::hps) {
                CUDA_CHECK(cudaMemcpy(ct2_Ql_ptr, encrypted2_ptr, size_Ql * n * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice));
                rns_tool.base_Ql_to_Rl_conv().bConv_HPS(ct2_Rl_ptr, ct2_Ql_ptr, n);
            } else if (mul_tech == mul_tech_type::hps_overq || mul_tech == mul_tech_type::hps_overq_leveled) {
                if (levelsDropped)
                    rns_tool.base_Q_to_Rl_conv().bConv_BEHZ_var1(ct2_Rl_ptr, encrypted2_ptr, n);
                else
                    rns_tool.base_Ql_to_Rl_conv().bConv_BEHZ_var1(ct2_Rl_ptr, encrypted2_ptr, n);
                rns_tool.base_Rl_to_Ql_conv().bConv_HPS(ct2_Ql_ptr, ct2_Rl_ptr, n);
            }
        }

        /* --------------------------------- ct1 *= ct2 -------------------------------- */
        // forward NTT
        for (size_t i = 0; i < ct1_size; i++) {
            uint64_t* ct1_ptr = ct1.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0);
        }

        for (size_t i = 0; i < ct2_size; i++) {
            uint64_t* ct2_ptr = ct2.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct2_ptr, gpu_QlRl_tables, size_QlRl, 0);
        }

        // (c0, c1, c2, ...) * (c0', c1', c2', ...)
        //    = (c0 * c0', c0*c1' + c1*c0', c0*c2'+c1*c1'+c2*c0', ...)
        uint64_t gridDimGlb = n * size_QlRl / blockDimGlb.x;
        tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb>>>(ct1.get(), ct2.get(), base_QlRl, ct1.get(), n, size_QlRl);
    }

    // inverse NTT
    for (size_t i = 0; i < dest_size; i++) {
        uint64_t* ct1_ptr = ct1.get() + i * size_QlRl * n;
        nwt_2d_radix8_backward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0);
    }

    /* --------------------------------- ct1 BConv -------------------------------- */
    // scale and round
    for (size_t i = 0; i < dest_size; i++) {
        uint64_t* encrypted1_ptr = encrypted1.data() + i * size_Q * n;
        const uint64_t* ct1_ptr = ct1.get() + i * size_QlRl * n;
        if (mul_tech == mul_tech_type::hps) {
            Pointer<uint64_t> temp;
            temp.acquire(allocate<uint64_t>(global_pool(), size_Rl * n));
            // scale and round QlRl to Rl
            rns_tool.scaleAndRound_HPS_QR_R(temp.get(), ct1_ptr);
            // Rl -> Ql
            rns_tool.base_Rl_to_Ql_conv().bConv_HPS(encrypted1_ptr, temp.get(), n);
        } else if (mul_tech == mul_tech_type::hps_overq || mul_tech == mul_tech_type::hps_overq_leveled) {
            // scale and round QlRl to Ql
            rns_tool.scaleAndRound_HPS_QlRl_Ql(encrypted1_ptr, ct1_ptr);

            if (levelsDropped && i != dest_size - 1) {
                rns_tool.ExpandCRTBasis_Ql_Q(encrypted1_ptr, encrypted1_ptr);
            }
        }
    }

    if (mul_tech == mul_tech_type::hps_overq_leveled) {
        encrypted1.SetNoiseScaleDeg(std::max(encrypted1.GetNoiseScaleDeg(), encrypted2.GetNoiseScaleDeg()) + 1);
    }

    // Extract encryption parameters.
    const size_t decomp_modulus_size = parms.coeff_modulus().size();
    const auto& key_vector = relin_keys.public_keys_;
    const auto key_component_count = key_vector[0].pk_.size_;
    const auto scheme = parms.scheme();

    // Verify parameters.
    if (encrypted1.size() != 3) {
        throw invalid_argument("destination_size must be 3");
    }
    if (scheme == scheme_type::bfv && encrypted1.is_ntt_form_) {
        throw invalid_argument("BFV encrypted cannot be in NTT form");
    }
    if (key_component_count != 2) {
        throw invalid_argument("destination_size must be equal to key_component_count");
    }

    // only c2 is not scale&round to Ql
    const uint64_t* c2 = encrypted1.data() + 2 * size_Q * n;

    // Extract encryption parameters.
    const auto& key_context_data = context.get_context_data(0);
    const auto& key_parms = key_context_data.parms();
    const auto& key_modulus = key_parms.coeff_modulus();
    const auto& modulus_QP = context.gpu_rns_tables().modulus();
    const size_t size_P = key_parms.special_modulus_size();
    const size_t size_QP = key_modulus.size();
    const size_t size_QlP = size_Ql + size_P;
    const size_t size_Ql_n = size_Ql * n;
    const size_t size_QP_n = size_QP * n;
    const size_t size_QlP_n = size_QlP * n;
    const size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    // mod up
    Pointer<uint64_t> t_mod_up;
    t_mod_up.acquire(allocate<uint64_t>(global_pool(), beta * size_QlP_n));
    rns_tool.modup(t_mod_up.get(), c2, context.gpu_rns_tables(), scheme);

    // key switch
    Pointer<uint64_t> cx;
    cx.acquire(allocate<uint64_t>(global_pool(), 2 * size_QlP_n));
    auto reduction_threshold = (1 << (bits_per_uint64 - rns_tool.qMSB() - 1)) - 1;
    key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb>>>(
        cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr_.get(), modulus_QP, n, size_QP, size_QP_n, size_QlP,
        size_QlP_n, size_Q, size_Ql, beta, reduction_threshold);

    // mod down
    for (size_t i = 0; i < 2; i++) {
        const auto cx_i = cx.get() + i * size_QlP_n;
        rns_tool.moddown_from_NTT(cx_i, cx_i, context.gpu_rns_tables(), scheme);
    }

    for (size_t i = 0; i < 2; i++) {
        const auto cx_i = cx.get() + i * size_QlP_n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
            auto ct_i = encrypted1.data() + i * size_Q * n;
            rns_tool.ExpandCRTBasis_Ql_Q_add_to_ct(ct_i, cx_i);
        } else {
            auto ct_i = encrypted1.data() + i * size_Ql_n;
            add_to_ct_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb>>>(ct_i, cx_i, rns_tool.base_Ql().base(), n,
                                                                         size_Ql);
        }
    }

    // update the encrypted
    encrypted1.resize(key_component_count, decomp_modulus_size, n);
}

// encrypted1 = encrypted1 * encrypted2
void multiply_inplace(const PhantomContext& context, PhantomCiphertext& encrypted1,
                      const PhantomCiphertext& encrypted2) {
    // Verify parameters.
    if (encrypted1.parms_id() != encrypted2.parms_id()) {
        throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    }
    if (encrypted1.chain_index() != encrypted2.chain_index())
        throw std::invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        throw std::invalid_argument("NTT form mismatch");
    if (encrypted1.scale() != encrypted2.scale())
        throw std::invalid_argument("scale mismatch");
    if (encrypted1.size_ != encrypted2.size_)
        throw std::invalid_argument("poly number mismatch");

    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted1.chain_index());
    auto& parms = context_data.parms();

    switch (parms.scheme()) {
        case scheme_type::bfv:
            bfv_multiply(context, encrypted1, encrypted2);
            break;
        case scheme_type::ckks:
        case scheme_type::bgv:
            bgv_ckks_multiply(context, encrypted1, encrypted2);
            break;

        default:
            throw invalid_argument("unsupported scheme");
    }
}

// encrypted1 = encrypted1 * encrypted2
// relin(encrypted1)
void multiply_and_relin_inplace(const PhantomContext& context, PhantomCiphertext& encrypted1,
                                const PhantomCiphertext& encrypted2, const PhantomRelinKey& relin_keys) {
    // Verify parameters.
    if (encrypted1.parms_id() != encrypted2.parms_id()) {
        throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    }
    if (encrypted1.chain_index() != encrypted2.chain_index())
        throw std::invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        throw std::invalid_argument("NTT form mismatch");
    if (encrypted1.scale() != encrypted2.scale())
        throw std::invalid_argument("scale mismatch");
    if (encrypted1.size_ != encrypted2.size_)
        throw std::invalid_argument("poly number mismatch");

    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted1.chain_index());
    auto& parms = context_data.parms();
    auto scheme = parms.scheme();
    auto mul_tech = parms.mul_tech();

    switch (scheme) {
        case scheme_type::bfv:
            if (mul_tech == mul_tech_type::hps || mul_tech == mul_tech_type::hps_overq ||
                mul_tech == mul_tech_type::hps_overq_leveled) {
                // enable fast mul&relin
                bfv_mul_relin_hps(context, encrypted1, encrypted2, relin_keys);
            } else if (mul_tech == mul_tech_type::behz) {
                bfv_multiply_behz(context, encrypted1, encrypted2);
                relinearize_inplace(context, encrypted1, relin_keys);
            } else {
                throw invalid_argument("unsupported mul tech in BFV mul&relin");
            }
            break;

        case scheme_type::ckks:
        case scheme_type::bgv:
            bgv_ckks_multiply(context, encrypted1, encrypted2);
            relinearize_inplace(context, encrypted1, relin_keys);
            break;

        default:
            throw invalid_argument("unsupported scheme");
    }
}

void add_plain_inplace(const PhantomContext& context, PhantomCiphertext& encrypted, const PhantomPlaintext& plain) {
    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();

    if (parms.scheme() == scheme_type::bfv && encrypted.is_ntt_form()) {
        throw std::invalid_argument("BFV encrypted cannot be in NTT form");
    }
    if (parms.scheme() == scheme_type::ckks && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("CKKS encrypted must be in NTT form");
    }
    if (parms.scheme() == scheme_type::bgv && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("BGV encrypted must be in NTT form");
    }
    if (encrypted.scale() != plain.scale()) {
        // TODO: be more precious
        throw std::invalid_argument("scale mismatch");
    }

    auto& coeff_modulus = parms.coeff_modulus();
    auto coeff_mod_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto base_rns = context.gpu_rns_tables().modulus();

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;

    switch (parms.scheme()) {
        case scheme_type::bfv: {
            multiply_add_plain_with_scaling_variant(context, plain, encrypted.chain_index(), encrypted);
            break;
        }
        case scheme_type::ckks: {
            // (c0 + pt, c1)
            add_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted.data(), plain.data(), base_rns, encrypted.data(),
                                                      poly_degree, coeff_mod_size);
            break;
        }
        case scheme_type::bgv: {
            // TODO: make bgv plaintext is_ntt_form true?
            // c0 = c0 + plaintext
            Pointer<uint64_t> plain_copy;
            plain_copy.acquire(allocate<uint64_t>(global_pool(), coeff_mod_size * poly_degree));
            for (size_t i = 0; i < coeff_mod_size; i++) {
                // modup t -> {q0, q1, ...., qj}
                nwt_2d_radix8_forward_modup_fuse(plain_copy.get() + i * poly_degree, plain.data(), i,
                                                 context.gpu_rns_tables(), 1, 0);
            }
            // (c0 + pt, c1)
            multiply_scalar_and_add_rns_poly<<<gridDimGlb, blockDimGlb>>>(
                encrypted.data(), plain_copy.get(), encrypted.correction_factor(), base_rns, encrypted.data(),
                poly_degree, coeff_mod_size);
            break;
        }
        default:
            throw invalid_argument("unsupported scheme");
    }
}

void sub_plain_inplace(const PhantomContext& context, PhantomCiphertext& encrypted, const PhantomPlaintext& plain) {
    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();

    if (parms.scheme() == scheme_type::bfv && encrypted.is_ntt_form()) {
        throw std::invalid_argument("BFV encrypted cannot be in NTT form");
    }
    if (parms.scheme() == scheme_type::ckks && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("CKKS encrypted must be in NTT form");
    }
    if (parms.scheme() == scheme_type::bgv && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("BGV encrypted must be in NTT form");
    }
    if (encrypted.scale() != plain.scale()) {
        // TODO: be more precious
        throw std::invalid_argument("scale mismatch");
    }

    auto& coeff_modulus = parms.coeff_modulus();
    auto coeff_mod_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto base_rns = context.gpu_rns_tables().modulus();
    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;

    switch (parms.scheme()) {
        case scheme_type::bfv: {
            multiply_sub_plain_with_scaling_variant(context, plain, encrypted.chain_index(), encrypted);
            break;
        }
        case scheme_type::ckks: {
            // (c0 - pt, c1)
            sub_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted.data(), plain.data(), base_rns, encrypted.data(),
                                                      poly_degree, coeff_mod_size);
            break;
        }
        case scheme_type::bgv: {
            // TODO: make bgv plaintext is_ntt_form true?
            // c0 = c0 - plaintext
            Pointer<uint64_t> plain_copy;
            plain_copy.acquire(allocate<uint64_t>(global_pool(), coeff_mod_size * poly_degree));
            for (size_t i = 0; i < coeff_mod_size; i++) {
                // modup t -> {q0, q1, ...., qj}
                nwt_2d_radix8_forward_modup_fuse(plain_copy.get() + i * poly_degree, plain.data(), i,
                                                 context.gpu_rns_tables(), 1, 0);
            }
            // (c0 - pt, c1)
            multiply_scalar_and_sub_rns_poly<<<gridDimGlb, blockDimGlb>>>(
                encrypted.data(), plain_copy.get(), encrypted.correction_factor(), base_rns, encrypted.data(),
                poly_degree, coeff_mod_size);
            break;
        }
        default:
            throw invalid_argument("unsupported scheme");
    }
}

void multiply_plain_ntt(const PhantomContext& context, PhantomCiphertext& encrypted, const PhantomPlaintext& plain) {
    if (!plain.is_ntt_form()) {
        throw invalid_argument("plain_ntt is not in NTT form");
    }
    if (encrypted.chain_index() != plain.chain_index()) {
        throw std::invalid_argument("encrypted and plain parameter mismatch");
    }
    if (encrypted.parms_id() != plain.parms_id()) {
        throw invalid_argument("encrypted_ntt and plain_ntt parameter mismatch");
    }

    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();
    auto& coeff_modulus = parms.coeff_modulus();
    auto coeff_mod_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto base_rns = context.gpu_rns_tables().modulus();
    auto rns_coeff_count = poly_degree * coeff_mod_size;

    double new_scale = encrypted.scale() * plain.scale();

    //(c0 * pt, c1 * pt)
    for (size_t i = 0; i < encrypted.size(); i++) {
        uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
        multiply_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted.data() + i * rns_coeff_count, plain.data(), base_rns,
                                                       encrypted.data() + i * rns_coeff_count, poly_degree,
                                                       coeff_mod_size);
    }

    encrypted.scale() = new_scale;
}

void multiply_plain_normal(const PhantomContext& context, PhantomCiphertext& encrypted, const PhantomPlaintext& plain) {
    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();
    auto& coeff_modulus = parms.coeff_modulus();
    auto coeff_mod_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto rns_coeff_count = poly_degree * coeff_mod_size;
    auto base_rns = context.gpu_rns_tables().modulus();
    auto encrypted_size = encrypted.size();

    auto plain_upper_half_threshold = context_data.plain_upper_half_threshold();
    auto plain_upper_half_increment = context.plain_upper_half_increment();

    double new_scale = encrypted.scale() * plain.scale();

    /*
    !!! This optimizations is removed also due to the access of device memory in host
    Optimizations for constant / monomial multiplication can lead to the presence of a timing side-channel,
    as the processing time varies with the plaintext, and therefore leaks plaintext, which may be sensitive.
    */
    /*if (plain_nonzero_coeff_count == 1)
    {
    }*/

    uint64_t gridDimGlb = rns_coeff_count / blockDimGlb.x;
    // Generic case: any plaintext polynomial
    // Allocate temporary space for an entire RNS polynomial
    Pointer<uint64_t> temp;
    temp.acquire(allocate<uint64_t>(global_pool(), rns_coeff_count));

    // if (context_data.qualifiers().using_fast_plain_lift) {
    // if t is smaller than every qi
    abs_plain_rns_poly<<<gridDimGlb, blockDimGlb>>>(plain.data(), plain_upper_half_threshold,
                                                    plain_upper_half_increment, temp.get(), poly_degree,
                                                    coeff_mod_size);
    // }
    // else {
    //     // need to perform decompose
    //     // N-slot plain, for each slot,
    //     // temp["coeff_mod_size"] = plain[tid] + plain_upper_half_increment (i.e., q-t) when (plain[tid] >=
    //     plain_upper_half_threshold)
    //     // otherwise temp["coeff_mod_size"] = plain[tid]
    //     auto &rns_tool = context.get_context_data(encrypted.chain_index()).gpu_rns_tool();
    //     auto &base_q = rns_tool.base_Ql_;
    //     // each block (num is N) of coeff_mod_size size, is decomposed into N * coeff_mod_size data
    //     base_q.decompose_array(temp.get(), plain.data(), base_rns, poly_degree, plain_upper_half_increment,
    //                            plain_upper_half_threshold);
    // }

    nwt_2d_radix8_forward_inplace(temp.get(), context.gpu_rns_tables(), coeff_mod_size, 0);

    // (c0 * pt, c1 * pt)
    for (size_t i = 0; i < encrypted_size; i++) {
        uint64_t* ci = encrypted.data() + i * rns_coeff_count;
        // NTT
        nwt_2d_radix8_forward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0);
        // Pointwise multiplication
        multiply_rns_poly<<<gridDimGlb, blockDimGlb>>>(ci, temp.get(), base_rns, ci, poly_degree, coeff_mod_size);
        // inverse NTT
        nwt_2d_radix8_backward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0);
    }

    encrypted.scale() = new_scale;
}

void multiply_plain_inplace(const PhantomContext& context, PhantomCiphertext& encrypted,
                            const PhantomPlaintext& plain) {
    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();
    auto scheme = parms.scheme();

    if (scheme == scheme_type::bfv) {
        multiply_plain_normal(context, encrypted, plain);
    } else if (scheme == scheme_type::ckks) {
        multiply_plain_ntt(context, encrypted, plain);
    } else if (scheme == scheme_type::bgv) {
        // Extract encryption parameters.
        auto& coeff_modulus = parms.coeff_modulus();
        auto coeff_mod_size = coeff_modulus.size();
        auto poly_degree = parms.poly_modulus_degree();
        auto base_rns = context.gpu_rns_tables().modulus();
        auto rns_coeff_count = poly_degree * coeff_mod_size;

        Pointer<uint64_t> plain_copy;
        plain_copy.acquire(allocate<uint64_t>(global_pool(), coeff_mod_size * poly_degree));
        for (size_t i = 0; i < coeff_mod_size; i++) {
            // modup t -> {q0, q1, ...., qj}
            nwt_2d_radix8_forward_modup_fuse(plain_copy.get() + i * poly_degree, plain.data(), i,
                                             context.gpu_rns_tables(), 1, 0);
        }

        double new_scale = encrypted.scale() * plain.scale();

        //(c0 * pt, c1 * pt)
        for (size_t i = 0; i < encrypted.size(); i++) {
            uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
            multiply_rns_poly<<<gridDimGlb, blockDimGlb>>>(encrypted.data() + i * rns_coeff_count, plain_copy.get(),
                                                           base_rns, encrypted.data() + i * rns_coeff_count,
                                                           poly_degree, coeff_mod_size);
        }

        encrypted.scale() = new_scale;
    } else {
        throw std::invalid_argument("unsupported scheme");
    }
}

void transform_to_ntt_inplace(const PhantomContext& context, PhantomCiphertext& encrypted) {
    if (encrypted.is_ntt_form()) {
        throw invalid_argument("encrypted is already in NTT form");
    }

    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();
    size_t poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = parms.coeff_modulus().size();

    for (size_t i = 0; i < encrypted.size(); i++) {
        // Transform each polynomial to NTT domain
        auto ci = encrypted.data() + i * coeff_mod_size * poly_degree;
        nwt_2d_radix8_forward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0);
    }

    // Finally change the is_ntt_transformed flag
    encrypted.is_ntt_form() = true;
}

void transform_from_ntt_inplace(const PhantomContext& context, PhantomCiphertext& encrypted_ntt) {
    if (!encrypted_ntt.is_ntt_form()) {
        throw invalid_argument("encrypted_ntt is not in NTT form");
    }

    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted_ntt.chain_index());
    auto& parms = context_data.parms();
    size_t poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = parms.coeff_modulus().size();

    for (size_t i = 0; i < encrypted_ntt.size(); i++) {
        // Transform each polynomial to NTT domain
        auto ci = encrypted_ntt.data() + i * coeff_mod_size * poly_degree;
        nwt_2d_radix8_backward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0);
    }

    // Finally change the is_ntt_transformed flag
    encrypted_ntt.is_ntt_form() = false;
}

void relinearize_inplace(const PhantomContext& context, PhantomCiphertext& encrypted,
                         const PhantomRelinKey& relin_keys) {
    // Extract encryption parameters.
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();
    size_t decomp_modulus_size = parms.coeff_modulus().size();
    size_t n = parms.poly_modulus_degree();
    auto& key_vector = relin_keys.public_keys_;
    auto key_component_count = key_vector[0].pk_.size_;

    // Verify parameters.
    auto scheme = parms.scheme();
    auto encrypted_size = encrypted.size_;
    if (encrypted_size != 3) {
        throw invalid_argument("destination_size must be 3");
    }
    if (scheme == scheme_type::bfv && encrypted.is_ntt_form_) {
        throw invalid_argument("BFV encrypted cannot be in NTT form");
    }
    if (scheme == scheme_type::ckks && !encrypted.is_ntt_form_) {
        throw invalid_argument("CKKS encrypted must be in NTT form");
    }
    if (scheme == scheme_type::bgv && !encrypted.is_ntt_form_) {
        throw invalid_argument("BGV encrypted must be in NTT form");
    }
    if (key_component_count != 2) {
        throw invalid_argument("destination_size must be equal to key_component_count");
    }

    uint64_t* c2 = encrypted.data() + 2 * decomp_modulus_size * n;
    switch_key_inplace(context, encrypted, c2, relin_keys, true);

    // update the encrypted
    encrypted.resize(key_component_count, decomp_modulus_size, n);
}

void rescale_to_next(const PhantomContext& context, const PhantomCiphertext& encrypted,
                     PhantomCiphertext& destination) {
    auto& context_data = context.get_context_data(context.get_first_index());
    auto& parms = context_data.parms();
    auto max_chain_index = parms.coeff_modulus().size();
    auto scheme = parms.scheme();

    // Verify parameters.
    if (encrypted.chain_index() == max_chain_index) {
        throw invalid_argument("end of modulus switching chain reached");
    }

    switch (scheme) {
        case scheme_type::bfv:
            throw invalid_argument("unsupported operation for scheme type");

        case scheme_type::ckks:
            // Modulus switching with scaling
            mod_switch_scale_to_next(context, encrypted, destination);
            break;

        default:
            throw invalid_argument("unsupported scheme");
    }
}

void mod_switch_to_inplace(const PhantomContext& context, PhantomPlaintext& plain, size_t chain_index) {
    if (!plain.is_ntt_form()) {
        throw invalid_argument("plain is not in NTT form");
    }
    if (plain.chain_index() > chain_index) {
        throw invalid_argument("cannot switch to higher level modulus");
    }

    while (plain.chain_index() != chain_index) {
        mod_switch_to_next_inplace(context, plain);
    }
}

void mod_switch_to_next_inplace(const PhantomContext& context, PhantomPlaintext& plain) {
    auto& context_data = context.get_context_data(context.get_first_index());
    auto& parms = context_data.parms();
    auto coeff_modulus_size = parms.coeff_modulus().size();

    auto max_chain_index = coeff_modulus_size;
    if (!plain.is_ntt_form()) {
        throw invalid_argument("plain is not in NTT form");
    }
    if (plain.chain_index() == max_chain_index) {
        throw invalid_argument("end of modulus switching chain reached");
    }

    auto next_chain_index = plain.chain_index() + 1;
    auto& next_context_data = context.get_context_data(next_chain_index);
    auto& next_parms = next_context_data.parms();

    // q_1,...,q_{k-1}
    auto& next_coeff_modulus = next_parms.coeff_modulus();
    size_t next_coeff_modulus_size = next_coeff_modulus.size();
    size_t coeff_count = next_parms.poly_modulus_degree();

    // Compute destination size first for exception safety
    auto dest_size = next_coeff_modulus_size * coeff_count;

    Pointer<uint64_t> data_copy;
    data_copy.acquire(plain.data_);
    plain.data_.acquire(allocate<uint64_t>(global_pool(), dest_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(plain.data(), data_copy.get(), dest_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

    plain.chain_index() = next_chain_index;
}

void mod_switch_to_inplace(const PhantomContext& context, PhantomCiphertext& encrypted, size_t chain_index) {
    if (encrypted.chain_index() > chain_index) {
        throw invalid_argument("cannot switch to higher level modulus");
    }

    while (encrypted.chain_index() != chain_index) {
        mod_switch_to_next_inplace(context, encrypted);
    }
}

void mod_switch_to_next(const PhantomContext& context, const PhantomCiphertext& encrypted,
                        PhantomCiphertext& destination) {
    // Assuming at this point encrypted is already validated.
    auto& context_data = context.get_context_data(context.get_first_index());
    auto& parms = context_data.parms();
    auto coeff_modulus_size = parms.coeff_modulus().size();
    auto scheme = parms.scheme();

    auto max_chain_index = coeff_modulus_size;
    if (encrypted.chain_index() == max_chain_index) {
        throw invalid_argument("end of modulus switching chain reached");
    }
    if (parms.scheme() == scheme_type::bfv && encrypted.is_ntt_form()) {
        throw std::invalid_argument("BFV encrypted cannot be in NTT form");
    }
    if (parms.scheme() == scheme_type::ckks && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("CKKS encrypted must be in NTT form");
    }

    switch (scheme) {
        case scheme_type::bfv:
            // Modulus switching with scaling
            mod_switch_scale_to_next(context, encrypted, destination);
            break;

        case scheme_type::ckks:
            // Modulus switching without scaling
            mod_switch_drop_to_next(context, encrypted, destination);
            break;

        default:
            throw invalid_argument("unsupported scheme");
    }
}

void mod_switch_drop_to_next(const PhantomContext& context, const PhantomCiphertext& encrypted,
                             PhantomCiphertext& destination) {
    // Assuming at this point encrypted is already validated.
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();
    auto coeff_modulus_size = parms.coeff_modulus().size();
    size_t N = parms.poly_modulus_degree();

    // Extract encryption parameters.
    auto next_chain_index = encrypted.chain_index() + 1;
    auto& next_context_data = context.get_context_data(next_chain_index);
    auto& next_parms = next_context_data.parms();

    // q_1,...,q_{k-1}
    size_t encrypted_size = encrypted.size();
    size_t next_coeff_modulus_size = next_parms.coeff_modulus().size();

    if (&encrypted == &destination) {
        Pointer<uint64_t> temp;
        temp.acquire(destination.data_);
        destination.data_.acquire(allocate<uint64_t>(global_pool(), encrypted_size * next_coeff_modulus_size * N));
        for (size_t i{0}; i < encrypted_size; i++) {
            auto temp_iter = temp.get() + i * coeff_modulus_size * N;
            auto encrypted_iter = encrypted.data() + i * next_coeff_modulus_size * N;
            CUDA_CHECK(cudaMemcpy(encrypted_iter, temp_iter, next_coeff_modulus_size * N * sizeof(uint64_t),
                cudaMemcpyDeviceToDevice));
        }
        // Set other attributes
        destination.chain_index() = next_chain_index;
        destination.coeff_modulus_size_ = next_coeff_modulus_size;
    } else {
        // Resize destination before writing
        destination.resize(context, next_chain_index, encrypted_size);
        // Copy data over to destination; only copy the RNS components relevant after modulus drop
        for (size_t i = 0; i < encrypted_size; i++) {
            auto destination_iter = destination.data() + i * next_coeff_modulus_size * N;
            auto encrypted_iter = encrypted.data() + i * coeff_modulus_size * N;
            CUDA_CHECK(cudaMemcpy(destination_iter, encrypted_iter, next_coeff_modulus_size * N * sizeof(uint64_t),
                cudaMemcpyDeviceToDevice));
        }
        // Set other attributes
        destination.scale() = encrypted.scale();
        destination.is_ntt_form() = encrypted.is_ntt_form();
    }
}

void mod_switch_scale_to_next(const PhantomContext& context, const PhantomCiphertext& encrypted,
                              PhantomCiphertext& destination) {
    // Assuming at this point encrypted is already validated.
    auto& context_data = context.get_context_data(encrypted.chain_index());
    auto& parms = context_data.parms();
    auto& rns_tool = context.get_context_data(encrypted.chain_index()).gpu_rns_tool();

    // Extract encryption parameters.
    size_t coeff_mod_size = parms.coeff_modulus().size();
    size_t poly_degree = parms.poly_modulus_degree();
    size_t encrypted_size = encrypted.size();

    auto next_index_id = context.get_next_index(encrypted.chain_index());
    auto& next_context_data = context.get_context_data(next_index_id);
    auto& next_parms = next_context_data.parms();

    //    size_t next_coeff_modulus_size = next_parms.coeff_modulus().size();

    Pointer<uint64_t> encrypted_copy;
    encrypted_copy.acquire(allocate<uint64_t>(global_pool(), encrypted_size * coeff_mod_size * poly_degree));
    CUDA_CHECK(cudaMemcpy(encrypted_copy.get(), encrypted.data(),
        encrypted_size * coeff_mod_size * poly_degree * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    // resize and empty the data array
    destination.resize(context, next_index_id, encrypted_size);

    switch (next_parms.scheme()) {
        case scheme_type::bfv:
            rns_tool.divide_and_round_q_last(encrypted_copy.get(), encrypted_size, destination.data());
            break;

        case scheme_type::ckks:
            rns_tool.divide_and_round_q_last_ntt(encrypted_copy.get(), encrypted_size, context.gpu_rns_tables(),
                                                 destination.data());
            break;

        default:
            throw invalid_argument("unsupported scheme");
    }

    // Set other attributes
    destination.is_ntt_form() = encrypted.is_ntt_form();
    if (next_parms.scheme() == scheme_type::ckks) {
        // Change the scale when using CKKS
        destination.scale() = encrypted.scale() / static_cast<double>(parms.coeff_modulus().back().value());
    }
}

void rotate_internal(const PhantomContext& context, PhantomCiphertext& encrypted, int step,
                     const PhantomGaloisKey& galois_key) {
    auto& context_data = context.get_context_data(encrypted.chain_index_);

    // Is there anything to do?
    if (step == 0) {
        return;
    }

    size_t coeff_count = context_data.parms().poly_modulus_degree();
    auto& key_galois_tool = context.key_galois_tool_;
    auto& galois_elts = key_galois_tool->galois_elts_;
    auto step_galois_elt = key_galois_tool->get_elt_from_step(step);

    auto iter = find(galois_elts.begin(), galois_elts.end(), step_galois_elt);
    if (iter != galois_elts.end()) {
        auto galois_elt_index = iter - galois_elts.begin();
        // Perform rotation and key switching
        apply_galois_inplace(context, encrypted, galois_elt_index, galois_key);
    } else {
        // Convert the steps to NAF: guarantees using smallest HW
        vector<int> naf_step = naf(step);

        // If naf_steps contains only one element, then this is a power-of-two
        // rotation and we would have expected not to get to this part of the
        // if-statement.
        if (naf_step.size() == 1) {
            throw invalid_argument("Galois key not present");
        }
        for (auto temp_step: naf_step) {
            if (static_cast<size_t>(abs(temp_step)) != (coeff_count >> 1)) {
                rotate_internal(context, encrypted, temp_step, galois_key);
            }
        }
    }
}

void hoisting_inplace(const PhantomContext& context, PhantomCiphertext& ct, const PhantomGaloisKey& glk,
                      const std::vector<int>& steps) {
    if (ct.size() > 2)
        throw invalid_argument("ciphertext size must be 2");

    auto& context_data = context.get_context_data(ct.chain_index_);
    auto& key_context_data = context.get_context_data(0);
    auto& key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto& key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();

    // HPS and HPSOverQ does not drop modulus
    uint32_t levelsDropped;

    if (scheme == scheme_type::bfv) {
        levelsDropped = 0;
        if (mul_tech == mul_tech_type::hps_overq_leveled) {
            size_t depth = ct.GetNoiseScaleDeg();
            bool isKeySwitch = true;
            bool is_Asymmetric = ct.is_asymmetric();
            size_t levels = depth - 1;
            auto dcrtBits = static_cast<double>(context.get_context_data(1).gpu_rns_tool().qMSB());

            // how many levels to drop
            levelsDropped = FindLevelsToDrop(context, levels, dcrtBits, isKeySwitch, is_Asymmetric);
        }
    } else if (scheme == scheme_type::bgv || scheme == scheme_type::ckks) {
        levelsDropped = ct.chain_index() - 1;
    } else {
        throw invalid_argument("unsupported scheme in switch_key_inplace");
    }

    auto& rns_tool = context.get_context_data(1 + levelsDropped).gpu_rns_tool();
    auto& parms = context_data.parms();
    auto& key_galois_tool = context.key_galois_tool_;
    auto& galois_elts = key_galois_tool->galois_elts_;

    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;

    auto size_Q_n = size_Q * n;
    auto size_Ql_n = size_Ql * n;
    auto size_QP_n = size_QP * n;
    auto size_QlP_n = size_QlP * n;

    Pointer<uint64_t> c0;
    c0.acquire(allocate<uint64_t>(global_pool(), size_Ql_n));

    Pointer<uint64_t> c1;
    c1.acquire(allocate<uint64_t>(global_pool(), size_Ql_n));

    auto elts = key_galois_tool->get_elts_from_steps(steps);

    // ------------------------------------------ automorphism c0 ------------------------------------------------------

    // specific operations for HPSOverQLeveled
    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        rns_tool.scaleAndRound_HPS_Q_Ql(c0.get(), ct.data());
    } else {
        CUDA_CHECK(cudaMemcpy(c0.get(), ct.data(), size_Ql_n * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    }

    Pointer<uint64_t> acc_c0;
    acc_c0.acquire(allocate<uint64_t>(global_pool(), size_Ql_n));

    auto first_elt = elts[0];
    auto first_iter = find(galois_elts.begin(), galois_elts.end(), first_elt);
    if (first_iter == galois_elts.end())
        throw std::logic_error("Galois key not present in hoisting");
    auto first_elt_index = first_iter - galois_elts.begin();

    if (parms.scheme() == scheme_type::bfv) {
        key_galois_tool->apply_galois(c0.get(), context.gpu_rns_tables(), size_Ql, first_elt_index, acc_c0.get());
    } else if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv) {
        key_galois_tool->apply_galois_ntt(c0.get(), size_Ql, first_elt_index, acc_c0.get());
    } else {
        throw logic_error("scheme not implemented");
    }

    // ----------------------------------------------- modup c1 --------------------------------------------------------

    // specific operations for HPSOverQLeveled
    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        rns_tool.scaleAndRound_HPS_Q_Ql(c1.get(), ct.data() + size_Q_n);
    } else {
        CUDA_CHECK(cudaMemcpy(c1.get(), ct.data() + size_Ql_n, size_Ql_n * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    }

    // Prepare key
    auto& key_vector = glk.relin_keys_[first_elt_index].public_keys_;
    auto key_poly_num = key_vector[0].pk_.size_;
    if (key_poly_num != 2)
        throw std::invalid_argument("key_poly_num must be 2 in hoisting");

    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    // mod up
    Pointer<uint64_t> modup_c1;
    modup_c1.acquire(allocate<uint64_t>(global_pool(), beta * size_QlP_n));
    rns_tool.modup(modup_c1.get(), c1.get(), context.gpu_rns_tables(), scheme);

    // ------------------------------------------ automorphism c1 ------------------------------------------------------

    Pointer<uint64_t> temp_modup_c1;
    temp_modup_c1.acquire(allocate<uint64_t>(global_pool(), beta * size_QlP_n));

    for (size_t b = 0; b < beta; b++) {
        key_galois_tool->apply_galois_ntt(modup_c1.get() + b * size_QlP_n, size_QlP, first_elt_index,
                                          temp_modup_c1.get() + b * size_QlP_n);
    }

    // ----------------------------------------- inner product c1 ------------------------------------------------------

    Pointer<uint64_t> acc_cx;
    acc_cx.acquire(allocate<uint64_t>(global_pool(), 2 * size_QlP_n));

    auto reduction_threshold =
            (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;
    key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb>>>(
        acc_cx.get(), temp_modup_c1.get(), glk.relin_keys_[first_elt_index].public_keys_ptr_.get(), modulus_QP, n,
        size_QP, size_QP_n, size_QlP, size_QlP_n, size_Q, size_Ql, beta, reduction_threshold);

    // ------------------------------------------ loop accumulate ------------------------------------------------------
    Pointer<uint64_t> temp_c0;
    temp_c0.acquire(allocate<uint64_t>(global_pool(), size_Ql_n));

    for (size_t i = 1; i < elts.size(); i++) {
        // automorphism c0

        auto elt = elts[i];
        auto iter = find(galois_elts.begin(), galois_elts.end(), elt);
        if (iter == galois_elts.end())
            throw std::logic_error("Galois key not present in hoisting");
        auto elt_index = iter - galois_elts.begin();

        if (parms.scheme() == scheme_type::bfv) {
            key_galois_tool->apply_galois(c0.get(), context.gpu_rns_tables(), size_Ql, elt_index, temp_c0.get());
        } else if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv) {
            key_galois_tool->apply_galois_ntt(c0.get(), size_Ql, elt_index, temp_c0.get());
        } else {
            throw logic_error("scheme not implemented");
        }

        // add to acc_c0
        uint64_t gridDimGlb = size_Ql_n / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb>>>(acc_c0.get(), temp_c0.get(), rns_tool.base_Ql().base(), acc_c0.get(),
                                                  n, size_Ql);

        // automorphism c1

        for (size_t b = 0; b < beta; b++) {
            key_galois_tool->apply_galois_ntt(modup_c1.get() + b * size_QlP_n, size_QlP, elt_index,
                                              temp_modup_c1.get() + b * size_QlP_n);
        }

        // inner product c1

        Pointer<uint64_t> temp_cx;
        temp_cx.acquire(allocate<uint64_t>(global_pool(), 2 * size_QlP_n));

        key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb>>>(
            temp_cx.get(), temp_modup_c1.get(), glk.relin_keys_[elt_index].public_keys_ptr_.get(), modulus_QP, n,
            size_QP, size_QP_n, size_QlP, size_QlP_n, size_Q, size_Ql, beta, reduction_threshold);

        // add to acc_cx
        gridDimGlb = size_QlP_n / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb>>>(acc_cx.get(), temp_cx.get(), rns_tool.base_QlP().base(), acc_cx.get(),
                                                  n, size_QlP);
        add_rns_poly<<<gridDimGlb, blockDimGlb>>>(acc_cx.get() + size_QlP_n, temp_cx.get() + size_QlP_n,
                                                  rns_tool.base_QlP().base(), acc_cx.get() + size_QlP_n, n, size_QlP);
    }

    // -------------------------------------------- mod down c1 --------------------------------------------------------
    rns_tool.moddown_from_NTT(acc_cx.get(), acc_cx.get(), context.gpu_rns_tables(), scheme);
    rns_tool.moddown_from_NTT(acc_cx.get() + size_QlP_n, acc_cx.get() + size_QlP_n, context.gpu_rns_tables(), scheme);

    // new c0
    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        add_rns_poly<<<size_Ql_n / blockDimGlb.x, blockDimGlb>>>(acc_c0.get(), acc_cx.get(), rns_tool.base_Ql().base(),
                                                                 acc_cx.get(), n, size_Ql);
        rns_tool.ExpandCRTBasis_Ql_Q(ct.data(), acc_cx.get());
    } else {
        add_rns_poly<<<size_Ql_n / blockDimGlb.x, blockDimGlb>>>(acc_c0.get(), acc_cx.get(), rns_tool.base_Ql().base(),
                                                                 ct.data(), n, size_Ql);
    }

    // new c1
    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        rns_tool.ExpandCRTBasis_Ql_Q(ct.data() + size_Q_n, acc_cx.get() + size_QlP_n);
    } else {
        CUDA_CHECK(cudaMemcpy(ct.data() + size_Ql_n, acc_cx.get() + size_QlP_n, size_Ql_n * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice));
    }
}

void apply_galois_inplace(const PhantomContext& context, PhantomCiphertext& encrypted, size_t galois_elt_index,
                          const PhantomGaloisKey& galois_keys) {
    auto& context_data = context.get_context_data(encrypted.chain_index_);
    auto& parms = context_data.parms();
    auto& coeff_modulus = parms.coeff_modulus();
    size_t N = parms.poly_modulus_degree();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t encrypted_size = encrypted.size();
    if (encrypted_size > 2) {
        throw invalid_argument("encrypted size must be 2");
    }
    auto c0 = encrypted.data();
    auto c1 = encrypted.data() + encrypted.coeff_modulus_size_ * encrypted.poly_modulus_degree_;
    // Use key_context_data where permutation tables exist since previous runs.
    auto& key_galois_tool = context.key_galois_tool_;

    Pointer<uint64_t> temp;
    temp.acquire(allocate<uint64_t>(global_pool(), coeff_modulus_size * N));

    // DO NOT CHANGE EXECUTION ORDER OF FOLLOWING SECTION
    // BEGIN: Apply Galois for each ciphertext
    // Execution order is sensitive, since apply_galois is not inplace!
    if (parms.scheme() == scheme_type::bfv) {
        // !!! DO NOT CHANGE EXECUTION ORDER!!!
        // First transform c0
        key_galois_tool->apply_galois(c0, context.gpu_rns_tables(), coeff_modulus_size, galois_elt_index, temp.get());
        // Copy result to c0
        CUDA_CHECK(cudaMemcpy(c0, temp.get(), coeff_modulus_size * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        // Next transform c1
        key_galois_tool->apply_galois(c1, context.gpu_rns_tables(), coeff_modulus_size, galois_elt_index, temp.get());
    } else if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv) {
        // !!! DO NOT CHANGE EXECUTION ORDER!!
        // First transform c0
        key_galois_tool->apply_galois_ntt(c0, coeff_modulus_size, galois_elt_index, temp.get());
        // Copy result to c0
        CUDA_CHECK(cudaMemcpy(c0, temp.get(), coeff_modulus_size * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        // Next transform c1
        key_galois_tool->apply_galois_ntt(c1, coeff_modulus_size, galois_elt_index, temp.get());
    } else {
        throw logic_error("scheme not implemented");
    }

    // Wipe c1
    CUDA_CHECK(cudaMemset(c1, 0, coeff_modulus_size * N * sizeof(uint64_t)));

    // END: Apply Galois for each ciphertext
    // REORDERING IS SAFE NOW
    // Calculate (temp * galois_key[0], temp * galois_key[1]) + (c0, 0)
    switch_key_inplace(context, encrypted, temp.get(), galois_keys.relin_keys_[galois_elt_index]);
}

void conjugate_internal(const PhantomContext& context, PhantomCiphertext& encrypted,
                        const PhantomGaloisKey& galois_key) {
    constexpr size_t galois_elt_index = 0;
    apply_galois_inplace(context, encrypted, galois_elt_index, galois_key);
}

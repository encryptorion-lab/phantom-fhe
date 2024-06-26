#include "evaluate.cuh"

#include "rns_bconv.cuh"
#include "scalingvariant.cuh"
#include "util.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

/**
Returns (f, e1, e2) such that
(1) e1 * factor1 = e2 * factor2 = f mod p;
(2) gcd(e1, p) = 1 and gcd(e2, p) = 1;
(3) abs(e1_bal) + abs(e2_bal) is minimal, where e1_bal and e2_bal represent e1 and e2 in (-p/2, p/2].
*/
[[nodiscard]] static auto balance_correction_factors(uint64_t factor1, uint64_t factor2,
                                                     const phantom::arith::Modulus &plain_modulus)
-> std::tuple<uint64_t, uint64_t, uint64_t> {
    uint64_t t = plain_modulus.value();
    uint64_t half_t = t / 2;

    auto sum_abs = [&](uint64_t x, uint64_t y) {
        int64_t x_bal = static_cast<int64_t>(x > half_t ? x - t : x);
        int64_t y_bal = static_cast<int64_t>(y > half_t ? y - t : y);
        return abs(x_bal) + abs(y_bal);
    };

    // ratio = f2 / f1 mod p
    uint64_t ratio = 1;
    if (!phantom::arith::try_invert_uint_mod(factor1, plain_modulus, ratio)) {
        throw std::logic_error("invalid correction factor1");
    }
    ratio = phantom::arith::multiply_uint_mod(ratio, factor2, plain_modulus);
    uint64_t e1 = ratio;
    uint64_t e2 = 1;
    int64_t sum = sum_abs(e1, e2);

    // Extended Euclidean
    auto prev_a = static_cast<int64_t>(plain_modulus.value());
    auto prev_b = static_cast<int64_t>(0);
    auto a = static_cast<int64_t>(ratio);
    int64_t b = 1;

    while (a != 0) {
        int64_t q = prev_a / a;
        int64_t temp = prev_a % a;
        prev_a = a;
        a = temp;

        temp = phantom::arith::sub_safe(prev_b, phantom::arith::mul_safe(b, q));
        prev_b = b;
        b = temp;

        uint64_t a_mod = phantom::arith::barrett_reduce_64(static_cast<uint64_t>(abs(a)), plain_modulus);
        if (a < 0) {
            a_mod = phantom::arith::negate_uint_mod(a_mod, plain_modulus);
        }
        uint64_t b_mod = phantom::arith::barrett_reduce_64(static_cast<uint64_t>(abs(b)), plain_modulus);
        if (b < 0) {
            b_mod = phantom::arith::negate_uint_mod(b_mod, plain_modulus);
        }
        if (a_mod != 0 && phantom::arith::gcd(a_mod, t) == 1) // which also implies gcd(b_mod, t) == 1
        {
            int64_t new_sum = sum_abs(a_mod, b_mod);
            if (new_sum < sum) {
                sum = new_sum;
                e1 = a_mod;
                e2 = b_mod;
            }
        }
    }
    return std::make_tuple(phantom::arith::multiply_uint_mod(e1, factor1, plain_modulus), e1, e2);
}

// https://github.com/microsoft/SEAL/blob/3a05febe18e8a096668cd82c75190255eda5ca7d/native/src/seal/evaluator.cpp#L24
template<typename T, typename S>
[[nodiscard]] inline bool are_same_scale(const T &value1, const S &value2) noexcept {
    return are_close<double>(value1.scale(), value2.scale());
}

static void negate_internal(const PhantomContext &context, PhantomCiphertext &encrypted, const cudaStream_t &stream) {
    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    const auto coeff_mod_size = coeff_modulus.size();
    const auto poly_degree = parms.poly_modulus_degree();
    const auto base_rns = context.gpu_rns_tables().modulus();
    const auto rns_coeff_count = poly_degree * coeff_mod_size;

    uint64_t gridDimGlb = rns_coeff_count / blockDimGlb.x;
    for (size_t i = 0; i < encrypted.size(); i++) {
        negate_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                encrypted.data() + i * rns_coeff_count, base_rns,
                encrypted.data() + i * rns_coeff_count,
                poly_degree,
                coeff_mod_size);
    }
}

void negate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                    const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    negate_internal(context, encrypted, stream_wrapper.get_stream());
}

/**
 * Adds two ciphertexts. This function adds together encrypted1 and encrypted2 and stores the result in encrypted1.
 * @param[in] encrypted1 The first ciphertext to add
 * @param[in] encrypted2 The second ciphertext to add
 */
void add_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                 const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    if (encrypted1.chain_index() != encrypted2.chain_index()) {
        throw std::invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    }
    if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form()) {
        throw std::invalid_argument("NTT form mismatch");
    }
    if (!are_same_scale(encrypted1, encrypted2)) {
        throw std::invalid_argument("scale mismatch");
    }
    if (encrypted1.size() != encrypted2.size()) {
        throw std::invalid_argument("poly number mismatch");
    }

    const auto &s = stream_wrapper.get_stream();

    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted1.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &plain_modulus = parms.plain_modulus();
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
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted1.data() + i * rns_coeff_count, get<1>(factors), base_rns,
                    encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }

        PhantomCiphertext encrypted2_copy = encrypted2;
        for (size_t i = 0; i < encrypted2.size(); i++) {
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted2_copy.data() + i * rns_coeff_count, get<2>(factors), base_rns,
                    encrypted2_copy.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }

        // Set new correction factor
        encrypted1.set_correction_factor(get<0>(factors));
        encrypted2_copy.set_correction_factor(get<0>(factors));

        // Prepare destination
        encrypted1.resize(context, context_data.chain_index(), max_size, s);
        for (size_t i = 0; i < min_size; i++) {
            add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted1.data() + i * rns_coeff_count, encrypted2_copy.data() + i * rns_coeff_count, base_rns,
                    encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }
        if (encrypted1_size < encrypted2_size) {
            cudaMemcpyAsync(encrypted1.data() + min_size * rns_coeff_count,
                            encrypted2_copy.data() + min_size * rns_coeff_count,
                            (encrypted2_size - encrypted1_size) * rns_coeff_count * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, s);
        }
    } else {
        // Prepare destination
        encrypted1.resize(context, context_data.chain_index(), max_size, s);
        for (size_t i = 0; i < min_size; i++) {
            add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted1.data() + i * rns_coeff_count, encrypted2.data() + i * rns_coeff_count, base_rns,
                    encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }
        if (encrypted1_size < encrypted2_size) {
            cudaMemcpyAsync(encrypted1.data() + min_size * rns_coeff_count,
                            encrypted2.data() + min_size * rns_coeff_count,
                            (encrypted2_size - encrypted1_size) * rns_coeff_count * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, s);
        }
    }
}

// TODO: fixme
void add_many(const PhantomContext &context, const vector<PhantomCiphertext> &encrypteds,
              PhantomCiphertext &destination, const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    const auto &s = stream_wrapper.get_stream();

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
        if (!are_same_scale(encrypteds[0], encrypteds[i])) {
            throw std::invalid_argument("scale mismatch");
        }
        if (encrypteds[0].size() != encrypteds[i].size()) {
            throw std::invalid_argument("poly number mismatch");
        }
    }

    auto &context_data = context.get_context_data(encrypteds[0].chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto coeff_mod_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto poly_num = encrypteds[0].size();
    auto base_rns = context.gpu_rns_tables().modulus();
    // reduction_threshold = 2 ^ (64 - max modulus bits)
    // max modulus bits = static_cast<uint64_t>(log2(coeff_modulus.front().value())) + 1
    uint64_t reduction_threshold =
            (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(coeff_modulus.front().value())) - 1)) - 1;

    destination.resize(context, encrypteds[0].chain_index(), encrypteds[0].size(), s);
    destination.set_ntt_form(encrypteds[0].is_ntt_form());
    destination.set_scale(encrypteds[0].scale());

    if (parms.scheme() == scheme_type::bgv) // TODO: any optimizations?
    {
        cudaMemcpyAsync(destination.data(), encrypteds[0].data(),
                        poly_degree * coeff_mod_size * encrypteds[0].size() * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, s);
        for (size_t i = 1; i < encrypteds.size(); i++) {
            add_inplace(context, destination, encrypteds[i], stream_wrapper);
        }
    } else {
        auto enc_device_ptr = make_cuda_auto_ptr<uint64_t *>(encrypteds.size(), s);
        std::vector<uint64_t *> enc_host_ptr(encrypteds.size());
        for (size_t i = 0; i < encrypteds.size(); i++) {
            enc_host_ptr[i] = encrypteds[i].data();
        }
        cudaMemcpyAsync(enc_device_ptr.get(), enc_host_ptr.data(), sizeof(uint64_t *) * encrypteds.size(),
                        cudaMemcpyHostToDevice, s);

        uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
        for (size_t i = 0; i < poly_num; i++) {
            add_many_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    enc_device_ptr.get(), encrypteds.size(), base_rns,
                    destination.data(), i, poly_degree, coeff_mod_size,
                    reduction_threshold);
        }
    }
}

void sub_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                 const bool &negate, const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    if (encrypted1.parms_id() != encrypted2.parms_id()) {
        throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    }
    if (encrypted1.chain_index() != encrypted2.chain_index())
        throw std::invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        throw std::invalid_argument("NTT form mismatch");
    if (!are_same_scale(encrypted1, encrypted2))
        throw std::invalid_argument("scale mismatch");
    if (encrypted1.size() != encrypted2.size())
        throw std::invalid_argument("poly number mismatch");

    const auto &s = stream_wrapper.get_stream();

    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted1.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &plain_modulus = parms.plain_modulus();
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
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted1.data() + i * rns_coeff_count, get<1>(factors), base_rns,
                    encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }

        PhantomCiphertext encrypted2_copy = encrypted2;
        for (size_t i = 0; i < encrypted2.size(); i++) {
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted2_copy.data() + i * rns_coeff_count, get<2>(factors), base_rns,
                    encrypted2_copy.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
        }

        // Set new correction factor
        encrypted1.set_correction_factor(get<0>(factors));
        encrypted2_copy.set_correction_factor(get<0>(factors));

        if (negate) {
            for (size_t i = 0; i < encrypted1.size(); i++) {
                sub_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                        encrypted2_copy.data() + i * rns_coeff_count, encrypted1.data() + i * rns_coeff_count, base_rns,
                        encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
            }
        } else {
            for (size_t i = 0; i < encrypted1.size(); i++) {
                sub_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                        encrypted1.data() + i * rns_coeff_count, encrypted2_copy.data() + i * rns_coeff_count, base_rns,
                        encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
            }
        }
    } else {
        if (negate) {
            for (size_t i = 0; i < encrypted1.size(); i++) {
                sub_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                        encrypted2.data() + i * rns_coeff_count, encrypted1.data() + i * rns_coeff_count, base_rns,
                        encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
            }
        } else {
            for (size_t i = 0; i < encrypted1.size(); i++) {
                sub_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                        encrypted1.data() + i * rns_coeff_count, encrypted2.data() + i * rns_coeff_count, base_rns,
                        encrypted1.data() + i * rns_coeff_count, poly_degree, coeff_modulus_size);
            }
        }
    }
}

static void bgv_ckks_multiply(const PhantomContext &context, PhantomCiphertext &encrypted1,
                              const PhantomCiphertext &encrypted2, const cudaStream_t &stream) {
    if (!(encrypted1.is_ntt_form() && encrypted2.is_ntt_form()))
        throw invalid_argument("encrypted1 and encrypted2 must be in NTT form");

    if (encrypted1.chain_index() != encrypted2.chain_index())
        throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");

    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted1.chain_index());
    auto &parms = context_data.parms();
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
    encrypted1.resize(context, encrypted1.chain_index(), dest_size, stream);

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;

    if (dest_size == 3) {
        if (&encrypted1 == &encrypted2) {
            // square
            tensor_square_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    encrypted1.data(), base_rns, encrypted1.data(), poly_degree, coeff_mod_size);
        } else {
            // standard multiply
            tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    encrypted1.data(), encrypted2.data(), base_rns, encrypted1.data(), poly_degree, coeff_mod_size);
        }
    } else {
        tensor_prod_mxn_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                encrypted1.data(), encrypted1_size, encrypted2.data(), encrypted2_size, base_rns, encrypted1.data(),
                dest_size, poly_degree, coeff_mod_size);
    }

    // CKKS needs to do scaling
    if (parms.scheme() == scheme_type::ckks)
        encrypted1.set_scale(encrypted1.scale() * encrypted2.scale());

    // BGV needs to update correction factor
    if (parms.scheme() == scheme_type::bgv)
        encrypted1.set_correction_factor(multiply_uint_mod(encrypted1.correction_factor(),
                                                           encrypted2.correction_factor(),
                                                           parms.plain_modulus()));
}

// Perform BEHZ steps (1)-(3) for PhantomCiphertext
// (1) Lift encrypted (initially in base q) to an extended base q U Bsk U {m_tilde}
// (2) Remove extra multiples of q from the results with Montgomery reduction, switching base to q U Bsk
// (3) Transform the data to NTT form
// @notice: temp is used to avoid memory malloc in sm_mrq
static void BEHZ_mul_1(const PhantomContext &context, const PhantomCiphertext &encrypted, uint64_t *encrypted_q,
                       uint64_t *encrypted_Bsk, const cudaStream_t &stream) {
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();
    auto poly_degree = parms.poly_modulus_degree();
    auto &rns_tool = context.get_context_data(encrypted.chain_index()).gpu_rns_tool();

    size_t base_q_size = rns_tool.base_Ql().size();
    size_t base_Bsk_size = rns_tool.base_Bsk().size();
    size_t base_Bsk_m_tilde_size = rns_tool.base_Bsk_m_tilde().size();

    size_t q_coeff_count = poly_degree * base_q_size;
    size_t bsk_coeff_count = poly_degree * base_Bsk_size;

    auto temp_base_Bsk_m_tilde = make_cuda_auto_ptr<uint64_t>(poly_degree * base_Bsk_m_tilde_size, stream);

    cudaMemcpyAsync(encrypted_q, encrypted.data(), encrypted.size() * q_coeff_count * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);

    for (size_t i = 0; i < encrypted.size(); i++) {
        uint64_t *encrypted_ptr = encrypted.data() + i * q_coeff_count;
        uint64_t *encrypted_q_ptr = encrypted_q + i * q_coeff_count;
        uint64_t *encrypted_bsk_ptr = encrypted_Bsk + i * bsk_coeff_count;
        // NTT forward
        nwt_2d_radix8_forward_inplace(encrypted_q_ptr, context.gpu_rns_tables(), base_q_size, 0, stream);
        // (1) Convert from base q to base Bsk U {m_tilde}
        rns_tool.fastbconv_m_tilde(temp_base_Bsk_m_tilde.get(), encrypted_ptr, stream);
        // (2) Reduce q-overflows in with Montgomery reduction, switching base to Bsk
        rns_tool.sm_mrq(encrypted_bsk_ptr, temp_base_Bsk_m_tilde.get(), stream);
        // NTT forward
        nwt_2d_radix8_forward_inplace_include_temp_mod(encrypted_bsk_ptr, rns_tool.gpu_Bsk_tables(), base_Bsk_size, 0,
                                                       rns_tool.gpu_Bsk_tables().size(), stream);
    }
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
static void bfv_multiply_behz(const PhantomContext &context, PhantomCiphertext &encrypted1,
                              const PhantomCiphertext &encrypted2, const cudaStream_t &stream) {
    if (encrypted1.is_ntt_form() || encrypted2.is_ntt_form()) {
        throw std::invalid_argument("encrypted1 or encrypted2 cannot be in NTT form");
    }

    // Extract encryption parameters.
    const auto &context_data = context.get_context_data(encrypted1.chain_index());
    const auto &parms = context_data.parms();
    const auto &rns_tool = context.get_context_data(encrypted1.chain_index()).gpu_rns_tool();

    const size_t poly_degree = parms.poly_modulus_degree();
    const size_t encrypted1_size = encrypted1.size();
    const size_t encrypted2_size = encrypted2.size();
    const size_t base_q_size = rns_tool.base_Ql().size();
    const size_t base_Bsk_size = rns_tool.base_Bsk().size();
    const size_t dest_size = encrypted1_size + encrypted2_size - 1;

    const DModulus *base_rns = context.gpu_rns_tables().modulus();
    const DModulus *base_Bsk = rns_tool.base_Bsk().base();

    auto encrypted1_q = make_cuda_auto_ptr<uint64_t>(dest_size * poly_degree * base_q_size, stream);
    auto encrypted1_Bsk = make_cuda_auto_ptr<uint64_t>(dest_size * poly_degree * base_Bsk_size, stream);
    auto encrypted2_q = make_cuda_auto_ptr<uint64_t>(encrypted2_size * poly_degree * base_q_size, stream);
    auto encrypted2_Bsk = make_cuda_auto_ptr<uint64_t>(encrypted2_size * poly_degree * base_Bsk_size, stream);

    // BEHZ, step 1-3
    BEHZ_mul_1(context, encrypted1, encrypted1_q.get(), encrypted1_Bsk.get(), stream);
    if (dest_size != 3 || &encrypted1 != &encrypted2)
        BEHZ_mul_1(context, encrypted2, encrypted2_q.get(), encrypted2_Bsk.get(), stream);

    uint64_t gridDimGlb;
    // BEHZ, step 4 Compute the ciphertext polynomial product using dyadic multiplication
    // (c0, c1, c2, ...) * (c0', c1', c2', ...)
    //    = (c0 * c0', c0*c1' + c1*c0', c0*c2'+c1*c1'+c2*c0', ...)
    if (dest_size == 3) {
        gridDimGlb = poly_degree * base_q_size / blockDimGlb.x;
        if (&encrypted1 == &encrypted2)
            tensor_square_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    encrypted1_q.get(), base_rns, encrypted1_q.get(),
                    poly_degree, base_q_size);

        else
            tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    encrypted1_q.get(), encrypted2_q.get(), base_rns,
                    encrypted1_q.get(), poly_degree, base_q_size);

        gridDimGlb = poly_degree * base_Bsk_size / blockDimGlb.x;
        if (&encrypted1 == &encrypted2)
            tensor_square_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    encrypted1_Bsk.get(), base_Bsk,
                    encrypted1_Bsk.get(), poly_degree, base_Bsk_size);
        else
            tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    encrypted1_Bsk.get(), encrypted2_Bsk.get(), base_Bsk,
                    encrypted1_Bsk.get(), poly_degree, base_Bsk_size);
    } else {
        gridDimGlb = poly_degree * base_q_size / blockDimGlb.x;
        tensor_prod_mxn_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                encrypted1_q.get(), encrypted1_size, encrypted2_q.get(),
                encrypted2_size, base_rns, encrypted1_q.get(), dest_size,
                poly_degree, base_q_size);

        gridDimGlb = poly_degree * base_Bsk_size / blockDimGlb.x;
        tensor_prod_mxn_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                encrypted1_Bsk.get(), encrypted1_size, encrypted2_Bsk.get(), encrypted2_size, base_Bsk,
                encrypted1_Bsk.get(), dest_size, poly_degree, base_Bsk_size);
    }

    // BEHZ, step 5: NTT backward
    // Step (6): multiply base q components by t (plain_modulus)
    for (size_t i = 0; i < dest_size; i++) {
        nwt_2d_radix8_backward_inplace_scale(encrypted1_q.get() + i * poly_degree * base_q_size,
                                             context.gpu_rns_tables(), base_q_size, 0, context.plain_modulus(),
                                             context.plain_modulus_shoup(), stream);
    }
    for (size_t i = 0; i < dest_size; i++) {
        nwt_2d_radix8_backward_inplace_include_temp_mod_scale(
                encrypted1_Bsk.get() + i * poly_degree * base_Bsk_size, rns_tool.gpu_Bsk_tables(), base_Bsk_size, 0,
                rns_tool.gpu_Bsk_tables().size(), rns_tool.tModBsk(), rns_tool.tModBsk_shoup(), stream);
    }

    // Resize encrypted1 to destination size
    encrypted1.resize(context, encrypted1.chain_index(), dest_size, stream);

    auto temp = make_cuda_auto_ptr<uint64_t>(poly_degree * base_Bsk_size, stream);
    for (size_t i = 0; i < dest_size; i++) {
        uint64_t *encrypted1_q_iter = encrypted1_q.get() + i * base_q_size * poly_degree;
        uint64_t *encrypted1_Bsk_iter = encrypted1_Bsk.get() + i * base_Bsk_size * poly_degree;
        uint64_t *encrypted1_iter = encrypted1.data() + i * base_q_size * poly_degree;
        // Step (7): divide by q and floor, producing a result(stored in encrypted2_Bsk) in base Bsk
        rns_tool.fast_floor(encrypted1_q_iter, encrypted1_Bsk_iter, temp.get(), stream);
        // encrypted1_q is used to avoid malloc in fastbconv_sk
        // Step (8): use Shenoy-Kumaresan method to convert the result to base q and write to encrypted1
        rns_tool.fastbconv_sk(temp.get(), encrypted1_iter, stream);
        // encrypted1_q is used to avoid malloc in fastbconv_sk
    }
}

size_t FindLevelsToDrop(const PhantomContext &context, size_t multiplicativeDepth, double dcrtBits, bool isKeySwitch,
                        bool isAsymmetric) {
    // Extract encryption parameters.
    auto &context_data = context.get_context_data(0);
    auto &parms = context_data.parms();
    auto n = parms.poly_modulus_degree();

    // handle no relin scenario
    size_t gpu_rns_tool_index = 0;
    if (context.using_keyswitching()) {
        gpu_rns_tool_index = 1;
    }

    auto &rns_tool = context.get_context_data(gpu_rns_tool_index).gpu_rns_tool(); // BFV does not drop modulus
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
static void
bfv_multiply_hps(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                 const cudaStream_t &stream) {

    if (encrypted1.is_ntt_form() || encrypted2.is_ntt_form()) {
        throw std::invalid_argument("encrypted1 or encrypted2 cannot be in NTT form");
    }

    // Extract encryption parameters.
    const auto &context_data = context.get_context_data(encrypted1.chain_index());
    const auto &parms = context_data.parms();
    const auto n = parms.poly_modulus_degree();
    const auto mul_tech = parms.mul_tech();

    const size_t ct1_size = encrypted1.size();
    const size_t ct2_size = encrypted2.size();
    const size_t dest_size = ct1_size + ct2_size - 1;
    if (dest_size != 3)
        throw std::logic_error("dest_size must be 3 when computing BFV multiplication using HPS");

    // Resize encrypted1 to destination size
    encrypted1.resize(context, encrypted1.chain_index(), dest_size, stream);

    // HPS and HPSOverQ does not drop modulus
    uint32_t levelsDropped = 0;

    // handle no relin scenario
    size_t gpu_rns_tool_index = 0;
    if (context.using_keyswitching()) {
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

    const auto &rns_tool = context.get_context_data(gpu_rns_tool_index + levelsDropped).gpu_rns_tool();
    const DModulus *base_QlRl = rns_tool.base_QlRl().base();
    const auto &gpu_QlRl_tables = rns_tool.gpu_QlRl_tables();
    const size_t size_Q = rns_tool.base_Q().size();
    const size_t size_Ql = rns_tool.base_Ql().size();
    const size_t size_Rl = rns_tool.base_Rl().size();
    const size_t size_QlRl = size_Ql + size_Rl;

    /* --------------------------------- ct1 BConv -------------------------------- */
    auto ct1 = make_cuda_auto_ptr<uint64_t>(dest_size * size_QlRl * n, stream);
    for (size_t i = 0; i < ct1_size; i++) {
        const uint64_t *encrypted1_ptr = encrypted1.data() + i * size_Q * n;
        uint64_t *ct1_ptr = ct1.get() + i * size_QlRl * n;
        uint64_t *ct1_Ql_ptr = ct1_ptr;
        uint64_t *ct1_Rl_ptr = ct1_Ql_ptr + size_Ql * n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped)
            rns_tool.scaleAndRound_HPS_Q_Ql(ct1_Ql_ptr, encrypted1_ptr, stream);
        else
            cudaMemcpyAsync(ct1_Ql_ptr, encrypted1_ptr, size_Ql * n * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, stream);

        rns_tool.base_Ql_to_Rl_conv().bConv_HPS(ct1_Rl_ptr, ct1_Ql_ptr, n, stream);
    }

    if (&encrypted1 == &encrypted2) {
        // if square, no need to compute ct2
        /* --------------------------------- ct1 *= ct1 -------------------------------- */
        // forward NTT
        for (size_t i = 0; i < ct1_size; i++) {
            uint64_t *ct1_ptr = ct1.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0, stream);
        }

        // (c0, c1, c2, ...) * (c0', c1', c2', ...)
        //    = (c0 * c0', c0*c1' + c1*c0', c0*c2'+c1*c1'+c2*c0', ...)
        uint64_t gridDimGlb = n * size_QlRl / blockDimGlb.x;
        tensor_square_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                ct1.get(), base_QlRl, ct1.get(), n, size_QlRl);
    } else {
        /* --------------------------------- ct2 BConv -------------------------------- */
        auto ct2 = make_cuda_auto_ptr<uint64_t>(ct2_size * size_QlRl * n, stream);
        for (size_t i = 0; i < ct2_size; i++) {
            const uint64_t *encrypted2_ptr = encrypted2.data() + i * size_Q * n;
            uint64_t *ct2_ptr = ct2.get() + i * size_QlRl * n;
            uint64_t *ct2_Ql_ptr = ct2_ptr;
            uint64_t *ct2_Rl_ptr = ct2_Ql_ptr + size_Ql * n;

            if (mul_tech == mul_tech_type::hps) {
                cudaMemcpyAsync(ct2_Ql_ptr, encrypted2_ptr, size_Ql * n * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, stream);
                rns_tool.base_Ql_to_Rl_conv().bConv_HPS(ct2_Rl_ptr, ct2_Ql_ptr, n, stream);
            } else if (mul_tech == mul_tech_type::hps_overq || mul_tech == mul_tech_type::hps_overq_leveled) {
                if (levelsDropped)
                    rns_tool.base_Q_to_Rl_conv().bConv_BEHZ_var1(ct2_Rl_ptr, encrypted2_ptr, n, stream);
                else
                    rns_tool.base_Ql_to_Rl_conv().bConv_BEHZ_var1(ct2_Rl_ptr, encrypted2_ptr, n, stream);
                rns_tool.base_Rl_to_Ql_conv().bConv_HPS(ct2_Ql_ptr, ct2_Rl_ptr, n, stream);
            }
        }

        /* --------------------------------- ct1 *= ct2 -------------------------------- */
        // forward NTT
        for (size_t i = 0; i < ct1_size; i++) {
            uint64_t *ct1_ptr = ct1.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0, stream);
        }

        for (size_t i = 0; i < ct2_size; i++) {
            uint64_t *ct2_ptr = ct2.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct2_ptr, gpu_QlRl_tables, size_QlRl, 0, stream);
        }

        // (c0, c1, c2, ...) * (c0', c1', c2', ...)
        //    = (c0 * c0', c0*c1' + c1*c0', c0*c2'+c1*c1'+c2*c0', ...)
        uint64_t gridDimGlb = n * size_QlRl / blockDimGlb.x;
        tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                ct1.get(), ct2.get(), base_QlRl, ct1.get(), n, size_QlRl);
    }

    // inverse NTT
    for (size_t i = 0; i < dest_size; i++) {
        uint64_t *ct1_ptr = ct1.get() + i * size_QlRl * n;
        nwt_2d_radix8_backward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0, stream);
    }

    /* --------------------------------- ct1 BConv -------------------------------- */
    // scale and round
    for (size_t i = 0; i < dest_size; i++) {
        uint64_t *encrypted1_ptr = encrypted1.data() + i * size_Q * n;
        const uint64_t *ct1_ptr = ct1.get() + i * size_QlRl * n;
        if (mul_tech == mul_tech_type::hps) {
            auto temp = make_cuda_auto_ptr<uint64_t>(size_Rl * n, stream);
            // scale and round QlRl to Rl
            rns_tool.scaleAndRound_HPS_QR_R(temp.get(), ct1_ptr, stream);
            // Rl -> Ql
            rns_tool.base_Rl_to_Ql_conv().bConv_HPS(encrypted1_ptr, temp.get(), n, stream);
        } else if (mul_tech == mul_tech_type::hps_overq || mul_tech == mul_tech_type::hps_overq_leveled) {
            // scale and round QlRl to Ql
            rns_tool.scaleAndRound_HPS_QlRl_Ql(encrypted1_ptr, ct1_ptr, stream);
            if (levelsDropped)
                rns_tool.ExpandCRTBasis_Ql_Q(encrypted1_ptr, encrypted1_ptr, stream);
        }
    }

    if (mul_tech == mul_tech_type::hps_overq_leveled) {
        encrypted1.SetNoiseScaleDeg(std::max(encrypted1.GetNoiseScaleDeg(), encrypted2.GetNoiseScaleDeg()) + 1);
    }
}

// encrypted1 = encrypted1 * encrypted2
// (c0, c1) * (c0', c1') = (c0*c0', c0'c1+c0c1', c1c1')
static void
bfv_multiply(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
             const cudaStream_t &stream) {
    auto mul_tech = context.mul_tech();
    if (mul_tech == mul_tech_type::behz) {
        bfv_multiply_behz(context, encrypted1, encrypted2, stream);
    } else if (mul_tech == mul_tech_type::hps || mul_tech == mul_tech_type::hps_overq ||
               mul_tech == mul_tech_type::hps_overq_leveled) {
        bfv_multiply_hps(context, encrypted1, encrypted2, stream);
    } else {
        throw invalid_argument("mul_tech not supported for bfv_multiply");
    }
}

static void
bfv_mul_relin_hps(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                  const PhantomRelinKey &relin_keys, const cudaStream_t &stream) {

    if (encrypted1.is_ntt_form() || encrypted2.is_ntt_form()) {
        throw std::invalid_argument("encrypted1 or encrypted2 cannot be in NTT form");
    }

    // Extract encryption parameters.
    const auto &context_data = context.get_context_data(encrypted1.chain_index());
    const auto &parms = context_data.parms();
    const auto n = parms.poly_modulus_degree();
    const auto mul_tech = parms.mul_tech();

    const size_t ct1_size = encrypted1.size();
    const size_t ct2_size = encrypted2.size();
    const size_t dest_size = ct1_size + ct2_size - 1;
    if (dest_size != 3)
        throw std::logic_error("dest_size must be 3 when computing BFV multiplication using HPS");

    // Resize encrypted1 to destination size
    encrypted1.resize(context, encrypted1.chain_index(), dest_size, stream);

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

    const auto &rns_tool = context.get_context_data(1 + levelsDropped).gpu_rns_tool();
    const DModulus *base_QlRl = rns_tool.base_QlRl().base();
    const auto &gpu_QlRl_tables = rns_tool.gpu_QlRl_tables();
    const size_t size_Q = rns_tool.base_Q().size();
    const size_t size_Ql = rns_tool.base_Ql().size();
    const size_t size_Rl = rns_tool.base_Rl().size();
    const size_t size_QlRl = size_Ql + size_Rl;

    /* --------------------------------- ct1 BConv -------------------------------- */
    auto ct1 = make_cuda_auto_ptr<uint64_t>(dest_size * size_QlRl * n, stream);
    for (size_t i = 0; i < ct1_size; i++) {
        const uint64_t *encrypted1_ptr = encrypted1.data() + i * size_Q * n;
        uint64_t *ct1_ptr = ct1.get() + i * size_QlRl * n;
        uint64_t *ct1_Ql_ptr = ct1_ptr;
        uint64_t *ct1_Rl_ptr = ct1_Ql_ptr + size_Ql * n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped)
            rns_tool.scaleAndRound_HPS_Q_Ql(ct1_Ql_ptr, encrypted1_ptr, stream);
        else
            cudaMemcpyAsync(ct1_Ql_ptr, encrypted1_ptr, size_Ql * n * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, stream);

        rns_tool.base_Ql_to_Rl_conv().bConv_HPS(ct1_Rl_ptr, ct1_Ql_ptr, n, stream);
    }

    if (&encrypted1 == &encrypted2) {
        // square
        /* --------------------------------- ct1 *= ct1 -------------------------------- */
        // forward NTT
        for (size_t i = 0; i < ct1_size; i++) {
            uint64_t *ct1_ptr = ct1.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0, stream);
        }

        // (c0, c1, c2, ...) * (c0', c1', c2', ...)
        //    = (c0 * c0', c0*c1' + c1*c0', c0*c2'+c1*c1'+c2*c0', ...)
        uint64_t gridDimGlb = n * size_QlRl / blockDimGlb.x;
        tensor_square_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                ct1.get(), base_QlRl, ct1.get(), n, size_QlRl);
    } else {
        /* --------------------------------- ct2 BConv -------------------------------- */
        auto ct2 = make_cuda_auto_ptr<uint64_t>(ct2_size * size_QlRl * n, stream);
        for (size_t i = 0; i < ct2_size; i++) {
            const uint64_t *encrypted2_ptr = encrypted2.data() + i * size_Q * n;
            uint64_t *ct2_ptr = ct2.get() + i * size_QlRl * n;
            uint64_t *ct2_Ql_ptr = ct2_ptr;
            uint64_t *ct2_Rl_ptr = ct2_Ql_ptr + size_Ql * n;

            if (mul_tech == mul_tech_type::hps) {
                cudaMemcpyAsync(ct2_Ql_ptr, encrypted2_ptr, size_Ql * n * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, stream);
                rns_tool.base_Ql_to_Rl_conv().bConv_HPS(ct2_Rl_ptr, ct2_Ql_ptr, n, stream);
            } else if (mul_tech == mul_tech_type::hps_overq || mul_tech == mul_tech_type::hps_overq_leveled) {
                if (levelsDropped)
                    rns_tool.base_Q_to_Rl_conv().bConv_BEHZ_var1(ct2_Rl_ptr, encrypted2_ptr, n, stream);
                else
                    rns_tool.base_Ql_to_Rl_conv().bConv_BEHZ_var1(ct2_Rl_ptr, encrypted2_ptr, n, stream);
                rns_tool.base_Rl_to_Ql_conv().bConv_HPS(ct2_Ql_ptr, ct2_Rl_ptr, n, stream);
            }
        }

        /* --------------------------------- ct1 *= ct2 -------------------------------- */
        // forward NTT
        for (size_t i = 0; i < ct1_size; i++) {
            uint64_t *ct1_ptr = ct1.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0, stream);
        }

        for (size_t i = 0; i < ct2_size; i++) {
            uint64_t *ct2_ptr = ct2.get() + i * size_QlRl * n;
            nwt_2d_radix8_forward_inplace(ct2_ptr, gpu_QlRl_tables, size_QlRl, 0, stream);
        }

        // (c0, c1, c2, ...) * (c0', c1', c2', ...)
        //    = (c0 * c0', c0*c1' + c1*c0', c0*c2'+c1*c1'+c2*c0', ...)
        uint64_t gridDimGlb = n * size_QlRl / blockDimGlb.x;
        tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                ct1.get(), ct2.get(), base_QlRl, ct1.get(), n, size_QlRl);
    }

    // inverse NTT
    for (size_t i = 0; i < dest_size; i++) {
        uint64_t *ct1_ptr = ct1.get() + i * size_QlRl * n;
        nwt_2d_radix8_backward_inplace(ct1_ptr, gpu_QlRl_tables, size_QlRl, 0, stream);
    }

    /* --------------------------------- ct1 BConv -------------------------------- */
    // scale and round
    for (size_t i = 0; i < dest_size; i++) {
        uint64_t *encrypted1_ptr = encrypted1.data() + i * size_Q * n;
        const uint64_t *ct1_ptr = ct1.get() + i * size_QlRl * n;
        if (mul_tech == mul_tech_type::hps) {
            auto temp = make_cuda_auto_ptr<uint64_t>(size_Rl * n, stream);
            // scale and round QlRl to Rl
            rns_tool.scaleAndRound_HPS_QR_R(temp.get(), ct1_ptr, stream);
            // Rl -> Ql
            rns_tool.base_Rl_to_Ql_conv().bConv_HPS(encrypted1_ptr, temp.get(), n, stream);
        } else if (mul_tech == mul_tech_type::hps_overq || mul_tech == mul_tech_type::hps_overq_leveled) {
            // scale and round QlRl to Ql
            rns_tool.scaleAndRound_HPS_QlRl_Ql(encrypted1_ptr, ct1_ptr, stream);
            if (levelsDropped && i != dest_size - 1)
                rns_tool.ExpandCRTBasis_Ql_Q(encrypted1_ptr, encrypted1_ptr, stream);
        }
    }

    if (mul_tech == mul_tech_type::hps_overq_leveled) {
        encrypted1.SetNoiseScaleDeg(std::max(encrypted1.GetNoiseScaleDeg(), encrypted2.GetNoiseScaleDeg()) + 1);
    }

    // Extract encryption parameters.
    const size_t decomp_modulus_size = parms.coeff_modulus().size();
    const auto scheme = parms.scheme();

    // Verify parameters.
    if (encrypted1.size() != 3) {
        throw invalid_argument("destination_size must be 3");
    }
    if (scheme == scheme_type::bfv && encrypted1.is_ntt_form()) {
        throw invalid_argument("BFV encrypted cannot be in NTT form");
    }

    // only c2 is not scale&round to Ql
    const uint64_t *c2 = encrypted1.data() + 2 * size_Q * n;

    // Extract encryption parameters.
    const auto &key_context_data = context.get_context_data(0);
    const auto &key_parms = key_context_data.parms();
    const auto &key_modulus = key_parms.coeff_modulus();
    const auto &modulus_QP = context.gpu_rns_tables().modulus();
    const size_t size_P = key_parms.special_modulus_size();
    const size_t size_QP = key_modulus.size();
    const size_t size_QlP = size_Ql + size_P;
    const size_t size_Ql_n = size_Ql * n;
    const size_t size_QP_n = size_QP * n;
    const size_t size_QlP_n = size_QlP * n;
    const size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    // mod up
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, stream);
    rns_tool.modup(t_mod_up.get(), c2, context.gpu_rns_tables(), scheme, stream);

    // key switch
    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, stream);
    auto reduction_threshold = (1 << (bits_per_uint64 - rns_tool.qMSB() - 1)) - 1;
    key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(
            cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr(), modulus_QP, n, size_QP, size_QP_n, size_QlP,
            size_QlP_n, size_Q, size_Ql, beta, reduction_threshold);

    // mod down
    for (size_t i = 0; i < 2; i++) {
        const auto cx_i = cx.get() + i * size_QlP_n;
        rns_tool.moddown_from_NTT(cx_i, cx_i, context.gpu_rns_tables(), scheme, stream);
    }

    for (size_t i = 0; i < 2; i++) {
        const auto cx_i = cx.get() + i * size_QlP_n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
            auto ct_i = encrypted1.data() + i * size_Q * n;
            rns_tool.ExpandCRTBasis_Ql_Q_add_to_ct(ct_i, cx_i, stream);
        } else {
            auto ct_i = encrypted1.data() + i * size_Ql_n;
            add_to_ct_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                    ct_i, cx_i, rns_tool.base_Ql().base(), n, size_Ql);
        }
    }

    // update the encrypted
    encrypted1.resize(2, decomp_modulus_size, n, stream);
}

// encrypted1 = encrypted1 * encrypted2
void multiply_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                      const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    // Verify parameters.
    if (encrypted1.parms_id() != encrypted2.parms_id()) {
        throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    }
    if (encrypted1.chain_index() != encrypted2.chain_index())
        throw std::invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        throw std::invalid_argument("NTT form mismatch");
    if (!are_same_scale(encrypted1, encrypted2))
        throw std::invalid_argument("scale mismatch");
    if (encrypted1.size() != encrypted2.size())
        throw std::invalid_argument("poly number mismatch");

    const auto &s = stream_wrapper.get_stream();
    auto &context_data = context.get_context_data(encrypted1.chain_index());

    switch (context_data.parms().scheme()) {
        case scheme_type::bfv:
            bfv_multiply(context, encrypted1, encrypted2, s);
            break;
        case scheme_type::ckks:
        case scheme_type::bgv:
            bgv_ckks_multiply(context, encrypted1, encrypted2, s);
            break;

        default:
            throw invalid_argument("unsupported scheme");
    }
}

// encrypted1 = encrypted1 * encrypted2
// relin(encrypted1)
void multiply_and_relin_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1,
                                const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys,
                                const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    // Verify parameters.
    if (encrypted1.parms_id() != encrypted2.parms_id()) {
        throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    }
    if (encrypted1.chain_index() != encrypted2.chain_index())
        throw std::invalid_argument("encrypted1 and encrypted2 parameter mismatch");
    if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        throw std::invalid_argument("NTT form mismatch");
    if (!are_same_scale(encrypted1, encrypted2))
        throw std::invalid_argument("scale mismatch");
    if (encrypted1.size() != encrypted2.size())
        throw std::invalid_argument("poly number mismatch");

    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted1.chain_index());
    auto &params = context_data.parms();
    auto scheme = params.scheme();
    auto mul_tech = params.mul_tech();

    const auto &s = stream_wrapper.get_stream();

    switch (scheme) {
        case scheme_type::bfv:
            if (mul_tech == mul_tech_type::hps || mul_tech == mul_tech_type::hps_overq ||
                mul_tech == mul_tech_type::hps_overq_leveled) {
                // enable fast mul&relin
                bfv_mul_relin_hps(context, encrypted1, encrypted2, relin_keys, s);
            } else if (mul_tech == mul_tech_type::behz) {
                bfv_multiply_behz(context, encrypted1, encrypted2, s);
                relinearize_inplace(context, encrypted1, relin_keys, stream_wrapper);
            } else {
                throw invalid_argument("unsupported mul tech in BFV mul&relin");
            }
            break;

        case scheme_type::ckks:
        case scheme_type::bgv:
            bgv_ckks_multiply(context, encrypted1, encrypted2, s);
            relinearize_inplace(context, encrypted1, relin_keys, stream_wrapper);
            break;

        default:
            throw invalid_argument("unsupported scheme");
    }
}

void add_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                       const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    const auto &s = stream_wrapper.get_stream();

    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();

    if (parms.scheme() == scheme_type::bfv && encrypted.is_ntt_form()) {
        throw std::invalid_argument("BFV encrypted cannot be in NTT form");
    }
    if (parms.scheme() == scheme_type::ckks && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("CKKS encrypted must be in NTT form");
    }
    if (parms.scheme() == scheme_type::bgv && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("BGV encrypted must be in NTT form");
    }
    if (!are_same_scale(encrypted, plain)) {
        // TODO: be more precious
        throw std::invalid_argument("scale mismatch");
    }

    auto &coeff_modulus = parms.coeff_modulus();
    auto coeff_mod_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto base_rns = context.gpu_rns_tables().modulus();

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;

    switch (parms.scheme()) {
        case scheme_type::bfv: {
            multiply_add_plain_with_scaling_variant(context, plain, encrypted.chain_index(), encrypted, s);
            break;
        }
        case scheme_type::ckks: {
            // (c0 + pt, c1)
            add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted.data(), plain.data(), base_rns, encrypted.data(),
                    poly_degree, coeff_mod_size);
            break;
        }
        case scheme_type::bgv: {
            // TODO: make bgv plaintext is_ntt_form true?
            // c0 = c0 + plaintext
            auto plain_copy = make_cuda_auto_ptr<uint64_t>(coeff_mod_size * poly_degree, s);
            for (size_t i = 0; i < coeff_mod_size; i++) {
                // modup t -> {q0, q1, ...., qj}
                nwt_2d_radix8_forward_modup_fuse(plain_copy.get() + i * poly_degree, plain.data(), i,
                                                 context.gpu_rns_tables(), 1, 0, s);
            }
            // (c0 + pt, c1)
            multiply_scalar_and_add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted.data(), plain_copy.get(), encrypted.correction_factor(), base_rns, encrypted.data(),
                    poly_degree, coeff_mod_size);
            break;
        }
        default:
            throw invalid_argument("unsupported scheme");
    }
}

void sub_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                       const phantom::util::cuda_stream_wrapper &stream_wrapper) {

    const auto &s = stream_wrapper.get_stream();
    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();

    if (parms.scheme() == scheme_type::bfv && encrypted.is_ntt_form()) {
        throw std::invalid_argument("BFV encrypted cannot be in NTT form");
    }
    if (parms.scheme() == scheme_type::ckks && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("CKKS encrypted must be in NTT form");
    }
    if (parms.scheme() == scheme_type::bgv && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("BGV encrypted must be in NTT form");
    }
    if (!are_same_scale(encrypted, plain)) {
        throw std::invalid_argument("scale mismatch");
    }

    auto &coeff_modulus = parms.coeff_modulus();
    auto coeff_mod_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto base_rns = context.gpu_rns_tables().modulus();
    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;

    switch (parms.scheme()) {
        case scheme_type::bfv: {
            multiply_sub_plain_with_scaling_variant(context, plain, encrypted.chain_index(), encrypted, s);
            break;
        }
        case scheme_type::ckks: {
            // (c0 - pt, c1)
            sub_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted.data(), plain.data(), base_rns, encrypted.data(),
                    poly_degree, coeff_mod_size);
            break;
        }
        case scheme_type::bgv: {
            // TODO: make bgv plaintext is_ntt_form true?
            // c0 = c0 - plaintext
            auto plain_copy = make_cuda_auto_ptr<uint64_t>(coeff_mod_size * poly_degree, s);
            for (size_t i = 0; i < coeff_mod_size; i++) {
                // modup t -> {q0, q1, ...., qj}
                nwt_2d_radix8_forward_modup_fuse(plain_copy.get() + i * poly_degree, plain.data(), i,
                                                 context.gpu_rns_tables(), 1, 0, s);
            }
            // (c0 - pt, c1)
            multiply_scalar_and_sub_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted.data(), plain_copy.get(), encrypted.correction_factor(), base_rns, encrypted.data(),
                    poly_degree, coeff_mod_size);
            break;
        }
        default:
            throw invalid_argument("unsupported scheme");
    }
}

static void multiply_plain_ntt(const PhantomContext &context, PhantomCiphertext &encrypted,
                               const PhantomPlaintext &plain, const cudaStream_t &stream) {
    if (encrypted.chain_index() != plain.chain_index()) {
        throw std::invalid_argument("encrypted and plain parameter mismatch");
    }
    if (encrypted.parms_id() != plain.parms_id()) {
        throw invalid_argument("encrypted_ntt and plain_ntt parameter mismatch");
    }

    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto coeff_mod_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto base_rns = context.gpu_rns_tables().modulus();
    auto rns_coeff_count = poly_degree * coeff_mod_size;

    double new_scale = encrypted.scale() * plain.scale();

    //(c0 * pt, c1 * pt)
    for (size_t i = 0; i < encrypted.size(); i++) {
        uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
        multiply_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                encrypted.data() + i * rns_coeff_count, plain.data(), base_rns,
                encrypted.data() + i * rns_coeff_count, poly_degree, coeff_mod_size);
    }

    encrypted.set_scale(new_scale);
}

static void multiply_plain_normal(const PhantomContext &context, PhantomCiphertext &encrypted,
                                  const PhantomPlaintext &plain, const cudaStream_t &stream) {
    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto coeff_mod_size = coeff_modulus.size();
    auto poly_degree = parms.poly_modulus_degree();
    auto rns_coeff_count = poly_degree * coeff_mod_size;
    auto base_rns = context.gpu_rns_tables().modulus();
    auto encrypted_size = encrypted.size();

    auto plain_upper_half_threshold = context_data.plain_upper_half_threshold();
    auto plain_upper_half_increment = context.plain_upper_half_increment();

    double new_scale = encrypted.scale() * plain.scale();

    uint64_t gridDimGlb = rns_coeff_count / blockDimGlb.x;
    // Generic case: any plaintext polynomial
    // Allocate temporary space for an entire RNS polynomial
    auto temp = make_cuda_auto_ptr<uint64_t>(rns_coeff_count, stream);

    // if (context_data.qualifiers().using_fast_plain_lift) {
    // if t is smaller than every qi
    abs_plain_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            plain.data(), plain_upper_half_threshold, plain_upper_half_increment, temp.get(), poly_degree,
            coeff_mod_size);

    nwt_2d_radix8_forward_inplace(temp.get(), context.gpu_rns_tables(), coeff_mod_size, 0, stream);

    // (c0 * pt, c1 * pt)
    for (size_t i = 0; i < encrypted_size; i++) {
        uint64_t *ci = encrypted.data() + i * rns_coeff_count;
        // NTT
        nwt_2d_radix8_forward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0, stream);
        // Pointwise multiplication
        multiply_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                ci, temp.get(), base_rns, ci, poly_degree, coeff_mod_size);
        // inverse NTT
        nwt_2d_radix8_backward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0, stream);
    }

    encrypted.set_scale(new_scale);
}

void multiply_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                            const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    const auto &s = stream_wrapper.get_stream();

    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();
    auto scheme = parms.scheme();

    if (scheme == scheme_type::bfv) {
        multiply_plain_normal(context, encrypted, plain, s);
    } else if (scheme == scheme_type::ckks) {
        multiply_plain_ntt(context, encrypted, plain, s);
    } else if (scheme == scheme_type::bgv) {
        // Extract encryption parameters.
        auto &coeff_modulus = parms.coeff_modulus();
        auto coeff_mod_size = coeff_modulus.size();
        auto poly_degree = parms.poly_modulus_degree();
        auto base_rns = context.gpu_rns_tables().modulus();
        auto rns_coeff_count = poly_degree * coeff_mod_size;

        auto plain_copy = make_cuda_auto_ptr<uint64_t>(coeff_mod_size * poly_degree, s);
        for (size_t i = 0; i < coeff_mod_size; i++) {
            // modup t -> {q0, q1, ...., qj}
            nwt_2d_radix8_forward_modup_fuse(plain_copy.get() + i * poly_degree, plain.data(), i,
                                             context.gpu_rns_tables(), 1, 0, s);
        }

        double new_scale = encrypted.scale() * plain.scale();

        //(c0 * pt, c1 * pt)
        for (size_t i = 0; i < encrypted.size(); i++) {
            uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
            multiply_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    encrypted.data() + i * rns_coeff_count, plain_copy.get(),
                    base_rns, encrypted.data() + i * rns_coeff_count,
                    poly_degree, coeff_mod_size);
        }

        encrypted.set_scale(new_scale);
    } else {
        throw std::invalid_argument("unsupported scheme");
    }
}

void relinearize_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                         const PhantomRelinKey &relin_keys, const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    // Extract encryption parameters.
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();
    size_t decomp_modulus_size = parms.coeff_modulus().size();
    size_t n = parms.poly_modulus_degree();

    // Verify parameters.
    auto scheme = parms.scheme();
    auto encrypted_size = encrypted.size();
    if (encrypted_size != 3) {
        throw invalid_argument("destination_size must be 3");
    }
    if (scheme == scheme_type::bfv && encrypted.is_ntt_form()) {
        throw invalid_argument("BFV encrypted cannot be in NTT form");
    }
    if (scheme == scheme_type::ckks && !encrypted.is_ntt_form()) {
        throw invalid_argument("CKKS encrypted must be in NTT form");
    }
    if (scheme == scheme_type::bgv && !encrypted.is_ntt_form()) {
        throw invalid_argument("BGV encrypted must be in NTT form");
    }

    uint64_t *c2 = encrypted.data() + 2 * decomp_modulus_size * n;

    const auto &s = stream_wrapper.get_stream();

    keyswitch_inplace(context, encrypted, c2, relin_keys, true, s);

    // update the encrypted
    encrypted.resize(2, decomp_modulus_size, n, s);
}

static void mod_switch_scale_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                     PhantomCiphertext &destination, const cudaStream_t &stream) {
    // Assuming at this point encrypted is already validated.
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();
    auto &rns_tool = context_data.gpu_rns_tool();

    // Extract encryption parameters.
    size_t coeff_mod_size = parms.coeff_modulus().size();
    size_t poly_degree = parms.poly_modulus_degree();
    size_t encrypted_size = encrypted.size();

    auto next_index_id = context.get_next_index(encrypted.chain_index());
    auto &next_context_data = context.get_context_data(next_index_id);
    auto &next_parms = next_context_data.parms();

    auto encrypted_copy = make_cuda_auto_ptr<uint64_t>(encrypted_size * coeff_mod_size * poly_degree, stream);
    cudaMemcpyAsync(encrypted_copy.get(), encrypted.data(),
                    encrypted_size * coeff_mod_size * poly_degree * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);

    // resize and empty the data array
    destination.resize(context, next_index_id, encrypted_size, stream);

    switch (next_parms.scheme()) {
        case scheme_type::bfv:
            rns_tool.divide_and_round_q_last(encrypted_copy.get(), encrypted_size, destination.data(), stream);
            break;
        case scheme_type::bgv:
            rns_tool.mod_t_and_divide_q_last_ntt(encrypted_copy.get(), encrypted_size, context.gpu_rns_tables(),
                                                 destination.data(), stream);
            break;
        case scheme_type::ckks:
            rns_tool.divide_and_round_q_last_ntt(encrypted_copy.get(), encrypted_size, context.gpu_rns_tables(),
                                                 destination.data(), stream);
            break;
        default:
            throw invalid_argument("unsupported scheme");
    }

    // Set other attributes
    destination.set_ntt_form(encrypted.is_ntt_form());
    if (next_parms.scheme() == scheme_type::ckks) {
        // Change the scale when using CKKS
        destination.set_scale(encrypted.scale() / static_cast<double>(parms.coeff_modulus().back().value()));
    } else if (next_parms.scheme() == scheme_type::bgv) {
        // Change the correction factor when using BGV
        destination.set_correction_factor(
                multiply_uint_mod(encrypted.correction_factor(), rns_tool.inv_q_last_mod_t(),
                                  next_parms.plain_modulus()));
    }
}

static void mod_switch_drop_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                    PhantomCiphertext &destination, const cudaStream_t &stream) {
    // Assuming at this point encrypted is already validated.
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();
    auto coeff_modulus_size = parms.coeff_modulus().size();
    size_t N = parms.poly_modulus_degree();

    // Extract encryption parameters.
    auto next_chain_index = encrypted.chain_index() + 1;
    auto &next_context_data = context.get_context_data(next_chain_index);
    auto &next_parms = next_context_data.parms();

    // q_1,...,q_{k-1}
    size_t encrypted_size = encrypted.size();
    size_t next_coeff_modulus_size = next_parms.coeff_modulus().size();

    if (&encrypted == &destination) {
        auto temp = std::move(destination.data_ptr());
        destination.data_ptr() = make_cuda_auto_ptr<uint64_t>(encrypted_size * next_coeff_modulus_size * N, stream);
        for (size_t i{0}; i < encrypted_size; i++) {
            auto temp_iter = temp.get() + i * coeff_modulus_size * N;
            auto encrypted_iter = encrypted.data() + i * next_coeff_modulus_size * N;
            cudaMemcpyAsync(encrypted_iter, temp_iter, next_coeff_modulus_size * N * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, stream);
        }
        // Set other attributes
        destination.set_chain_index(next_chain_index);
        destination.set_coeff_modulus_size(next_coeff_modulus_size);
    } else {
        // Resize destination before writing
        destination.resize(context, next_chain_index, encrypted_size, stream);
        // Copy data over to destination; only copy the RNS components relevant after modulus drop
        for (size_t i = 0; i < encrypted_size; i++) {
            auto destination_iter = destination.data() + i * next_coeff_modulus_size * N;
            auto encrypted_iter = encrypted.data() + i * coeff_modulus_size * N;
            cudaMemcpyAsync(destination_iter, encrypted_iter, next_coeff_modulus_size * N * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, stream);
        }
        // Set other attributes
        destination.set_scale(encrypted.scale());
        destination.set_ntt_form(encrypted.is_ntt_form());
    }
}

void mod_switch_to_next_inplace(const PhantomContext &context, PhantomPlaintext &plain,
                                const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    const auto &s = stream_wrapper.get_stream();

    auto &context_data = context.get_context_data(context.get_first_index());
    auto &parms = context_data.parms();
    auto coeff_modulus_size = parms.coeff_modulus().size();

    auto max_chain_index = coeff_modulus_size;
    if (plain.chain_index() == max_chain_index) {
        throw invalid_argument("end of modulus switching chain reached");
    }

    auto next_chain_index = plain.chain_index() + 1;
    auto &next_context_data = context.get_context_data(next_chain_index);
    auto &next_parms = next_context_data.parms();

    // q_1,...,q_{k-1}
    auto &next_coeff_modulus = next_parms.coeff_modulus();
    size_t next_coeff_modulus_size = next_coeff_modulus.size();
    size_t coeff_count = next_parms.poly_modulus_degree();

    // Compute destination size first for exception safety
    auto dest_size = next_coeff_modulus_size * coeff_count;

    auto data_copy = std::move(plain.data_ptr());
    plain.data_ptr() = make_cuda_auto_ptr<uint64_t>(dest_size, s);
    cudaMemcpyAsync(plain.data(), data_copy.get(), dest_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);

    plain.set_chain_index(next_chain_index);
}

PhantomCiphertext mod_switch_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                     const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    const auto &s = stream_wrapper.get_stream();

    auto &context_data = context.get_context_data(context.get_first_index());
    auto &parms = context_data.parms();
    auto coeff_modulus_size = parms.coeff_modulus().size();
    auto scheme = parms.scheme();

    auto max_chain_index = coeff_modulus_size;
    if (encrypted.chain_index() == max_chain_index) {
        throw invalid_argument("end of modulus switching chain reached");
    }
    if (parms.scheme() == scheme_type::bfv && encrypted.is_ntt_form()) {
        throw std::invalid_argument("BFV encrypted cannot be in NTT form");
    }
    if (parms.scheme() == scheme_type::bgv && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("BGV encrypted must be in NTT form");
    }
    if (parms.scheme() == scheme_type::ckks && !(encrypted.is_ntt_form())) {
        throw std::invalid_argument("CKKS encrypted must be in NTT form");
    }

    // Modulus switching with scaling
    PhantomCiphertext destination;
    switch (scheme) {
        case scheme_type::bfv:
        case scheme_type::bgv:
            // Modulus switching with scaling
            mod_switch_scale_to_next(context, encrypted, destination, s);
            break;
        case scheme_type::ckks:
            // Modulus switching without scaling
            mod_switch_drop_to_next(context, encrypted, destination, s);
            break;
        default:
            throw invalid_argument("unsupported scheme");
    }
    return destination;
}

PhantomCiphertext rescale_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                  const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    const auto &s = stream_wrapper.get_stream();

    auto &context_data = context.get_context_data(context.get_first_index());
    auto &parms = context_data.parms();
    auto max_chain_index = parms.coeff_modulus().size();
    auto scheme = parms.scheme();

    if (scheme != scheme_type::ckks)
        throw invalid_argument("unsupported scheme");

    // Verify parameters.
    if (encrypted.chain_index() == max_chain_index) {
        throw invalid_argument("end of modulus switching chain reached");
    }

    // Modulus switching with scaling
    PhantomCiphertext destination;
    mod_switch_scale_to_next(context, encrypted, destination, s);
    return destination;
}

void apply_galois_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t galois_elt_index,
                          const PhantomGaloisKey &galois_keys,
                          const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    size_t N = parms.poly_modulus_degree();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t encrypted_size = encrypted.size();
    if (encrypted_size > 2) {
        throw invalid_argument("encrypted size must be 2");
    }
    auto c0 = encrypted.data();
    auto c1 = encrypted.data() + encrypted.coeff_modulus_size() * encrypted.poly_modulus_degree();
    // Use key_context_data where permutation tables exist since previous runs.
    auto &key_galois_tool = context.key_galois_tool_;

    const auto &s = stream_wrapper.get_stream();

    auto temp = make_cuda_auto_ptr<uint64_t>(coeff_modulus_size * N, s);

    // DO NOT CHANGE EXECUTION ORDER OF FOLLOWING SECTION
    // BEGIN: Apply Galois for each ciphertext
    // Execution order is sensitive, since apply_galois is not inplace!
    if (parms.scheme() == scheme_type::bfv) {
        // !!! DO NOT CHANGE EXECUTION ORDER!!!
        // First transform c0
        key_galois_tool->apply_galois(c0, context.gpu_rns_tables(), coeff_modulus_size, galois_elt_index, temp.get(),
                                      s);
        // Copy result to c0
        cudaMemcpyAsync(c0, temp.get(), coeff_modulus_size * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);
        // Next transform c1
        key_galois_tool->apply_galois(c1, context.gpu_rns_tables(), coeff_modulus_size, galois_elt_index, temp.get(),
                                      s);
    } else if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv) {
        // !!! DO NOT CHANGE EXECUTION ORDER!!
        // First transform c0
        key_galois_tool->apply_galois_ntt(c0, coeff_modulus_size, galois_elt_index, temp.get(), s);
        // Copy result to c0
        cudaMemcpyAsync(c0, temp.get(), coeff_modulus_size * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);
        // Next transform c1
        key_galois_tool->apply_galois_ntt(c1, coeff_modulus_size, galois_elt_index, temp.get(), s);
    } else {
        throw logic_error("scheme not implemented");
    }

    // Wipe c1
    cudaMemsetAsync(c1, 0, coeff_modulus_size * N * sizeof(uint64_t), s);

    // END: Apply Galois for each ciphertext
    // REORDERING IS SAFE NOW
    // Calculate (temp * galois_key[0], temp * galois_key[1]) + (c0, 0)
    keyswitch_inplace(context, encrypted, temp.get(), galois_keys.get_relin_keys(galois_elt_index), false, s);
}

// TODO: remove recursive chain
static void rotate_internal(const PhantomContext &context, PhantomCiphertext &encrypted, int step,
                            const PhantomGaloisKey &galois_key,
                            const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    auto &context_data = context.get_context_data(encrypted.chain_index());

    // Is there anything to do?
    if (step == 0) {
        return;
    }

    size_t coeff_count = context_data.parms().poly_modulus_degree();
    auto &key_galois_tool = context.key_galois_tool_;
    auto &galois_elts = key_galois_tool->galois_elts();
    auto step_galois_elt = key_galois_tool->get_elt_from_step(step);

    auto iter = find(galois_elts.begin(), galois_elts.end(), step_galois_elt);
    if (iter != galois_elts.end()) {
        auto galois_elt_index = iter - galois_elts.begin();
        // Perform rotation and key switching
        apply_galois_inplace(context, encrypted, galois_elt_index, galois_key, stream_wrapper);
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
                rotate_internal(context, encrypted, temp_step, galois_key, stream_wrapper);
            }
        }
    }
}

void rotate_rows_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int steps,
                         const PhantomGaloisKey &galois_key, const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    if (context.key_context_data().parms().scheme() != phantom::scheme_type::bfv &&
        context.key_context_data().parms().scheme() != phantom::scheme_type::bgv) {
        throw std::logic_error("unsupported scheme");
    }
    rotate_internal(context, encrypted, steps, galois_key, stream_wrapper);
}

void rotate_columns_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                            const PhantomGaloisKey &galois_key,
                            const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    if (context.key_context_data().parms().scheme() != phantom::scheme_type::bfv &&
        context.key_context_data().parms().scheme() != phantom::scheme_type::bgv) {
        throw std::logic_error("unsupported scheme");
    }
    apply_galois_inplace(context, encrypted, 0, galois_key, stream_wrapper);
}

void rotate_vector_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int step,
                           const PhantomGaloisKey &galois_key,
                           const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    if (context.key_context_data().parms().scheme() != phantom::scheme_type::ckks) {
        throw std::logic_error("unsupported scheme");
    }
    rotate_internal(context, encrypted, step, galois_key, stream_wrapper);
}

void complex_conjugate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                               const PhantomGaloisKey &galois_key,
                               const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    if (context.key_context_data().parms().scheme() != phantom::scheme_type::ckks) {
        throw std::logic_error("unsupported scheme");
    }
    apply_galois_inplace(context, encrypted, 0, galois_key, stream_wrapper);
}

void hoisting_inplace(const PhantomContext &context, PhantomCiphertext &ct, const PhantomGaloisKey &glk,
                      const std::vector<int> &steps, const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    const auto &s = stream_wrapper.get_stream();

    if (ct.size() > 2)
        throw invalid_argument("ciphertext size must be 2");

    auto &context_data = context.get_context_data(ct.chain_index());
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto &key_modulus = key_parms.coeff_modulus();
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
        throw invalid_argument("unsupported scheme in keyswitch_inplace");
    }

    auto &rns_tool = context.get_context_data(1 + levelsDropped).gpu_rns_tool();
    auto &parms = context_data.parms();
    auto &key_galois_tool = context.key_galois_tool_;
    auto &galois_elts = key_galois_tool->galois_elts();

    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;

    auto size_Q_n = size_Q * n;
    auto size_Ql_n = size_Ql * n;
    auto size_QP_n = size_QP * n;
    auto size_QlP_n = size_QlP * n;

    auto c0 = make_cuda_auto_ptr<uint64_t>(size_Ql_n, s);
    auto c1 = make_cuda_auto_ptr<uint64_t>(size_Ql_n, s);

    auto elts = key_galois_tool->get_elts_from_steps(steps);

    // ------------------------------------------ automorphism c0 ------------------------------------------------------

    // specific operations for HPSOverQLeveled
    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        rns_tool.scaleAndRound_HPS_Q_Ql(c0.get(), ct.data(), s);
    } else {
        cudaMemcpyAsync(
                c0.get(), ct.data(), size_Ql_n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);
    }

    auto acc_c0 = make_cuda_auto_ptr<uint64_t>(size_Ql_n, s);

    auto first_elt = elts[0];
    auto first_iter = find(galois_elts.begin(), galois_elts.end(), first_elt);
    if (first_iter == galois_elts.end())
        throw std::logic_error("Galois key not present in hoisting");
    auto first_elt_index = first_iter - galois_elts.begin();

    if (parms.scheme() == scheme_type::bfv) {
        key_galois_tool->apply_galois(c0.get(), context.gpu_rns_tables(), size_Ql, first_elt_index, acc_c0.get(), s);
    } else if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv) {
        key_galois_tool->apply_galois_ntt(c0.get(), size_Ql, first_elt_index, acc_c0.get(), s);
    } else {
        throw logic_error("scheme not implemented");
    }

    // ----------------------------------------------- modup c1 --------------------------------------------------------

    // specific operations for HPSOverQLeveled
    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        rns_tool.scaleAndRound_HPS_Q_Ql(c1.get(), ct.data() + size_Q_n, s);
    } else {
        cudaMemcpyAsync(
                c1.get(), ct.data() + size_Ql_n, size_Ql_n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);
    }

    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    // mod up
    auto modup_c1 = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
    rns_tool.modup(modup_c1.get(), c1.get(), context.gpu_rns_tables(), scheme, s);

    // ------------------------------------------ automorphism c1 ------------------------------------------------------

    auto temp_modup_c1 = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);

    for (size_t b = 0; b < beta; b++) {
        key_galois_tool->apply_galois_ntt(modup_c1.get() + b * size_QlP_n, size_QlP, first_elt_index,
                                          temp_modup_c1.get() + b * size_QlP_n, s);
    }

    // ----------------------------------------- inner product c1 ------------------------------------------------------

    auto acc_cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);

    auto reduction_threshold =
            (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;
    key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
            acc_cx.get(), temp_modup_c1.get(), glk.get_relin_keys(first_elt_index).public_keys_ptr(), modulus_QP, n,
            size_QP, size_QP_n, size_QlP, size_QlP_n, size_Q, size_Ql, beta, reduction_threshold);

    // ------------------------------------------ loop accumulate ------------------------------------------------------

    auto temp_c0 = make_cuda_auto_ptr<uint64_t>(size_Ql_n, s);

    for (size_t i = 1; i < elts.size(); i++) {
        // automorphism c0

        auto elt = elts[i];
        auto iter = find(galois_elts.begin(), galois_elts.end(), elt);
        if (iter == galois_elts.end())
            throw std::logic_error("Galois key not present in hoisting");
        auto elt_index = iter - galois_elts.begin();

        if (parms.scheme() == scheme_type::bfv) {
            key_galois_tool->apply_galois(c0.get(), context.gpu_rns_tables(), size_Ql, elt_index, temp_c0.get(), s);
        } else if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv) {
            key_galois_tool->apply_galois_ntt(c0.get(), size_Ql, elt_index, temp_c0.get(), s);
        } else {
            throw logic_error("scheme not implemented");
        }

        // add to acc_c0
        uint64_t gridDimGlb = size_Ql_n / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                acc_c0.get(), temp_c0.get(), rns_tool.base_Ql().base(), acc_c0.get(), n, size_Ql);

        // automorphism c1

        for (size_t b = 0; b < beta; b++) {
            key_galois_tool->apply_galois_ntt(modup_c1.get() + b * size_QlP_n, size_QlP, elt_index,
                                              temp_modup_c1.get() + b * size_QlP_n, s);
        }

        // inner product c1
        auto temp_cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);
        key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
                temp_cx.get(), temp_modup_c1.get(), glk.get_relin_keys(elt_index).public_keys_ptr(), modulus_QP, n,
                size_QP, size_QP_n, size_QlP, size_QlP_n, size_Q, size_Ql, beta, reduction_threshold);

        // add to acc_cx
        gridDimGlb = size_QlP_n / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                acc_cx.get(), temp_cx.get(), rns_tool.base_QlP().base(), acc_cx.get(),
                n, size_QlP);
        add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                acc_cx.get() + size_QlP_n, temp_cx.get() + size_QlP_n,
                rns_tool.base_QlP().base(), acc_cx.get() + size_QlP_n, n, size_QlP);
    }

    // -------------------------------------------- mod down c1 --------------------------------------------------------
    rns_tool.moddown_from_NTT(acc_cx.get(), acc_cx.get(), context.gpu_rns_tables(), scheme, s);
    rns_tool.moddown_from_NTT(acc_cx.get() + size_QlP_n, acc_cx.get() + size_QlP_n, context.gpu_rns_tables(), scheme,
                              s);

    // new c0
    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        add_rns_poly<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
                acc_c0.get(), acc_cx.get(), rns_tool.base_Ql().base(),
                acc_cx.get(), n, size_Ql);
        rns_tool.ExpandCRTBasis_Ql_Q(ct.data(), acc_cx.get(), s);
    } else {
        add_rns_poly<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
                acc_c0.get(), acc_cx.get(), rns_tool.base_Ql().base(),
                ct.data(), n, size_Ql);
    }

    // new c1
    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        rns_tool.ExpandCRTBasis_Ql_Q(ct.data() + size_Q_n, acc_cx.get() + size_QlP_n, s);
    } else {
        cudaMemcpyAsync(ct.data() + size_Ql_n, acc_cx.get() + size_QlP_n, size_Ql_n * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, s);
    }
}

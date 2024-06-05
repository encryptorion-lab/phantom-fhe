#include "mempool.cuh"
#include "ntt.cuh"
#include "rns.cuh"
#include "scalingvariant.cuh"
#include "secretkey.h"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

// Compute the infty norm of poly
void poly_infty_norm_coeffmod(const uint64_t *poly, size_t coeff_count, size_t coeff_uint64_count,
                              const uint64_t *modulus, uint64_t *result) {
    // Construct negative threshold: (modulus + 1) / 2
    auto modulus_neg_threshold = std::vector<uint64_t>(coeff_uint64_count);

    half_round_up_uint(modulus, coeff_uint64_count, modulus_neg_threshold.data());

    // Mod out the poly coefficients and choose a symmetric representative from [-modulus,modulus)
    set_zero_uint(coeff_uint64_count, result);
    auto coeff_abs_value = std::vector<uint64_t>(coeff_uint64_count);

    for (size_t i{0}; i < coeff_count; i++) {
        if (is_greater_than_or_equal_uint(poly + i * coeff_uint64_count, modulus_neg_threshold.data(),
                                          coeff_uint64_count)) {
            sub_uint(modulus, poly + i * coeff_uint64_count, coeff_uint64_count, coeff_abs_value.data());
        } else {
            set_uint(poly + i * coeff_uint64_count, coeff_uint64_count, coeff_abs_value.data());
        }

        if (is_greater_than_uint(coeff_abs_value.data(), result, coeff_uint64_count)) {
            // Store the new max
            set_uint(coeff_abs_value.data(), coeff_uint64_count, result);
        }
    }
}

/** Encrypt zero using the public key, internal function, no modulus switch here.
 * @param[in] context PhantomContext
 * @param[inout] cipher The generated ciphertext
 * @param[in] chain_index The id of the corresponding context data
 * @param[in] is_ntt_from Whether the ciphertext should be in NTT form
 */
void PhantomPublicKey::encrypt_zero_asymmetric(const PhantomContext &context, PhantomCiphertext &cipher,
                                               size_t chain_index, bool is_ntt_form) const {
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = coeff_modulus.size();
    auto base_rns = context.gpu_rns_tables().modulus();

    cipher.resize(context, chain_index, pk_.size());
    cipher.is_ntt_form() = is_ntt_form;
    cipher.scale() = 1.0;
    cipher.correction_factor() = 1;

    // c[j] = public_key[j] * u + e[j] in BFV/CKKS
    //      = public_key[j] * u + p * e[j] in BGV
    // where e[j] <-- chi, u <-- R_3

    // first generate the ternary random u
    random_bytes(context.prng_seed(), phantom::util::global_variables::prng_seed_byte_count);

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    // in <-- ternary
    sample_ternary_poly<<<gridDimGlb, blockDimGlb>>>(context.in(), context.prng_seed(), base_rns, poly_degree,
                                                     coeff_mod_size);

    // transform u into NTT
    nwt_2d_radix8_forward_inplace(context.in(), context.gpu_rns_tables(), coeff_mod_size, 0);

    // then, generate the cbd error
    random_bytes(context.prng_seed(), phantom::util::global_variables::prng_seed_byte_count);

    if (is_ntt_form) {
        for (size_t i = 0; i < cipher.size_; i++) {
            // CAUTION: pk_ contains two polys with max modulus size, use it with caution when chain_index != 0
            uint64_t *ci = cipher.data() + i * poly_degree * coeff_mod_size;
            uint64_t *pki = pk_.data() + i * poly_degree * pk_.coeff_modulus_size_;
            // transform e into NTT, res stored in cipher
            sample_error_poly<<<gridDimGlb, blockDimGlb>>>(ci, context.prng_seed(), base_rns, poly_degree,
                                                           coeff_mod_size);
            if (parms.scheme() == scheme_type::bgv) {
                // noise = te instead of e in BGV
                multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(ci, context.plain_modulus(),
                                                                      context.plain_modulus_shoup(), base_rns, ci,
                                                                      poly_degree, coeff_mod_size);
            }

            nwt_2d_radix8_forward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0);
            // u * pk + e or (u*pk + te for BGV)
            multiply_and_add_rns_poly<<<gridDimGlb, blockDimGlb>>>(context.in(), pki, ci, base_rns, ci, poly_degree,
                                                                   coeff_mod_size);
        }
    } else {
        Pointer<uint64_t> error;
        error.acquire(allocate<uint64_t>(global_pool(), coeff_mod_size * poly_degree));

        for (size_t i = 0; i < cipher.size(); i++) {
            uint64_t *ci = cipher.data() + i * poly_degree * coeff_mod_size;
            uint64_t *pki = pk_.data() + i * poly_degree * pk_.coeff_modulus_size_;

            multiply_rns_poly<<<gridDimGlb, blockDimGlb>>>(pki, context.in(), base_rns, ci, poly_degree,
                                                           coeff_mod_size);

            nwt_2d_radix8_backward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0);
            // obtain e res stored in cipher
            sample_error_poly<<<gridDimGlb, blockDimGlb>>>(error.get(), context.prng_seed(), base_rns, poly_degree,
                                                           coeff_mod_size);
            add_rns_poly<<<gridDimGlb, blockDimGlb>>>(ci, error.get(), base_rns, ci, poly_degree, coeff_mod_size);
        }
    }
}

/** Encrypt zero using the public key, and perform the model switch is necessary
 * @brief pk [pk0, pk1], ternary variable u, cbd (gauss) noise e0, e1, return [pk0*u+e0, pk1*u+e1]
 * @param[in] context PhantomContext
 * @param[inout] cipher The generated ciphertext
 * @param[in] chain_index The id of the corresponding context data
 * @param[in] save_seed Save random seed in ciphertext
 */
void PhantomPublicKey::encrypt_zero_asymmetric_internal(const PhantomContext &context, PhantomCiphertext &cipher,
                                                        size_t chain_index, bool save_seed) const {
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    // first data level
    auto coeff_mod_size = coeff_modulus.size();

    bool is_ntt_form = false;
    if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv) {
        is_ntt_form = true;
    } else if (parms.scheme() != scheme_type::bfv) {
        throw invalid_argument("unsupported scheme");
    }

    cipher.resize(context, chain_index, 2);

    auto prev_index = context.get_previous_index(chain_index);
    if (prev_index == chain_index) {
        // Does not require modulus switching
        return encrypt_zero_asymmetric(context, cipher, chain_index, is_ntt_form);
    }

    PhantomCiphertext temp_cipher;
    encrypt_zero_asymmetric(context, temp_cipher, prev_index, is_ntt_form);
    // moddown
    size_t size_P = parms.special_modulus_size();

    for (size_t i = 0; i < temp_cipher.size(); i++) {
        uint64_t *cx_i = temp_cipher.data() + i * poly_degree * (coeff_mod_size + size_P);
        uint64_t *ct_i = cipher.data() + i * poly_degree * coeff_mod_size;
        cudaDeviceSynchronize();
        context.get_context_data(1).gpu_rns_tool().moddown(ct_i, cx_i, context.gpu_rns_tables(), parms.scheme(),
                                                           context.get_cuda_stream(0));
        cudaDeviceSynchronize();
    }

    cipher.is_ntt_form() = is_ntt_form;
    cipher.scale() = temp_cipher.scale();
    cipher.chain_index() = chain_index;
}

/** asymmetric encryption.
 * @brief: asymmetric encryption requires modulus switching.
 * @param[in] context PhantomContext
 * @param[in] plain The data to be encrypted
 * @param[out] cipher The generated ciphertext
 * @param[in] save_seed Save random seed in ciphertext
 */
void PhantomPublicKey::encrypt_asymmetric(const PhantomContext &context, const PhantomPlaintext &plain,
                                          PhantomCiphertext &cipher, bool save_seed) {
    auto &context_data = context.get_context_data(0); // i.e. 0 is key_param_id
    auto &parms = context_data.parms();
    auto scheme = parms.scheme();
    if (scheme == scheme_type::bfv) {
        if (plain.is_ntt_form()) {
            throw std::invalid_argument("BFV plain must not be in NTT form");
        }
        encrypt_zero_asymmetric_internal(context, cipher, context.get_first_index(), save_seed);
        // calcuate [plain * coeff / plain-modulus].
        // return [plain * coeff / plain-modulus + c0, c1]
        multiply_add_plain_with_scaling_variant(context, plain, context.get_first_index(), cipher);
    } else if (scheme == scheme_type::ckks) {
        if (!plain.is_ntt_form()) {
            throw std::invalid_argument("CKKS plain must be in NTT form");
        }

        // [c0, c1] is the encrytion of 0
        encrypt_zero_asymmetric_internal(context, cipher, plain.chain_index(), save_seed);
        // [c0 + plaintext, c1]
        auto &ckks_context_data = context.get_context_data(plain.chain_index());
        auto &ckks_parms = ckks_context_data.parms();
        auto poly_degree = ckks_parms.poly_modulus_degree();
        auto ckks_coeff_mod_size = ckks_parms.coeff_modulus().size();
        auto base_rns = context.gpu_rns_tables().modulus();

        // c0 = c0 + plaintext

        uint64_t gridDimGlb = poly_degree * ckks_coeff_mod_size / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb>>>(cipher.data(), plain.data(), base_rns, cipher.data(), poly_degree,
                                                  ckks_coeff_mod_size);

        pk_.chain_index() = plain.chain_index();
        cipher.scale() = plain.scale();
    } else if (scheme == scheme_type::bgv) {
        if (plain.is_ntt_form()) {
            throw std::invalid_argument("BGV plain must not be in NTT form");
        }
        // c0 = pl_0*u + t*e_0
        // c1 = pk_1*u + t*e_1
        encrypt_zero_asymmetric_internal(context, cipher, context.get_first_index(), save_seed);

        auto &bgv_context_data = context.get_context_data(context.get_first_index());
        auto &bgv_parms = bgv_context_data.parms();
        auto poly_degree = bgv_parms.poly_modulus_degree();
        auto bgv_coeff_mod_size = bgv_parms.coeff_modulus().size();
        auto base_rns = context.gpu_rns_tables().modulus();

        // c0 = c0 + plaintext
        Pointer<uint64_t> plain_copy;
        plain_copy.acquire(allocate<uint64_t>(global_pool(), bgv_coeff_mod_size * poly_degree));
        for (size_t i = 0; i < bgv_coeff_mod_size; i++) {
            // modup t -> Q
            nwt_2d_radix8_forward_modup_fuse(plain_copy.get() + i * poly_degree, plain.data(), i,
                                             context.gpu_rns_tables(), 1, 0);
        }

        uint64_t gridDimGlb = poly_degree * bgv_coeff_mod_size / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb>>>(cipher.data(), plain_copy.get(), base_rns, cipher.data(), poly_degree,
                                                  bgv_coeff_mod_size);
    } else {
        throw std::invalid_argument("unsupported scheme.");
    }

    cipher.is_asymmetric() = true;
}

/** Generate the secret key for the specified param
 * @notice: use the context data corresponding to the key_param_id (i.e. 0)
 * As one salsa20 invocation generates 64 bytes, for key generation, one byte could generate one ternary
 * Therefore, we only need random number of length poly_degree/64 bytes.
 * @param[in] context PhantomContext
 */
void PhantomSecretKey::gen_secretkey(const PhantomContext &context) {
    auto &context_data = context.get_context_data(0);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = coeff_modulus.size();
    auto base_rns = context.gpu_rns_tables().modulus();

    random_bytes(context.prng_seed(), phantom::util::global_variables::prng_seed_byte_count);

    secret_key_array_.acquire(allocate<uint64_t>(global_pool(), poly_degree * coeff_mod_size));

    // Copy constant data to device constatnt memory
    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    sample_ternary_poly<<<gridDimGlb, blockDimGlb>>>(secret_key_array_.get(), context.prng_seed(), base_rns,
                                                     poly_degree, coeff_mod_size);

    // Compute the NTT form of secret key and
    // save secret_key to the first coeff_mod_size * N elements of secret_key_array
    nwt_2d_radix8_forward_inplace(secret_key_array_.get(), context.gpu_rns_tables(), coeff_mod_size, 0);

    gen_flag_ = true;
    chain_index_ = 0;
    sk_max_power_ = 1;

    return;
}

void PhantomSecretKey::compute_secret_key_array(const PhantomContext &context, size_t max_power) {
    // Check to see if secret key and public key have been generated
    if (!gen_flag_) {
        throw logic_error("cannot generate relinearization keys for unspecified secret key");
    }
    if (max_power <= 1 || max_power <= sk_max_power_) {
        return;
    }

    // Extract encryption parameters.
    auto &key_context_data = context.get_context_data(0);
    auto &parms = key_context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = coeff_modulus.size();
    auto base_rns = context.gpu_rns_tables().modulus();

    Pointer<uint64_t> new_secret_key_array;
    new_secret_key_array.acquire(allocate<uint64_t>(global_pool(), max_power * poly_degree * coeff_mod_size));
    // Copy the power of secret key
    PHANTOM_CHECK_CUDA(cudaMemcpy(new_secret_key_array.get(), secret_key_array(),
                                  sk_max_power_ * poly_degree * coeff_mod_size * sizeof(uint64_t),
                                  cudaMemcpyDeviceToDevice));

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    for (size_t i = 0; i < max_power - sk_max_power_; i++) {
        uint64_t *prev_power = new_secret_key_array.get() + (i + sk_max_power_ - 1) * coeff_mod_size * poly_degree;
        uint64_t *curr_power = new_secret_key_array.get() + (i + sk_max_power_) * coeff_mod_size * poly_degree;

        multiply_rns_poly<<<gridDimGlb, blockDimGlb>>>(prev_power, secret_key_array(), base_rns, curr_power,
                                                       poly_degree, coeff_mod_size);
    }

    // Release the old secret_key_array_
    secret_key_array_.acquire(new_secret_key_array);
    sk_max_power_ = max_power;
}

void PhantomSecretKey::gen_publickey(const PhantomContext &context, PhantomPublicKey &pk, bool save_seed) const {
    encrypt_zero_symmetric(context, pk.pk_, 0, true, save_seed); // use key_param_id (i.e., 0)
    pk.pk_.chain_index() = 0;
}

void PhantomSecretKey::generate_one_kswitch_key(const PhantomContext &context, uint64_t *new_key,
                                                PhantomRelinKey &relin_keys, bool save_seed) const {
    // Extract encryption parameters.
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto &key_modulus = key_parms.coeff_modulus();
    auto poly_degree = key_parms.poly_modulus_degree();
    auto base_rns = context.gpu_rns_tables().modulus();

    size_t dnum = relin_keys.pk_num_;
    size_t size_QP = key_modulus.size();
    size_t size_P = key_parms.special_modulus_size();
    size_t alpha = size_P;

    auto bigP_mod_q = context.get_context_data(0).gpu_rns_tool().bigP_mod_q();
    auto bigP_mod_q_shoup = context.get_context_data(0).gpu_rns_tool().bigP_mod_q_shoup();

    // Every pk_ = [P_{w,q}(s^2)+(-(as+e)), a]
    relin_keys.public_keys_.resize(dnum);
    relin_keys.public_keys_ptr_.acquire(allocate<uint64_t *>(global_pool(), dnum));
    uint64_t *pk_ptr[dnum];

    // First initiate the pk_ = [-(as+e), a]
    for (size_t twr = 0; twr < dnum; twr++) {
        PhantomPublicKey public_key(context);
        gen_publickey(context, public_key, save_seed);
        pk_ptr[twr] = public_key.pk_.data();
        relin_keys.public_keys_[twr] = std::move(public_key);
    }
    PHANTOM_CHECK_CUDA(
            cudaMemcpy(relin_keys.public_keys_ptr_.get(), pk_ptr, sizeof(uint64_t *) * dnum, cudaMemcpyHostToDevice));

    // Second compute P_{w,q}(s^2)+(-(as+e))
    uint64_t gridDimGlb = poly_degree * dnum * alpha / blockDimGlb.x;
    multiply_temp_mod_and_add_rns_poly<<<gridDimGlb, blockDimGlb>>>(
            new_key, relin_keys.public_keys_ptr_.get(), base_rns, relin_keys.public_keys_ptr_.get(), poly_degree, dnum,
            alpha, bigP_mod_q, bigP_mod_q_shoup);

    // Set the gen flag
    relin_keys.gen_flag_ = true;
}

void PhantomSecretKey::gen_relinkey(const PhantomContext &context, PhantomRelinKey &relin_key, bool save_seed) {
    // Check to see if secret key and public key have been generated
    if (!gen_flag_) {
        throw logic_error("cannot generate relinearization keys for unspecified secret key");
    }

    // Extract encryption parameters.
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto &key_modulus = key_parms.coeff_modulus();
    auto poly_degree = key_parms.poly_modulus_degree();
    auto coeff_mod_size = key_modulus.size();

    size_t max_power = 2;
    if (max_power > sk_max_power_) {
        compute_secret_key_array(context, max_power);
    }

    // Make sure we have enough secret keys computed
    uint64_t *sk_square = secret_key_array() + coeff_mod_size * poly_degree;
    generate_one_kswitch_key(context, sk_square, relin_key, save_seed);
}

void PhantomSecretKey::create_galois_keys(const PhantomContext &context, PhantomGaloisKey &galois_keys,
                                          bool save_seed) const {
    // Check to see if secret key and public key have been generated
    if (!gen_flag_) {
        throw logic_error("cannot generate relinearization keys for unspecified secret key");
    }
    // Extract encryption parameters.
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto &key_modulus = key_parms.coeff_modulus();
    auto &key_galois_tool = context.key_galois_tool_;
    auto poly_degree = key_parms.poly_modulus_degree();
    auto key_mod_size = key_modulus.size();

    // get galois_elts
    auto &galois_elts = key_galois_tool->galois_elts_;

    Pointer<uint64_t> rotated_secret_key;
    rotated_secret_key.acquire(allocate<uint64_t>(global_pool(), key_mod_size * poly_degree));
    auto secret_key = secret_key_array_.get();

    galois_keys.relin_key_num_ = galois_elts.size();
    galois_keys.relin_keys_.resize(galois_keys.relin_key_num_);

    for (size_t galois_elt_idx{0}; galois_elt_idx < galois_keys.relin_key_num_; galois_elt_idx++) {
        auto galois_elt = galois_elts[galois_elt_idx];

        // Verify coprime conditions.
        if (!(galois_elt & 1) || (galois_elt >= poly_degree << 1)) {
            throw invalid_argument("Galois element is not valid");
        }
        // Rotate secret key for each coeff_modulus
        key_galois_tool->apply_galois_ntt(secret_key, key_mod_size, galois_elt_idx, rotated_secret_key.get());

        PhantomRelinKey relin_key(context);
        generate_one_kswitch_key(context, rotated_secret_key.get(), relin_key, save_seed);
        galois_keys.relin_keys_[galois_elt_idx] = std::move(relin_key);
    }
    galois_keys.gen_flag_ = true;
}

/** Encrypt zero using the secret key
 * @param[in] context PhantomContext
 * @param[inout] cipher The generated ciphertext
 * @param[in] chain_index The index of the used context data
 * @param[in] is_ntt_form Whether the ciphertext needs to be in NTT form
 * @param[in] save_seed Save random seed in ciphertext (current version not support)
 */
void PhantomSecretKey::encrypt_zero_symmetric(const PhantomContext &context, PhantomCiphertext &cipher,
                                              size_t chain_index, bool is_ntt_form, bool save_seed) const {
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = coeff_modulus.size();
    auto base_rns = context.gpu_rns_tables().modulus();

    cipher.resize(context, chain_index, 2); // size = 2
    cipher.is_ntt_form() = is_ntt_form;
    cipher.scale() = 1.0;
    cipher.correction_factor() = 1;

    // Generate ciphertext: (c[0], c[1]) = ([-(as+ e)]_q, a) in BFV/CKKS
    // Generate ciphertext: (c[0], c[1]) = ([-(as+te)]_q, a) in BGV
    uint64_t *c0 = cipher.data();
    uint64_t *c1 = cipher.data() + poly_degree * coeff_mod_size;

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    // first generate the error e
    random_bytes(context.prng_seed(), phantom::util::global_variables::prng_seed_byte_count);

    sample_error_poly<<<gridDimGlb, blockDimGlb>>>(context.in(), context.prng_seed(), base_rns, poly_degree,
                                                   coeff_mod_size);

    // then, we auto save the seed for c1 in the prng seed
    random_bytes(context.prng_seed(), phantom::util::global_variables::prng_seed_byte_count);

    if (is_ntt_form) {
        if (parms.scheme() == scheme_type::bgv) {
            // noise = te instead of e in BGV
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(context.in(), context.plain_modulus(),
                                                                  context.plain_modulus_shoup(), base_rns, context.in(),
                                                                  poly_degree, coeff_mod_size);
        }
        // transform e into NTT, here coeff_mod_size corresponding to the chain index
        nwt_2d_radix8_forward_inplace(context.in(), context.gpu_rns_tables(), coeff_mod_size, 0);
        // uniform random generator
        sample_uniform_poly<<<gridDimGlb, blockDimGlb>>>(c1, context.prng_seed(), base_rns, poly_degree,
                                                         coeff_mod_size);
        // c0 = -(as + e) or c0 = -(as + te), c1 = a
        multiply_and_add_negate_rns_poly<<<gridDimGlb, blockDimGlb>>>(c1, secret_key_array(), context.in(), base_rns,
                                                                      c0, poly_degree, coeff_mod_size);
    } else {
        // uniform random generator
        sample_uniform_poly<<<gridDimGlb, blockDimGlb>>>(c1, context.prng_seed(), base_rns, poly_degree,
                                                         coeff_mod_size);
        // c0 = c1 * s
        multiply_rns_poly<<<gridDimGlb, blockDimGlb>>>(c1, secret_key_array(), base_rns, c0, poly_degree,
                                                       coeff_mod_size);
        // c0 backward, here coeff_mod_size corresponding to the chain index
        nwt_2d_radix8_backward_inplace(c0, context.gpu_rns_tables(), coeff_mod_size, 0);
        // c0 = -(c0 + e)
        add_and_negate_rns_poly<<<gridDimGlb, blockDimGlb>>>(c0, context.in(), base_rns, c0, poly_degree,
                                                             coeff_mod_size);
        // c1 backward
        nwt_2d_radix8_backward_inplace(c1, context.gpu_rns_tables(), coeff_mod_size, 0);
    }
}

/** Symmetric encryption
 * @brief: symmetric encryption does not require modulus switching.
 * @param[in] context PhantomContext
 * @param[in] plain The data to be encrypted
 * @param[out] cipher The generated ciphertext
 * @param[in] save_seed Save random seed in ciphertext
 */
void PhantomSecretKey::encrypt_symmetric(const PhantomContext &context, const PhantomPlaintext &plain,
                                         PhantomCiphertext &cipher, bool save_seed) const {
    auto &context_data = context.get_context_data(0); // Use key_parm_id for obtaining scheme
    auto &parms = context_data.parms();
    auto scheme = parms.scheme();
    bool is_ntt_form = false;
    if (scheme == phantom::scheme_type::bfv) {
        if (plain.is_ntt_form()) {
            throw std::invalid_argument("BFV plain must not be in NTT form");
        }
        encrypt_zero_symmetric(context, cipher, context.get_first_index(), is_ntt_form, save_seed);
        // calcuate [plain * coeff / plain-modulus].
        // return [plain * coeff / plain-modulus + c0, c1]
        multiply_add_plain_with_scaling_variant(context, plain, context.get_first_index(), cipher);
    } else if (scheme == phantom::scheme_type::ckks) {
        is_ntt_form = true;
        if (!plain.is_ntt_form()) {
            throw std::invalid_argument("CKKS plain must be in NTT form");
        }
        // [c0, c1] is the encrytion of 0, key idea is use the plain's chain_index to find corresponding data
        encrypt_zero_symmetric(context, cipher, plain.chain_index(), is_ntt_form, save_seed);

        // [c0 + plaintext, c1]
        auto &ckks_context_data = context.get_context_data(plain.chain_index());
        auto &ckks_parms = ckks_context_data.parms();
        auto poly_degree = ckks_parms.poly_modulus_degree();
        size_t ckks_coeff_mod_size = ckks_parms.coeff_modulus().size();

        // c0 = c0 + plaintext
        uint64_t gridDimGlb = poly_degree * ckks_coeff_mod_size / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb>>>(cipher.data(), plain.data(), context.gpu_rns_tables().modulus(),
                                                  cipher.data(), poly_degree, ckks_coeff_mod_size);

        cipher.scale() = plain.scale();
    } else if (scheme == phantom::scheme_type::bgv) {
        is_ntt_form = true;
        if (plain.is_ntt_form()) {
            throw std::invalid_argument("BGV plain must not be in NTT form");
        }
        // (c[0], c[1]) = ([-(as+te)]_q, a)
        encrypt_zero_symmetric(context, cipher, context.get_first_index(), is_ntt_form, save_seed);

        auto &bgv_context_data = context.get_context_data(context.get_first_index());
        auto &bgv_parms = bgv_context_data.parms();
        auto poly_degree = bgv_parms.poly_modulus_degree();
        auto bgv_coeff_mod_size = bgv_parms.coeff_modulus().size();
        auto base_rns = context.gpu_rns_tables().modulus();

        // c0 = c0 + plaintext
        Pointer<uint64_t> plain_copy;
        plain_copy.acquire(allocate<uint64_t>(global_pool(), bgv_coeff_mod_size * poly_degree));
        for (size_t i = 0; i < bgv_coeff_mod_size; i++) {
            PHANTOM_CHECK_CUDA(
                    cudaMemcpy(plain_copy.get() + i * poly_degree, plain.data(), sizeof(uint64_t) * poly_degree,
                               cudaMemcpyDeviceToDevice));
        }

        nwt_2d_radix8_forward_inplace(plain_copy.get(), context.gpu_rns_tables(), bgv_coeff_mod_size, 0);
        uint64_t gridDimGlb = poly_degree * bgv_coeff_mod_size / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb>>>(cipher.data(), plain_copy.get(), base_rns, cipher.data(), poly_degree,
                                                  bgv_coeff_mod_size);
    } else {
        throw std::invalid_argument("unsupported scheme.");
    }
}

void PhantomSecretKey::ckks_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                    PhantomPlaintext &destination) {
    if (!encrypted.is_ntt_form()) {
        throw invalid_argument("encrypted must be in NTT form");
    }

    // We already know that the parameters are valid
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = coeff_modulus.size();
    auto base_rns = context.gpu_rns_tables().modulus();
    auto needed_sk_power = encrypted.size() - 1;

    if (needed_sk_power > sk_max_power_) {
        compute_secret_key_array(context, needed_sk_power);
    }

    uint64_t *c0 = encrypted.data();

    // Since we overwrite plain, we zeroize plain parameters
    // This is necessary, otherwise resize will throw an exception.
    // Resize plain to appropriate size
    destination.chain_index() = 0;
    destination.resize(coeff_mod_size, poly_degree);
    PHANTOM_CHECK_CUDA(cudaMemcpy(destination.data(), c0, coeff_mod_size * poly_degree * sizeof(uint64_t),
                                  cudaMemcpyDeviceToDevice));
    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    for (size_t i = 1; i <= needed_sk_power; i++) {
        uint64_t *ci = encrypted.data() + i * coeff_mod_size * poly_degree;
        uint64_t *si = secret_key_array() + (i - 1) * coeff_modulus_size_ * poly_degree;
        // c_0 += c_j * s^{j}
        multiply_and_add_rns_poly<<<gridDimGlb, blockDimGlb>>>(ci, si, destination.data(), base_rns, destination.data(),
                                                               poly_degree, coeff_mod_size);
    }

    // Set destination parameters as in ciphertext
    destination.chain_index() = encrypted.chain_index();
    destination.scale() = encrypted.scale();
}

void PhantomSecretKey::bfv_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                   PhantomPlaintext &destination) {
    auto chain_index = encrypted.chain_index_;
    auto coeff_mod_size = encrypted.coeff_modulus_size_;
    auto poly_degree = encrypted.poly_modulus_degree_;
    auto base_rns = context.gpu_rns_tables().modulus();
    auto poly_num = encrypted.size_;
    auto needed_sk_power = poly_num - 1;

    if (needed_sk_power > sk_max_power_) {
        compute_secret_key_array(context, needed_sk_power);
    }

    uint64_t *c0 = encrypted.data();
    uint64_t *c1 = encrypted.data() + coeff_mod_size * poly_degree;

    // Firstly find c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q
    // This is equal to Delta m + v where ||v|| < Delta/2.
    // Add Delta / 2 and now we have something which is Delta * (m + epsilon) where epsilon < 1
    // Therefore, we can (integer) divide by Delta and the answer will round down to m.

    // Compute c_1 *s, ..., c_{count-1} * s^{count-1} mod q
    // The secret key powers are already NTT transformed.
    Pointer<uint64_t> inner_prod;
    inner_prod.acquire(allocate<uint64_t>(global_pool(), coeff_mod_size * poly_degree));
    PHANTOM_CHECK_CUDA(cudaMemset(inner_prod.get(), 0UL, coeff_mod_size * poly_degree * sizeof(uint64_t)));
    Pointer<uint64_t> temp;
    temp.acquire(allocate<uint64_t>(global_pool(), coeff_mod_size * poly_degree));

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    for (size_t i = 1; i <= needed_sk_power; i++) {
        uint64_t *ci = encrypted.data() + i * coeff_mod_size * poly_degree;
        uint64_t *si = secret_key_array() + (i - 1) * coeff_modulus_size_ * poly_degree;
        PHANTOM_CHECK_CUDA(
                cudaMemcpy(temp.get(), ci, coeff_mod_size * poly_degree * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        // Change ci to NTT form
        nwt_2d_radix8_forward_inplace(temp.get(), context.gpu_rns_tables(), coeff_mod_size, 0);
        // c1 = c1*s^1 + c2*s^2 + ......
        multiply_and_add_rns_poly<<<gridDimGlb, blockDimGlb>>>(temp.get(), si, inner_prod.get(), base_rns,
                                                               inner_prod.get(), poly_degree, coeff_mod_size);
    }

    // change c_1 to normal form
    nwt_2d_radix8_backward_inplace(inner_prod.get(), context.gpu_rns_tables(), coeff_mod_size, 0);

    // finally, c_0 = c_0 + c_1
    add_rns_poly<<<gridDimGlb, blockDimGlb>>>(c0, inner_prod.get(), base_rns, inner_prod.get(), poly_degree,
                                              coeff_mod_size);

    auto mul_tech = context.mul_tech();

    if (mul_tech == mul_tech_type::behz) {
        // Divide scaling variant using BEHZ FullRNS techniques
        cudaDeviceSynchronize();
        context.get_context_data(chain_index)
                .gpu_rns_tool()
                .behz_decrypt_scale_and_round(inner_prod.get(), temp.get(), context.gpu_rns_tables(), coeff_mod_size,
                                              poly_degree, destination.data(), context.get_cuda_stream(0));
        cudaDeviceSynchronize();
    } else if (mul_tech == mul_tech_type::hps || mul_tech == mul_tech_type::hps_overq ||
               mul_tech == mul_tech_type::hps_overq_leveled) {
        // HPS scale and round
        context.get_context_data(chain_index)
                .gpu_rns_tool()
                .hps_decrypt_scale_and_round(destination.data(), inner_prod.get());
    } else {
        throw std::invalid_argument("BFV decrypt mul_tech not supported");
    }

    // Set destination parameters as in ciphertext
    destination.chain_index() = 0;
    destination.scale() = encrypted.scale();
}

void PhantomSecretKey::bgv_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                   PhantomPlaintext &destination) {
    if (!encrypted.is_ntt_form()) {
        throw invalid_argument("encrypted must be in NTT form");
    }

    // We already know that the parameters are valid
    auto &context_data = context.get_context_data(encrypted.chain_index());
    auto &rns_tool = context.get_context_data(encrypted.chain_index()).gpu_rns_tool();
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &plain_modulus = parms.plain_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = coeff_modulus.size();
    auto base_rns = context.gpu_rns_tables().modulus();
    auto needed_sk_power = encrypted.size() - 1;

    if (needed_sk_power > sk_max_power_) {
        compute_secret_key_array(context, needed_sk_power);
    }

    uint64_t *c0 = encrypted.data();
    Pointer<uint64_t> inner_prod;
    inner_prod.acquire(allocate<uint64_t>(global_pool(), coeff_mod_size * poly_degree));
    PHANTOM_CHECK_CUDA(cudaMemcpy(inner_prod.get(), c0, coeff_mod_size * poly_degree * sizeof(uint64_t),
                                  cudaMemcpyDeviceToDevice));

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    for (size_t i = 1; i <= needed_sk_power; i++) {
        uint64_t *ci = encrypted.data() + i * coeff_mod_size * poly_degree;
        uint64_t *si = secret_key_array() + (i - 1) * coeff_modulus_size_ * poly_degree;
        // c_0 += c_j * s^{j}
        multiply_and_add_rns_poly<<<gridDimGlb, blockDimGlb>>>(ci, si, inner_prod.get(), base_rns, inner_prod.get(),
                                                               poly_degree, coeff_mod_size);
    }

    nwt_2d_radix8_backward_inplace(inner_prod.get(), context.gpu_rns_tables(), coeff_mod_size, 0);

    // Set destination parameters as in ciphertext
    destination.chain_index() = 0;
    destination.scale() = encrypted.scale();
    rns_tool.decrypt_mod_t(destination.data(), inner_prod.get(), poly_degree);

    if (encrypted.correction_factor() != 1) {
        uint64_t fix = 1;
        if (!try_invert_uint_mod(encrypted.correction_factor(), plain_modulus, fix)) {
            throw logic_error("invalid correction factor");
        }

        gridDimGlb = poly_degree / blockDimGlb.x;
        multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(
                destination.data(), fix, context.gpu_plain_tables().modulus(), destination.data(), poly_degree, 1);
    }
}

void PhantomSecretKey::decrypt(const PhantomContext &context, const PhantomCiphertext &cipher,
                               PhantomPlaintext &plain) {
    auto &context_data = context.get_context_data(0); // Use key_parm_id for obtaining scheme
    auto &parms = context_data.parms();
    auto scheme = parms.scheme();

    if (scheme == phantom::scheme_type::bfv) {
        bfv_decrypt(context, cipher, plain);
    } else if (scheme == phantom::scheme_type::ckks) {
        ckks_decrypt(context, cipher, plain);
    } else if (scheme == phantom::scheme_type::bgv) {
        bgv_decrypt(context, cipher, plain);
    } else {
        throw std::invalid_argument("unsupported scheme.");
    }
}

[[nodiscard]] int PhantomSecretKey::invariant_noise_budget(const PhantomContext &context,
                                                           const PhantomCiphertext &cipher) {
    auto chain_index = cipher.chain_index();
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &plain_modulus = parms.plain_modulus();

    auto coeff_mod_size = cipher.coeff_modulus_size_;
    auto poly_degree = cipher.poly_modulus_degree_;
    auto base_rns = context.gpu_rns_tables().modulus();
    auto poly_num = cipher.size_;
    auto needed_sk_power = poly_num - 1;
    if (needed_sk_power > sk_max_power_) {
        compute_secret_key_array(context, needed_sk_power);
    }

    auto cipher_copy(cipher);
    uint64_t *c0 = cipher_copy.data();
    uint64_t *c1 = cipher_copy.data() + coeff_mod_size * poly_degree;
    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;

    // Compute c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q
    // First compute c_1 *s, ..., c_{count-1} * s^{count-1} mod q
    for (size_t i = 1; i <= needed_sk_power; i++) {
        uint64_t *ci = cipher_copy.data() + i * coeff_mod_size * poly_degree;
        uint64_t *si = secret_key_array() + (i - 1) * coeff_modulus_size_ * poly_degree;
        // Change ci to NTT form
        nwt_2d_radix8_forward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0);
        // ci * s^{i} in NTT form
        if (i == 1) {
            // c1 = c1 * s^1
            multiply_rns_poly<<<gridDimGlb, blockDimGlb>>>(ci, si, base_rns, ci, poly_degree, coeff_mod_size);
        } else {
            // c1 = c1*s^1 + c2*s^2 + ......
            multiply_and_add_rns_poly<<<gridDimGlb, blockDimGlb>>>(ci, si, c1, base_rns, c1, poly_degree,
                                                                   coeff_mod_size);
        }
    }

    // change c_1 to normal form
    nwt_2d_radix8_backward_inplace(c1, context.gpu_rns_tables(), coeff_mod_size, 0);
    // finally, c_0 = c_0 + c_1
    add_rns_poly<<<gridDimGlb, blockDimGlb>>>(c0, c1, base_rns, c0, poly_degree, coeff_mod_size);

    // compute c0 * plain_modulus
    multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(c0, plain_modulus.value(), base_rns, c0, poly_degree,
                                                          coeff_mod_size);

    // Copy noise_poly to Host
    uint64_t *host_noise_poly;
    host_noise_poly = (uint64_t *) malloc(coeff_mod_size * poly_degree * sizeof(uint64_t));
    PHANTOM_CHECK_CUDA(
            cudaMemcpy(host_noise_poly, c0, coeff_mod_size * poly_degree * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // CRT-compose the noise
    auto &base_q = context.get_context_data_rns_tool(chain_index).host_base_Ql();
    base_q.compose_array(host_noise_poly, poly_degree);

    // Next we compute the infinity norm mod parms.coeff_modulus()
    std::vector<std::uint64_t> norm(coeff_mod_size);
    std::vector<uint64_t> modulus;
    for (size_t idx{0}; idx < coeff_mod_size; idx++) {
        modulus.push_back(parms.coeff_modulus().at(idx).value());
    }
    auto total_coeff_modulus = context_data.total_coeff_modulus();
    poly_infty_norm_coeffmod(host_noise_poly, poly_degree, coeff_mod_size, total_coeff_modulus.data(), norm.data());

    // The -1 accounts for scaling the invariant noise by 2;
    // note that we already took plain_modulus into account in compose
    // so no need to subtract log(plain_modulus) from this
    int bit_count_diff = context_data.total_coeff_modulus_bit_count() -
                         get_significant_bit_count_uint(norm.data(), coeff_mod_size) - 1;
    free(host_noise_poly);

    return max(0, bit_count_diff);
}

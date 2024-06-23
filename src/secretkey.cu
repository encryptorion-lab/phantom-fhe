#include "ntt.cuh"
#include "rns.cuh"
#include "scalingvariant.cuh"
#include "secretkey.h"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

void
PhantomPublicKey::encrypt_zero_asymmetric_internal_internal(const PhantomContext &context, PhantomCiphertext &cipher,
                                                            size_t chain_index, bool is_ntt_form,
                                                            const cudaStream_t &stream) const {
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = coeff_modulus.size();
    auto base_rns = context.gpu_rns_tables().modulus();

    cipher.resize(context, chain_index, pk_.size(), stream);
    cipher.is_ntt_form_ = is_ntt_form;
    cipher.scale_ = 1.0;
    cipher.correction_factor_ = 1;

    // c[j] = public_key[j] * u + e[j] in BFV/CKKS
    //      = public_key[j] * u + p * e[j] in BGV
    // where e[j] <-- chi, u <-- R_3

    // first generate the ternary random u
    auto prng_seed_error = make_cuda_auto_ptr<uint8_t>(phantom::util::global_variables::prng_seed_byte_count, stream);
    random_bytes(prng_seed_error.get(), phantom::util::global_variables::prng_seed_byte_count, stream);

    auto u = make_cuda_auto_ptr<uint64_t>(coeff_mod_size * poly_degree, stream);

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    // in <-- ternary
    sample_ternary_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            u.get(), prng_seed_error.get(), base_rns, poly_degree, coeff_mod_size);

    // transform u into NTT
    nwt_2d_radix8_forward_inplace(u.get(), context.gpu_rns_tables(), coeff_mod_size, 0, stream);

    // then, generate the cbd error
    random_bytes(prng_seed_error.get(), phantom::util::global_variables::prng_seed_byte_count, stream);

    if (is_ntt_form) {
        for (size_t i = 0; i < cipher.size(); i++) {
            // CAUTION: pk_ contains two polys with max modulus size, use it with caution when chain_index != 0
            uint64_t *ci = cipher.data() + i * poly_degree * coeff_mod_size;
            uint64_t *pki = pk_.data() + i * poly_degree * pk_.coeff_modulus_size();
            // transform e into NTT, res stored in cipher
            sample_error_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    ci, prng_seed_error.get(), base_rns, poly_degree, coeff_mod_size);
            if (parms.scheme() == scheme_type::bgv) {
                // noise = te instead of e in BGV
                multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        ci, context.plain_modulus(), context.plain_modulus_shoup(),
                        base_rns, ci, poly_degree, coeff_mod_size);
            }

            nwt_2d_radix8_forward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0, stream);
            // u * pk + e or (u*pk + te for BGV)
            multiply_and_add_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    u.get(), pki, ci, base_rns, ci, poly_degree, coeff_mod_size);
        }
    } else {
        auto error = make_cuda_auto_ptr<uint64_t>(coeff_mod_size * poly_degree, stream);

        for (size_t i = 0; i < cipher.size(); i++) {
            uint64_t *ci = cipher.data() + i * poly_degree * coeff_mod_size;
            uint64_t *pki = pk_.data() + i * poly_degree * pk_.coeff_modulus_size_;

            multiply_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    pki, u.get(), base_rns, ci, poly_degree, coeff_mod_size);

            nwt_2d_radix8_backward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0, stream);
            // obtain e res stored in cipher
            sample_error_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    error.get(), prng_seed_error.get(), base_rns, poly_degree, coeff_mod_size);
            add_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    ci, error.get(), base_rns, ci, poly_degree, coeff_mod_size);
        }
    }
}

void PhantomPublicKey::encrypt_zero_asymmetric_internal(const PhantomContext &context, PhantomCiphertext &cipher,
                                                        size_t chain_index, const cudaStream_t &stream) const {
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

    cipher.resize(context, chain_index, 2, stream);

    auto prev_index = context.get_previous_index(chain_index);
    if (prev_index == chain_index) {
        // Does not require modulus switching
        return encrypt_zero_asymmetric_internal_internal(context, cipher, chain_index, is_ntt_form, stream);
    }

    PhantomCiphertext temp_cipher;
    encrypt_zero_asymmetric_internal_internal(context, temp_cipher, prev_index, is_ntt_form, stream);
    // moddown
    size_t size_P = parms.special_modulus_size();

    for (size_t i = 0; i < temp_cipher.size(); i++) {
        uint64_t *cx_i = temp_cipher.data() + i * poly_degree * (coeff_mod_size + size_P);
        uint64_t *ct_i = cipher.data() + i * poly_degree * coeff_mod_size;
        context.get_context_data(1).gpu_rns_tool().moddown(
                ct_i, cx_i, context.gpu_rns_tables(), parms.scheme(), stream);
    }

    cipher.is_ntt_form_ = is_ntt_form;
    cipher.scale_ = temp_cipher.scale();
    cipher.chain_index_ = chain_index;
}


void PhantomPublicKey::encrypt_asymmetric(const PhantomContext &context, const PhantomPlaintext &plain,
                                          PhantomCiphertext &cipher,
                                          const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    auto &context_data = context.get_context_data(0); // i.e. 0 is key_param_id
    auto &parms = context_data.parms();
    auto scheme = parms.scheme();

    const auto &s = stream_wrapper.get_stream();

    cipher.scale_ = 1.0;
    cipher.correction_factor_ = 1;
    cipher.noiseScaleDeg_ = 1;

    if (scheme == scheme_type::bfv) {
        encrypt_zero_asymmetric_internal(context, cipher, context.get_first_index(), s);
        // calculate [plain * coeff / plain-modulus].
        // return [plain * coeff / plain-modulus + c0, c1]
        multiply_add_plain_with_scaling_variant(context, plain, context.get_first_index(), cipher, s);
    } else if (scheme == scheme_type::ckks) {
        // [c0, c1] is the encryption of 0
        encrypt_zero_asymmetric_internal(context, cipher, plain.chain_index(), s);

        // [c0 + plaintext, c1]
        auto &ckks_context_data = context.get_context_data(plain.chain_index());
        auto &ckks_parms = ckks_context_data.parms();
        auto poly_degree = ckks_parms.poly_modulus_degree();
        auto ckks_coeff_mod_size = ckks_parms.coeff_modulus().size();
        auto base_rns = context.gpu_rns_tables().modulus();

        // c0 = c0 + plaintext
        uint64_t gridDimGlb = poly_degree * ckks_coeff_mod_size / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                cipher.data(), plain.data(), base_rns, cipher.data(), poly_degree, ckks_coeff_mod_size);

        pk_.chain_index_ = plain.chain_index();
        cipher.scale_ = plain.scale();
    } else if (scheme == scheme_type::bgv) {
        // c0 = pl_0*u + t*e_0
        // c1 = pk_1*u + t*e_1
        encrypt_zero_asymmetric_internal(context, cipher, context.get_first_index(), s);

        auto &bgv_context_data = context.get_context_data(context.get_first_index());
        auto &bgv_parms = bgv_context_data.parms();
        auto poly_degree = bgv_parms.poly_modulus_degree();
        auto bgv_coeff_mod_size = bgv_parms.coeff_modulus().size();
        auto base_rns = context.gpu_rns_tables().modulus();

        // c0 = c0 + plaintext
        auto plain_copy = make_cuda_auto_ptr<uint64_t>(bgv_coeff_mod_size * poly_degree, s);
        for (size_t i = 0; i < bgv_coeff_mod_size; i++) {
            // modup t -> Q
            nwt_2d_radix8_forward_modup_fuse(plain_copy.get() + i * poly_degree, plain.data(), i,
                                             context.gpu_rns_tables(), 1, 0, s);
        }

        uint64_t gridDimGlb = poly_degree * bgv_coeff_mod_size / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                cipher.data(), plain_copy.get(), base_rns, cipher.data(), poly_degree, bgv_coeff_mod_size);
    } else {
        throw std::invalid_argument("unsupported scheme.");
    }

    cipher.is_asymmetric_ = true;
}

/************************************ PhantomSecretKey ************************************************/

void PhantomSecretKey::compute_secret_key_array(const PhantomContext &context, size_t max_power,
                                                const cudaStream_t &stream) {

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

    auto new_secret_key_array = make_cuda_auto_ptr<uint64_t>(max_power * poly_degree * coeff_mod_size, stream);
    // Copy the power of secret key
    cudaMemcpyAsync(new_secret_key_array.get(), secret_key_array(),
                    sk_max_power_ * poly_degree * coeff_mod_size * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    for (size_t i = 0; i < max_power - sk_max_power_; i++) {
        uint64_t *prev_power = new_secret_key_array.get() + (i + sk_max_power_ - 1) * coeff_mod_size * poly_degree;
        uint64_t *curr_power = new_secret_key_array.get() + (i + sk_max_power_) * coeff_mod_size * poly_degree;

        multiply_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                prev_power, secret_key_array(), base_rns, curr_power,
                poly_degree, coeff_mod_size);
    }

    // Release the old secret_key_array_
    secret_key_array_ = std::move(new_secret_key_array);
    sk_max_power_ = max_power;
}

void PhantomSecretKey::encrypt_zero_symmetric(const PhantomContext &context, PhantomCiphertext &cipher,
                                              const uint8_t *prng_seed_a, size_t chain_index, bool is_ntt_form,
                                              const cudaStream_t &stream) const {
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = coeff_modulus.size();
    auto base_rns = context.gpu_rns_tables().modulus();

    cipher.resize(context, chain_index, 2, stream); // size = 2
    cipher.is_ntt_form_ = is_ntt_form;
    cipher.scale_ = 1.0;
    cipher.correction_factor_ = 1;

    // Generate ciphertext: (c[0], c[1]) = ([-(as+ e)]_q, a) in BFV/CKKS
    // Generate ciphertext: (c[0], c[1]) = ([-(as+te)]_q, a) in BGV
    uint64_t *c0 = cipher.data();
    uint64_t *c1 = cipher.data() + poly_degree * coeff_mod_size;

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    // first generate the error e
    auto prng_seed_error = make_cuda_auto_ptr<uint8_t>(phantom::util::global_variables::prng_seed_byte_count, stream);
    random_bytes(prng_seed_error.get(), phantom::util::global_variables::prng_seed_byte_count, stream);

    auto u = make_cuda_auto_ptr<uint64_t>(coeff_mod_size * poly_degree, stream);

    sample_error_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            u.get(), prng_seed_error.get(), base_rns, poly_degree,
            coeff_mod_size);

    if (is_ntt_form) {
        if (parms.scheme() == scheme_type::bgv) {
            // noise = te instead of e in BGV
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    u.get(), context.plain_modulus(), context.plain_modulus_shoup(), base_rns, u.get(),
                    poly_degree, coeff_mod_size);
        }
        // transform e into NTT, here coeff_mod_size corresponding to the chain index
        nwt_2d_radix8_forward_inplace(u.get(), context.gpu_rns_tables(), coeff_mod_size, 0, stream);
        // uniform random generator
        sample_uniform_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                c1, prng_seed_a, base_rns, poly_degree, coeff_mod_size);
        // c0 = -(as + e) or c0 = -(as + te), c1 = a
        multiply_and_add_negate_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                c1, secret_key_array(), u.get(), base_rns, c0, poly_degree, coeff_mod_size);
    } else {
        // uniform random generator
        sample_uniform_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                c1, prng_seed_a, base_rns, poly_degree, coeff_mod_size);
        // c0 = c1 * s
        multiply_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                c1, secret_key_array(), base_rns, c0, poly_degree, coeff_mod_size);
        // c0 backward, here coeff_mod_size corresponding to the chain index
        nwt_2d_radix8_backward_inplace(c0, context.gpu_rns_tables(), coeff_mod_size, 0, stream);
        // c0 = -(c0 + e)
        add_and_negate_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                c0, u.get(), base_rns, c0, poly_degree, coeff_mod_size);
        // c1 backward
        nwt_2d_radix8_backward_inplace(c1, context.gpu_rns_tables(), coeff_mod_size, 0, stream);
    }
}

void PhantomSecretKey::generate_one_kswitch_key(const PhantomContext &context, uint64_t *new_key,
                                                PhantomRelinKey &relin_keys, const cudaStream_t &stream) const {
    // Extract encryption parameters.
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto &key_modulus = key_parms.coeff_modulus();
    auto poly_degree = key_parms.poly_modulus_degree();
    auto base_rns = context.gpu_rns_tables().modulus();

    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();
    size_t size_Q = size_QP - size_P;
    size_t dnum = size_Q / size_P;
    size_t alpha = size_P;

    auto bigP_mod_q = context.get_context_data(0).gpu_rns_tool().bigP_mod_q();
    auto bigP_mod_q_shoup = context.get_context_data(0).gpu_rns_tool().bigP_mod_q_shoup();

    // Every pk_ = [P_{w,q}(s^2)+(-(as+e)), a]

    relin_keys.public_keys_.resize(dnum);
    relin_keys.public_keys_ptr_ = make_cuda_auto_ptr<uint64_t *>(dnum, stream);

    // First initiate the pk_ = [-(as+e), a]
    for (size_t twr = 0; twr < dnum; twr++) {
        auto prng_seed_a = make_cuda_auto_ptr<uint8_t>(phantom::util::global_variables::prng_seed_byte_count, stream);
        random_bytes(prng_seed_a.get(), phantom::util::global_variables::prng_seed_byte_count, stream);
        PhantomCiphertext pk;
        encrypt_zero_symmetric(context, pk, prng_seed_a.get(), 0, true, stream);
        relin_keys.public_keys_[twr] = std::move(pk.data_ptr());
    }

    std::vector<uint64_t *> pk_ptr(dnum);
    for (size_t twr = 0; twr < dnum; twr++)
        pk_ptr[twr] = relin_keys.public_keys_[twr].get();
    cudaMemcpyAsync(relin_keys.public_keys_ptr_.get(), pk_ptr.data(), sizeof(uint64_t *) * dnum, cudaMemcpyHostToDevice,
                    stream);

    // Second compute P_{w,q}(s^2)+(-(as+e))
    uint64_t gridDimGlb = poly_degree * dnum * alpha / blockDimGlb.x;
    multiply_temp_mod_and_add_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            new_key, relin_keys.public_keys_ptr_.get(), base_rns, relin_keys.public_keys_ptr_.get(), poly_degree, dnum,
            alpha, bigP_mod_q, bigP_mod_q_shoup);
}

void PhantomSecretKey::gen_secretkey(const PhantomContext &context, const cudaStream_t &stream) {
    if (gen_flag_) {
        throw std::logic_error("cannot generate secret key twice");
    }

    auto &context_data = context.get_context_data(0);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = coeff_modulus.size();
    auto base_rns = context.gpu_rns_tables().modulus();

    poly_modulus_degree_ = parms.poly_modulus_degree();
    coeff_modulus_size_ = parms.coeff_modulus().size();

    const auto &s = stream;

    data_rns_ = phantom::util::make_cuda_auto_ptr<uint64_t>(poly_modulus_degree_ * coeff_modulus_size_, s);

    auto prng_seed_error = make_cuda_auto_ptr<uint8_t>(phantom::util::global_variables::prng_seed_byte_count, s);
    random_bytes(prng_seed_error.get(), phantom::util::global_variables::prng_seed_byte_count, s);

    secret_key_array_ = make_cuda_auto_ptr<uint64_t>(poly_degree * coeff_mod_size, s);

    // Copy constant data to device constant memory
    random_bytes(prng_seed_error.get(), phantom::util::global_variables::prng_seed_byte_count, s);
    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    sample_ternary_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
            secret_key_array_.get(), prng_seed_error.get(), base_rns,
            poly_degree, coeff_mod_size);

    // Compute the NTT form of secret key and
    // save secret_key to the first coeff_mod_size * N elements of secret_key_array
    nwt_2d_radix8_forward_inplace(secret_key_array_.get(), context.gpu_rns_tables(), coeff_mod_size, 0, s);

    chain_index_ = 0;
    sk_max_power_ = 1;
    gen_flag_ = true;
}

PhantomPublicKey PhantomSecretKey::gen_publickey(const PhantomContext &context) const {
    PhantomPublicKey pk;

    const auto &s = phantom::util::global_variables::default_stream->get_stream();
    pk.prng_seed_a_ = make_cuda_auto_ptr<uint8_t>(phantom::util::global_variables::prng_seed_byte_count, s);
    random_bytes(pk.prng_seed_a_.get(), phantom::util::global_variables::prng_seed_byte_count, s);
    encrypt_zero_symmetric(context, pk.pk_, pk.prng_seed_a_.get(), 0, true, s);
    pk.pk_.chain_index_ = 0;

    pk.gen_flag_ = true;

    return pk;
}

PhantomRelinKey PhantomSecretKey::gen_relinkey(const PhantomContext &context) {
    PhantomRelinKey relin_key;

    const auto &s = phantom::util::global_variables::default_stream->get_stream();

    // Extract encryption parameters.
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto &key_modulus = key_parms.coeff_modulus();
    auto poly_degree = key_parms.poly_modulus_degree();
    auto coeff_mod_size = key_modulus.size();

    size_t max_power = 2;
    if (max_power > sk_max_power_) {
        compute_secret_key_array(context, max_power, s);
    }

    // Make sure we have enough secret keys computed
    uint64_t *sk_square = secret_key_array() + coeff_mod_size * poly_degree;
    generate_one_kswitch_key(context, sk_square, relin_key, s);
    relin_key.gen_flag_ = true;

    return relin_key;
}

PhantomGaloisKey PhantomSecretKey::create_galois_keys(const PhantomContext &context) const {
    PhantomGaloisKey galois_keys;

    // Extract encryption parameters.
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto &key_modulus = key_parms.coeff_modulus();
    auto &key_galois_tool = context.key_galois_tool_;
    auto poly_degree = key_parms.poly_modulus_degree();
    auto key_mod_size = key_modulus.size();

    const auto &s = phantom::util::global_variables::default_stream->get_stream();

    // get galois_elts
    auto &galois_elts = key_galois_tool->galois_elts();

    auto rotated_secret_key = make_cuda_auto_ptr<uint64_t>(key_mod_size * poly_degree, s);
    auto secret_key = secret_key_array_.get();

    auto relin_key_num = galois_elts.size();
    galois_keys.relin_keys_.resize(relin_key_num);

    for (size_t galois_elt_idx{0}; galois_elt_idx < relin_key_num; galois_elt_idx++) {
        auto galois_elt = galois_elts[galois_elt_idx];

        // Verify coprime conditions.
        if (!(galois_elt & 1) || (galois_elt >= poly_degree << 1)) {
            throw invalid_argument("Galois element is not valid");
        }
        // Rotate secret key for each coeff_modulus
        key_galois_tool->apply_galois_ntt(secret_key, key_mod_size, galois_elt_idx, rotated_secret_key.get(),
                                          s);
        PhantomRelinKey relin_key;
        generate_one_kswitch_key(context, rotated_secret_key.get(), relin_key, s);
        galois_keys.relin_keys_[galois_elt_idx] = std::move(relin_key);
    }
    galois_keys.gen_flag_ = true;

    return galois_keys;
}

void PhantomSecretKey::encrypt_symmetric(const PhantomContext &context, const PhantomPlaintext &plain,
                                         PhantomCiphertext &cipher,
                                         const phantom::util::cuda_stream_wrapper &stream_wrapper) const {
    auto &context_data = context.get_context_data(0); // Use key_parm_id for obtaining scheme
    auto &parms = context_data.parms();
    auto scheme = parms.scheme();
    bool is_ntt_form = false;

    const auto &s = stream_wrapper.get_stream();

    auto prng_seed_a = make_cuda_auto_ptr<uint8_t>(phantom::util::global_variables::prng_seed_byte_count, s);
    random_bytes(prng_seed_a.get(), phantom::util::global_variables::prng_seed_byte_count, s);

    cipher.scale_ = 1.0;
    cipher.correction_factor_ = 1;
    cipher.noiseScaleDeg_ = 1;

    if (scheme == phantom::scheme_type::bfv) {
        encrypt_zero_symmetric(context, cipher, prng_seed_a.get(), context.get_first_index(), is_ntt_form, s);
        // calculate [plain * coeff / plain-modulus].
        // return [plain * coeff / plain-modulus + c0, c1]
        multiply_add_plain_with_scaling_variant(context, plain, context.get_first_index(), cipher, s);
    } else if (scheme == phantom::scheme_type::ckks) {
        is_ntt_form = true;
        // [c0, c1] is the encrytion of 0, key idea is use the plain's chain_index to find corresponding data
        encrypt_zero_symmetric(context, cipher, prng_seed_a.get(), plain.chain_index(), is_ntt_form, s);

        // [c0 + plaintext, c1]
        auto &ckks_context_data = context.get_context_data(plain.chain_index());
        auto &ckks_parms = ckks_context_data.parms();
        auto poly_degree = ckks_parms.poly_modulus_degree();
        size_t ckks_coeff_mod_size = ckks_parms.coeff_modulus().size();

        // c0 = c0 + plaintext
        uint64_t gridDimGlb = poly_degree * ckks_coeff_mod_size / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                cipher.data(), plain.data(), context.gpu_rns_tables().modulus(),
                cipher.data(), poly_degree, ckks_coeff_mod_size);

        cipher.scale_ = plain.scale();
    } else if (scheme == phantom::scheme_type::bgv) {
        is_ntt_form = true;
        // (c[0], c[1]) = ([-(as+te)]_q, a)
        encrypt_zero_symmetric(context, cipher, prng_seed_a.get(), context.get_first_index(), is_ntt_form, s);

        auto &bgv_context_data = context.get_context_data(context.get_first_index());
        auto &bgv_parms = bgv_context_data.parms();
        auto poly_degree = bgv_parms.poly_modulus_degree();
        auto bgv_coeff_mod_size = bgv_parms.coeff_modulus().size();
        auto base_rns = context.gpu_rns_tables().modulus();

        // c0 = c0 + plaintext
        auto plain_copy = make_cuda_auto_ptr<uint64_t>(bgv_coeff_mod_size * poly_degree, s);
        for (size_t i = 0; i < bgv_coeff_mod_size; i++) {
            cudaMemcpyAsync(plain_copy.get() + i * poly_degree, plain.data(), sizeof(uint64_t) * poly_degree,
                            cudaMemcpyDeviceToDevice, s);
        }

        nwt_2d_radix8_forward_inplace(plain_copy.get(), context.gpu_rns_tables(), bgv_coeff_mod_size, 0, s);
        uint64_t gridDimGlb = poly_degree * bgv_coeff_mod_size / blockDimGlb.x;
        add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                cipher.data(), plain_copy.get(), base_rns, cipher.data(), poly_degree,
                bgv_coeff_mod_size);
    } else {
        throw std::invalid_argument("unsupported scheme.");
    }
}

void PhantomSecretKey::ckks_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                    PhantomPlaintext &destination, const cudaStream_t &stream) {
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
        compute_secret_key_array(context, needed_sk_power, stream);
    }

    uint64_t *c0 = encrypted.data();

    cudaMemcpyAsync(destination.data(), c0, coeff_mod_size * poly_degree * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);
    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    for (size_t i = 1; i <= needed_sk_power; i++) {
        uint64_t *ci = encrypted.data() + i * coeff_mod_size * poly_degree;
        uint64_t *si = secret_key_array() + (i - 1) * coeff_modulus_size_ * poly_degree;
        // c_0 += c_j * s^{j}
        multiply_and_add_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                ci, si, destination.data(), base_rns, destination.data(),
                poly_degree, coeff_mod_size);
    }

    // Set destination parameters as in ciphertext
    destination.chain_index_ = encrypted.chain_index();
    destination.scale_ = encrypted.scale();
}

void PhantomSecretKey::bfv_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                   PhantomPlaintext &destination, const cudaStream_t &stream) {
    auto chain_index = encrypted.chain_index_;
    auto coeff_mod_size = encrypted.coeff_modulus_size_;
    auto poly_degree = encrypted.poly_modulus_degree_;
    auto base_rns = context.gpu_rns_tables().modulus();
    auto poly_num = encrypted.size_;
    auto needed_sk_power = poly_num - 1;

    if (needed_sk_power > sk_max_power_) {
        compute_secret_key_array(context, needed_sk_power, stream);
    }

    uint64_t *c0 = encrypted.data();

    // Firstly find c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q
    // This is equal to Delta m + v where ||v|| < Delta/2.
    // Add Delta / 2 and now we have something which is Delta * (m + epsilon) where epsilon < 1
    // Therefore, we can (integer) divide by Delta and the answer will round down to m.

    // Compute c_1 *s, ..., c_{count-1} * s^{count-1} mod q
    // The secret key powers are already NTT transformed.
    auto inner_prod = make_cuda_auto_ptr<uint64_t>(coeff_mod_size * poly_degree, stream);
    cudaMemsetAsync(inner_prod.get(), 0UL, coeff_mod_size * poly_degree * sizeof(uint64_t), stream);
    auto temp = make_cuda_auto_ptr<uint64_t>(coeff_mod_size * poly_degree, stream);

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    for (size_t i = 1; i <= needed_sk_power; i++) {
        uint64_t *ci = encrypted.data() + i * coeff_mod_size * poly_degree;
        uint64_t *si = secret_key_array() + (i - 1) * coeff_modulus_size_ * poly_degree;
        cudaMemcpyAsync(temp.get(), ci, coeff_mod_size * poly_degree * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                        stream);
        // Change ci to NTT form
        nwt_2d_radix8_forward_inplace(temp.get(), context.gpu_rns_tables(), coeff_mod_size, 0, stream);
        // c1 = c1*s^1 + c2*s^2 + ......
        multiply_and_add_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                temp.get(), si, inner_prod.get(), base_rns,
                inner_prod.get(), poly_degree, coeff_mod_size);
    }

    // change c_1 to normal form
    nwt_2d_radix8_backward_inplace(inner_prod.get(), context.gpu_rns_tables(), coeff_mod_size, 0, stream);

    // finally, c_0 = c_0 + c_1
    add_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            c0, inner_prod.get(), base_rns, inner_prod.get(), poly_degree, coeff_mod_size);

    auto mul_tech = context.mul_tech();

    if (mul_tech == mul_tech_type::behz) {
        // Divide scaling variant using BEHZ FullRNS techniques
        context.get_context_data(chain_index).gpu_rns_tool().behz_decrypt_scale_and_round(
                inner_prod.get(), temp.get(), context.gpu_rns_tables(), coeff_mod_size,
                poly_degree, destination.data(), stream);
    } else if (mul_tech == mul_tech_type::hps || mul_tech == mul_tech_type::hps_overq ||
               mul_tech == mul_tech_type::hps_overq_leveled) {
        // HPS scale and round
        context.get_context_data(chain_index).gpu_rns_tool().hps_decrypt_scale_and_round(
                destination.data(), inner_prod.get(), stream);
    } else {
        throw std::invalid_argument("BFV decrypt mul_tech not supported");
    }

    // Set destination parameters as in ciphertext
    destination.set_chain_index(0);
}

void PhantomSecretKey::bgv_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                   PhantomPlaintext &destination, const cudaStream_t &stream) {
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
        compute_secret_key_array(context, needed_sk_power, stream);
    }

    uint64_t *c0 = encrypted.data();
    auto inner_prod = make_cuda_auto_ptr<uint64_t>(coeff_mod_size * poly_degree, stream);
    cudaMemcpyAsync(inner_prod.get(), c0, coeff_mod_size * poly_degree * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    for (size_t i = 1; i <= needed_sk_power; i++) {
        uint64_t *ci = encrypted.data() + i * coeff_mod_size * poly_degree;
        uint64_t *si = secret_key_array() + (i - 1) * coeff_modulus_size_ * poly_degree;
        // c_0 += c_j * s^{j}
        multiply_and_add_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                ci, si, inner_prod.get(), base_rns, inner_prod.get(),
                poly_degree, coeff_mod_size);
    }

    nwt_2d_radix8_backward_inplace(inner_prod.get(), context.gpu_rns_tables(), coeff_mod_size, 0, stream);

    // Set destination parameters as in ciphertext
    destination.chain_index_ = 0;
    destination.scale_ = encrypted.scale();
    rns_tool.decrypt_mod_t(destination.data(), inner_prod.get(), poly_degree, stream);

    if (encrypted.correction_factor() != 1) {
        uint64_t fix = 1;
        if (!try_invert_uint_mod(encrypted.correction_factor(), plain_modulus, fix)) {
            throw logic_error("invalid correction factor");
        }

        gridDimGlb = poly_degree / blockDimGlb.x;
        multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                destination.data(), fix, context.gpu_plain_tables().modulus(), destination.data(), poly_degree, 1);
    }
}

void PhantomSecretKey::decrypt(const PhantomContext &context, const PhantomCiphertext &cipher,
                               PhantomPlaintext &plain, const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    auto &context_data = context.get_context_data(cipher.chain_index());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto poly_degree = parms.poly_modulus_degree();
    auto coeff_mod_size = coeff_modulus.size();
    auto scheme = parms.scheme();

    const auto &s = stream_wrapper.get_stream();

    // init plaintext
    if (parms.scheme() == phantom::scheme_type::bfv || parms.scheme() == phantom::scheme_type::bgv)
        plain.coeff_modulus_size_ = 1;
    else if (parms.scheme() == phantom::scheme_type::ckks)
        plain.coeff_modulus_size_ = coeff_mod_size;
    else
        throw std::invalid_argument("Unsupported FHE scheme.");
    plain.poly_modulus_degree_ = poly_degree;
    plain.resize(plain.coeff_modulus_size_, plain.poly_modulus_degree_, s);

    if (scheme == phantom::scheme_type::bfv) {
        bfv_decrypt(context, cipher, plain, s);
    } else if (scheme == phantom::scheme_type::ckks) {
        ckks_decrypt(context, cipher, plain, s);
    } else if (scheme == phantom::scheme_type::bgv) {
        bgv_decrypt(context, cipher, plain, s);
    } else {
        throw std::invalid_argument("unsupported scheme.");
    }
}

// Compute the infinity norm of poly
static void poly_infinity_norm_coeffmod(const uint64_t *poly, size_t coeff_count, size_t coeff_uint64_count,
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

int PhantomSecretKey::invariant_noise_budget(const PhantomContext &context,
                                             const PhantomCiphertext &cipher,
                                             const phantom::util::cuda_stream_wrapper &stream_wrapper) {
    const auto &s = stream_wrapper.get_stream();

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
        compute_secret_key_array(context, needed_sk_power, s);
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
        nwt_2d_radix8_forward_inplace(ci, context.gpu_rns_tables(), coeff_mod_size, 0, s);
        // ci * s^{i} in NTT form
        if (i == 1) {
            // c1 = c1 * s^1
            multiply_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    ci, si, base_rns, ci, poly_degree, coeff_mod_size);
        } else {
            // c1 = c1*s^1 + c2*s^2 + ......
            multiply_and_add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
                    ci, si, c1, base_rns, c1, poly_degree, coeff_mod_size);
        }
    }

    // change c_1 to normal form
    nwt_2d_radix8_backward_inplace(c1, context.gpu_rns_tables(), coeff_mod_size, 0, s);
    // finally, c_0 = c_0 + c_1
    add_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
            c0, c1, base_rns, c0, poly_degree, coeff_mod_size);

    // compute c0 * plain_modulus
    multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(
            c0, plain_modulus.value(), base_rns, c0, poly_degree, coeff_mod_size);

    // Copy noise_poly to Host
    std::vector<uint64_t> host_noise_poly(coeff_mod_size * poly_degree);
    cudaMemcpyAsync(host_noise_poly.data(), c0, coeff_mod_size * poly_degree * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                    s);

    // CRT-compose the noise
    auto &base_q = context.get_context_data_rns_tool(chain_index).host_base_Ql();
    base_q.compose_array(host_noise_poly.data(), poly_degree);

    // Next we compute the infinity norm mod parms.coeff_modulus()
    std::vector<uint64_t> norm(coeff_mod_size);
    std::vector<uint64_t> modulus;
    for (size_t idx{0}; idx < coeff_mod_size; idx++) {
        modulus.push_back(parms.coeff_modulus().at(idx).value());
    }
    auto total_coeff_modulus = context_data.total_coeff_modulus();
    poly_infinity_norm_coeffmod(host_noise_poly.data(), poly_degree, coeff_mod_size, total_coeff_modulus.data(),
                                norm.data());

    // The -1 accounts for scaling the invariant noise by 2;
    // note that we already took plain_modulus into account in compose
    // so no need to subtract log(plain_modulus) from this
    int bit_count_diff = context_data.total_coeff_modulus_bit_count() -
                         get_significant_bit_count_uint(norm.data(), coeff_mod_size) - 1;

    return max(0, bit_count_diff);
}

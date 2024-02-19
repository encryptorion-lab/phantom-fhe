#pragma once

#include <cmath>

#include "ciphertext.h"
#include "context.cuh"
#include "ntt.cuh"
#include "plaintext.h"
#include "secretkey.h"

/**
 * Negates a ciphertext.
 * @param[in] encrypted The ciphertext to negate
 * @throws std::invalid_argument if encrypted is not valid for the encryption parameters
 */
void negate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted);

inline void negate(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomCiphertext &destination) {
    destination = encrypted;
    negate_inplace(context, destination);
}

/**
 * Adds two ciphertexts. This function adds together encrypted1 and encrypted2 and stores the result in encrypted1.
 * @param[in] encrypted1 The first ciphertext to add
 * @param[in] encrypted2 The second ciphertext to add
 */
void add_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);

inline void add(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                PhantomCiphertext &destination) {
    if (&encrypted2 == &destination) {
        add_inplace(context, destination, encrypted1);
    }
    else {
        destination = encrypted1;
        add_inplace(context, destination, encrypted2);
    }
}

void add_many(const PhantomContext &context, const std::vector<PhantomCiphertext> &encrypteds,
              PhantomCiphertext &destination);

void sub_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                 const bool &negate);

inline void sub(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                PhantomCiphertext &destination) {
    if (&encrypted2 == &destination) {
        sub_inplace(context, destination, encrypted1, true);
    }
    else {
        destination = encrypted1;
        sub_inplace(context, destination, encrypted2, false);
    }
}

// encrypted = encrypted + plain
void add_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

inline void add_plain(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                      PhantomCiphertext &destination) {
    if (&encrypted == &destination)
        add_plain_inplace(context, encrypted, plain);
    else {
        destination = encrypted;
        add_plain_inplace(context, destination, plain);
    }
}

// encrypted = encrypted - plain
void sub_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

inline void sub_plain(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                      PhantomCiphertext &destination) {
    if (&encrypted == &destination)
        sub_plain_inplace(context, encrypted, plain);
    else {
        destination = encrypted;
        sub_plain_inplace(context, destination, plain);
    }
}

// encrypted1 = encrypted1 * encrypted2
void bfv_multiply(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);

void bfv_multiply_behz(const PhantomContext &context, PhantomCiphertext &encrypted1,
                       const PhantomCiphertext &encrypted2);

size_t FindLevelsToDrop(const PhantomContext &context, size_t multiplicativeDepth, double dcrtBits, bool isKeySwitch,
                        bool isAsymmetric);

void bfv_multiply_hps(const PhantomContext &context, PhantomCiphertext &encrypted1,
                      const PhantomCiphertext &encrypted2);

void bfv_mul_relin_hps(const PhantomContext &context, PhantomCiphertext &encrypted1,
                       const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys);

// encrypted1 = encrypted1 * encrypted2
void bgv_ckks_multiply(const PhantomContext &context, PhantomCiphertext &encrypted1,
                       const PhantomCiphertext &encrypted2);

// encrypted1 = encrypted1 * encrypted2
void multiply_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1,
                      const PhantomCiphertext &encrypted2);

inline void multiply(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                     PhantomCiphertext &destination) {
    if (&encrypted2 == &destination)
        multiply_inplace(context, destination, encrypted1);
    else {
        destination = encrypted1;
        multiply_inplace(context, destination, encrypted2);
    }
}

void multiply_and_relin_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1,
                                const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys);

inline void multiply_and_relin(const PhantomContext &context, PhantomCiphertext &encrypted1,
                               const PhantomCiphertext &encrypted2, PhantomCiphertext &destination,
                               const PhantomRelinKey &relin_keys) {
    if (&encrypted2 == &destination)
        multiply_and_relin_inplace(context, destination, encrypted1, relin_keys);
    else {
        destination = encrypted1;
        multiply_and_relin_inplace(context, destination, encrypted2, relin_keys);
    }
}

void multiply_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

inline void multiply_plain(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                           PhantomCiphertext &destination) {
    if (&encrypted == &destination)
        multiply_plain_inplace(context, encrypted, plain);
    else {
        destination = encrypted;
        multiply_plain_inplace(context, destination, plain);
    }
}

__global__ void key_switch_inner_prod_c2_and_evk(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                                                 const DModulus *modulus, size_t n, size_t size_QP, size_t size_QP_n,
                                                 size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                                                 size_t beta, size_t reduction_threshold);

// used by relinearize_internal
void key_switch_inner_prod(uint64_t *p_cx, const uint64_t *p_t_mod_up, const uint64_t *const *rlk,
                           const phantom::DRNSTool &rns_tool, const DModulus *modulus_QP,
                           const size_t reduction_threshold);

void switch_key_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
                        const PhantomRelinKey &relin_keys, bool is_relin = false);

/**
 * Relinearizes a ciphertext. This functions relinearizes encrypted, reducing its size down to 2.
 * @param[in] context PhantomContext
 * @param[inout] encrypted The ciphertext to relinearize
 * @param[in] relin_keys The relinearization keys
 * @param[in] destination_size The size of ciphertext
 */
void relinearize_internal(const PhantomContext &context, PhantomCiphertext &encrypted,
                          const PhantomRelinKey &relin_keys, size_t destination_size);

/**
 * Relinearizes a ciphertext. This functions relinearizes encrypted, reducing its size down to 2.
 * @param[in] context PhantomContext
 * @param[inout] encrypted The ciphertext to relinearize
 * @param[in] relin_keys The relinearization keys
 */
inline void relinearize_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                                const PhantomRelinKey &relin_keys) {
    relinearize_internal(context, encrypted, relin_keys, 2);
}

inline void relinearize(const PhantomContext &context, PhantomCiphertext &encrypted, PhantomCiphertext &destination,
                        const PhantomRelinKey &relin_keys) {
    if (&encrypted == &destination)
        relinearize_inplace(context, encrypted, relin_keys);
    else {
        destination = encrypted;
        relinearize_inplace(context, destination, relin_keys);
    }
}

/** Modulus switches an NTT transformed plaintext from modulo q_1...q_k down to modulo q_1...q_{k-1}.
 * @param[in] context PhantomContext
 * @param[inout] plain PhantomPlaintext
 */
void mod_switch_to_next_inplace(const PhantomContext &context, PhantomPlaintext &plain);

/** Modulus switches an NTT transformed plaintext from modulo q_1...q_k down to modulo q_1...q_{k-1}.
 * @param[in] context PhantomContext
 * @param[in] plain PhantomPlaintext
 * @param[out] destination The result
 */
void mod_switch_to_next(const PhantomContext &context, const PhantomPlaintext &plain, PhantomPlaintext &destination);

/** Given an NTT transformed plaintext modulo q_1...q_k, this function switches the modulus down until the
parameters reach the given chain_index.

 * @param[in] plain The plaintext to be switched to a smaller modulus
 * @param[in] chain_index The target chain_index
 * @throws std::invalid_argument if plain is not in NTT form
 * @throws std::invalid_argument if plain is not valid for the encryption parameters
 * @throws std::invalid_argument if parms_id is not valid for the encryption parameters
 * @throws std::invalid_argument if plain is already at lower level in modulus chain than the parameters
corresponding to parms_id
 * @throws std::invalid_argument if the scale is too large for the new encryption parameters
  */
void mod_switch_to_inplace(const PhantomContext &context, PhantomPlaintext &plain, size_t chain_index);

inline void mod_switch_to(const PhantomContext &context, const PhantomPlaintext &plain, PhantomPlaintext &destination,
                          size_t chain_index) {
    if (&plain == &destination) {
        mod_switch_to_inplace(context, destination, chain_index);
    }
    else {
        destination = plain;
        mod_switch_to_inplace(context, destination, chain_index);
    }
}

// /** Given an NTT transformed plaintext modulo q_1...q_k, this function switches the modulus down until the
// parameters reach the given chain_index.
//
//  * @param[in] context PhantomContext
//  * @param[in] plain The plaintext to be switched to a smaller modulus
//  * @param[in] parms_id The target parms_id
//  * @throws std::invalid_argument if plain is not in NTT form
//  * @throws std::invalid_argument if plain is not valid for the encryption parameters
//  * @throws std::invalid_argument if parms_id is not valid for the encryption parameters
//  * @throws std::invalid_argument if plain is already at lower level in modulus chain than the parameters
// corresponding to parms_id
//  * @throws std::invalid_argument if the scale is too large for the new encryption parameters
//   */
// inline void mod_switch_to_inplace(const PhantomContext &context, PhantomPlaintext &plain,
//                                   phantom::parms_id_type parms_id) {
//     auto parms_id_chain_index = context.get_chain_index(parms_id);
//     mod_switch_to_inplace(context, plain, parms_id_chain_index);
// }

// inline void mod_switch_to(const PhantomContext &context, const PhantomPlaintext &plain, phantom::parms_id_type
// parms_id,
//                           PhantomPlaintext &destination) {
//     if (&plain == &destination) {
//         mod_switch_to_inplace(context, destination, parms_id);
//     }
//     else {
//         destination = plain;
//         mod_switch_to_inplace(context, destination, parms_id);
//     }
// }

void mod_switch_scale_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted,
                              PhantomCiphertext &destination);

void mod_switch_drop_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted,
                             PhantomCiphertext &destination);

/** Given a ciphertext encrypted modulo q_1...q_k, this function switches the modulus down to q_1...q_{k-1} and
stores the result in the destination parameter.
 * @param[in] context PhantomContext
 * @param[in] encrypted The ciphertext to mod switch
 * @param[out] destination The result
 */
void mod_switch_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted,
                        PhantomCiphertext &destination);

/**
 * @param[in] context PhantomContext
 * @param[in] encrypted The ciphertext to be switched to a smaller modulus
 * @throws std::invalid_argument if encrypted is not valid for the encryption parameters
 * @throws std::invalid_argument if encrypted is not in the default NTT form
 * @throws std::invalid_argument if encrypted is already at lowest level
 * @throws std::invalid_argument if the scale is too large for the new encryption parameters
 * @throws std::invalid_argument if pool is uninitialized
 * @throws std::logic_error if result ciphertext is transparent
 */
inline void mod_switch_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted) {
    mod_switch_to_next(context, encrypted, encrypted);
}

/** Given a ciphertext encrypted modulo q_1...q_k, this function switches the modulus down until the parameters
reach the given chain_index.

 * @param[in] context PhantomContext
 * @param[in] encrypted The ciphertext to be switched to a smaller modulus
 * @param[in] chain_index The target chain_index
 */
void mod_switch_to_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t chain_index);

inline void mod_switch_to(const PhantomContext &context, const PhantomCiphertext &encrypted,
                          PhantomCiphertext &destination, size_t chain_index) {
    if (&encrypted == &destination)
        mod_switch_to_inplace(context, destination, chain_index);
    else {
        destination = encrypted;
        mod_switch_to_inplace(context, destination, chain_index);
    }
}

// /** Given a ciphertext encrypted modulo q_1...q_k, this function switches the modulus down until the parameters
// reach the given chain_index.
//
//  * @param[in] context PhantomContext
//  * @param[in] encrypted The ciphertext to be switched to a smaller modulus
//  * @param[in] parms_id The target parms_id
//  */
// inline void mod_switch_to_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
//                                   phantom::parms_id_type parms_id) {
//     auto parms_id_chain_index = context.get_chain_index(parms_id);
//     mod_switch_to_inplace(context, encrypted, parms_id_chain_index);
// }

// inline void mod_switch_to(const PhantomContext &context, const PhantomCiphertext &encrypted,
//                           phantom::parms_id_type parms_id, PhantomCiphertext &destination) {
//     if (&encrypted == &destination)
//         mod_switch_to_inplace(context, destination, parms_id);
//     else {
//         destination = encrypted;
//         mod_switch_to_inplace(context, destination, parms_id);
//     }
// }

/**Given a ciphertext encrypted modulo q_1...q_k, this function switches the modulus down to q_1...q_{k-1}, scales
  the message down accordingly, and stores the result in the destination parameter.
 * @param[in] context PhantomContext
 * @param[in] encrypted The ciphertext to  Rescale
 * @param[out] destination The result
 */
void rescale_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomCiphertext &destination);

inline void rescale_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted) {
    rescale_to_next(context, encrypted, encrypted);
}

/**Transforms a ciphertext to NTT domain. This functions applies David Harvey's Number Theoretic Transform
   separately to each polynomial of a ciphertext.

 * @param context PhantomContext
 * @param encrypted The ciphertext to transform
 */
void transform_to_ntt_inplace(const PhantomContext &context, PhantomCiphertext &encrypted);

inline void transform_to_ntt(const PhantomContext &context, PhantomCiphertext &encrypted,
                             PhantomCiphertext &destination_ntt) {
    if (&encrypted == &destination_ntt) {
        transform_to_ntt_inplace(context, encrypted);
    }
    else {
        destination_ntt = encrypted;
        transform_to_ntt_inplace(context, destination_ntt);
    }
}

/**Transforms a ciphertext to NTT domain. This functions applies David Harvey's Number Theoretic Transform
   separately to each polynomial of a ciphertext.

 * @param context PhantomContext
 * @param encrypted The ciphertext to transform
 */
void transform_from_ntt_inplace(const PhantomContext &context, PhantomCiphertext &encrypted_ntt);

inline void transform_from_ntt(const PhantomContext &context, PhantomCiphertext &encrypted_ntt,
                               PhantomCiphertext &destination) {
    if (&encrypted_ntt == &destination) {
        transform_from_ntt_inplace(context, encrypted_ntt);
    }
    else {
        destination = encrypted_ntt;
        transform_from_ntt_inplace(context, destination);
    }
}

void hoisting_inplace(const PhantomContext &context, PhantomCiphertext &ct, const PhantomGaloisKey &glk,
                      const std::vector<int> &steps);

inline auto hoisting(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomGaloisKey &glk,
                     const std::vector<int> &steps) {
    PhantomCiphertext destination = encrypted;
    hoisting_inplace(context, destination, glk, steps);
    return destination;
}

/**Applies a Galois automorphism to a ciphertext. To evaluate the Galois automorphism, an appropriate set of Galois
   keys must also be provided. Dynamic memory allocations in the process are allocated from the memory pool pointed
   to by the given MemoryPoolHandle.

   The desired Galois automorphism is given as a Galois element, and must be an odd integer in the interval
   [1, M-1], where M = 2*N, and N = poly_modulus_degree. Used with batching, a Galois element 3^i % M corresponds
   to a cyclic row rotation i steps to the left, and a Galois element 3^(N/2-i) % M corresponds to a cyclic row
   rotation i steps to the right. The Galois element M-1 corresponds to a column rotation (row swap) in BFV, and
   complex conjugation in CKKS. In the polynomial view (not batching), a Galois automorphism by a Galois element p
   changes Enc(plain(x)) to Enc(plain(x^p)).
 * @param context PhantomContext
 * @param encrypted The ciphertext to apply a Galois automorphism
 * @param galois_elt_index The index of galois_elt in galois_elts
 * @param galois_key The Galois keys
 */
void apply_galois_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t galois_elt_index,
                          const PhantomGaloisKey &galois_keys);

inline void apply_galois(const PhantomContext &context, PhantomCiphertext &encrypted, size_t galois_elt_index,
                         const PhantomGaloisKey &galois_keys, PhantomCiphertext &destination) {
    if (&encrypted == &destination)
        apply_galois_inplace(context, encrypted, galois_elt_index, galois_keys);
    else {
        destination = encrypted;
        apply_galois_inplace(context, destination, galois_elt_index, galois_keys);
    }
}

// using by complex_conjugate_inplace and rotate_columns_inplace
void conjugate_internal(const PhantomContext &context, PhantomCiphertext &encrypted,
                        const PhantomGaloisKey &galois_key);

// used by other rotate functions
void rotate_internal(const PhantomContext &context, PhantomCiphertext &encrypted, int step,
                     const PhantomGaloisKey &galois_key);

/**
  Rotates plaintext matrix rows cyclically. When batching is used with the BFV scheme, this function rotates the
  encrypted plaintext matrix rows cyclically to the left (steps > 0) or to the right (steps < 0). Since the size
  of the batched matrix is 2-by-(N/2), where N is the degree of the polynomial modulus, the number of steps to
  rotate must have absolute value at most N/2-1.
 * @param[in] context PhantomContext
 * @param[in] encrypted The ciphertext to rotate
 * @param[in] galois_keys The Galois keys
 */
inline void rotate_rows_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int steps,
                                const PhantomGaloisKey &galois_key) {
    if (context.key_context_data().parms().scheme() != phantom::scheme_type::bfv &&
        context.key_context_data().parms().scheme() != phantom::scheme_type::bgv) {
        throw std::logic_error("unsupported scheme");
    }
    rotate_internal(context, encrypted, steps, galois_key);
}

inline void rotate_rows(const PhantomContext &context, PhantomCiphertext &encrypted, int steps,
                        const PhantomGaloisKey &galois_key, PhantomCiphertext &destination) {
    if (&encrypted == &destination)
        rotate_rows_inplace(context, encrypted, steps, galois_key);
    else {
        destination = encrypted;
        rotate_rows_inplace(context, destination, steps, galois_key);
    }
}

/**
 Rotates plaintext matrix columns cyclically. When batching is used with the BFV scheme, this function rotates
 the encrypted plaintext matrix columns cyclically. Since the size of the batched matrix is 2-by-(N/2), where N
 is the degree of the polynomial modulus, this means simply swapping the two rows.
 * @param[in] context PhantomContext
 * @param[in] encrypted The ciphertext to rotate
 * @param[in] galois_keys The Galois keys
 */
inline void rotate_columns_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                                   const PhantomGaloisKey &galois_key) {
    if (context.key_context_data().parms().scheme() != phantom::scheme_type::bfv &&
        context.key_context_data().parms().scheme() != phantom::scheme_type::bgv) {
        throw std::logic_error("unsupported scheme");
    }
    conjugate_internal(context, encrypted, galois_key);
}

inline void rotate_columns(const PhantomContext &context, PhantomCiphertext &encrypted,
                           const PhantomGaloisKey &galois_key, PhantomCiphertext &destination) {
    if (&encrypted == &destination)
        rotate_columns_inplace(context, encrypted, galois_key);
    else {
        destination = encrypted;
        rotate_columns_inplace(context, destination, galois_key);
    }
}

/**
 Rotates plaintext vector cyclically. When using the CKKS scheme, this function rotates the encrypted plaintext
 vector cyclically to the left (steps > 0) or to the right (steps < 0). Since the size of the batched matrix is
 2-by-(N/2), where N is the degree of the polynomial modulus, the number of steps to rotate must have absolute
 value at most N/2-1.
 * @param[in] context PhantomContext
 * @param[in] encrypted The ciphertext to rotate
 * @param[in] steps The number of steps to rotate (positive left, negative right)
 * @param[in] galois_keys The Galois keys
 */
inline void rotate_vector_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int step,
                                  const PhantomGaloisKey &galois_key) {
    if (context.key_context_data().parms().scheme() != phantom::scheme_type::ckks) {
        throw std::logic_error("unsupported scheme");
    }
    rotate_internal(context, encrypted, step, galois_key);
}

inline void rotate_vector(const PhantomContext &context, PhantomCiphertext &encrypted, int step,
                          const PhantomGaloisKey &galois_key, PhantomCiphertext &destination) {
    if (&encrypted == &destination)
        rotate_vector_inplace(context, encrypted, step, galois_key);
    else {
        destination = encrypted;
        rotate_vector_inplace(context, destination, step, galois_key);
    }
}

/**
 Complex conjugates plaintext slot values. When using the CKKS scheme, this function complex conjugates all
 values in the underlying plaintext.
 * @param[in] context PhantomContext
 * @param[in] encrypted The ciphertext to rotate
 * @param[in] galois_keys The Galois keys
 */
inline void complex_conjugate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                                      const PhantomGaloisKey &galois_key) {
    if (context.key_context_data().parms().scheme() != phantom::scheme_type::ckks) {
        throw std::logic_error("unsupported scheme");
    }
    conjugate_internal(context, encrypted, galois_key);
}

inline void complex_conjugate(const PhantomContext &context, PhantomCiphertext &encrypted,
                              const PhantomGaloisKey &galois_key, PhantomCiphertext &destination) {
    if (&encrypted == &destination)
        complex_conjugate_inplace(context, encrypted, galois_key);
    else {
        destination = encrypted;
        complex_conjugate_inplace(context, destination, galois_key);
    }
}

[[nodiscard]] inline bool is_scale_within_bounds(double scale, const phantom::ContextData &context_data) noexcept {
    int scale_bit_count_bound = 0;
    switch (context_data.parms().scheme()) {
        case phantom::scheme_type::bfv:
        case phantom::scheme_type::bgv:
            scale_bit_count_bound = context_data.parms().plain_modulus().bit_count();
            break;
        case phantom::scheme_type::ckks:
            scale_bit_count_bound = context_data.total_coeff_modulus_bit_count();
            break;
        default:
            // Unsupported scheme; check will fail
            scale_bit_count_bound = -1;
    };

    return !(scale <= 0 || (static_cast<int>(log2(scale)) >= scale_bit_count_bound));
}

/**
Returns (f, e1, e2) such that
(1) e1 * factor1 = e2 * factor2 = f mod p;
(2) gcd(e1, p) = 1 and gcd(e2, p) = 1;
(3) abs(e1_bal) + abs(e2_bal) is minimal, where e1_bal and e2_bal represent e1 and e2 in (-p/2, p/2].
*/
[[nodiscard]] inline auto balance_correction_factors(uint64_t factor1, uint64_t factor2,
                                                     const phantom::Modulus &plain_modulus)
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
    if (!phantom::util::try_invert_uint_mod(factor1, plain_modulus, ratio)) {
        throw std::logic_error("invalid correction factor1");
    }
    ratio = phantom::util::multiply_uint_mod(ratio, factor2, plain_modulus);
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

        temp = phantom::util::sub_safe(prev_b, phantom::util::mul_safe(b, q));
        prev_b = b;
        b = temp;

        uint64_t a_mod = phantom::util::barrett_reduce_64(static_cast<uint64_t>(abs(a)), plain_modulus);
        if (a < 0) {
            a_mod = phantom::util::negate_uint_mod(a_mod, plain_modulus);
        }
        uint64_t b_mod = phantom::util::barrett_reduce_64(static_cast<uint64_t>(abs(b)), plain_modulus);
        if (b < 0) {
            b_mod = phantom::util::negate_uint_mod(b_mod, plain_modulus);
        }
        if (a_mod != 0 && phantom::util::gcd(a_mod, t) == 1) // which also implies gcd(b_mod, t) == 1
        {
            int64_t new_sum = sum_abs(a_mod, b_mod);
            if (new_sum < sum) {
                sum = new_sum;
                e1 = a_mod;
                e2 = b_mod;
            }
        }
    }
    return std::make_tuple(phantom::util::multiply_uint_mod(e1, factor1, plain_modulus), e1, e2);
}

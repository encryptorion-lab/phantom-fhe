#pragma once

#include <cmath>

#include "ciphertext.h"
#include "context.cuh"
#include "ntt.cuh"
#include "plaintext.h"
#include "secretkey.h"
#include "cuda_wrapper.cuh"

size_t FindLevelsToDrop(const PhantomContext &context, size_t multiplicativeDepth, double dcrtBits, bool isKeySwitch,
                        bool isAsymmetric);

__global__ void key_switch_inner_prod_c2_and_evk(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                                                 const DModulus *modulus, size_t n, size_t size_QP, size_t size_QP_n,
                                                 size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                                                 size_t beta, size_t reduction_threshold);

// used by keyswitch_inplace
void key_switch_inner_prod(uint64_t *p_cx, const uint64_t *p_t_mod_up, const uint64_t *const *rlk,
                           const phantom::DRNSTool &rns_tool, const DModulus *modulus_QP,
                           size_t reduction_threshold, const cudaStream_t &stream);

void keyswitch_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
                       const PhantomRelinKey &relin_keys,
                       bool is_relin, // false
                       const cudaStream_t &stream);

/***************************************************** Core APIs ******************************************************/

// encrypted = -encrypted
void negate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                    const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto negate(const PhantomContext &context, const PhantomCiphertext &encrypted,
                   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    negate_inplace(context, destination, stream_wrapper);
    return destination;
}

// encrypted1 += encrypted2
void add_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                 const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto add(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted1;
    add_inplace(context, destination, encrypted2, stream_wrapper);
    return destination;
}

// destination = encrypteds[0] + encrypteds[1] + ...
void add_many(const PhantomContext &context, const std::vector<PhantomCiphertext> &encrypteds,
              PhantomCiphertext &destination,
              const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

// if negate = false (default): encrypted1 -= encrypted2
// if negate = true: encrypted1 = encrypted2 - encrypted1
void sub_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                 const bool &negate = false,
                 const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto sub(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                const bool &negate = false,
                const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted1;
    sub_inplace(context, destination, encrypted2, negate, stream_wrapper);
    return destination;
}

// encrypted += plain
void add_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                       const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto add_plain(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                      const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    add_plain_inplace(context, destination, plain, stream_wrapper);
    return destination;
}

// encrypted -= plain
void sub_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                       const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto sub_plain(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                      const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    sub_plain_inplace(context, destination, plain, stream_wrapper);
    return destination;
}

// encrypted *= plain
void multiply_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                            const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto multiply_plain(const PhantomContext &context, const PhantomCiphertext &encrypted,
                           const PhantomPlaintext &plain,
                           const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    multiply_plain_inplace(context, destination, plain, stream_wrapper);
    return destination;
}

// encrypted1 *= encrypted2
void multiply_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                      const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto
multiply(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
         const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted1;
    multiply_inplace(context, destination, encrypted2, stream_wrapper);
    return destination;
}

// encrypted1 *= encrypted2
void multiply_and_relin_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1,
                                const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys,
                                const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto multiply_and_relin(const PhantomContext &context, const PhantomCiphertext &encrypted1,
                               const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys,
                               const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted1;
    multiply_and_relin_inplace(context, destination, encrypted2, relin_keys, stream_wrapper);
    return destination;
}

void relinearize_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                         const PhantomRelinKey &relin_keys,
                         const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto relinearize(const PhantomContext &context, const PhantomCiphertext &encrypted,
                        const PhantomRelinKey &relin_keys,
                        const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    relinearize_inplace(context, destination, relin_keys, stream_wrapper);
    return destination;
}

// ciphertext
[[nodiscard]]
PhantomCiphertext rescale_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                  const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

// ciphertext
inline void rescale_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                                    const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    encrypted = rescale_to_next(context, encrypted, stream_wrapper);
}

// ciphertext
[[nodiscard]]
PhantomCiphertext mod_switch_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                     const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

// ciphertext
inline void mod_switch_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                                       const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    encrypted = mod_switch_to_next(context, encrypted, stream_wrapper);
}

// ciphertext
inline auto mod_switch_to(const PhantomContext &context, const PhantomCiphertext &encrypted, size_t chain_index,
                          const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    if (encrypted.chain_index() > chain_index) {
        throw std::invalid_argument("cannot switch to higher level modulus");
    }

    PhantomCiphertext destination = encrypted;

    while (destination.chain_index() != chain_index) {
        mod_switch_to_next_inplace(context, destination, stream_wrapper);
    }

    return destination;
}

// ciphertext
inline void mod_switch_to_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t chain_index,
                                  const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    if (encrypted.chain_index() > chain_index) {
        throw std::invalid_argument("cannot switch to higher level modulus");
    }

    while (encrypted.chain_index() != chain_index) {
        mod_switch_to_next_inplace(context, encrypted, stream_wrapper);
    }
}

// plaintext
void mod_switch_to_next_inplace(const PhantomContext &context, PhantomPlaintext &plain,
                                const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

// plaintext
inline auto mod_switch_to_next(const PhantomContext &context, const PhantomPlaintext &plain,
                               const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomPlaintext destination = plain;
    mod_switch_to_next_inplace(context, destination, stream_wrapper);
    return destination;
}

// plaintext
inline void mod_switch_to_inplace(const PhantomContext &context, PhantomPlaintext &plain, size_t chain_index,
                                  const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    if (plain.chain_index() > chain_index) {
        throw std::invalid_argument("cannot switch to higher level modulus");
    }

    while (plain.chain_index() != chain_index) {
        mod_switch_to_next_inplace(context, plain, stream_wrapper);
    }
}

// plaintext
inline auto mod_switch_to(const PhantomContext &context, const PhantomPlaintext &plain, size_t chain_index,
                          const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    if (plain.chain_index() > chain_index) {
        throw std::invalid_argument("cannot switch to higher level modulus");
    }

    PhantomPlaintext destination = plain;

    while (destination.chain_index() != chain_index) {
        mod_switch_to_next_inplace(context, destination, stream_wrapper);
    }

    return destination;
}

void apply_galois_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t galois_elt_index,
                          const PhantomGaloisKey &galois_keys,
                          const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto apply_galois(const PhantomContext &context, const PhantomCiphertext &encrypted, size_t galois_elt_index,
                         const PhantomGaloisKey &galois_keys,
                         const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    apply_galois_inplace(context, destination, galois_elt_index, galois_keys, stream_wrapper);
    return destination;
}

void rotate_rows_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int steps,
                         const PhantomGaloisKey &galois_key,
                         const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto rotate_rows(const PhantomContext &context, const PhantomCiphertext &encrypted, int steps,
                        const PhantomGaloisKey &galois_key,
                        const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    rotate_rows_inplace(context, destination, steps, galois_key, stream_wrapper);
    return destination;
}

void rotate_columns_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                            const PhantomGaloisKey &galois_key,
                            const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto rotate_columns(const PhantomContext &context, const PhantomCiphertext &encrypted,
                           const PhantomGaloisKey &galois_key,
                           const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    rotate_columns_inplace(context, destination, galois_key, stream_wrapper);
    return destination;
}

void rotate_vector_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int step,
                           const PhantomGaloisKey &galois_key,
                           const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto rotate_vector(const PhantomContext &context, const PhantomCiphertext &encrypted, int step,
                          const PhantomGaloisKey &galois_key,
                          const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    rotate_vector_inplace(context, destination, step, galois_key, stream_wrapper);
    return destination;
}

void complex_conjugate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                               const PhantomGaloisKey &galois_key,
                               const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto complex_conjugate(const PhantomContext &context, const PhantomCiphertext &encrypted,
                              const PhantomGaloisKey &galois_key,
                              const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    complex_conjugate_inplace(context, destination, galois_key, stream_wrapper);
    return destination;
}

/*************************************************** Advanced APIs ****************************************************/

void hoisting_inplace(const PhantomContext &context, PhantomCiphertext &ct, const PhantomGaloisKey &glk,
                      const std::vector<int> &steps,
                      const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

inline auto hoisting(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomGaloisKey &glk,
                     const std::vector<int> &steps,
                     const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
    PhantomCiphertext destination = encrypted;
    hoisting_inplace(context, destination, glk, steps, stream_wrapper);
    return destination;
}

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

// used by switch_key_inplace
void key_switch_inner_prod(uint64_t *p_cx, const uint64_t *p_t_mod_up, const uint64_t *const *rlk,
                           const phantom::DRNSTool &rns_tool, const DModulus *modulus_QP,
                           size_t reduction_threshold, const cudaStream_t &stream);

/***************************************************** Core APIs ******************************************************/

// encrypted = -encrypted
void negate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                    const cudaStream_t &stream = nullptr);

// encrypted1 += encrypted2
void add_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                 const cudaStream_t &stream = nullptr);

// destination = encrypteds[0] + encrypteds[1] + ...
void add_many(const PhantomContext &context, const std::vector<PhantomCiphertext> &encrypteds,
              PhantomCiphertext &destination, const cudaStream_t &stream = nullptr);

// if negate = false (default): encrypted1 -= encrypted2
// if negate = true: encrypted1 = encrypted2 - encrypted1
void sub_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                 const bool &negate = false, const cudaStream_t &stream = nullptr);

// encrypted += plain
void add_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                       const cudaStream_t &stream = nullptr);

// encrypted -= plain
void sub_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                       const cudaStream_t &stream = nullptr);

// encrypted *= plain
void multiply_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain,
                            const cudaStream_t &stream = nullptr);

// encrypted1 *= encrypted2
void multiply_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                      const cudaStream_t &stream = nullptr);

// encrypted1 *= encrypted2
void multiply_and_relin_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1,
                                const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys,
                                const cudaStream_t &stream = nullptr);

void switch_key_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
                        const PhantomRelinKey &relin_keys,
                        bool is_relin, // false
                        const cudaStream_t &stream = nullptr);

void relinearize_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                         const PhantomRelinKey &relin_keys,
                         const cudaStream_t &stream = nullptr);

// ciphertext
void mod_switch_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted,
                        PhantomCiphertext &destination,
                        const cudaStream_t &stream = nullptr);

// ciphertext
inline void mod_switch_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                                       const cudaStream_t &stream = nullptr) {
    mod_switch_to_next(context, encrypted, encrypted, stream);
}

// ciphertext
inline void mod_switch_to_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t chain_index,
                                  const cudaStream_t &stream = nullptr) {
    if (encrypted.chain_index() > chain_index) {
        throw std::invalid_argument("cannot switch to higher level modulus");
    }

    while (encrypted.chain_index() != chain_index) {
        mod_switch_to_next_inplace(context, encrypted, stream);
    }
}

// plaintext
void mod_switch_to_next_inplace(const PhantomContext &context, PhantomPlaintext &plain,
                                const cudaStream_t &stream = nullptr);

// plaintext
inline void mod_switch_to_inplace(const PhantomContext &context, PhantomPlaintext &plain, size_t chain_index,
                                  const cudaStream_t &stream = nullptr) {
    if (plain.chain_index() > chain_index) {
        throw std::invalid_argument("cannot switch to higher level modulus");
    }

    while (plain.chain_index() != chain_index) {
        mod_switch_to_next_inplace(context, plain, stream);
    }
}

void rescale_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomCiphertext &destination,
                     const cudaStream_t &stream = nullptr);

inline void rescale_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                                    const cudaStream_t &stream = nullptr) {
    rescale_to_next(context, encrypted, encrypted, stream);
}

void apply_galois_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t galois_elt_index,
                          const PhantomGaloisKey &galois_keys, const cudaStream_t &stream = nullptr);

void rotate_rows_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int steps,
                         const PhantomGaloisKey &galois_key, const cudaStream_t &stream = nullptr);

void rotate_columns_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                            const PhantomGaloisKey &galois_key, const cudaStream_t &stream = nullptr);

void rotate_vector_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int step,
                           const PhantomGaloisKey &galois_key, const cudaStream_t &stream = nullptr);

void complex_conjugate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                               const PhantomGaloisKey &galois_key, const cudaStream_t &stream = nullptr);

/*************************************************** Advanced APIs ****************************************************/

void hoisting_inplace(const PhantomContext &context, PhantomCiphertext &ct, const PhantomGaloisKey &glk,
                      const std::vector<int> &steps, const cudaStream_t &stream = nullptr);

inline auto hoisting(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomGaloisKey &glk,
                     const std::vector<int> &steps, const cudaStream_t &stream = nullptr) {
    PhantomCiphertext destination = encrypted;
    hoisting_inplace(context, destination, glk, steps, stream);
    return destination;
}

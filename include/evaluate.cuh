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

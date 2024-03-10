#include "polymath.cuh"

//#include <rmm/device_scalar.hpp>
//#include <rmm/device_vector.hpp>
//#include <thrust/device_vector.h>
//#include <thrust/reduce.h>

using namespace phantom::util;
using namespace phantom::arith;

/**  res = - operand % coeff_mod
 * @param[in] operand Operand1
 * @param[in] DModulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] poly_degree The degree of poly
 */
__global__ void negate_rns_poly(const uint64_t *operand,
                                const DModulus *modulus,
                                uint64_t *result,
                                const uint64_t poly_degree,
                                const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();
        uint64_t op = operand[tid];

        uint64_t non_zero = (op != 0);
        result[tid] = (mod_value - op) & static_cast<uint64_t>(-non_zero);
    }
}

/**  res = operand1 + operand2 % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] poly_degree The degree of poly
 */
__global__ void add_rns_poly(const uint64_t *operand1,
                             const uint64_t *operand2,
                             const DModulus *modulus,
                             uint64_t *result,
                             const uint64_t poly_degree,
                             const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();

        result[tid] = add_uint64_uint64_mod(operand1[tid], operand2[tid], mod_value);
    }
}

__global__ void add_std_cipher(const uint64_t *cipher1,
                               const uint64_t *cipher2,
                               const DModulus *modulus,
                               uint64_t *result,
                               const uint64_t poly_degree,
                               const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();
        uint64_t rns_coeff_count = poly_degree * coeff_mod_size;

        result[tid] = add_uint64_uint64_mod(cipher1[tid], cipher2[tid], mod_value);
        result[tid + rns_coeff_count] = add_uint64_uint64_mod(cipher1[tid + rns_coeff_count],
                                                              cipher2[tid + rns_coeff_count], mod_value);
    }
}

/**  res = -(operand1 + operand2) % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void add_and_negate_rns_poly(const uint64_t *operand1,
                                        const uint64_t *operand2,
                                        const DModulus *modulus,
                                        uint64_t *result,
                                        const uint64_t poly_degree,
                                        const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();

        uint64_t temp;
        temp = add_uint64_uint64_mod(operand1[tid], operand2[tid], mod_value);
        uint64_t non_zero = (temp != 0);
        result[tid] = (mod_value - temp) & static_cast<uint64_t>(-non_zero);
    }
}

/**  res = operand1 - operand2 % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void sub_rns_poly(const uint64_t *operand1,
                             const uint64_t *operand2,
                             const DModulus *modulus,
                             uint64_t *result,
                             const uint64_t poly_degree,
                             const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();

        result[tid] = sub_uint64_uint64_mod(operand1[tid], operand2[tid], mod_value);
    }
}

__global__ void add_many_rns_poly(const uint64_t *const *operands,
                                  const uint64_t add_size,
                                  const DModulus *modulus,
                                  uint64_t *result,
                                  const uint32_t poly_index,
                                  const uint32_t poly_degree,
                                  const uint32_t coeff_mod_size,
                                  const uint64_t reduction_threshold) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        uint32_t rns_offset = poly_index * poly_degree * coeff_mod_size;
        DModulus mod = modulus[twr];

        uint64_t temp = operands[0U][tid + rns_offset];
        for (uint32_t i = 1; i < add_size; i++) {
            if (i && reduction_threshold == 0) {
                // in case of overflow
                temp = barrett_reduce_uint64_uint64(temp, mod.value(), mod.const_ratio()[1]);
            }
            temp += operands[i][tid + rns_offset];
        }
        temp = barrett_reduce_uint64_uint64(temp, mod.value(), mod.const_ratio()[1]);

        result[tid + rns_offset] = temp;
    }
}

/**  res = operand1 * operand2 % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void multiply_rns_poly(const uint64_t *operand1,
                                  const uint64_t *operand2,
                                  const DModulus *modulus,
                                  uint64_t *result,
                                  const uint64_t poly_degree,
                                  const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        result[tid] = multiply_and_barrett_reduce_uint64(operand1[tid],
                                                         operand2[tid],
                                                         mod.value(),
                                                         mod.const_ratio());
    }
}

/**  res = operand1 * operand2 % coeff_mod
 * @param[in] operand Operand1
 * @param[in] scalar Operand2
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void multiply_scalar_rns_poly(const uint64_t *operand,
                                         const uint64_t scale,
                                         const DModulus *modulus,
                                         uint64_t *result,
                                         const uint64_t poly_degree,
                                         const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        result[tid] = multiply_and_barrett_reduce_uint64(operand[tid], scale, mod.value(), mod.const_ratio());
    }
}

__global__ void multiply_scalar_rns_poly(const uint64_t *operand,
                                         const uint64_t *scalar,
                                         const uint64_t *scalar_shoup,
                                         const DModulus *modulus,
                                         uint64_t *result,
                                         const uint64_t poly_degree,
                                         const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        const size_t twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();
        uint64_t scalar_value = scalar[twr];
        uint64_t scalar_value_shoup = scalar_shoup[twr];

        result[tid] = multiply_and_reduce_shoup(operand[tid], scalar_value, scalar_value_shoup, mod_value);
    }
}

/**  res = (operand1 * operand2 + operand3) % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] operand3 Operand3
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void multiply_and_add_rns_poly(const uint64_t *operand1,
                                          const uint64_t *operand2,
                                          const uint64_t *operand3,
                                          const DModulus *modulus,
                                          uint64_t *result,
                                          const uint64_t poly_degree,
                                          const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        uint128_t prod, sum;

        prod = multiply_uint64_uint64(operand1[tid], operand2[tid]);
        sum = add_uint128_uint64(prod, operand3[tid]);
        result[tid] = barrett_reduce_uint128_uint64(sum, mod.value(), mod.const_ratio());
    }
}

__global__ void multiply_scalar_and_add_rns_poly(const uint64_t *operand1,
                                                 const uint64_t *operand2,
                                                 const uint64_t scalar,
                                                 const DModulus *modulus,
                                                 uint64_t *result,
                                                 const uint64_t poly_degree,
                                                 const uint64_t coeff_mod_size) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        int twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        uint128_t prod, sum;

        prod = multiply_uint64_uint64(operand2[tid], scalar);
        sum = add_uint128_uint64(prod, operand1[tid]);
        result[tid] = barrett_reduce_uint128_uint64(sum, mod.value(), mod.const_ratio());
    }
}

__global__ void multiply_scalar_and_sub_rns_poly(const uint64_t *operand1,
                                                 const uint64_t *operand2,
                                                 const uint64_t scalar,
                                                 const DModulus *modulus,
                                                 uint64_t *result,
                                                 const uint64_t poly_degree,
                                                 const uint64_t coeff_mod_size) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        int twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        uint64_t res;

        res = multiply_and_barrett_reduce_uint64(operand2[tid], scalar, mod.value(), mod.const_ratio());
        result[tid] = sub_uint64_uint64_mod(operand1[tid], res, mod.value());
    }
}

/**  res = (operand1 * operand2 + operand3 * scale) % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] operand3 Operand3
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void multiply_and_scale_add_rns_poly(const uint64_t *operand1,
                                                const uint64_t *operand2,
                                                const uint64_t *operand3,
                                                const uint64_t &scale,
                                                const DModulus *modulus,
                                                uint64_t *result,
                                                const uint64_t poly_degree,
                                                const uint64_t coeff_mod_size) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        int twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        uint128_t prod, sum;

        prod = multiply_uint64_uint64(operand1[tid], operand2[tid]);
        sum = multiply_uint64_uint64(operand3[tid], scale);
        add_uint128_uint128(sum, prod, sum);
        result[tid] = barrett_reduce_uint128_uint64(sum, mod.value(), mod.const_ratio());
    }
}

__global__ void multiply_temp_mod_and_add_rns_poly(const uint64_t *operand1,
                                                   const uint64_t *const *operand2,
                                                   const DModulus *modulus,
                                                   uint64_t **result,
                                                   const size_t n,
                                                   const size_t dnum,
                                                   const size_t alpha,
                                                   const uint64_t *bigP_mod_q,
                                                   const uint64_t *bigP_mod_q_shoup) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n * dnum * alpha;
         tid += blockDim.x * gridDim.x) {
        const size_t twr = tid / (alpha * n);
        const size_t mod_idx = tid / n;
        uint64_t qi = modulus[mod_idx].value();
        uint64_t factor = bigP_mod_q[mod_idx];
        uint64_t factor_shoup = bigP_mod_q_shoup[mod_idx];
        uint64_t tmp = multiply_and_reduce_shoup(operand1[tid], factor, factor_shoup, qi);
        result[twr][tid] = add_uint64_uint64_mod(operand2[twr][tid], tmp, qi);
    }
}

/**  res = -(c1 * s + e) % coeff_mod
 * rlwr distribution: (a, b), generate b
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] operand3 Operand3
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void multiply_and_add_negate_rns_poly(const uint64_t *operand1,
                                                 const uint64_t *operand2,
                                                 const uint64_t *operand3,
                                                 const DModulus *modulus,
                                                 uint64_t *result,
                                                 const uint64_t poly_degree,
                                                 const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        uint128_t product, sum;
        uint64_t red;
        product = multiply_uint64_uint64(operand1[tid], operand2[tid]);
        sum = add_uint128_uint64(product, operand3[tid]);
        red = barrett_reduce_uint128_uint64(sum, mod.value(), mod.const_ratio());
        uint64_t non_zero = (red != 0);
        result[tid] = (mod.value() - red) & static_cast<uint64_t>(-non_zero);
    }
}

// out = (operand1 - operand2) * scale % modulus
__global__ void sub_and_scale_single_mod_poly(const uint64_t *operand1,
                                              const uint64_t *operand2,
                                              const uint64_t scale,
                                              const uint64_t scale_shoup,
                                              const uint64_t modulus,
                                              uint64_t *result,
                                              const uint64_t poly_degree) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree;
         tid += blockDim.x * gridDim.x) {
        uint64_t temp;
        sub_uint64_uint64(modulus, operand2[tid], temp);
        add_uint64_uint64(operand1[tid], temp, temp);
        // temp = sub_uint64_uint64_mod(operand1[tid], operand2[tid], modulus);
        result[tid] = multiply_and_reduce_shoup(temp, scale, scale_shoup, modulus);
    }
}

__global__ void sub_and_scale_rns_poly(const uint64_t *operand1,
                                       const uint64_t *operand2,
                                       const uint64_t *scale,
                                       const uint64_t *scale_shoup,
                                       const DModulus *modulus,
                                       uint64_t *result,
                                       const uint64_t poly_degree,
                                       const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();

        uint64_t temp;
        // temp = sub_uint64_uint64_mod(operand1[tid], operand2[tid], mod.value());
        sub_uint64_uint64(mod_value, operand2[tid], temp);
        add_uint64_uint64(operand1[tid], temp, temp);
        result[tid] = multiply_and_reduce_shoup(temp, scale[twr], scale_shoup[twr], mod_value);
    }
}

__global__ void bfv_add_timesQ_overt_kernel(uint64_t *ct,
                                            const uint64_t *pt,
                                            uint64_t negQl_mod_t,
                                            uint64_t negQl_mod_t_shoup,
                                            const uint64_t *tInv_mod_q,
                                            const uint64_t *tInv_mod_q_shoup,
                                            const DModulus *modulus_Ql,
                                            uint64_t t,
                                            uint64_t n,
                                            uint64_t size_Ql) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n * size_Ql;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / n;
        uint64_t qi = modulus_Ql[twr].value();
        uint64_t tInv_mod_qi = tInv_mod_q[twr];
        uint64_t tInv_mod_qi_shoup = tInv_mod_q_shoup[twr];

        uint64_t m = pt[tid % n];
        uint64_t mQl_mod_t = multiply_and_reduce_shoup(m, negQl_mod_t, negQl_mod_t_shoup, t);
        ct[tid] += multiply_and_reduce_shoup(mQl_mod_t, tInv_mod_qi, tInv_mod_qi_shoup, qi);
        if (ct[tid] >= qi) ct[tid] -= qi;
    }
}

__global__ void bfv_sub_timesQ_overt_kernel(uint64_t *ct,
                                            const uint64_t *pt,
                                            uint64_t negQl_mod_t,
                                            uint64_t negQl_mod_t_shoup,
                                            const uint64_t *tInv_mod_q,
                                            const uint64_t *tInv_mod_q_shoup,
                                            const DModulus *modulus_Ql,
                                            uint64_t t,
                                            uint64_t n,
                                            uint64_t size_Ql) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n * size_Ql;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / n;
        uint64_t qi = modulus_Ql[twr].value();
        uint64_t tInv_mod_qi = tInv_mod_q[twr];
        uint64_t tInv_mod_qi_shoup = tInv_mod_q_shoup[twr];

        uint64_t m = pt[tid % n];
        uint64_t mQl_mod_t = multiply_and_reduce_shoup(m, negQl_mod_t, negQl_mod_t_shoup, t);
        ct[tid] = ct[tid] + qi - multiply_and_reduce_shoup(mQl_mod_t, tInv_mod_qi, tInv_mod_qi_shoup, qi);
        if (ct[tid] >= qi) ct[tid] -= qi;
    }
}

__global__ void tensor_prod_2x2_rns_poly(const uint64_t *operand1,
                                         const uint64_t *operand2,
                                         const DModulus *modulus,
                                         uint64_t *result,
                                         uint32_t poly_degree,
                                         uint32_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        uint64_t c0_0, c0_1, c1_0, c1_1;
        uint64_t d0, d1, d2;
        uint64_t rns_coeff_count = poly_degree * coeff_mod_size;

        c0_0 = operand1[tid];
        c0_1 = operand1[tid + rns_coeff_count];
        c1_0 = operand2[tid];
        c1_1 = operand2[tid + rns_coeff_count];

        // d0 <- c0 * c'0
        d0 = multiply_and_barrett_reduce_uint64(c0_0, c1_0, mod.value(), mod.const_ratio());
        // d2 <- c1 * c'1
        d2 = multiply_and_barrett_reduce_uint64(c0_1, c1_1, mod.value(), mod.const_ratio());
        // d1 <- (c0 + c1) * (c'0 + c'1) - c0 * c'0 - c1 * c'1
        d1 = multiply_and_barrett_reduce_uint64(c0_0 + c0_1, c1_0 + c1_1, mod.value(), mod.const_ratio());
        d1 = d1 + 2 * mod.value() - d0 - d2;
        if (d1 >= mod.value()) d1 -= mod.value();
        if (d1 >= mod.value()) d1 -= mod.value();

        result[tid] = d0;
        result[tid + rns_coeff_count] = d1;
        result[tid + 2 * rns_coeff_count] = d2;
    }
}

__global__ void tensor_square_2x2_rns_poly(const uint64_t *operand,
                                           const DModulus *modulus,
                                           uint64_t *result,
                                           const uint32_t poly_degree,
                                           const uint32_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        uint64_t c0, c1;
        uint64_t d0, d1, d2;
        uint64_t rns_coeff_count = poly_degree * coeff_mod_size;

        c0 = operand[tid];
        c1 = operand[tid + rns_coeff_count];

        // d0 <- c0 * c'0
        d0 = multiply_and_barrett_reduce_uint64(c0, c0, mod.value(), mod.const_ratio());
        // d1 <- c0 * c'1 + c1 * c'0
        uint128_t prod;
        prod = multiply_uint64_uint64(c0, c1);
        shift_left_uint128(prod, 1, prod);
        d1 = barrett_reduce_uint128_uint64(prod, mod.value(), mod.const_ratio());
        // d2 <- c1 * c'1
        d2 = multiply_and_barrett_reduce_uint64(c1, c1, mod.value(), mod.const_ratio());

        result[tid] = d0;
        result[tid + rns_coeff_count] = d1;
        result[tid + 2 * rns_coeff_count] = d2;
    }
}

/** Compute the ciphertext product for BFV and CKKS multiplication, in general case
 *  The multiplication of individual polynomials is done using a dyadic product where the inputs are already in NTT form.
 * @param[in] operand1 Ciphertext1 (in NTT form)
 * @param[in] op1_size number of polys in Ciphertext1
 * @param[in] operand2 Ciphertext2 (in NTT form)
 * @param[in] op2_size number of polys in Ciphertext2
 * @param[in] modulus the modulus
 * @param[out] result Store the result
 * @param[in] res_size the number of polys in the ciphertext product
 * @param[in] poly_degree poly degree (i.e., 4096, 8192, ...)
 * @param[in] coeff_mod_size total number of coeffs in RNS form
 */
__global__ void tensor_prod_mxn_rns_poly(const uint64_t *operand1, uint32_t op1_size,
                                         const uint64_t *operand2, uint32_t op2_size,
                                         const DModulus *modulus,
                                         uint64_t *result, uint64_t res_size,
                                         uint32_t poly_degree, uint32_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];
        uint64_t rns_coeff_count = poly_degree * coeff_mod_size;
        // ATTENTION: cipher size can not be too large,
        // each gpu block 32b is limited (less than 64k)
        // if cipher size exceeds certain value,
        // we recommend to read directly from global memory instead
        auto *c1 = new uint64_t[op1_size];
        auto *c2 = new uint64_t[op2_size];

        uint128_t temp_acc, temp_prod;

        for (uint32_t i = 0; i < op1_size; i++) {
            c1[i] = operand1[tid + rns_coeff_count * i];
        }
        for (uint32_t i = 0; i < op2_size; i++) {
            c2[i] = operand2[tid + rns_coeff_count * i];
        }

        uint32_t enc1_first, enc1_last, enc2_first, prod_idx;

        for (uint32_t j = 0; j < res_size; j++) {
            enc1_last = min(j, op1_size - 1);
            enc2_first = min(j, op2_size - 1);
            enc1_first = j - enc2_first;

            // The total number of dyadic products for this poly_index
            prod_idx = enc1_last - enc1_first + 1;

            temp_acc = {0UL, 0UL};
            for (uint32_t i = 0; i < prod_idx; i++) {
                temp_prod = multiply_uint64_uint64(c1[enc1_first + i], c2[enc2_first - i]);
                add_uint128_uint128(temp_prod, temp_acc, temp_acc);
            }
            result[tid + rns_coeff_count * j] =
                    barrett_reduce_uint128_uint64(temp_acc, mod.value(), mod.const_ratio());
        }
        delete[] c1;
        delete[] c2;
    }
}

/** used in BEHZ to compute
 * FastBconvSK(x, Bsk, q) = (FastBconv(x, B, q) - alpha_sk * B) mod q
 * @param[in] operand1 alpha_sk, need to be centered reduced
 * @param[in] negate m_sk
 * @param[in] operand2 prod_B_mod_q
 * @param[in] operand3 result of FastBconv(x, B, q)
 * @param[out] result FastBconvSK(x, Bsk, q)
 * @param[in] poly_degree poly degree (i.e., 4096, 8192, ...)
 * @param[in] coeff_mod_size total number of coeffs in RNS form
 */
__global__ void multiply_and_negated_add_rns_poly(const uint64_t *operand1, const uint64_t negate,
                                                  const uint64_t *operand2,
                                                  const uint64_t *operand3,
                                                  const DModulus *modulus,
                                                  uint64_t *result,
                                                  const uint64_t poly_degree,
                                                  const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        uint64_t op1 = operand1[tid % poly_degree];
        uint64_t prod_B_mod_q_elt = operand2[twr];

        if (op1 > negate >> 1) {
            // alpha_sk within [-m_sk/2, 0), therefore, -alpha_sk * B is positive
            op1 = negate - op1;
        }
        else {
            // alpha_sk within [0, m_sk/2), therefore, -alpha_sk * B = alpha_sk * (-B mod q)
            prod_B_mod_q_elt = mod.value() - prod_B_mod_q_elt;
        }

        op1 = multiply_and_barrett_reduce_uint64(op1, prod_B_mod_q_elt, mod.value(), mod.const_ratio());
        result[tid] = add_uint64_uint64_mod(operand3[tid], op1, mod.value());
    }
}

/** if input < plain_upper_half_threshold res = input,
 * else res = input + plain_upper_half_increment
 * @param[in]: input, The array to perform abs, in size N
 * @param[in]: N, the degree of poly
 * @param[in]: coeff_mod_size, The number of coeff prime
 * @param[in]: plain_upper_half_threshold, the threshold means the input is negative or positive
 * @param[in]: q - t corresponding to this coeff prime
 * @param[out]: res, To store the result.
 */
__global__ void abs_plain_rns_poly(const uint64_t *operand,
                                   const uint64_t plain_upper_half_threshold,
                                   const uint64_t *plain_upper_half_increment,
                                   uint64_t *result,
                                   const uint64_t poly_degree,
                                   const uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        uint64_t qi_minus_t = plain_upper_half_increment[twr];

        uint64_t op = operand[tid % poly_degree];

        if (op >= plain_upper_half_threshold)
            op += qi_minus_t;

        result[tid] = op;
    }
}

// TODO: optimize this function using warp vote
__global__ void zero_coeff_count_kernel(uint32_t *dst,
                                        const uint64_t *src,
                                        const uint64_t ct1_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ct1_size) {
        __shared__ uint32_t buffer;
        buffer = 0;
        // if (tid == 0)
        //     printf(">>> zero: %d\n", static_cast<uint32_t>(nums[tid] == 0));
        atomicAdd(&buffer, static_cast<uint32_t>(src[tid] == 0));
        __syncthreads();

        // result[blockIdx.x] = buffer;
        if (threadIdx.x == 0) {
            atomicAdd(dst, buffer);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int warpReduceSum(unsigned int mask, int mySum) {
#if __CUDA_ARCH__ >= 800
    // Specialize warpReduceFunc for int inputs to use __reduce_add_sync intrinsic
    // when on SM 8.0 or higher
    mySum = __reduce_add_sync(mask, mySum);
#else
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        mySum += __shfl_down_sync(mask, mySum, offset);
    }
#endif
    return mySum;
}

template<typename T, unsigned int blockSize>
__global__ void zero_count_kernel(const T *__restrict__ g_idata,
                                  int *__restrict__ g_odata,
                                  unsigned int n) {
    extern __shared__ int sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize * gridDim.x;
    unsigned int maskLength = (blockSize & 31); // 31 = warpSize-1
    maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
    const unsigned int mask = (0xffffffff) >> maskLength;

    int mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread

    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
        mySum += (g_idata[i] == 0);
        // ensure we don't read out of bounds -- this is optimized away for
        // powerOf2 sized arrays
        if ((i + blockSize) < n) {
            mySum += (g_idata[i + blockSize] == 0);
        }
        i += gridSize;
    }

    // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum(mask, mySum);

    // each thread puts its local sum into shared memory
    if ((tid % warpSize) == 0) {
        sdata[tid / warpSize] = mySum;
    }

    __syncthreads();

    const unsigned int shmem_extent =
            (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
    const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
    if (tid < shmem_extent) {
        mySum = sdata[tid];
        // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
        // SM 8.0
        mySum = warpReduceSum(ballot_result, mySum);
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

template<class T>
void reduce(int size, T *d_idata, int *d_odata) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // For reduce7 kernel we require only blockSize/warpSize
    // number of elements in shared memory
    int smemSize = ((threads / 32) + 1) * sizeof(T);

    zero_count_kernel<uint64_t, 256>
            <<<dimGrid, dimBlock, smemSize>>>
            (d_idata, d_odata, size);
}

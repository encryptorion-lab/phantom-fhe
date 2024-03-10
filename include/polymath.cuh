#pragma once

#include "uintmodmath.cuh"
#include "gputype.h"

__global__ void negate_rns_poly(const uint64_t* operand,
                                const DModulus* modulus,
                                uint64_t* result,
                                uint64_t poly_degree,
                                uint64_t coeff_mod_size);

/**  res = operand1 + operand2 % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void add_rns_poly(const uint64_t* operand1,
                             const uint64_t* operand2,
                             const DModulus* modulus,
                             uint64_t* result,
                             uint64_t poly_degree,
                             uint64_t coeff_mod_size);

__global__ void add_std_cipher(const uint64_t* cipher1,
                               const uint64_t* cipher2,
                               const DModulus* modulus,
                               uint64_t* result,
                               uint64_t poly_degree,
                               uint64_t coeff_mod_size);

/**  res = operand1 - operand2 % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void sub_rns_poly(const uint64_t* operand1,
                             const uint64_t* operand2,
                             const DModulus* modulus,
                             uint64_t* result,
                             uint64_t poly_degree,
                             uint64_t coeff_mod_size);

/**  res = -(operand1 + operand2) % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] DModulus Coeff modulus
 * @param[out] res The buff to hold the result
 * @param[in] N The degree of poly
 */
__global__ void add_and_negate_rns_poly(const uint64_t* operand1,
                                        const uint64_t* operand2,
                                        const DModulus* modulus,
                                        uint64_t* result,
                                        uint64_t poly_degree,
                                        uint64_t coeff_mod_size);

__global__ void add_many_rns_poly(const uint64_t* const * operands,
                                  uint64_t add_size,
                                  const DModulus* modulus,
                                  uint64_t* result,
                                  uint32_t poly_index,
                                  uint32_t poly_degree,
                                  uint32_t coeff_mod_size,
                                  uint64_t reduction_threshold);

/**  res = operand1 * operand2 % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void multiply_rns_poly(const uint64_t* operand1,
                                  const uint64_t* operand2,
                                  const DModulus* modulus,
                                  uint64_t* result,
                                  uint64_t poly_degree,
                                  uint64_t coeff_mod_size);

/**  result = (operand1 * scalar) % modulus
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void multiply_scalar_rns_poly(const uint64_t* operand,
                                         uint64_t scalar,
                                         const DModulus* modulus,
                                         uint64_t* result,
                                         uint64_t poly_degree,
                                         uint64_t coeff_mod_size);

__global__ void multiply_scalar_rns_poly(const uint64_t* operand,
                                         const uint64_t* scalar,
                                         const uint64_t* scalar_shoup,
                                         const DModulus* modulus,
                                         uint64_t* result,
                                         uint64_t poly_degree,
                                         uint64_t coeff_mod_size);

/**  res = (operand1 * operand2 + operand3) % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] operand3 Operand3
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void multiply_and_add_rns_poly(const uint64_t* operand1,
                                          const uint64_t* operand2,
                                          const uint64_t* operand3,
                                          const DModulus* modulus,
                                          uint64_t* result,
                                          uint64_t poly_degree,
                                          uint64_t coeff_mod_size);

/**  res = (operand1 + operand2 * scalar) % coeff_mod **/
__global__ void multiply_scalar_and_add_rns_poly(const uint64_t* operand1,
                                                 const uint64_t* operand2,
                                                 const uint64_t scalar,
                                                 const DModulus* modulus,
                                                 uint64_t* result,
                                                 const uint64_t poly_degree,
                                                 const uint64_t coeff_mod_size);

/**  res = (operand1 - operand2 * scalar) % coeff_mod **/
__global__ void multiply_scalar_and_sub_rns_poly(const uint64_t* operand1,
                                                 const uint64_t* operand2,
                                                 const uint64_t scalar,
                                                 const DModulus* modulus,
                                                 uint64_t* result,
                                                 const uint64_t poly_degree,
                                                 const uint64_t coeff_mod_size);

/**  res = (operand1 * operand2 + operand3 * scale) % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] operand3 Operand3
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void multiply_and_scale_add_rns_poly(const uint64_t* operand1,
                                                const uint64_t* operand2,
                                                const uint64_t* operand3,
                                                const uint64_t& scale,
                                                const DModulus* modulus,
                                                uint64_t* result,
                                                const uint64_t poly_degree,
                                                const uint64_t coeff_mod_size);

/**  res = (operand1 * modulus[size_QP - 1] + operand3) % coeff_mod
 * modulus[size_QP - 1] is the temp modulus p
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] n The degree of poly
 * @param[in] dnum The decompose modulus size (first_index)
 * @param[in] alpha
 * @param[in] factor
 * @param[in] factor_shoup
 */
__global__ void multiply_temp_mod_and_add_rns_poly(const uint64_t* operand1,
                                                   const uint64_t* const * operand2,
                                                   const DModulus* modulus,
                                                   uint64_t** result,
                                                   size_t n,
                                                   size_t dnum,
                                                   size_t alpha,
                                                   const uint64_t* factor,
                                                   const uint64_t* factor_shoup);

/**  res = -(operand1 * operand2 + operand3) % coeff_mod
 * @param[in] operand1 Operand1
 * @param[in] operand2 Operand2
 * @param[in] operand3 Operand3
 * @param[in] modulus Coeff modulus
 * @param[out] result The buff to hold the result
 * @param[in] coeff_count The degree of poly
 */
__global__ void multiply_and_add_negate_rns_poly(const uint64_t* operand1,
                                                 const uint64_t* operand2,
                                                 const uint64_t* operand3,
                                                 const DModulus* modulus,
                                                 uint64_t* result,
                                                 uint64_t poly_degree,
                                                 uint64_t coeff_mod_size);

// out = (operand1 - operand2) * scale % modulus
__global__ void sub_and_scale_single_mod_poly(const uint64_t* operand1,
                                              const uint64_t* operand2,
                                              uint64_t scale,
                                              uint64_t scale_shoup,
                                              uint64_t modulus,
                                              uint64_t* result,
                                              uint64_t poly_degree);

__global__ void sub_and_scale_rns_poly(const uint64_t* operand1,
                                       const uint64_t* operand2,
                                       const uint64_t* scale,
                                       const uint64_t* scale_shoup,
                                       const DModulus* modulus,
                                       uint64_t* result,
                                       uint64_t poly_degree,
                                       uint64_t coeff_mod_size);

__global__ void bfv_add_timesQ_overt_kernel(uint64_t* ct,
                                            const uint64_t* pt,
                                            uint64_t negQl_mod_t,
                                            uint64_t negQl_mod_t_shoup,
                                            const uint64_t* tInv_mod_q,
                                            const uint64_t* tInv_mod_q_shoup,
                                            const DModulus* modulus_Ql,
                                            uint64_t t,
                                            uint64_t n,
                                            uint64_t size_Ql);

__global__ void bfv_sub_timesQ_overt_kernel(uint64_t* ct,
                                            const uint64_t* pt,
                                            uint64_t negQl_mod_t,
                                            uint64_t negQl_mod_t_shoup,
                                            const uint64_t* tInv_mod_q,
                                            const uint64_t* tInv_mod_q_shoup,
                                            const DModulus* modulus_Ql,
                                            uint64_t t,
                                            uint64_t n,
                                            uint64_t size_Ql);

// used in BEHZ
// out = operand1 * (neg_)mul_operand2 + add_operand
// whether perform the neg operation depends on neg_condition, neg is performed on operand1 with neg_operand
// @notice: operand1 value must not be modified!!!
__global__ void multiply_and_negated_add_rns_poly(const uint64_t* operand1, uint64_t negate,
                                                  const uint64_t* operand2,
                                                  const uint64_t* operand3,
                                                  const DModulus* modulus,
                                                  uint64_t* result,
                                                  uint64_t poly_degree,
                                                  uint64_t coeff_mod_size);

/** if input < plain_upper_half_threshold res = input,
 * else res = input + plain_upper_half_increment
 * @param[in]: input, The array to perform abs, in size N
 * @param[in]: N, the degree of poly
 * @param[in]: coeff_mod_size, The number of coeff prime
 * @param[in]: plain_upper_half_threshold, the threshold means the input is negative or positive
 * @param[in]: q - t corresponding to this coeff prime
 * @param[out]: res, To store the result.
 */
__global__ void abs_plain_rns_poly(const uint64_t* operand,
                                   uint64_t plain_upper_half_threshold,
                                   const uint64_t* plain_upper_half_increment,
                                   uint64_t* result,
                                   uint64_t poly_degree,
                                   uint64_t coeff_mod_size);

/** Compute the ciphertext product for BFV and CKKS multiplication, in general case
 *  The multiplication of individual polynomials is done using a dyadic product where the inputs are already in NTT form.
 * @param[in] operand1 Ciphertext1 (in NTT form)
 * @param[in] operand2 Ciphertext2 (in NTT form)
 * @param[in] modulus the modulus
 * @param[out] result Store the result
 * @param[in] poly_degree poly degree (i.e., 4096, 8192, ...)
 * @param[in] coeff_mod_size total number of coeffs in RNS form
 */

__global__ void tensor_prod_2x2_rns_poly(const uint64_t* operand1,
                                         const uint64_t* operand2,
                                         const DModulus* modulus,
                                         uint64_t* result,
                                         uint32_t poly_degree,
                                         uint32_t coeff_mod_size);

__global__ void tensor_prod_2x2_rns_poly_lazy(const uint64_t* operand1,
                                              const uint64_t* operand2,
                                              uint64_t* result,
                                              uint32_t poly_degree,
                                              uint32_t coeff_mod_size);

__global__ void tensor_square_2x2_rns_poly(const uint64_t* operand,
                                           const DModulus* modulus,
                                           uint64_t* result,
                                           const uint32_t poly_degree,
                                           const uint32_t coeff_mod_size);

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
__global__ void tensor_prod_mxn_rns_poly(const uint64_t* operand1, uint32_t op1_size,
                                         const uint64_t* operand2, uint32_t op2_size,
                                         const DModulus* modulus,
                                         uint64_t* result, uint64_t res_size,
                                         uint32_t poly_degree, uint32_t coeff_mod_size);

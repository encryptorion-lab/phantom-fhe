#pragma once

#include "gputype.h"

#define SAMPLE_SIZE(n)                                                                                                 \
    ({                                                                                                                 \
        size_t SAMPLE_SIZE;                                                                                            \
        switch (n) {                                                                                                   \
            case 2048:                                                                                                 \
            case 4096:                                                                                                 \
                SAMPLE_SIZE = 64;                                                                                      \
                break;                                                                                                 \
            case 8192:                                                                                                 \
                SAMPLE_SIZE = 128;                                                                                     \
                break;                                                                                                 \
            case 16384:                                                                                                \
            case 32768:                                                                                                \
            case 65536:                                                                                                \
            case 131072:                                                                                               \
                SAMPLE_SIZE = 256;                                                                                     \
                break;                                                                                                 \
            default:                                                                                                   \
                throw std::invalid_argument("unsupported polynomial degree when selecting sample size");               \
                break;                                                                                                 \
        };                                                                                                             \
        SAMPLE_SIZE;                                                                                                   \
    })

//==================================================== NWT 1D ==========================================================

void fnwt_1d(uint64_t *inout, const uint64_t *twiddles, const uint64_t *twiddles_shoup, const DModulus *modulus,
             size_t dim, size_t coeff_modulus_size, size_t start_modulus_idx);

void fnwt_1d_opt(uint64_t *inout, const uint64_t *twiddles, const uint64_t *twiddles_shoup, const DModulus *modulus,
                 size_t dim, size_t coeff_modulus_size, size_t start_modulus_idx);

void inwt_1d(uint64_t *inout, const uint64_t *itwiddles, const uint64_t *itwiddles_shoup, const DModulus *modulus,
             const uint64_t *scalar, const uint64_t *scalar_shoup, size_t dim, size_t coeff_modulus_size,
             size_t start_modulus_idx);

void inwt_1d_opt(uint64_t *inout, const uint64_t *itwiddles, const uint64_t *itwiddles_shoup, const DModulus *modulus,
                 const uint64_t *scalar, const uint64_t *scalar_shoup, size_t dim, size_t coeff_modulus_size,
                 size_t start_modulus_idx);

//==================================================== NWT 2D ==========================================================
// void forward_bired_ntt(uint64_t *inout, const uint64_t *twiddles, const uint64_t *twiddles_shoup,
//                        const DModulus *modulus, size_t dim, size_t coeff_modulus_size);

// void inverse_bired_ntt(uint64_t *inout, const uint64_t *itwiddles, const uint64_t *itwiddles_shoup,
//                        const DModulus *modulus, const uint64_t *scalar, const uint64_t *scalar_shoup, size_t dim,
//                        size_t coeff_modulus_size);

//==================================================== NWT 2D ==========================================================

void nwt_2d_radix8_forward_inplace(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                   size_t start_modulus_idx);

void nwt_2d_radix8_forward_inplace_fuse_moddown(uint64_t *ct, const uint64_t *cx, const uint64_t *bigPInv_mod_q,
                                                const uint64_t *bigPInv_mod_q_shoup, uint64_t *delta,
                                                const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                                size_t start_modulus_idx);

void nwt_2d_radix8_forward_inplace_include_temp_mod(uint64_t *inout, const DNTTTable &ntt_tables,
                                                    size_t coeff_modulus_size, size_t start_modulus_idx,
                                                    size_t total_modulus_size);

void nwt_2d_radix8_forward_inplace_include_special_mod(uint64_t *inout, const DNTTTable &ntt_tables,
                                                       size_t coeff_modulus_size, size_t start_modulus_idx,
                                                       size_t size_QP, size_t size_P);

void nwt_2d_radix8_forward_inplace_include_special_mod_exclude_range(uint64_t *inout, const DNTTTable &ntt_tables,
                                                                     size_t coeff_modulus_size,
                                                                     size_t start_modulus_idx, size_t size_QP,
                                                                     size_t size_P, size_t excluded_range_start,
                                                                     size_t excluded_range_end);

void nwt_2d_radix8_forward_inplace_single_mod(uint64_t *inout, size_t modulus_index, const DNTTTable &ntt_tables,
                                              size_t coeff_modulus_size, size_t start_modulus_idx);

void nwt_2d_radix8_forward_modup_fuse(uint64_t *out, const uint64_t *in, size_t modulus_index,
                                      const DNTTTable &ntt_tables, size_t coeff_modulus_size, size_t start_modulus_idx);

void nwt_2d_radix8_forward_moddown_fuse(uint64_t *encrypted, const uint64_t *cx, const uint64_t *cx_last,
                                        uint64_t *reduced_cx_last, const DNTTTable &ntt_tables,
                                        const uint64_t *inv_q_last_mod_q, const uint64_t *inv_q_last_mod_q_shoup,
                                        size_t coeff_modulus_size, size_t q_coeff_count, size_t pq_coeff_count,
                                        size_t start_modulus_idx);

void nwt_2d_radix8_backward_inplace(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                    size_t start_modulus_idx);

void nwt_2d_radix8_backward(uint64_t *out, const uint64_t *in, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                            size_t start_modulus_idx);

void nwt_2d_radix8_backward_scale(uint64_t *out, const uint64_t *in, const DNTTTable &ntt_tables,
                                  size_t coeff_modulus_size, size_t start_modulus_idx, const uint64_t *scale,
                                  const uint64_t *scale_shoup);

void nwt_2d_radix8_backward_inplace_scale(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                          size_t start_modulus_idx, const uint64_t *scale, const uint64_t *scale_shoup);

void nwt_2d_radix8_backward_inplace_include_special_mod(uint64_t *inout, const DNTTTable &ntt_tables,
                                                        size_t coeff_modulus_size, size_t start_modulus_idx,
                                                        size_t size_QP, size_t size_P);

void nwt_2d_radix8_backward_inplace_include_temp_mod_scale(uint64_t *inout, const DNTTTable &ntt_tables,
                                                           size_t coeff_modulus_size, size_t start_modulus_idx,
                                                           size_t total_modulus_size, const uint64_t *scale,
                                                           const uint64_t *scale_shoup);

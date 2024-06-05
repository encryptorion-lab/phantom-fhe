#include "ntt.cuh"
#include "butterfly.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

__global__ static void
inwt_radix8_phase1(uint64_t *out,
                   const uint64_t *in,
                   const uint64_t *itwiddles,
                   const uint64_t *itwiddles_shoup,
                   const DModulus *modulus,
                   const size_t coeff_mod_size,
                   const size_t start_mod_idx,
                   const size_t n,
                   const size_t n1,
                   const size_t n2) {
    extern __shared__ uint64_t buffer[];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (n / 8 * coeff_mod_size);
         i += blockDim.x * gridDim.x) {
        size_t group = n2 / 8;
        size_t set = threadIdx.x / group;
        // size of a block
        uint64_t samples[8];
        size_t t = n / 2 / n1;
        // prime idx
        size_t twr_idx = i / (n / 8) + start_mod_idx;
        // index in N/2 range
        size_t n_idx = i % (n / 8);
        // i'th block
        size_t m_idx = n_idx / (t / 4);
        size_t t_idx = n_idx % (t / 4);
        // base address
        auto in_ptr = in + twr_idx * n;
        auto out_ptr = out + twr_idx * n;
        const uint64_t *psi = itwiddles + n * twr_idx;
        const uint64_t *psi_shoup = itwiddles_shoup + n * twr_idx;
        const DModulus *modulus_table = modulus;
        uint64_t modulus = modulus_table[twr_idx].value();
        size_t n_init = 2 * m_idx * t + t_idx;

#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            buffer[set * n2 + t_idx + t / 4 * j] = *(in_ptr + n_init + t / 4 * j);
        }
        __syncthreads();

#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            samples[l] = buffer[set * n2 + 8 * t_idx + l];
        }
        size_t tw_idx = n1 + m_idx;
        size_t tw_idx2 = (t / 4) * tw_idx + t_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus);
#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            buffer[set * n2 + 8 * t_idx + l] = samples[l];
        }
        size_t tail = 0;
        __syncthreads();

#pragma unroll
        for (size_t j = t / 32, k = 32; j > 0; j >>= 3, k *= 8) {
            size_t m_idx2 = t_idx / (k / 4);
            size_t t_idx2 = t_idx % (k / 4);
#pragma unroll
            for (size_t l = 0; l < 8; l++) {
                samples[l] =
                        buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
            }
            tw_idx2 = j * tw_idx + m_idx2;
            intt8(samples, psi, psi_shoup, tw_idx2, modulus);
#pragma unroll
            for (size_t l = 0; l < 8; l++) {
                buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                        samples[l];
            }
            if (j == 2)
                tail = 1;
            if (j == 4)
                tail = 2;
            __syncthreads();
        }

#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            samples[j] = buffer[set * n2 + t_idx + t / 4 * j];
        }
        if (tail == 1) {
            gs_butterfly(samples[0], samples[4], psi[tw_idx], psi_shoup[tw_idx], modulus);
            gs_butterfly(samples[1], samples[5], psi[tw_idx], psi_shoup[tw_idx], modulus);
            gs_butterfly(samples[2], samples[6], psi[tw_idx], psi_shoup[tw_idx], modulus);
            gs_butterfly(samples[3], samples[7], psi[tw_idx], psi_shoup[tw_idx], modulus);
        } else if (tail == 2) {
            intt4(samples, psi, psi_shoup, tw_idx, modulus);
            intt4(samples + 1, psi, psi_shoup, tw_idx, modulus);
        }
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(out_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}

__global__ static void
inplace_inwt_radix8_phase2(uint64_t *inout,
                           const uint64_t *itwiddles,
                           const uint64_t *itwiddles_shoup,
                           const uint64_t *inv_degree_modulo,
                           const uint64_t *inv_degree_modulo_shoup,
                           const DModulus *modulus,
                           const size_t coeff_mod_size,
                           const size_t start_mod_idx,
                           const size_t n,
                           const size_t n1,
                           const size_t pad) {
    extern __shared__ uint64_t buffer[];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n / 8 * coeff_mod_size);
         i += blockDim.x * gridDim.x) {
        // pad address
        size_t pad_tid = threadIdx.x % pad;
        size_t pad_idx = threadIdx.x / pad;

        size_t group = n1 / 8;
        // size of a block
        uint64_t samples[8];
        size_t t = n / 2;
        // prime idx
        size_t twr_idx = i / (n / 8) + start_mod_idx;
        // index in N/2 range
        size_t n_idx = i % (n / 8);

        // base address
        uint64_t *data_ptr = inout + twr_idx * n;
        const uint64_t *psi = itwiddles + n * twr_idx;
        const uint64_t *psi_shoup = itwiddles_shoup + n * twr_idx;
        const DModulus *modulus_table = modulus;
        uint64_t modulus_value = modulus_table[twr_idx].value();
        uint64_t inv_degree_mod = inv_degree_modulo[twr_idx];
        uint64_t inv_degree_mod_shoup = inv_degree_modulo_shoup[twr_idx];

        size_t n_init = 2 * t / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));

#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            samples[j] = *(data_ptr + n_init + t / 4 / group * j);
        }
        size_t tw_idx = 1;
        size_t tw_idx2 = group * tw_idx + pad_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            buffer[pad_tid * (n1 + pad) + 8 * pad_idx + j] = samples[j];
        }
        size_t tail = 0;
        __syncthreads();

#pragma unroll
        for (size_t j = group / 8, k = 32; j > 0; j >>= 3, k *= 8) {
            size_t m_idx2 = pad_idx / (k / 4);
            size_t t_idx2 = pad_idx % (k / 4);
#pragma unroll
            for (size_t l = 0; l < 8; l++) {
                samples[l] = buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 +
                                    (k / 4) * l];
            }
            size_t tw_idx2 = j * tw_idx + m_idx2;
            intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
#pragma unroll
            for (size_t l = 0; l < 8; l++) {
                buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] = samples[l];
            }
            if (j == 2)
                tail = 1;
            if (j == 4)
                tail = 2;
            __syncthreads();
        }
        if (group < 8)
            tail = (group == 4) ? 2 : 1;
#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            samples[l] = buffer[pad_tid * (n1 + pad) + pad_idx + group * l];
        }
        if (tail == 1) {
            gs_butterfly(samples[0], samples[4], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[1], samples[5], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[2], samples[6], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[3], samples[7], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
        } else if (tail == 2) {
            intt4(samples, psi, psi_shoup, tw_idx, modulus_value);
            intt4(samples + 1, psi, psi_shoup, tw_idx, modulus_value);
        }

        for (size_t j = 0; j < 4; j++) {
            samples[j] =
                    multiply_and_reduce_shoup_lazy(samples[j], inv_degree_mod, inv_degree_mod_shoup, modulus_value);
        }

        n_init = t / 4 / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            csub_q(samples[j], modulus_value);
            *(data_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}

__global__ static void
inplace_inwt_radix8_phase2_scale(uint64_t *inout,
                                 const uint64_t *itwiddles,
                                 const uint64_t *itwiddles_shoup,
                                 const uint64_t *inv_degree_modulo,
                                 const uint64_t *inv_degree_modulo_shoup,
                                 const DModulus *modulus,
                                 const size_t coeff_mod_size,
                                 const size_t start_mod_idx,
                                 const size_t n,
                                 const size_t n1,
                                 const size_t pad,
                                 const uint64_t *scale,
                                 const uint64_t *scale_shoup) {
    extern __shared__ uint64_t buffer[];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n / 8 * coeff_mod_size);
         i += blockDim.x * gridDim.x) {
        // pad address
        size_t pad_tid = threadIdx.x % pad;
        size_t pad_idx = threadIdx.x / pad;

        size_t group = n1 / 8;
        // size of a block
        uint64_t samples[8];
        size_t t = n / 2;
        // prime idx
        size_t twr_idx = i / (n / 8) + start_mod_idx;
        // index in N/2 range
        size_t n_idx = i % (n / 8);

        // base address
        uint64_t *data_ptr = inout + twr_idx * n;
        const uint64_t *psi = itwiddles + n * twr_idx;
        const uint64_t *psi_shoup = itwiddles_shoup + n * twr_idx;
        const DModulus *modulus_table = modulus;
        uint64_t modulus_value = modulus_table[twr_idx].value();
        uint64_t inv_degree_mod = inv_degree_modulo[twr_idx];
        uint64_t inv_degree_mod_shoup = inv_degree_modulo_shoup[twr_idx];

        size_t n_init = 2 * t / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));

#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            samples[j] = *(data_ptr + n_init + t / 4 / group * j);
        }
        size_t tw_idx = 1;
        size_t tw_idx2 = group * tw_idx + pad_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            buffer[pad_tid * (n1 + pad) + 8 * pad_idx + j] = samples[j];
        }
        size_t tail = 0;
        __syncthreads();

#pragma unroll
        for (size_t j = group / 8, k = 32; j > 0; j >>= 3, k *= 8) {
            size_t m_idx2 = pad_idx / (k / 4);
            size_t t_idx2 = pad_idx % (k / 4);
#pragma unroll
            for (size_t l = 0; l < 8; l++) {
                samples[l] = buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 +
                                    (k / 4) * l];
            }
            size_t tw_idx2 = j * tw_idx + m_idx2;
            intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
#pragma unroll
            for (size_t l = 0; l < 8; l++) {
                buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] = samples[l];
            }
            if (j == 2)
                tail = 1;
            if (j == 4)
                tail = 2;
            __syncthreads();
        }
        if (group < 8)
            tail = (group == 4) ? 2 : 1;
#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            samples[l] = buffer[pad_tid * (n1 + pad) + pad_idx + group * l];
        }
        if (tail == 1) {
            gs_butterfly(samples[0], samples[4], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[1], samples[5], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[2], samples[6], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[3], samples[7], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
        } else if (tail == 2) {
            intt4(samples, psi, psi_shoup, tw_idx, modulus_value);
            intt4(samples + 1, psi, psi_shoup, tw_idx, modulus_value);
        }

        for (size_t j = 0; j < 4; j++) {
            samples[j] =
                    multiply_and_reduce_shoup_lazy(samples[j], inv_degree_mod, inv_degree_mod_shoup, modulus_value);
        }

        n_init = t / 4 / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));
        uint64_t scale_q = scale[twr_idx];
        uint64_t scale_q_shoup = scale_shoup[twr_idx];
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = multiply_and_reduce_shoup(
                    samples[j], scale_q, scale_q_shoup, modulus_value);
        }
    }
}

void nwt_2d_radix8_backward(uint64_t *out,
                            const uint64_t *in,
                            const DNTTTable &ntt_tables,
                            size_t coeff_modulus_size,
                            size_t start_modulus_idx,
                            const cudaStream_t &stream) {
    size_t poly_degree = ntt_tables.n();
    size_t phase2_sample_size = SAMPLE_SIZE(poly_degree);

    const size_t phase1_sample_size = poly_degree / phase2_sample_size;
    const size_t per_block_memory = blockDimNTT.x * per_thread_sample_size * sizeof(uint64_t);
    inwt_radix8_phase1<<<gridDimNTT, blockDimNTT, per_block_memory, stream>>>(
            out,
            in,
            ntt_tables.itwiddle(),
            ntt_tables.itwiddle_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            poly_degree,
            phase1_sample_size,
            phase2_sample_size);
    inplace_inwt_radix8_phase2<<<
    gridDimNTT, (phase1_sample_size / 8) * per_block_pad,
    (phase1_sample_size + per_block_pad + 1) * per_block_pad * sizeof(uint64_t), stream>>>(
            out,
            ntt_tables.itwiddle(), ntt_tables.itwiddle_shoup(),
            ntt_tables.n_inv_mod_q(), ntt_tables.n_inv_mod_q_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            poly_degree,
            phase1_sample_size,
            per_block_pad);
}

void nwt_2d_radix8_backward_scale(uint64_t *out,
                                  const uint64_t *in,
                                  const DNTTTable &ntt_tables,
                                  size_t coeff_modulus_size,
                                  size_t start_modulus_idx,
                                  const uint64_t *scale,
                                  const uint64_t *scale_shoup,
                                  const cudaStream_t &stream) {
    size_t poly_degree = ntt_tables.n();
    size_t phase2_sample_size = SAMPLE_SIZE(poly_degree);

    const size_t phase1_sample_size = poly_degree / phase2_sample_size;
    const size_t per_block_memory = blockDimNTT.x * per_thread_sample_size * sizeof(uint64_t);
    inwt_radix8_phase1<<<gridDimNTT, blockDimNTT, per_block_memory, stream>>>(
            out,
            in,
            ntt_tables.itwiddle(),
            ntt_tables.itwiddle_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            poly_degree,
            phase1_sample_size,
            phase2_sample_size);
    inplace_inwt_radix8_phase2_scale<<<
    gridDimNTT, (phase1_sample_size / 8) * per_block_pad,
    (phase1_sample_size + per_block_pad + 1) * per_block_pad * sizeof(uint64_t), stream>>>(
            out,
            ntt_tables.itwiddle(), ntt_tables.itwiddle_shoup(),
            ntt_tables.n_inv_mod_q(), ntt_tables.n_inv_mod_q_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            poly_degree,
            phase1_sample_size,
            per_block_pad,
            scale, scale_shoup);
}

__global__ static void
inplace_fnwt_radix8_phase1_include_special_mod_exclude_range(uint64_t *inout,
                                                             const uint64_t *twiddles,
                                                             const uint64_t *twiddles_shoup,
                                                             const DModulus *modulus,
                                                             size_t coeff_mod_size,
                                                             size_t start_mod_idx,
                                                             size_t size_QP, size_t size_P,
                                                             size_t n,
                                                             size_t n1,
                                                             size_t pad,
                                                             size_t excluded_range_start, size_t excluded_range_end) {
    extern __shared__ uint64_t buffer[];

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n / 8 * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        // pad address
        size_t pad_tid = threadIdx.x % pad;
        size_t pad_idx = threadIdx.x / pad;

        size_t group = n1 / 8;
        // size of a block
        uint64_t samples[8];
        size_t t = n / 2;
        // modulus idx
        size_t twr_idx = tid / (n / 8) + start_mod_idx;
        if (twr_idx >= excluded_range_start && twr_idx < excluded_range_end) continue;
        size_t twr_idx2 = (twr_idx >= start_mod_idx + coeff_mod_size - size_P
                           ? size_QP - (start_mod_idx + coeff_mod_size - twr_idx)
                           : twr_idx);
        // index in n/8 range (in each tower)
        size_t n_idx = tid % (n / 8);
        // base address
        uint64_t *data_ptr = inout + twr_idx * n;
        const uint64_t *psi = twiddles + twr_idx2 * n;
        const uint64_t *psi_shoup = twiddles_shoup + twr_idx2 * n;
        const DModulus *modulus_table = modulus;
        uint64_t modulus = modulus_table[twr_idx2].value();
        size_t n_init = t / 4 / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));

#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            samples[j] = *(data_ptr + n_init + t / 4 * j);
        }
        size_t tw_idx = 1;
        fntt8(samples, psi, psi_shoup, tw_idx, modulus);
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            buffer[pad_tid * (n1 + pad) + pad_idx + group * j] = samples[j];
        }
        size_t remain_iters = 0;
        __syncthreads();
#pragma unroll
        for (size_t j = 8, k = group / 2; j < group + 1; j *= 8, k >>= 3) {
            size_t m_idx2 = pad_idx / (k / 4);
            size_t t_idx2 = pad_idx % (k / 4);
#pragma unroll
            for (size_t l = 0; l < 8; l++) {
                samples[l] = buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
            }
            size_t tw_idx2 = j * tw_idx + m_idx2;
            fntt8(samples, psi, psi_shoup, tw_idx2, modulus);
#pragma unroll
            for (size_t l = 0; l < 8; l++) {
                buffer[(n1 + pad) * pad_tid + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] = samples[l];
            }
            if (j == group / 2)
                remain_iters = 1;
            if (j == group / 4)
                remain_iters = 2;
            __syncthreads();
        }

        if (group < 8)
            remain_iters = (group == 4) ? 2 : 1;
#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            samples[l] = buffer[(n1 + pad) * pad_tid + 8 * pad_idx + l];
        }
        if (remain_iters == 1) {
            size_t tw_idx2 = 4 * group * tw_idx + 4 * pad_idx;
            ct_butterfly(samples[0], samples[1], psi[tw_idx2], psi_shoup[tw_idx2], modulus);
            ct_butterfly(samples[2], samples[3], psi[tw_idx2 + 1], psi_shoup[tw_idx2 + 1], modulus);
            ct_butterfly(samples[4], samples[5], psi[tw_idx2 + 2], psi_shoup[tw_idx2 + 2], modulus);
            ct_butterfly(samples[6], samples[7], psi[tw_idx2 + 3], psi_shoup[tw_idx2 + 3], modulus);
        } else if (remain_iters == 2) {
            size_t tw_idx2 = 2 * group * tw_idx + 2 * pad_idx;
            fntt4(samples, psi, psi_shoup, tw_idx2, modulus);
            fntt4(samples + 4, psi, psi_shoup, tw_idx2 + 1, modulus);
        }
#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            buffer[(n1 + pad) * pad_tid + 8 * pad_idx + l] = samples[l];
        }

        __syncthreads();
        for (size_t j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = buffer[pad_tid * (n1 + pad) + pad_idx + group * j];
        }
    }
}

__global__ static void
inplace_fnwt_radix8_phase2_include_special_mod_exclude_range(uint64_t *inout,
                                                             const uint64_t *twiddles,
                                                             const uint64_t *twiddles_shoup,
                                                             const DModulus *modulus,
                                                             size_t coeff_mod_size,
                                                             size_t start_mod_idx,
                                                             size_t size_QP, size_t size_P,
                                                             size_t n,
                                                             size_t n1,
                                                             size_t n2,
                                                             size_t excluded_range_start, size_t excluded_range_end) {
    extern __shared__ uint64_t buffer[];

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n / 8 * coeff_mod_size);
         tid += blockDim.x * gridDim.x) {
        size_t group = n2 / 8;
        size_t set = threadIdx.x / group;
        // size of a block
        uint64_t samples[8];
        size_t t = n2 / 2;
        // prime idx
        size_t twr_idx = coeff_mod_size - 1 - (tid / (n / 8)) + start_mod_idx;
        if (twr_idx >= excluded_range_start && twr_idx < excluded_range_end) continue;
        size_t twr_idx2 = (twr_idx >= start_mod_idx + coeff_mod_size - size_P
                           ? size_QP - (start_mod_idx + coeff_mod_size - twr_idx)
                           : twr_idx);
        // index in n/2 range
        size_t n_idx = tid % (n / 8);
        // tid'th block
        size_t m_idx = n_idx / (t / 4);
        size_t t_idx = n_idx % (t / 4);
        // base address
        uint64_t *data_ptr = inout + twr_idx * n;
        const uint64_t *psi = twiddles + n * twr_idx2;
        const uint64_t *psi_shoup = twiddles_shoup + n * twr_idx2;
        const DModulus *modulus_table = modulus;
        uint64_t modulus = modulus_table[twr_idx2].value();
        size_t n_init = 2 * m_idx * t + t_idx;
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            samples[j] = *(data_ptr + n_init + t / 4 * j);
        }
        size_t tw_idx = n1 + m_idx;
        fntt8(samples, psi, psi_shoup, tw_idx, modulus);
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            buffer[set * n2 + t_idx + t / 4 * j] = samples[j];
        }
        size_t tail = 0;
        __syncthreads();

#pragma unroll
        for (size_t j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
            size_t m_idx2 = t_idx / (k / 4);
            size_t t_idx2 = t_idx % (k / 4);
#pragma unroll
            for (size_t l = 0; l < 8; l++) {
                samples[l] =
                        buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
            }
            size_t tw_idx2 = j * tw_idx + m_idx2;
            fntt8(samples, psi, psi_shoup, tw_idx2, modulus);
#pragma unroll
            for (size_t l = 0; l < 8; l++) {
                buffer[set * n2 + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                        samples[l];
            }
            if (j == t / 8)
                tail = 1;
            if (j == t / 16)
                tail = 2;
            __syncthreads();
        }

#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            samples[l] = buffer[set * n2 + 8 * t_idx + l];
        }
        if (tail == 1) {
            size_t tw_idx2 = t * tw_idx + 4 * t_idx;
            ct_butterfly(samples[0], samples[1], psi[tw_idx2], psi_shoup[tw_idx2], modulus);
            ct_butterfly(samples[2], samples[3], psi[tw_idx2 + 1], psi_shoup[tw_idx2 + 1], modulus);
            ct_butterfly(samples[4], samples[5], psi[tw_idx2 + 2], psi_shoup[tw_idx2 + 2], modulus);
            ct_butterfly(samples[6], samples[7], psi[tw_idx2 + 3], psi_shoup[tw_idx2 + 3], modulus);
        } else if (tail == 2) {
            size_t tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
            fntt4(samples, psi, psi_shoup, tw_idx2, modulus);
            fntt4(samples + 4, psi, psi_shoup, tw_idx2 + 1, modulus);
        }
#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            buffer[set * n2 + 8 * t_idx + l] = samples[l];
        }
        __syncthreads();

        uint64_t modulus2 = modulus << 1;
        // final reduction
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            samples[j] = buffer[set * n2 + t_idx + t / 4 * j];
            csub_q(samples[j], modulus2);
            csub_q(samples[j], modulus);
        }
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}

void nwt_2d_radix8_forward_inplace_include_special_mod_exclude_range(
        uint64_t *inout,
        const DNTTTable &ntt_tables,
        size_t coeff_modulus_size,
        size_t start_modulus_idx,
        size_t size_QP, size_t size_P,
        size_t excluded_range_start, size_t excluded_range_end,
        const cudaStream_t &stream) {
    if (excluded_range_start < start_modulus_idx ||
        excluded_range_end > (start_modulus_idx + coeff_modulus_size)) {
        throw std::invalid_argument("Excluded range in NTT is invalid.");
    }

    size_t poly_degree = ntt_tables.n();
    size_t phase1_sample_size = SAMPLE_SIZE(poly_degree);

    const size_t phase2_sample_size = poly_degree / phase1_sample_size;
    const size_t per_block_memory = blockDimNTT.x * per_thread_sample_size * sizeof(uint64_t);
    //
    inplace_fnwt_radix8_phase1_include_special_mod_exclude_range<<<
    gridDimNTT, (phase1_sample_size / 8) * per_block_pad,
    (phase1_sample_size + per_block_pad + 1) * per_block_pad * sizeof(uint64_t), stream>>>(
            inout,
            ntt_tables.twiddle(),
            ntt_tables.twiddle_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            size_QP, size_P,
            poly_degree,
            phase1_sample_size,
            per_block_pad,
            excluded_range_start, excluded_range_end);
    // max 512 threads per block
    inplace_fnwt_radix8_phase2_include_special_mod_exclude_range<<<
    gridDimNTT, blockDimNTT, per_block_memory, stream>>>(
            inout,
            ntt_tables.twiddle(),
            ntt_tables.twiddle_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            size_QP, size_P,
            poly_degree,
            phase1_sample_size,
            phase2_sample_size,
            excluded_range_start, excluded_range_end);
}

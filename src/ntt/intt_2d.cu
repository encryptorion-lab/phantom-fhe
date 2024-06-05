#include "ntt.cuh"
#include "butterfly.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

__global__ static void
inplace_inwt_radix8_phase1(uint64_t *inout,
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
        uint64_t *data_ptr = inout + twr_idx * n;
        const uint64_t *psi = itwiddles + n * twr_idx;
        const uint64_t *psi_shoup = itwiddles_shoup + n * twr_idx;
        const DModulus *modulus_table = modulus;
        uint64_t modulus_value = modulus_table[twr_idx].value();
        size_t n_init = 2 * m_idx * t + t_idx;

#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            buffer[set * n2 + t_idx + t / 4 * j] = *(data_ptr + n_init + t / 4 * j);
        }
        __syncthreads();

#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            samples[l] = buffer[set * n2 + 8 * t_idx + l];
        }
        size_t tw_idx = n1 + m_idx;
        size_t tw_idx2 = (t / 4) * tw_idx + t_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
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
            intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
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
            gs_butterfly(samples[0], samples[4], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[1], samples[5], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[2], samples[6], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[3], samples[7], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
        } else if (tail == 2) {
            intt4(samples, psi, psi_shoup, tw_idx, modulus_value);
            intt4(samples + 1, psi, psi_shoup, tw_idx, modulus_value);
        }
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = samples[j];
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
            samples[j] = multiply_and_reduce_shoup_lazy(samples[j], inv_degree_mod, inv_degree_mod_shoup,
                                                        modulus_value);
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
inplace_inwt_radix8_phase2_and_scale(uint64_t *inout,
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
                                     const uint64_t *scale, const uint64_t *scale_shoup) {
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
            samples[j] = multiply_and_reduce_shoup_lazy(samples[j], inv_degree_mod, inv_degree_mod_shoup,
                                                        modulus_value);
        }

        n_init = t / 4 / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = multiply_and_reduce_shoup(
                    samples[j], scale[twr_idx], scale_shoup[twr_idx], modulus_value);
        }
    }
}

__global__ static void
inplace_inwt_radix8_phase1_include_temp_mod(uint64_t *inout,
                                            const uint64_t *itwiddles,
                                            const uint64_t *itwiddles_shoup,
                                            const DModulus *modulus,
                                            const size_t coeff_mod_size,
                                            const size_t start_mod_idx,
                                            const size_t total_mod_size,
                                            const size_t n,
                                            const size_t n1,
                                            const size_t n2) {
    extern __shared__ uint64_t buffer[];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n / 8 * coeff_mod_size);
         i += blockDim.x * gridDim.x) {
        size_t group = n2 / 8;
        size_t set = threadIdx.x / group;
        // size of a block
        uint64_t samples[8];
        size_t t = n / 2 / n1;
        // prime idx
        size_t twr_idx = i / (n / 8) + start_mod_idx;
        size_t twr_idx2 = (twr_idx == coeff_mod_size + start_mod_idx - 1 ? total_mod_size - 1 : twr_idx);
        // index in N/2 range
        size_t n_idx = i % (n / 8);
        // i'th block
        size_t m_idx = n_idx / (t / 4);
        size_t t_idx = n_idx % (t / 4);
        // base address
        uint64_t *data_ptr = inout + twr_idx * n;
        const uint64_t *psi = itwiddles + n * twr_idx2;
        const uint64_t *psi_shoup = itwiddles_shoup + n * twr_idx2;
        const DModulus *modulus_table = modulus;
        uint64_t modulus_value = modulus_table[twr_idx2].value();
        size_t n_init = 2 * m_idx * t + t_idx;

#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            buffer[set * n2 + t_idx + t / 4 * j] = *(data_ptr + n_init + t / 4 * j);
        }
        __syncthreads();

#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            samples[l] = buffer[set * n2 + 8 * t_idx + l];
        }
        size_t tw_idx = n1 + m_idx;
        size_t tw_idx2 = (t / 4) * tw_idx + t_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
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
            intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
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
            gs_butterfly(samples[0], samples[4], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[1], samples[5], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[2], samples[6], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[3], samples[7], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
        } else if (tail == 2) {
            intt4(samples, psi, psi_shoup, tw_idx, modulus_value);
            intt4(samples + 1, psi, psi_shoup, tw_idx, modulus_value);
        }
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}

__global__ static void
inplace_inwt_radix8_phase1_include_special_mod(uint64_t *inout,
                                               const uint64_t *itwiddles,
                                               const uint64_t *itwiddles_shoup,
                                               const DModulus *modulus,
                                               size_t coeff_mod_size,
                                               size_t start_mod_idx,
                                               size_t size_QP,
                                               size_t size_P,
                                               size_t n,
                                               size_t n1,
                                               size_t n2) {
    extern __shared__ uint64_t buffer[];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n / 8 * coeff_mod_size);
         i += blockDim.x * gridDim.x) {
        size_t group = n2 / 8;
        size_t set = threadIdx.x / group;
        // size of a block
        uint64_t samples[8];
        size_t t = n / 2 / n1;
        // prime idx
        size_t twr_idx = i / (n / 8) + start_mod_idx;
        size_t twr_idx2 = (twr_idx >= start_mod_idx + coeff_mod_size - size_P
                           ? size_QP - (start_mod_idx + coeff_mod_size - twr_idx)
                           : twr_idx);
        // index in N/2 range
        size_t n_idx = i % (n / 8);
        // i'th block
        size_t m_idx = n_idx / (t / 4);
        size_t t_idx = n_idx % (t / 4);
        // base address
        uint64_t *data_ptr = inout + twr_idx * n;
        const uint64_t *psi = itwiddles + n * twr_idx2;
        const uint64_t *psi_shoup = itwiddles_shoup + n * twr_idx2;
        const DModulus *modulus_table = modulus;
        uint64_t modulus_value = modulus_table[twr_idx2].value();
        size_t n_init = 2 * m_idx * t + t_idx;

#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            buffer[set * n2 + t_idx + t / 4 * j] = *(data_ptr + n_init + t / 4 * j);
        }
        __syncthreads();

#pragma unroll
        for (size_t l = 0; l < 8; l++) {
            samples[l] = buffer[set * n2 + 8 * t_idx + l];
        }
        size_t tw_idx = n1 + m_idx;
        size_t tw_idx2 = (t / 4) * tw_idx + t_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
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
            intt8(samples, psi, psi_shoup, tw_idx2, modulus_value);
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
            gs_butterfly(samples[0], samples[4], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[1], samples[5], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[2], samples[6], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
            gs_butterfly(samples[3], samples[7], psi[tw_idx], psi_shoup[tw_idx], modulus_value);
        } else if (tail == 2) {
            intt4(samples, psi, psi_shoup, tw_idx, modulus_value);
            intt4(samples + 1, psi, psi_shoup, tw_idx, modulus_value);
        }
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = samples[j];
        }
    }
}

__global__ static void
inplace_inwt_radix8_phase2_include_special_mod(uint64_t *inout,
                                               const uint64_t *itwiddles,
                                               const uint64_t *itwiddles_shoup,
                                               const uint64_t *inv_degree_modulo,
                                               const uint64_t *inv_degree_modulo_shoup,
                                               const DModulus *modulus,
                                               size_t coeff_mod_size,
                                               size_t start_mod_idx,
                                               size_t size_QP,
                                               size_t size_P,
                                               size_t n,
                                               size_t n1,
                                               size_t pad) {
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
        size_t twr_idx2 = (twr_idx >= start_mod_idx + coeff_mod_size - size_P
                           ? size_QP - (start_mod_idx + coeff_mod_size - twr_idx)
                           : twr_idx);
        // index in N/2 range
        size_t n_idx = i % (n / 8);
        // base address
        uint64_t *data_ptr = inout + twr_idx * n;
        const uint64_t *psi = itwiddles + n * twr_idx2;
        const uint64_t *psi_shoup = itwiddles_shoup + n * twr_idx2;
        const DModulus *modulus_table = modulus;
        uint64_t modulus_value = modulus_table[twr_idx2].value();
        uint64_t inv_degree_mod = inv_degree_modulo[twr_idx2];
        uint64_t inv_degree_mod_shoup = inv_degree_modulo_shoup[twr_idx2];
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
            samples[j] = multiply_and_reduce_shoup_lazy(samples[j], inv_degree_mod, inv_degree_mod_shoup,
                                                        modulus_value);
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
inplace_inwt_radix8_phase2_include_temp_mod_and_scale(uint64_t *inout,
                                                      const uint64_t *itwiddles,
                                                      const uint64_t *itwiddles_shoup,
                                                      const uint64_t *inv_degree_modulo,
                                                      const uint64_t *inv_degree_modulo_shoup,
                                                      const DModulus *modulus,
                                                      const size_t coeff_mod_size,
                                                      const size_t start_mod_idx,
                                                      const size_t total_mod_size,
                                                      const size_t n,
                                                      const size_t n1,
                                                      const size_t pad,
                                                      const uint64_t *scale, const uint64_t *scale_shoup) {
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
        size_t twr_idx2 = (twr_idx == coeff_mod_size - 1 ? total_mod_size - 1 : twr_idx);
        // index in N/2 range
        size_t n_idx = i % (n / 8);
        // base address
        uint64_t *data_ptr = inout + twr_idx * n;
        const uint64_t *psi = itwiddles + n * twr_idx2;
        const uint64_t *psi_shoup = itwiddles_shoup + n * twr_idx2;
        const DModulus *modulus_table = modulus;
        uint64_t modulus_value = modulus_table[twr_idx2].value();
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
            samples[j] = multiply_and_reduce_shoup_lazy(samples[j], inv_degree_mod, inv_degree_mod_shoup,
                                                        modulus_value);
        }

        n_init = t / 4 / group * pad_idx + pad_tid + pad * (n_idx / (group * pad));
#pragma unroll
        for (size_t j = 0; j < 8; j++) {
            *(data_ptr + n_init + t / 4 * j) = multiply_and_reduce_shoup(
                    samples[j], scale[twr_idx2], scale_shoup[twr_idx2], modulus_value);
        }
    }
}

void nwt_2d_radix8_backward_inplace(uint64_t *inout,
                                    const DNTTTable &ntt_tables,
                                    size_t coeff_modulus_size,
                                    size_t start_modulus_idx,
                                    const cudaStream_t &stream) {
    size_t poly_degree = ntt_tables.n();
    size_t phase2_sample_size = SAMPLE_SIZE(poly_degree);

    const size_t phase1_sample_size = poly_degree / phase2_sample_size;
    constexpr size_t per_block_memory = blockDimNTT.x * per_thread_sample_size * sizeof(uint64_t);
    inplace_inwt_radix8_phase1<<<
    gridDimNTT, blockDimNTT, per_block_memory, stream>>>(
            inout,
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
            inout,
            ntt_tables.itwiddle(), ntt_tables.itwiddle_shoup(),
            ntt_tables.n_inv_mod_q(), ntt_tables.n_inv_mod_q_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            poly_degree,
            phase1_sample_size,
            per_block_pad);
}

void nwt_2d_radix8_backward_inplace_scale(uint64_t *inout,
                                          const DNTTTable &ntt_tables,
                                          size_t coeff_modulus_size,
                                          size_t start_modulus_idx,
                                          const uint64_t *scale, const uint64_t *scale_shoup,
                                          const cudaStream_t &stream) {
    size_t poly_degree = ntt_tables.n();
    size_t phase2_sample_size = SAMPLE_SIZE(poly_degree);

    const size_t phase1_sample_size = poly_degree / phase2_sample_size;
    const size_t per_block_memory = blockDimNTT.x * per_thread_sample_size * sizeof(uint64_t);
    inplace_inwt_radix8_phase1<<<
    gridDimNTT, blockDimNTT, per_block_memory, stream>>>(
            inout,
            ntt_tables.itwiddle(),
            ntt_tables.itwiddle_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            poly_degree,
            phase1_sample_size,
            phase2_sample_size);
    inplace_inwt_radix8_phase2_and_scale<<<
    gridDimNTT, (phase1_sample_size / 8) * per_block_pad,
    (phase1_sample_size + per_block_pad + 1) * per_block_pad * sizeof(uint64_t), stream>>>(
            inout,
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

void nwt_2d_radix8_backward_inplace_include_special_mod(uint64_t *inout,
                                                        const DNTTTable &ntt_tables,
                                                        size_t coeff_modulus_size,
                                                        size_t start_modulus_idx,
                                                        size_t size_QP,
                                                        size_t size_P,
                                                        const cudaStream_t &stream) {
    size_t poly_degree = ntt_tables.n();
    size_t phase2_sample_size = SAMPLE_SIZE(poly_degree);
    const size_t phase1_sample_size = poly_degree / phase2_sample_size;
    const size_t per_block_memory = blockDimNTT.x * per_thread_sample_size * sizeof(uint64_t);
    inplace_inwt_radix8_phase1_include_special_mod<<<
    gridDimNTT, blockDimNTT, per_block_memory, stream>>>(
            inout,
            ntt_tables.itwiddle(),
            ntt_tables.itwiddle_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            size_QP,
            size_P,
            poly_degree,
            phase1_sample_size,
            phase2_sample_size);
    inplace_inwt_radix8_phase2_include_special_mod<<<
    gridDimNTT, (phase1_sample_size / 8) * per_block_pad,
    (phase1_sample_size + per_block_pad + 1) * per_block_pad * sizeof(uint64_t), stream>>>(
            inout,
            ntt_tables.itwiddle(), ntt_tables.itwiddle_shoup(),
            ntt_tables.n_inv_mod_q(), ntt_tables.n_inv_mod_q_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            size_QP,
            size_P,
            poly_degree,
            phase1_sample_size,
            per_block_pad);
}

void nwt_2d_radix8_backward_inplace_include_temp_mod_scale(uint64_t *inout,
                                                           const DNTTTable &ntt_tables,
                                                           size_t coeff_modulus_size,
                                                           size_t start_modulus_idx,
                                                           size_t total_modulus_size,
                                                           const uint64_t *scale, const uint64_t *scale_shoup,
                                                           const cudaStream_t &stream) {
    size_t poly_degree = ntt_tables.n();
    size_t phase2_sample_size = SAMPLE_SIZE(poly_degree);
    const size_t phase1_sample_size = poly_degree / phase2_sample_size;
    const size_t per_block_memory = blockDimNTT.x * per_thread_sample_size * sizeof(uint64_t);
    inplace_inwt_radix8_phase1_include_temp_mod<<<
    gridDimNTT, blockDimNTT, per_block_memory, stream>>>(
            inout,
            ntt_tables.itwiddle(),
            ntt_tables.itwiddle_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            total_modulus_size,
            poly_degree,
            phase1_sample_size,
            phase2_sample_size);
    inplace_inwt_radix8_phase2_include_temp_mod_and_scale<<<
    gridDimNTT, (phase1_sample_size / 8) * per_block_pad,
    (phase1_sample_size + per_block_pad + 1) * per_block_pad * sizeof(uint64_t), stream>>>(
            inout,
            ntt_tables.itwiddle(), ntt_tables.itwiddle_shoup(),
            ntt_tables.n_inv_mod_q(), ntt_tables.n_inv_mod_q_shoup(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            total_modulus_size,
            poly_degree,
            phase1_sample_size,
            per_block_pad,
            scale, scale_shoup);
}

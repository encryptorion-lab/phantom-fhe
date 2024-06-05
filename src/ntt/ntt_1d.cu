#include "ntt.cuh"
#include "butterfly.cuh"
#include "uintmodmath.cuh"

using namespace phantom::arith;

/** forward NTT transformation, with N (num of operands) up to 2048,
 * to ensure all operation completed in one block.
 * @param[inout] inout The value to operate and the returned result
 * @param[in] twiddles The pre-computated forward NTT table
 * @param[in] mod The coeff modulus value
 * @param[in] n The poly degreee
 * @param[in] logn The logarithm of n
 * @param[in] numOfGroups
 * @param[in] iter The current iteration in forward NTT transformation
 */
__global__ void inplace_fnwt_radix2(uint64_t *inout,
                                    const uint64_t *twiddles,
                                    const uint64_t *twiddles_shoup,
                                    const DModulus *modulus,
                                    size_t coeff_mod_size,
                                    size_t start_mod_idx,
                                    size_t n) {
    extern __shared__ uint64_t buffer[];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n / 2 * coeff_mod_size; // deal with 2 data per thread
         i += blockDim.x * gridDim.x) {
        size_t mod_idx = i / (n / 2) + start_mod_idx;
        size_t tid = i % (n / 2);

        // modulus
        const DModulus *modulus_table = modulus;
        uint64_t mod = modulus_table[mod_idx].value();
        uint64_t mod2 = mod << 1;

        size_t pairsInGroup;
        size_t k, j, glbIdx, bufIdx; // k = psi_step
        uint64_t samples[2];

        for (size_t numOfGroups = 1; numOfGroups < n; numOfGroups <<= 1) {
            pairsInGroup = n / numOfGroups / 2;

            k = tid / pairsInGroup;
            j = tid % pairsInGroup;
            glbIdx = 2 * k * pairsInGroup + j;
            bufIdx = glbIdx % n;
            glbIdx += mod_idx * n;

            uint64_t psi = twiddles[numOfGroups + k + n * mod_idx];
            uint64_t psi_shoup = twiddles_shoup[numOfGroups + k + n * mod_idx];

            if (numOfGroups == 1) {
                samples[0] = inout[glbIdx];
                samples[1] = inout[glbIdx + pairsInGroup];
            } else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }
            ct_butterfly(samples[0], samples[1], psi, psi_shoup, mod);

            if (numOfGroups == n >> 1) {
                csub_q(samples[0], mod2);
                csub_q(samples[0], mod);
                csub_q(samples[1], mod2);
                csub_q(samples[1], mod);
                inout[glbIdx] = samples[0];
                inout[glbIdx + pairsInGroup] = samples[1];
            } else {
                buffer[bufIdx] = samples[0];
                buffer[bufIdx + pairsInGroup] = samples[1];
                __syncthreads();
            }
        }
    }
}

__global__ void inplace_fnwt_radix2_opt(uint64_t *inout,
                                        const uint64_t *twiddles,
                                        const uint64_t *twiddles_shoup,
                                        const DModulus *modulus,
                                        const size_t n) {
    extern __shared__ uint64_t buffer[];

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t mod_idx = i / (n / 2);
    const size_t tid = i % (n / 2);

    // modulus
    const DModulus *modulus_table = modulus;
    const uint64_t mod = modulus_table[mod_idx].value();
    const uint64_t mod2 = mod << 1;

    constexpr size_t numOfGroups = 1;
    const size_t pairsInGroup = n / numOfGroups / 2;
    const size_t k = tid / pairsInGroup; // k = psi_step
    const uint64_t psi = twiddles[numOfGroups + k + n * mod_idx];
    const uint64_t psi_shoup = twiddles_shoup[numOfGroups + k + n * mod_idx];
    const size_t j = tid % pairsInGroup;
    size_t glbIdx = 2 * k * pairsInGroup + j;
    const size_t bufIdx = glbIdx % n;
    glbIdx += mod_idx * n;
    uint64_t samples0 = inout[glbIdx];
    uint64_t samples1 = inout[glbIdx + pairsInGroup];
    ct_butterfly(samples0, samples1, psi, psi_shoup, mod);
    buffer[bufIdx] = samples0;
    buffer[bufIdx + pairsInGroup] = samples1;
    __syncthreads();

    for (size_t numOfGroups = 2; numOfGroups < n / 2; numOfGroups <<= 1) {
        const size_t pairsInGroup = n / numOfGroups / 2;
        const size_t k = tid / pairsInGroup; // k = psi_step
        const uint64_t psi = twiddles[numOfGroups + k + n * mod_idx];
        const uint64_t psi_shoup = twiddles_shoup[numOfGroups + k + n * mod_idx];
        const size_t j = tid % pairsInGroup;
        const size_t glbIdx = 2 * k * pairsInGroup + j;
        const size_t bufIdx = glbIdx % n;
        uint64_t samples0 = buffer[bufIdx];
        uint64_t samples1 = buffer[bufIdx + pairsInGroup];
        ct_butterfly(samples0, samples1, psi, psi_shoup, mod);
        buffer[bufIdx] = samples0;
        buffer[bufIdx + pairsInGroup] = samples1;
        __syncthreads();
    }

    const size_t numOfGroups_last = n / 2;
    const size_t pairsInGroup_last = n / numOfGroups_last / 2;
    const size_t k_last = tid / pairsInGroup_last; // k = psi_step
    const uint64_t psi_last = twiddles[numOfGroups_last + k_last + n * mod_idx];
    const uint64_t psi_last_shoup = twiddles_shoup[numOfGroups_last + k_last + n * mod_idx];
    const size_t j_last = tid % pairsInGroup_last;
    size_t glbIdx_last = 2 * k_last * pairsInGroup_last + j_last;
    const size_t bufIdx_last = glbIdx_last % n;
    uint64_t samples0_last = buffer[bufIdx_last];
    uint64_t samples1_last = buffer[bufIdx_last + pairsInGroup_last];
    ct_butterfly(samples0_last, samples1_last, psi_last, psi_last_shoup, mod);
    csub_q(samples0_last, mod2);
    csub_q(samples0_last, mod);
    csub_q(samples1_last, mod2);
    csub_q(samples1_last, mod);
    glbIdx_last += mod_idx * n;
    inout[glbIdx_last] = samples0_last;
    inout[glbIdx_last + pairsInGroup_last] = samples1_last;
}

void fnwt_1d(uint64_t *inout,
             const uint64_t *twiddles,
             const uint64_t *twiddles_shoup,
             const DModulus *modulus,
             size_t dim,
             size_t coeff_modulus_size,
             size_t start_modulus_idx,
             const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);

    inplace_fnwt_radix2<<<coeff_modulus_size, dim / 2, per_block_memory, stream>>>(
            inout,
            twiddles,
            twiddles_shoup,
            modulus,
            coeff_modulus_size,
            start_modulus_idx,
            dim);
}

void fnwt_1d_opt(uint64_t *inout,
                 const uint64_t *twiddles,
                 const uint64_t *twiddles_shoup,
                 const DModulus *modulus,
                 size_t dim,
                 size_t coeff_modulus_size,
                 size_t start_modulus_idx,
                 const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);

    inplace_fnwt_radix2_opt<<<coeff_modulus_size, dim / 2, per_block_memory, stream>>>(
            inout,
            twiddles,
            twiddles_shoup,
            modulus,
            dim);
}

/** backward NTT transformation, with N (num of operands) up to 2048,
 * to ensure all operation completed in one block.
 * @param[inout] inout The value to operate and the returned result
 * @param[in] inverse_twiddles The pre-computated backward NTT table
 * @param[in] mod The coeff modulus value
 * @param[in] n The poly degreee
 * @param[in] logn The logarithm of n
 * @param[in] numOfGroups
 * @param[in] iter The current iteration in backward NTT transformation
 */
__global__ void inplace_inwt_radix2(uint64_t *inout,
                                    const uint64_t *itwiddles,
                                    const uint64_t *itwiddles_shoup,
                                    const DModulus *modulus,
                                    const uint64_t *scalar, const uint64_t *scalar_shoup,
                                    size_t coeff_mod_size,
                                    size_t start_mod_idx,
                                    size_t n) {
    extern __shared__ uint64_t buffer[];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n / 2 * coeff_mod_size;
         i += blockDim.x * gridDim.x) {
        size_t mod_idx = i / (n / 2) + start_mod_idx;
        size_t tid = i % (n / 2);

        size_t pairsInGroup;
        size_t k, j, glbIdx, bufIdx;
        uint64_t samples[2];

        const DModulus *modulus_table = modulus;
        uint64_t mod = modulus_table[mod_idx].value();

        const uint64_t scalar_ = scalar[mod_idx];
        const uint64_t scalar_shoup_ = scalar_shoup[mod_idx];

        for (size_t _numOfGroups = n / 2; _numOfGroups >= 1; _numOfGroups >>= 1) {
            pairsInGroup = n / _numOfGroups / 2;
            k = tid / pairsInGroup;
            j = tid % pairsInGroup;
            glbIdx = 2 * k * pairsInGroup + j;
            bufIdx = glbIdx % n;
            glbIdx += mod_idx * n;
            uint64_t psi = itwiddles[_numOfGroups + k + mod_idx * n];
            uint64_t psi_shoup = itwiddles_shoup[_numOfGroups + k + mod_idx * n];
            if (_numOfGroups == n >> 1) {
                samples[0] = inout[glbIdx];
                samples[1] = inout[glbIdx + pairsInGroup];
            } else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }

            gs_butterfly(samples[0], samples[1], psi, psi_shoup, mod);

            if (_numOfGroups == 1) {
                // final reduction
                csub_q(samples[0], mod);
                csub_q(samples[1], mod);
            }

            if (_numOfGroups == 1) {
                samples[0] = multiply_and_reduce_shoup(samples[0], scalar_, scalar_shoup_, mod);
                inout[glbIdx] = samples[0];
                inout[glbIdx + pairsInGroup] = samples[1];
            } else {
                buffer[bufIdx] = samples[0];
                buffer[bufIdx + pairsInGroup] = samples[1];
                __syncthreads();
            }
        }
    }
}

void inwt_1d(uint64_t *inout,
             const uint64_t *itwiddles, const uint64_t *itwiddles_shoup, const DModulus *modulus,
             const uint64_t *scalar, const uint64_t *scalar_shoup,
             size_t dim, size_t coeff_modulus_size, size_t start_modulus_idx,
             const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);

    inplace_inwt_radix2<<<coeff_modulus_size, dim / 2, per_block_memory, stream>>>(
            inout,
            itwiddles,
            itwiddles_shoup,
            modulus,
            scalar, scalar_shoup,
            coeff_modulus_size,
            start_modulus_idx,
            dim);
}

void inwt_1d_opt(uint64_t *inout,
                 const uint64_t *itwiddles, const uint64_t *itwiddles_shoup, const DModulus *modulus,
                 const uint64_t *scalar, const uint64_t *scalar_shoup,
                 size_t dim, size_t coeff_modulus_size, size_t start_modulus_idx,
                 const cudaStream_t &stream) {
    const size_t per_block_memory = dim * sizeof(uint64_t);

    inplace_inwt_radix2<<<coeff_modulus_size, dim / 2, per_block_memory, stream>>>(
            inout,
            itwiddles,
            itwiddles_shoup,
            modulus,
            scalar, scalar_shoup,
            coeff_modulus_size,
            start_modulus_idx,
            dim);
}

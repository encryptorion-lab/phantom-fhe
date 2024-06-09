#include "fft.h"
#include "context.cuh"

using namespace phantom::arith;

namespace phantom::util {
    // Required for C++14 compliance: static constexpr member variables are not necessarily inlined so need to
    // ensure symbol is created.
    constexpr double ComplexRoots::PI_;

    ComplexRoots::ComplexRoots(size_t degree_of_roots) : degree_of_roots_(degree_of_roots) {
        roots_ = (cuDoubleComplex *) malloc((degree_of_roots_ / 8 + 1) * sizeof(cuDoubleComplex));

        // Generate 1/8 of all roots.
        // Alternatively, choose from precomputed high-precision roots in files.
        for (size_t i = 0; i <= degree_of_roots_ / 8; i++) {
            roots_[i] = polar(1.0, 2 * PI_ * static_cast<double>(i) / static_cast<double>(degree_of_roots_));
        }
    }

    cuDoubleComplex ComplexRoots::get_root(size_t index) const {
        index &= degree_of_roots_ - 1;
        auto mirror = [](cuDoubleComplex a) {
            return make_cuDoubleComplex(a.y, a.x);
        };

        // This express the 8-fold symmetry of all n-th roots.
        if (index <= degree_of_roots_ / 8) {
            return roots_[index];
        } else if (index <= degree_of_roots_ / 4) {
            return mirror(roots_[degree_of_roots_ / 4 - index]);
        } else if (index <= degree_of_roots_ / 2) {
            return cuCsub({0, 0}, cuConj(get_root(degree_of_roots_ / 2 - index)));
        } else if (index <= 3 * degree_of_roots_ / 4) {
            return cuCsub({0, 0}, get_root(index - degree_of_roots_ / 2));
        } else {
            return cuConj(get_root(degree_of_roots_ - index));
        }
    }
}

/** Computer one butterfly in forward FFT
 * x[0] = x[0] + pow * x[1]
 * x[1] = x[0] - pow * x[1]
 * @param[inout] x Values to operate, two int64_t, x[0] and x[1]
 * @param[in] pow The pre-computated one twiddle
 */
__device__ __forceinline__ void ct_butterfly_cplx(cuDoubleComplex *x,
                                                  const cuDoubleComplex &pow) {
    cuDoubleComplex s[2];
    s[0] = x[0];
    s[1] = cuCmul(x[1], pow);
    x[0] = cuCadd(s[0], s[1]);
    x[1] = cuCsub(s[0], s[1]);
}

/** Computer one butterfly in inverse FFT
 * x[0] = x[0] + pow * x[1]
 * x[1] = x[0] - pow * x[1]
 * @param[inout] x Value to operate
 * @param[in] mod The modulus
 * @param[in] pow The pre-computated one twiddle
 */
__device__ __forceinline__ void gs_butterfly_cplx(cuDoubleComplex *x,
                                                  const cuDoubleComplex &pow) {
    cuDoubleComplex s[2];

    s[0] = cuCadd(x[0], x[1]);
    s[1] = cuCsub(x[0], x[1]);

    // x[0] = divide2_dc(s[0]); // div-2 mod
    // x[1] = divide2_dc(s[1]);
    x[0] = s[0];
    x[1] = cuCmul(s[1], pow);
}

/** forward FFT transformation, with N (num of operands) up to 2048,
 * to ensure all operation completed in one block.
 * @param[inout] inout The value to operate and the returned result
 * @param[in] twiddles The pre-computated forward NTT table
 * @param[in] group powers of 5
 * @param[in] n The slot
 * @param[in] logn The logarithm of n
 * @param[in] numOfGroups
 * @param[in] iter The current iteration in forward NTT transformation
 * @param[in] scalar ckks encoding scalar divided by n
 */
__global__ void inplace_special_ffft_base_kernel(cuDoubleComplex *inout,
                                                 const cuDoubleComplex *twiddles,
                                                 const uint32_t *group,
                                                 const uint32_t n, const uint32_t logn,
                                                 const uint32_t numOfGroups, const uint32_t iter,
                                                 const uint32_t M, const uint32_t logRatio) {
    extern __shared__ cuDoubleComplex buffer[];

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n >> 1); // deal with 2 data per thread
         tid += blockDim.x * gridDim.x) {
        uint32_t logPairs, pairsInGroup;
        uint32_t k, j, glbIdx, bufIdx; // k = psi_step
        cuDoubleComplex one_twiddle;
        cuDoubleComplex samples[2];
        //============================
        uint32_t psiIdx;
        //============================
        int _iter = iter;
        int _numOfGroups = numOfGroups;
        for (; _numOfGroups < n; _numOfGroups <<= 1) {
            logPairs = logn - _iter - 1;
            pairsInGroup = 1 << logPairs;

            k = tid >> logPairs;
            j = tid & (pairsInGroup - 1);
            glbIdx = 2 * k * pairsInGroup + j;
            bufIdx = glbIdx - blockIdx.x * (n >> iter);

            //============================
            // bit-reverse width = logn - 1 - sparseRatio
            psiIdx = group[__brev(k << logPairs) >> (33 - logn)];
            psiIdx <<= logPairs + logRatio;
            psiIdx &= M - 1; // %M
            // printf("blk = %d, thr = %d, %d, %d, %d\n", blockIdx.x, tid, _numOfGroups, pairsInGroup, psiIdx);
            one_twiddle = twiddles[psiIdx];
            //============================

            if (_numOfGroups == numOfGroups) {
                samples[0] = inout[glbIdx];
                samples[1] = inout[glbIdx + pairsInGroup];
            } else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }
            // if (tid == 0)
            // {
            //     printf("=======================input===================\n");
            //     printf("bufIdx: %d,  pairsInGroup: %d, %d,\n", bufIdx, pairsInGroup, _numOfGroups);
            //     printf("tid = %d, %lf+i%lf, %lf+i%lf, %lf+i%lf\n", tid, samples[0].x, samples[0].y, samples[1].x, samples[1].y, one_twiddle.x, one_twiddle.y);
            // }
            // if (_iter == 2)
            //     printf("iter = %d, tid = %d, %lf + i %lf,  %lf + i %lf\n\n", _iter, tid, samples[0].x, samples[0].y, samples[1].x, samples[1].y);
            ct_butterfly_cplx(samples, one_twiddle);

            _iter += 1;

            if (_numOfGroups == n >> 1) {
                // inout[__brev(glbIdx) >> (32 - logn)] = samples[0];
                // inout[__brev(glbIdx + pairsInGroup) >> (32 - logn)] = samples[1];
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

/** forward NTT transformation, with N (num of operands) is larger than 2048
 * @param[inout] inout The value to operate and the returned result
 * @param[in] twiddles The pre-computated forward NTT table
 * @param[in] n The poly degreee
 * @param[in] logn The logarithm of n
 * @param[in] numOfGroups
 * @param[in] iter The current iteration in forward NTT transformation
 */
__global__ void inplace_special_ffft_iter_kernel(cuDoubleComplex *inout,
                                                 const cuDoubleComplex *twiddles,
                                                 const uint32_t *group,
                                                 const uint32_t n, const uint32_t logn,
                                                 const uint32_t numOfGroups, const uint32_t iter,
                                                 const uint32_t M, const uint32_t logRatio) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n >> 1);
         tid += blockDim.x * gridDim.x) {
        uint32_t logPairs, pairsInGroup;
        uint32_t k, j, glbIdx;
        cuDoubleComplex one_twiddle;
        cuDoubleComplex samples[2];

        logPairs = logn - iter - 1;
        pairsInGroup = 1 << logPairs;

        k = tid >> logPairs;
        j = tid & (pairsInGroup - 1);
        glbIdx = 2 * k * pairsInGroup + j;

        //============================
        uint32_t psiIdx;
        psiIdx = group[__brev(k << logPairs) >> (33 - logn)];
        psiIdx <<= logPairs + logRatio;
        psiIdx &= M - 1; // %M
        one_twiddle = twiddles[psiIdx];
        //============================

        samples[0] = inout[glbIdx];
        samples[1] = inout[glbIdx + pairsInGroup];
        // if (iter == 0)
        //     printf("iter = %d, tid = %d, %lf + i %lf,  %lf + i %lf\n\n", iter, tid, samples[0].x, samples[0].y, samples[1].x, samples[1].y);
        ct_butterfly_cplx(samples, one_twiddle);

        inout[glbIdx] = samples[0];
        inout[glbIdx + pairsInGroup] = samples[1];
    }
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
__global__ void inplace_special_ifft_base_kernel(cuDoubleComplex *inout,
                                                 const cuDoubleComplex *twiddles,
                                                 const uint32_t *group,
                                                 const uint32_t n, const uint32_t logn,
                                                 const uint32_t numOfGroups, const int32_t iter,
                                                 const uint32_t M, const uint32_t logRatio,
                                                 double scalar) {
    extern __shared__ cuDoubleComplex buffer[];

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n >> 1);
         tid += blockDim.x * gridDim.x) {
        uint32_t logPairs, pairsInGroup;
        uint32_t k, j, glbIdx, bufIdx;
        cuDoubleComplex one_twiddle;
        cuDoubleComplex samples[2];
        int _iter = logn - 1;
        int _numOfGroups = n >> 1;
        //============================
        uint32_t psiIdx;
        //============================
        for (; _numOfGroups >= numOfGroups; _numOfGroups >>= 1) {
            logPairs = logn - _iter - 1;
            pairsInGroup = 1 << logPairs;
            k = tid >> logPairs;
            j = tid & (pairsInGroup - 1);
            glbIdx = 2 * k * pairsInGroup + j;
            bufIdx = glbIdx - blockIdx.x * (n >> iter);

            //============================
            psiIdx = group[__brev(k << logPairs) >> (33 - logn)];
            psiIdx <<= logPairs + logRatio;
            psiIdx &= M - 1; // %M
            // printf("blk = %d, thr = %d, %d, %d, %d\n", blockIdx.x, tid, _numOfGroups, pairsInGroup, (n << 2) - psiIdx);
            one_twiddle = twiddles[M - psiIdx];
            // printf("blk = %d, thr = %d, %d, %lf, %lf\n", blockIdx.x, tid, (n << 2) - psiIdx, one_twiddle.x, one_twiddle.y);

            //============================

            if (_numOfGroups == n >> 1) {
                // samples[0] = inout[__brev(glbIdx) >> (32 - logn)];
                // samples[1] = inout[__brev(glbIdx + pairsInGroup) >> (32 - logn)];
                samples[0] = inout[glbIdx];
                samples[1] = inout[glbIdx + pairsInGroup];
            } else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }

            gs_butterfly_cplx(samples, one_twiddle);
            _iter -= 1;
            // printf("tid = %d, %lf+i%lf, %lf+i%lf, %lf\n", tid, samples[0].x, samples[0].y, samples[1].x, samples[1].y, scalar);
            if (_numOfGroups == 1) {
                if (scalar != 0.0) {
                    samples[0] = scalar_multiply_cuDoubleComplex(samples[0], scalar);
                    samples[1] = scalar_multiply_cuDoubleComplex(samples[1], scalar);
                }
            }

            if (_numOfGroups == numOfGroups) {
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

/** backward NTT transformation, with N (num of operands) larger than 2048,
 * @param[inout] inout The value to operate and the returned result
 * @param[in] inverse_twiddles The pre-computated backward NTT table
 * @param[in] mod The coeff modulus value
 * @param[in] n The poly degreee
 * @param[in] logn The logarithm of n
 * @param[in] numOfGroups
 * @param[in] iter The current iteration in backward NTT transformation
 */
__global__ void inplace_special_ifft_iter_kernel(cuDoubleComplex *inout,
                                                 const cuDoubleComplex *twiddles,
                                                 const uint32_t *group,
                                                 const uint32_t n, const uint32_t logn,
                                                 const uint32_t numOfGroups, const int32_t iter,
                                                 const uint32_t M, const uint32_t logRatio,
                                                 double scalar) {
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n >> 1);
         tid += blockDim.x * gridDim.x) {
        uint32_t logPairs, pairsInGroup;
        uint32_t k, j, glbIdx;
        cuDoubleComplex one_twiddle;
        cuDoubleComplex samples[2];

        logPairs = logn - iter - 1;
        pairsInGroup = 1 << logPairs;

        k = tid >> logPairs;
        j = tid & (pairsInGroup - 1);
        glbIdx = 2 * k * pairsInGroup + j;

        //============================
        uint32_t psiIdx;
        psiIdx = group[__brev(k << logPairs) >> (33 - logn)];
        psiIdx <<= logPairs + logRatio;
        psiIdx &= M - 1; // %M
        one_twiddle = twiddles[M - psiIdx];
        //============================

        /* if (numOfGroups == n >> 1)
        {
            samples[0] = inout[__brev(glbIdx) >> (32 - logn)];
            samples[1] = inout[__brev(glbIdx + pairsInGroup) >> (32 - logn)];
        }
        else
        {
            samples[0] = inout[glbIdx];
            samples[1] = inout[glbIdx + pairsInGroup];
        } */

        samples[0] = inout[glbIdx];
        samples[1] = inout[glbIdx + pairsInGroup];

        gs_butterfly_cplx(samples, one_twiddle);
        if (numOfGroups == 1) {
            if (scalar != 0.0) {
                samples[0] = scalar_multiply_cuDoubleComplex(samples[0], scalar);
                samples[1] = scalar_multiply_cuDoubleComplex(samples[1], scalar);
            }
        }

        inout[glbIdx] = samples[0];
        inout[glbIdx + pairsInGroup] = samples[1];
    }
}

/** Perform forward NTT transformation
 * @param[inout] gpu_rns_vec_ The DRNSInfo stored in PhantomContext.
 * @param[in] coeff_mod_size The number of coeff modulus
 */
void special_fft_forward(DCKKSEncoderInfo &gp, const cudaStream_t &stream) {
    uint32_t threadsPerBlock, blocksPerGrid;

    if (gp.sparse_slots() == 00) {
        throw std::invalid_argument("the poly degree has not been set");
    }
    uint32_t logn = log2(gp.sparse_slots());
    uint32_t logRatio = log2(gp.m()) - logn - 2;

    if (gp.sparse_slots() <= SWITCH_POINT) {
        // max 1024 threads, max n = 2048
        threadsPerBlock = gp.sparse_slots() >> 1;
        blocksPerGrid = 1;
        uint32_t iter = 0;
        uint32_t numOfGroups = 1;
        inplace_special_ffft_base_kernel<<<blocksPerGrid, threadsPerBlock,
        gp.sparse_slots() * sizeof(cuDoubleComplex), stream>>>(
                gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter, gp.m(),
                logRatio);
    } else {
        uint32_t iter = 0;
        uint32_t numOfGroups = 1;
        threadsPerBlock = NTT_THREAD_PER_BLOCK;
        blocksPerGrid = ceil((float) gp.sparse_slots() / threadsPerBlock / 2);
        for (; numOfGroups < (gp.sparse_slots() / SWITCH_POINT); numOfGroups <<= 1) {
            inplace_special_ffft_iter_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter, gp.m(),
                    logRatio);

            iter++;
        }

        inplace_special_ffft_base_kernel<<<
        blocksPerGrid, threadsPerBlock, SWITCH_POINT * sizeof(cuDoubleComplex), stream>>>(
                gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter,
                gp.m(), logRatio);
    }
}

/** Perform backward FFT transformation
 * @param[inout] gp DCKKSEncoderInfo
 * @param[in] coeff_mod_size The number of coeff modulus
 */
void special_fft_backward(DCKKSEncoderInfo &gp, double scalar, const cudaStream_t &stream) {
    uint32_t threadsPerBlock, blocksPerGrid;
    uint32_t logn = log2(gp.sparse_slots());
    uint32_t logRatio = log2(gp.m()) - logn - 2;
    if (gp.sparse_slots() <= SWITCH_POINT) {
        //  max 1024 threads, max n = 2048
        uint32_t iter = 0;
        uint32_t numOfGroups = 1;
        threadsPerBlock = gp.sparse_slots() >> 1;
        blocksPerGrid = 1;
        inplace_special_ifft_base_kernel<<<
        blocksPerGrid, threadsPerBlock, gp.sparse_slots() * sizeof(cuDoubleComplex), stream>>>(
                gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter, gp.m(),
                logRatio, scalar);
    } else {
        int32_t iter = logn - log2(SWITCH_POINT);
        uint32_t numOfGroups = gp.sparse_slots() / SWITCH_POINT;
        if (iter < 0)
            iter = 0;
        if (numOfGroups < 1)
            numOfGroups = 1;

        threadsPerBlock = NTT_THREAD_PER_BLOCK;
        blocksPerGrid = ceil((float) gp.sparse_slots() / threadsPerBlock / 2);

        inplace_special_ifft_base_kernel<<<
        blocksPerGrid, threadsPerBlock, SWITCH_POINT * sizeof(cuDoubleComplex), stream>>>(
                gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter,
                gp.m(), logRatio, scalar);
        numOfGroups >>= 1;
        for (; numOfGroups >= 1; numOfGroups >>= 1) {
            iter--;

            inplace_special_ifft_iter_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter, gp.m(),
                    logRatio, scalar);
        }
    }
}

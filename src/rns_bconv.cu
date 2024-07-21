#include "ntt.cuh"
#include "polymath.cuh"
#include "rns.cuh"
#include "rns_bconv.cuh"
#include "util.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

/**
 * generic base conversion phase 1
 * dst = src * scale % base
 * @param dst
 * @param src
 * @param scale
 * @param base
 * @param base_size
 * @param n
 */
__global__ void bconv_mult_kernel(uint64_t *dst, const uint64_t *src, const uint64_t *scale,
                                  const uint64_t *scale_shoup, const DModulus *base, uint64_t base_size, uint64_t n) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n * base_size; tid += blockDim.x * gridDim.x) {
        size_t i = tid / n;
        auto modulus = base[i].value();
        dst[tid] = multiply_and_reduce_shoup(src[tid], scale[i], scale_shoup[i], modulus);
    }
}

/**
 * generic base conversion phase 1 unroll 2
 * @param dst
 * @param src
 * @param scale
 * @param base
 * @param base_size
 * @param n
 */
__global__ void bconv_mult_unroll2_kernel(uint64_t *dst, const uint64_t *src, const uint64_t *scale,
                                          const uint64_t *scale_shoup, const DModulus *base, uint64_t base_size,
                                          uint64_t n) {
    constexpr const int unroll_factor = 2;

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < base_size * n / unroll_factor;
         tid += blockDim.x * gridDim.x) {
        size_t i = tid / (n / unroll_factor);
        size_t coeff_idx = tid * unroll_factor;
        auto modulus = base[i].value();
        auto scale_factor = scale[i];
        auto scale_factor_shoup = scale_shoup[i];
        uint64_t in_x, in_y;
        uint64_t out_x, out_y;

        ld_two_uint64(in_x, in_y, src + coeff_idx);
        out_x = multiply_and_reduce_shoup(in_x, scale_factor, scale_factor_shoup, modulus);
        out_y = multiply_and_reduce_shoup(in_y, scale_factor, scale_factor_shoup, modulus);
        st_two_uint64(dst + coeff_idx, out_x, out_y);
    }
}

/**
 * generic base conversion phase 1 unroll 4
 * @param dst
 * @param src
 * @param scale
 * @param base
 * @param base_size
 * @param n
 */
__global__ void bconv_mult_unroll4_kernel(uint64_t *dst, const uint64_t *src, const uint64_t *scale,
                                          const uint64_t *scale_shoup, const DModulus *base, uint64_t base_size,
                                          uint64_t n) {
    constexpr const int unroll_number = 4;

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < base_size * n / unroll_number;
         tid += blockDim.x * gridDim.x) {
        size_t i = tid / (n / unroll_number);
        size_t coeff_idx = tid * unroll_number;
        auto modulus = base[i].value();
        auto scale_factor = scale[i];
        auto scale_factor_shoup = scale_shoup[i];
        uint64_t in_x, in_y;
        uint64_t out_x, out_y;

        ld_two_uint64(in_x, in_y, src + coeff_idx);
        out_x = multiply_and_reduce_shoup(in_x, scale_factor, scale_factor_shoup, modulus);
        out_y = multiply_and_reduce_shoup(in_y, scale_factor, scale_factor_shoup, modulus);
        st_two_uint64(dst + coeff_idx, out_x, out_y);

        ld_two_uint64(in_x, in_y, src + coeff_idx + 2);
        out_x = multiply_and_reduce_shoup(in_x, scale_factor, scale_factor_shoup, modulus);
        out_y = multiply_and_reduce_shoup(in_y, scale_factor, scale_factor_shoup, modulus);
        st_two_uint64(dst + coeff_idx + 2, out_x, out_y);
    }
}

/**
 * generic base conversion phase 2
 * @param dst
 * @param xi_qiHatInv_mod_qi
 * @param QHatModp
 * @param ibase
 * @param ibase_size
 * @param obase
 * @param obase_size
 * @param n
 */
__global__ void bconv_matmul_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *QHatModp,
                                    const DModulus *ibase, uint64_t ibase_size, const DModulus *obase,
                                    uint64_t obase_size, uint64_t n) {
    extern __shared__ uint64_t s_QHatModp[];
    for (size_t idx = threadIdx.x; idx < obase_size * ibase_size; idx += blockDim.x) {
        s_QHatModp[idx] = QHatModp[idx];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < obase_size * n; tid += blockDim.x * gridDim.x) {
        const size_t degree_idx = tid / obase_size;
        const size_t out_prime_idx = tid % obase_size;

        uint128_t accum = base_convert_acc(xi_qiHatInv_mod_qi, s_QHatModp, out_prime_idx, n, ibase_size, degree_idx);

        uint64_t obase_value = obase[out_prime_idx].value();
        uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};

        uint64_t out = barrett_reduce_uint128_uint64(accum, obase_value, obase_ratio);
        dst[out_prime_idx * n + degree_idx] = out;
    }
}

/**
 * generic base conversion phase 2 unroll 2
 * @param dst
 * @param xi_qiHatInv_mod_qi
 * @param QHatModp
 * @param ibase
 * @param ibase_size
 * @param obase
 * @param obase_size
 * @param n
 */
__global__ void bconv_matmul_unroll2_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *QHatModp,
                                            const DModulus *ibase, uint64_t ibase_size, const DModulus *obase,
                                            uint64_t obase_size, uint64_t n) {
    constexpr const int unroll_number = 2;
    extern __shared__ uint64_t s_QHatModp[];
    for (size_t idx = threadIdx.x; idx < obase_size * ibase_size; idx += blockDim.x) {
        s_QHatModp[idx] = QHatModp[idx];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < obase_size * n / unroll_number;
         tid += blockDim.x * gridDim.x) {
        const size_t degree_idx = unroll_number * (tid / obase_size);
        const size_t out_prime_idx = tid % obase_size;

        uint128_t2 accum =
                base_convert_acc_unroll2(xi_qiHatInv_mod_qi, s_QHatModp, out_prime_idx, n, ibase_size, degree_idx);

        uint64_t obase_value = obase[out_prime_idx].value();
        uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};

        uint64_t out = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
        uint64_t out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
        st_two_uint64(dst + out_prime_idx * n + degree_idx, out, out2);
    }
}

/**
 * generic base conversion phase 2 unroll 4
 * @param dst
 * @param xi_qiHatInv_mod_qi
 * @param QHatModp
 * @param ibase
 * @param ibase_size
 * @param obase
 * @param obase_size
 * @param n
 */
__global__ void bconv_matmul_unroll4_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *QHatModp,
                                            const DModulus *ibase, uint64_t ibase_size, const DModulus *obase,
                                            uint64_t obase_size, uint64_t n) {
    constexpr const int unroll_number = 4;
    extern __shared__ uint64_t s_QHatModp[];
    for (size_t idx = threadIdx.x; idx < obase_size * ibase_size; idx += blockDim.x) {
        s_QHatModp[idx] = QHatModp[idx];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (n * obase_size + unroll_number - 1) / unroll_number;
         tid += blockDim.x * gridDim.x) {
        const size_t degree_idx = unroll_number * (tid / obase_size);
        const size_t out_prime_idx = tid % obase_size;

        uint128_t4 accum =
                base_convert_acc_unroll4(xi_qiHatInv_mod_qi, s_QHatModp, out_prime_idx, n, ibase_size, degree_idx);

        uint64_t obase_value = obase[out_prime_idx].value();
        uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};

        uint64_t out = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
        uint64_t out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
        st_two_uint64(dst + out_prime_idx * n + degree_idx, out, out2);

        out = barrett_reduce_uint128_uint64(accum.z, obase_value, obase_ratio);
        out2 = barrett_reduce_uint128_uint64(accum.w, obase_value, obase_ratio);
        st_two_uint64(dst + out_prime_idx * n + degree_idx + 2, out, out2);
    }
}

void DBaseConverter::bConv_BEHZ(uint64_t *dst, const uint64_t *src, size_t n, const cudaStream_t &stream) const {
    size_t ibase_size = ibase_.size();
    size_t obase_size = obase_.size();

    auto temp = make_cuda_auto_ptr<uint64_t>(ibase_size * n, stream);

    constexpr int unroll_factor = 2;

    uint64_t gridDimGlb = ibase_size * n / unroll_factor / blockDimGlb.x;
    bconv_mult_unroll2_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            temp.get(), src, ibase_.QHatInvModq(),
            ibase_.QHatInvModq_shoup(), ibase_.base(), ibase_size, n);

    gridDimGlb = obase_size * n / unroll_factor / blockDimGlb.x;
    bconv_matmul_unroll2_kernel<<<
    gridDimGlb, blockDimGlb, sizeof(uint64_t) * obase_size * ibase_size, stream>>>(
            dst, temp.get(), QHatModp(), ibase_.base(), ibase_size, obase_.base(), obase_size, n);
}

void DBaseConverter::bConv_BEHZ_var1(uint64_t *dst, const uint64_t *src, size_t n, const cudaStream_t &stream) const {
    size_t ibase_size = ibase_.size();
    size_t obase_size = obase_.size();

    auto temp = make_cuda_auto_ptr<uint64_t>(ibase_size * n, stream);

    constexpr int unroll_factor = 2;

    uint64_t gridDimGlb = ibase_size * n / unroll_factor / blockDimGlb.x;
    bconv_mult_unroll2_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            temp.get(), src, negPQHatInvModq(), negPQHatInvModq_shoup(), ibase_.base(), ibase_size, n);

    gridDimGlb = obase_size * n / unroll_factor / blockDimGlb.x;
    bconv_matmul_unroll2_kernel<<<gridDimGlb, blockDimGlb, sizeof(uint64_t) * obase_size * ibase_size, stream>>>(
            dst, temp.get(), QInvModp(), ibase_.base(), ibase_size, obase_.base(), obase_size, n);
}

[[maybe_unused]] __global__ static void
base_convert_matmul_hps_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *qiHat_mod_pj,
                               const uint64_t *v_Q_mod_pj, const double *qiInv, const DModulus *ibase,
                               uint64_t ibase_size, const DModulus *obase, uint64_t obase_size, uint64_t n) {
    extern __shared__ uint64_t s_qiHat_mod_pj[];
    for (size_t i = threadIdx.x; i < obase_size * ibase_size; i += blockDim.x) {
        s_qiHat_mod_pj[i] = qiHat_mod_pj[i];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < obase_size * n; tid += blockDim.x * gridDim.x) {
        const size_t degree_idx = tid / obase_size;
        const size_t out_prime_idx = tid % obase_size;

        uint128_t accum =
                base_convert_acc(xi_qiHatInv_mod_qi, s_qiHat_mod_pj, out_prime_idx, n, ibase_size, degree_idx);

        double_t accum_frac = base_convert_acc_frac(xi_qiHatInv_mod_qi, qiInv, n, ibase_size, degree_idx);

        uint64_t obase_value = obase[out_prime_idx].value();
        uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};

        uint64_t out = barrett_reduce_uint128_uint64(accum, obase_value, obase_ratio);
        uint64_t vQ_mod_pj = v_Q_mod_pj[llround(accum_frac) * obase_size + out_prime_idx];
        out = sub_uint64_uint64_mod(out, vQ_mod_pj, obase_value);
        dst[out_prime_idx * n + degree_idx] = out;
    }
}

__global__ static void base_convert_matmul_hps_unroll2_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi,
                                                              const uint64_t *qiHat_mod_pj, const uint64_t *v_Q_mod_pj,
                                                              const double *qiInv, const DModulus *ibase,
                                                              uint64_t ibase_size, const DModulus *obase,
                                                              uint64_t obase_size, uint64_t n) {
    constexpr const int unroll_number = 2;
    extern __shared__ uint64_t s_qiHat_mod_pj[];
    for (size_t i = threadIdx.x; i < obase_size * ibase_size; i += blockDim.x) {
        s_qiHat_mod_pj[i] = qiHat_mod_pj[i];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < obase_size * n / unroll_number;
         tid += blockDim.x * gridDim.x) {
        const size_t degree_idx = unroll_number * (tid / obase_size);
        const size_t out_prime_idx = tid % obase_size;

        uint128_t2 accum =
                base_convert_acc_unroll2(xi_qiHatInv_mod_qi, s_qiHat_mod_pj, out_prime_idx, n, ibase_size, degree_idx);

        double_t2 accum_frac = base_convert_acc_frac_unroll2(xi_qiHatInv_mod_qi, qiInv, n, ibase_size, degree_idx);

        uint64_t obase_value = obase[out_prime_idx].value();
        uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};

        uint64_t out = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
        uint64_t out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
        uint64_t vQ_mod_pj = v_Q_mod_pj[llround(accum_frac.x) * obase_size + out_prime_idx];
        uint64_t vQ_mod_pj2 = v_Q_mod_pj[llround(accum_frac.y) * obase_size + out_prime_idx];
        out = sub_uint64_uint64_mod(out, vQ_mod_pj, obase_value);
        out2 = sub_uint64_uint64_mod(out2, vQ_mod_pj2, obase_value);
        st_two_uint64(dst + out_prime_idx * n + degree_idx, out, out2);
    }
}

[[maybe_unused]] __global__ static void
base_convert_matmul_hps_unroll4_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *qiHat_mod_pj,
                                       const uint64_t *v_Q_mod_pj, const double *qiInv, const DModulus *ibase,
                                       uint64_t ibase_size, const DModulus *obase, uint64_t obase_size, uint64_t n) {
    constexpr const int unroll_number = 4;
    extern __shared__ uint64_t s_qiHat_mod_pj[];
    for (size_t i = threadIdx.x; i < obase_size * ibase_size; i += blockDim.x) {
        s_qiHat_mod_pj[i] = qiHat_mod_pj[i];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (n * obase_size + unroll_number - 1) / unroll_number;
         tid += blockDim.x * gridDim.x) {
        const size_t degree_idx = unroll_number * (tid / obase_size);
        const size_t out_prime_idx = tid % obase_size;

        uint128_t4 accum =
                base_convert_acc_unroll4(xi_qiHatInv_mod_qi, s_qiHat_mod_pj, out_prime_idx, n, ibase_size, degree_idx);

        double_t4 accum_frac = base_convert_acc_frac_unroll4(xi_qiHatInv_mod_qi, qiInv, n, ibase_size, degree_idx);

        uint64_t obase_value = obase[out_prime_idx].value();
        uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};

        uint64_t out = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
        uint64_t out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
        uint64_t vQ_mod_pj = v_Q_mod_pj[llround(accum_frac.x) * obase_size + out_prime_idx];
        uint64_t vQ_mod_pj2 = v_Q_mod_pj[llround(accum_frac.y) * obase_size + out_prime_idx];
        out = sub_uint64_uint64_mod(out, vQ_mod_pj, obase_value);
        out2 = sub_uint64_uint64_mod(out2, vQ_mod_pj2, obase_value);
        st_two_uint64(dst + out_prime_idx * n + degree_idx, out, out2);

        out = barrett_reduce_uint128_uint64(accum.z, obase_value, obase_ratio);
        out2 = barrett_reduce_uint128_uint64(accum.w, obase_value, obase_ratio);
        vQ_mod_pj = v_Q_mod_pj[llround(accum_frac.z) * obase_size + out_prime_idx];
        vQ_mod_pj2 = v_Q_mod_pj[llround(accum_frac.w) * obase_size + out_prime_idx];
        out = sub_uint64_uint64_mod(out, vQ_mod_pj, obase_value);
        out2 = sub_uint64_uint64_mod(out2, vQ_mod_pj2, obase_value);
        st_two_uint64(dst + out_prime_idx * n + degree_idx + 2, out, out2);
    }
}

void DBaseConverter::bConv_HPS(uint64_t *dst, const uint64_t *src, size_t n, const cudaStream_t &stream) const {
    size_t ibase_size = ibase_.size();
    size_t obase_size = obase_.size();

    auto temp = make_cuda_auto_ptr<uint64_t>(ibase_size * n, stream);

    constexpr int unroll_factor = 2;

    uint64_t gridDimGlb = ibase_size * n / unroll_factor / blockDimGlb.x;
    bconv_mult_unroll2_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(temp.get(), src, ibase_.QHatInvModq(),
                                                                      ibase_.QHatInvModq_shoup(), ibase_.base(),
                                                                      ibase_size, n);

    gridDimGlb = obase_size * n / unroll_factor / blockDimGlb.x;
    base_convert_matmul_hps_unroll2_kernel<<<
    gridDimGlb, blockDimGlb, sizeof(uint64_t) * obase_size * ibase_size, stream>>>(
            dst, temp.get(), QHatModp(), alpha_Q_mod_pj(), ibase_.qiInv(), ibase_.base(), ibase_size, obase_.base(),
            obase_size, n);
}

__global__ static void exact_convert_array_kernel(uint64_t *dst, const uint64_t *src, const DModulus *ibase,
                                                  const uint64_t ibase_size, const DModulus *obase,
                                                  const uint64_t obase_size, const uint64_t *ibase_prod,
                                                  const uint64_t *inv_punctured_prod_mod_base_array,
                                                  const uint64_t *inv_punctured_prod_mod_base_array_shoup,
                                                  const uint64_t *base_change_matrix, const uint64_t poly_degree,
                                                  const uint64_t reduction_threshold) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree; tid += blockDim.x * gridDim.x) {
        uint64_t yi;
        double_t v = 0.0;
        uint64_t rounded_v;
        DModulus t = obase[0];
        uint128_t inner_prod = {0, 0};
        uint64_t q_mod_t = 0;

        for (uint64_t i = 0; i < ibase_size; i++) {
            uint64_t qi = ibase[i].value();
            // Calculate [x_{i} * \tilde{q_{i}}]_{q_{i}}
            yi = multiply_and_reduce_shoup(src[tid + i * poly_degree], inv_punctured_prod_mod_base_array[i],
                                           inv_punctured_prod_mod_base_array_shoup[i], qi);

            if ((i != 0) && (i && reduction_threshold == 0)) {
                inner_prod.lo = barrett_reduce_uint128_uint64(inner_prod, t.value(), t.const_ratio());
                inner_prod.hi = 0;
            }
            add_uint128_uint128(multiply_uint64_uint64(yi, base_change_matrix[i]), inner_prod, inner_prod);

            q_mod_t = barrett_reduce_uint128_uint64({q_mod_t, ibase_prod[ibase_size - i - 1]}, t.value(),
                                                    t.const_ratio());

            // sum of y_{i}/q_{i}
            v += static_cast<double_t>(yi) / static_cast<double_t>(qi);
        }
        // Inner product < [x_{i} * \tilde{q_{i}}]_{q_{i}}, [q*_{i}]_t >
        inner_prod.lo = barrett_reduce_uint128_uint64(inner_prod, t.value(), t.const_ratio());
        rounded_v = static_cast<uint64_t>(round(v));
        // [inner_prod - v*q]_t
        q_mod_t = multiply_and_barrett_reduce_uint64(rounded_v, q_mod_t, t.value(), t.const_ratio());
        dst[tid] = sub_uint64_uint64_mod(inner_prod.lo, q_mod_t, t.value());
    }
}

void DBaseConverter::exact_convert_array(uint64_t *dst, const uint64_t *src, uint64_t poly_degree,
                                         const cudaStream_t &stream) const {
    size_t ibase_size = ibase_.size();
    size_t obase_size = obase_.size();
    uint64_t gridDimGlb = poly_degree / blockDimGlb.x;
    // mask of reduction threshold
    auto reduction_threshold = 15;
    if (obase_size != 1) {
        throw invalid_argument("out base in exact_convert_array must be one.");
    }

    exact_convert_array_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            dst, src, ibase_.base(), ibase_size, obase_.base(), obase_size, ibase_.big_modulus(), ibase_.QHatInvModq(),
            ibase_.QHatInvModq_shoup(), QHatModp(), poly_degree, reduction_threshold);
}

__global__ static void modup_bconv_single_p_kernel(uint64_t *dst, const uint64_t *src_raw,
                                                   const uint64_t *src_normal_form, size_t in_prime_idx, size_t n,
                                                   const DModulus *base_QlP, uint64_t size_QlP) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n * size_QlP; tid += blockDim.x * gridDim.x) {
        const size_t out_prime_idx = tid / n;
        const size_t coeff_idx = tid % n;
        if (out_prime_idx != in_prime_idx) {
            const uint64_t in_prime = base_QlP[in_prime_idx].value();
            const uint64_t out_prime = base_QlP[out_prime_idx].value();
            const uint64_t barret_ratio = base_QlP[out_prime_idx].const_ratio()[1];
            const uint64_t coeff = src_normal_form[coeff_idx];
            uint64_t result;
            if (in_prime > out_prime)
                result = barrett_reduce_uint64_uint64(coeff, out_prime, barret_ratio);
            else
                result = coeff;
            dst[tid] = result;
        } else {
            dst[tid] = src_raw[coeff_idx];
        }
    }
}

__global__ static void bconv_matmul_padded_unroll2_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi,
                                                          const uint64_t *qiHat_mod_pj, const DModulus *ibase,
                                                          uint64_t ibase_size, const DModulus *obase,
                                                          uint64_t obase_size, uint64_t n, size_t startPartIdx,
                                                          size_t size_PartQl) {
    constexpr const int unroll_number = 2;
    extern __shared__ uint64_t s_qiHat_mod_pj[];
    for (size_t i = threadIdx.x; i < obase_size * ibase_size; i += blockDim.x) {
        s_qiHat_mod_pj[i] = qiHat_mod_pj[i];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (n * obase_size + unroll_number - 1) / unroll_number;
         tid += blockDim.x * gridDim.x) {
        const size_t degree_idx = unroll_number * (tid / obase_size);
        const size_t out_prime_idx = tid % obase_size;

        uint128_t2 accum =
                base_convert_acc_unroll2(xi_qiHatInv_mod_qi, s_qiHat_mod_pj, out_prime_idx, n, ibase_size, degree_idx);

        uint64_t obase_value = obase[out_prime_idx].value();
        uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};

        // Leap over the overlapped region.
        const size_t padded_out_prime_idx = out_prime_idx + ((out_prime_idx >= startPartIdx) ? size_PartQl : 0);

        uint64_t out = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
        uint64_t out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
        st_two_uint64(dst + padded_out_prime_idx * n + degree_idx, out, out2);
    }
}

[[maybe_unused]] __global__ static void
bconv_matmul_padded_unroll4_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *qiHat_mod_pj,
                                   const DModulus *ibase, uint64_t ibase_size, const DModulus *obase,
                                   uint64_t obase_size, uint64_t n, size_t startPartIdx, size_t size_PartQl) {
    constexpr const int unroll_number = 4;
    extern __shared__ uint64_t s_qiHat_mod_pj[];
    for (size_t i = threadIdx.x; i < obase_size * ibase_size; i += blockDim.x) {
        s_qiHat_mod_pj[i] = qiHat_mod_pj[i];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (n * obase_size + unroll_number - 1) / unroll_number;
         tid += blockDim.x * gridDim.x) {
        const size_t degree_idx = unroll_number * (tid / obase_size);
        const size_t out_prime_idx = tid % obase_size;

        uint128_t4 accum =
                base_convert_acc_unroll4(xi_qiHatInv_mod_qi, s_qiHat_mod_pj, out_prime_idx, n, ibase_size, degree_idx);

        uint64_t obase_value = obase[out_prime_idx].value();
        uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};

        // Leap over the overlapped region.
        const size_t padded_out_prime_idx = out_prime_idx + ((out_prime_idx >= startPartIdx) ? size_PartQl : 0);

        uint64_t out = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
        uint64_t out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
        st_two_uint64(dst + padded_out_prime_idx * n + degree_idx, out, out2);

        out = barrett_reduce_uint128_uint64(accum.z, obase_value, obase_ratio);
        out2 = barrett_reduce_uint128_uint64(accum.w, obase_value, obase_ratio);
        st_two_uint64(dst + padded_out_prime_idx * n + degree_idx + 2, out, out2);
    }
}

__global__ static void modup_copy_partQl_kernel(uint64_t *t_mod_up, const uint64_t *cks, size_t size_Ql_n,
                                                size_t size_QlP_n, size_t size_alpha_n) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size_Ql_n; tid += blockDim.x * gridDim.x) {
        const size_t beta_idx = tid / size_alpha_n;
        t_mod_up[beta_idx * size_QlP_n + tid] = cks[tid];
    }
}

void DRNSTool::modup(uint64_t *dst, const uint64_t *cks, const DNTTTable &ntt_tables, const scheme_type &scheme,
                     const cudaStream_t &stream) const {
    size_t n = n_;
    size_t size_Ql = base_Ql_.size();
    size_t size_P = size_P_;
    size_t size_QlP = size_Ql + size_P_;
    size_t size_QP = size_QP_;

    auto size_Ql_n = size_Ql * n;
    auto size_QlP_n = size_QlP * n;

    size_t alpha = size_P;
    size_t beta = v_base_part_Ql_to_compl_part_QlP_conv_.size();

    auto t_cks = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);

    // cks is in NTT domain, t_cks is in normal domain
    if (alpha == 1) {
        // In CKKS and BGV t_target is in NTT form; switch back to normal form
        if (scheme == scheme_type::ckks || scheme == scheme_type::bgv) {
            // no need to multiply qiHatInv_mod_qi
            nwt_2d_radix8_backward(t_cks.get(), cks, ntt_tables, size_Ql, 0, stream);
            // copy partQl to t_mod_up is fused with modup_bconv_single_p_kernel
        }
    } else {
        // In CKKS and BGV t_target is in NTT form; switch back to normal form
        if (scheme == scheme_type::ckks || scheme == scheme_type::bgv) {
            // fuse with base converter kernel 1 (multiply qiHatInv_mod_qi)
            nwt_2d_radix8_backward_scale(t_cks.get(), cks, ntt_tables, size_Ql, 0, partQlHatInv_mod_Ql_concat_.get(),
                                         partQlHatInv_mod_Ql_concat_shoup_.get(), stream);
        }

        // copy partQl to t_mod_up
        modup_copy_partQl_kernel<<<n * size_Ql / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                dst, cks, size_Ql_n, size_QlP_n, alpha * n);
    }

    for (size_t beta_idx = 0; beta_idx < beta; beta_idx++) {
        const size_t startPartIdx = alpha * beta_idx;
        const size_t size_PartQl = (beta_idx == beta - 1) ? (size_Ql - alpha * (beta - 1)) : alpha;
        const size_t endPartIdx = startPartIdx + size_PartQl;

        const uint64_t *cks_part_i = cks + startPartIdx * n;
        uint64_t *t_cks_part_i = t_cks.get() + startPartIdx * n;
        uint64_t *t_modup_part_i = dst + beta_idx * size_QlP_n;

        if (alpha == 1) {
            uint64_t gridDimGlb = n * size_QlP / blockDimGlb.x;

            if (scheme == scheme_type::ckks || scheme == scheme_type::bgv) {
                modup_bconv_single_p_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        t_modup_part_i, cks_part_i, t_cks.get() + startPartIdx * n,
                        startPartIdx, n, base_QlP_.base(), size_QlP);
            } else if (scheme == scheme_type::bfv) {
                modup_bconv_single_p_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        t_modup_part_i, cks_part_i, cks_part_i,
                        startPartIdx, n, base_QlP_.base(), size_QlP);
            } else {
                throw invalid_argument("unsupported scheme");
            }
        } else {
            auto &base_part_Ql_to_compl_part_QlP_conv = v_base_part_Ql_to_compl_part_QlP_conv_[beta_idx];

            auto &ibase = base_part_Ql_to_compl_part_QlP_conv.ibase();
            auto &obase = base_part_Ql_to_compl_part_QlP_conv.obase();
            const auto qiHat_mod_pj = base_part_Ql_to_compl_part_QlP_conv.QHatModp();
            const size_t ibase_size = ibase.size();
            const size_t obase_size = obase.size();
            uint64_t gridDimGlb;

            // bfv need to scale while ckks and bgv already scaled
            if (scheme == scheme_type::bfv) {
                gridDimGlb = n * ibase_size / blockDimGlb.x;
                bconv_mult_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        t_cks_part_i, cks_part_i, ibase.QHatInvModq(),
                        ibase.QHatInvModq_shoup(), ibase.base(), ibase_size, n);
            }

            constexpr int unroll_factor = 2;
            gridDimGlb = n * obase_size / blockDimGlb.x / unroll_factor;
            bconv_matmul_padded_unroll2_kernel<<<
            gridDimGlb, blockDimGlb, sizeof(uint64_t) * obase_size * ibase_size, stream>>>(
                    t_modup_part_i, t_cks_part_i, qiHat_mod_pj, ibase.base(), ibase_size, obase.base(), obase_size, n,
                    startPartIdx, size_PartQl);
        }

        if (scheme == scheme_type::ckks || scheme == scheme_type::bgv) {
            // some part of t_mod_up_i is already in NTT domain, no need to perform NTT
            nwt_2d_radix8_forward_inplace_include_special_mod_exclude_range(
                    t_modup_part_i, ntt_tables, size_QlP, 0,
                    size_QP, size_P, startPartIdx, endPartIdx, stream);
        } else if (scheme == scheme_type::bfv) {
            nwt_2d_radix8_forward_inplace_include_special_mod(
                    t_modup_part_i, ntt_tables, size_QlP, 0, size_QP, size_P, stream);
        } else {
            throw invalid_argument("unsupported scheme");
        }
    }
}

// delta = [Cp + [-Cp * pInv]_t * p]_qi
// ci' = [(ci - delta) * pInv]_qi
/*
 * ct: output ciphertext in base Ql
 * cx: input ciphertext in base QlP, also use as temporary storage
 */
__global__ static void bgv_moddown_kernel(uint64_t *dst, const uint64_t *cx, const uint64_t *delta,
                                          const uint64_t *cp_mod_t, const uint64_t *P_mod_qi,
                                          const uint64_t *P_mod_qi_shoup, const uint64_t *PInv_mod_qi,
                                          const uint64_t *PInv_mod_qi_shoup, const uint64_t PInv_mod_t,
                                          const uint64_t PInv_mod_t_shoup, const DModulus *base_Ql, size_t size_Ql,
                                          uint64_t t, uint64_t n) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n * size_Ql; tid += blockDim.x * gridDim.x) {
        size_t i = tid / n;
        auto qi = base_Ql[i].value();

        uint64_t temp = multiply_and_reduce_shoup(cp_mod_t[tid % n], PInv_mod_t, PInv_mod_t_shoup, t);
        uint64_t correction = multiply_and_reduce_shoup(temp, P_mod_qi[i], P_mod_qi_shoup[i], qi);
        temp = sub_uint64_uint64_mod(cx[tid], delta[tid], qi);
        temp = add_uint64_uint64_mod(temp, correction, qi);
        dst[tid] = multiply_and_reduce_shoup(temp, PInv_mod_qi[i], PInv_mod_qi_shoup[i], qi);
    }
}

// __global__ static void bgv_moddown_step0_kernel(uint64_t *correction, const uint64_t *cp_mod_t,
//                                                 const uint64_t *P_mod_qi, const uint64_t *P_mod_qi_shoup,
//                                                 const uint64_t PInv_mod_t, const uint64_t PInv_mod_t_shoup,
//                                                 const DModulus *base_Ql, size_t size_Ql, uint64_t t, uint64_t n) {
//     for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n * size_Ql; tid += blockDim.x * gridDim.x) {
//         size_t i = tid / n;
//         auto qi = base_Ql[i].value();
//
//         uint64_t temp = multiply_and_reduce_shoup(cp_mod_t[tid % n], PInv_mod_t, PInv_mod_t_shoup, t);
//         correction[tid] = multiply_and_reduce_shoup(temp, P_mod_qi[i], P_mod_qi_shoup[i], qi);
//     }
// }

// __global__ static void bgv_moddown_step1_kernel(uint64_t *dst, const uint64_t *cx, const uint64_t *delta,
//                                                 const uint64_t *correction, const DModulus *modulus,
//                                                 const uint64_t *PInv_mod_qi, const uint64_t *PInv_mod_qi_shoup,
//                                                 size_t n, size_t size_Ql) {
//     for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n * size_Ql; tid += blockDim.x * gridDim.x) {
//         size_t i = tid / n;
//         uint64_t mod = modulus[i].value();
//         uint64_t temp = sub_uint64_uint64_mod(cx[tid], delta[tid], mod);
//         temp = add_uint64_uint64_mod(temp, correction[tid], mod);
//         dst[tid] = multiply_and_reduce_shoup(temp, PInv_mod_qi[i], PInv_mod_qi_shoup[i], mod);
//     }
// }

__global__ static void moddown_kernel(uint64_t *dst, const uint64_t *cx, const uint64_t *delta, const DModulus *modulus,
                                      const uint64_t *PInv_mod_qi, const uint64_t *PInv_mod_qi_shoup, size_t n,
                                      size_t size_Ql) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n * size_Ql; tid += blockDim.x * gridDim.x) {
        size_t i = tid / n;
        uint64_t mod = modulus[i].value();
        uint64_t temp = sub_uint64_uint64_mod(cx[tid], delta[tid], mod);
        dst[tid] = multiply_and_reduce_shoup(temp, PInv_mod_qi[i], PInv_mod_qi_shoup[i], mod);
    }
}

__global__ static void moddown_bconv_single_p_kernel(uint64_t *dst, const uint64_t *src, size_t n,
                                                     const DModulus *base_QlP, uint64_t size_QlP) {
    const size_t size_Ql = size_QlP - 1;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t out_prime_idx = tid / n;
    const size_t coeff_idx = tid % n;
    const uint64_t in_prime = base_QlP[size_Ql].value(); // special prime
    const uint64_t out_prime = base_QlP[out_prime_idx].value();
    const uint64_t barret_ratio = base_QlP[out_prime_idx].const_ratio()[1];
    const uint64_t coeff = src[coeff_idx];
    uint64_t result;
    if (in_prime > out_prime)
        result = barrett_reduce_uint64_uint64(coeff, out_prime, barret_ratio);
    else
        result = coeff;
    dst[tid] = result;
}

/*
 * input: CKKS and BGV in NTT domain, BFV in normal domain
 */
void DRNSTool::moddown(uint64_t *ct_i, uint64_t *cx_i, const DNTTTable &ntt_tables, const scheme_type &scheme,
                       const cudaStream_t &stream) const {
    size_t n = n_;
    size_t size_Ql = base_Ql_.size();
    size_t size_QlP = size_Ql + size_P_;
    size_t size_Ql_n = size_Ql * n;

    auto delta = make_cuda_auto_ptr<uint64_t>(size_QlP * n, stream);

    if (scheme == scheme_type::ckks) {
        // Transform cx_i[P] to normal domain
        nwt_2d_radix8_backward_inplace_include_special_mod(
                cx_i, ntt_tables, size_P_, size_Ql, size_QP_, size_P_, stream);
    } else if (scheme == scheme_type::bgv) {
        // Transform cx_i[QlP] to normal domain
        nwt_2d_radix8_backward_inplace_include_special_mod(
                cx_i, ntt_tables, size_QlP, 0, size_QP_, size_P_, stream);
    }

    if (scheme == scheme_type::bgv) {
        base_P_to_Ql_conv_.bConv_BEHZ(delta.get(), cx_i + size_Ql_n, n, stream);

        auto temp_t = make_cuda_auto_ptr<uint64_t>(n, stream);

        base_P_to_t_conv_.bConv_BEHZ(temp_t.get(), cx_i + size_Ql_n, n, stream);

        bgv_moddown_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                ct_i, cx_i, delta.get(), temp_t.get(), bigP_mod_q(), bigP_mod_q_shoup(), bigPInv_mod_q(),
                bigPInv_mod_q_shoup(), bigPInv_mod_t_, bigPInv_mod_t_shoup_, ntt_tables.modulus(), size_Ql, t_.value(),
                n);

        nwt_2d_radix8_forward_inplace(ct_i, ntt_tables, size_Ql, 0, stream);
    } else if (scheme == scheme_type::bfv || scheme == scheme_type::ckks) {
        // BFV and CKKS
        base_P_to_Ql_conv_.bConv_BEHZ(delta.get(), cx_i + size_Ql_n, n, stream);

        if (scheme == scheme_type::ckks) {
            // CKKS can compute the last step in NTT domain
            nwt_2d_radix8_forward_inplace(delta.get(), ntt_tables, size_Ql, 0, stream);
        }

        // CKKS and BGV in NTT domain while BFV in normal domain
        // cx_i[k] = qk^(-1) * (cx_i[k] - (cx_i[last] mod qk)) mod qk
        moddown_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                ct_i, cx_i, delta.get(), ntt_tables.modulus(),
                bigPInv_mod_q(), bigPInv_mod_q_shoup(), n, size_Ql);
    } else {
        throw invalid_argument("unsupported scheme");
    }
}

__global__ void add_to_ct_kernel(uint64_t *ct, const uint64_t *cx, const DModulus *modulus, size_t n, size_t size_Ql) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n * size_Ql; tid += blockDim.x * gridDim.x) {
        size_t twr = tid / n;
        DModulus mod = modulus[twr];
        ct[tid] = add_uint64_uint64_mod(ct[tid], cx[tid], mod.value());
    }
}

/*
 * used in key switching
 * input: in NTT domain
 * output: CKKS and BGV in NTT domain, BFV in normal domain
 */
void DRNSTool::moddown_from_NTT(uint64_t *ct_i, uint64_t *cx_i, const DNTTTable &ntt_tables,
                                const scheme_type &scheme, const cudaStream_t &stream) const {
    size_t n = n_;
    size_t size_Ql = base_Ql_.size();
    size_t size_QlP = size_Ql + size_P_;
    size_t alpha = size_P_;
    size_t size_Ql_n = size_Ql * n;

    auto delta = make_cuda_auto_ptr<uint64_t>(size_Ql_n, stream);

    if (scheme == scheme_type::ckks) {
        // Transform cx_i[P] to normal domain
        nwt_2d_radix8_backward_inplace_include_special_mod(
                cx_i, ntt_tables, size_P_, size_Ql, size_QP_, size_P_, stream);
    } else if (scheme == scheme_type::bgv || scheme == scheme_type::bfv) {
        // Transform cx_i[QlP] to normal domain
        nwt_2d_radix8_backward_inplace_include_special_mod(
                cx_i, ntt_tables, size_QlP, 0, size_QP_, size_P_, stream);
    }

    if (alpha == 1) {
        uint64_t gridDimGlb = n * size_Ql / blockDimGlb.x;
        moddown_bconv_single_p_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                delta.get(), cx_i + size_Ql_n, n, base_QlP_.base(), size_QlP);
    } else {
        base_P_to_Ql_conv_.bConv_BEHZ(delta.get(), cx_i + size_Ql_n, n, stream);
    }

    if (scheme == scheme_type::bgv) {
        auto temp_t = make_cuda_auto_ptr<uint64_t>(n, stream);

        base_P_to_t_conv_.bConv_BEHZ(temp_t.get(), cx_i + size_Ql_n, n, stream);

        // delta = [Cp + [-Cp * pInv]_t * p]_qi
        // ci' = [(ci - delta) * pInv]_qi
        bgv_moddown_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                ct_i, cx_i, delta.get(), temp_t.get(), bigP_mod_q(), bigP_mod_q_shoup(), bigPInv_mod_q(),
                bigPInv_mod_q_shoup(), bigPInv_mod_t_, bigPInv_mod_t_shoup_, ntt_tables.modulus(), size_Ql, t_.value(),
                n);

        nwt_2d_radix8_forward_inplace(ct_i, ntt_tables, size_Ql, 0, stream);
    } else if (scheme == scheme_type::ckks) {
        // CKKS can compute the last step in NTT domain
        // ct_i += (cxi - delta) * factor mod qi
        nwt_2d_radix8_forward_inplace_fuse_moddown(ct_i, cx_i, bigPInv_mod_q_.get(), bigPInv_mod_q_shoup_.get(),
                                                   delta.get(), ntt_tables, size_Ql, 0, stream);
    } else if (scheme == scheme_type::bfv) {
        // ct_i += (cxi - delta) * factor mod qi
        moddown_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                ct_i, cx_i, delta.get(), ntt_tables.modulus(),
                bigPInv_mod_q(), bigPInv_mod_q_shoup(), n, size_Ql);
    }
}

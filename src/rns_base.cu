#include "ntt.cuh"
#include "polymath.cuh"
#include "rns.cuh"
#include "rns_base.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

namespace phantom::arith {

    void DRNSBase::init(const RNSBase &cpu_rns_base, const cudaStream_t &stream) {
        size_ = cpu_rns_base.size();

        base_ = phantom::util::make_cuda_auto_ptr<DModulus>(size_, stream);
        for (size_t idx = 0; idx < size_; idx++) {
            auto temp_modulus = *(cpu_rns_base.base() + idx);
            DModulus temp(temp_modulus.value(), temp_modulus.const_ratio().at(0), temp_modulus.const_ratio().at(1));
            cudaMemcpyAsync(base() + idx, &temp, sizeof(DModulus),
                            cudaMemcpyHostToDevice, stream);
        }

        big_Q_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_, stream);
        cudaMemcpyAsync(big_modulus(), cpu_rns_base.big_modulus(), size_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);

        big_qiHat_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_ * size_, stream);
        cudaMemcpyAsync(big_qiHat(), cpu_rns_base.big_qiHat(), size_ * size_ * sizeof(std::uint64_t),
                        cudaMemcpyHostToDevice, stream);

        qiHat_mod_qi_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_, stream);
        qiHat_mod_qi_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_, stream);
        cudaMemcpyAsync(qiHat_mod_qi_.get(), cpu_rns_base.qiHat_mod_qi(), size_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(qiHat_mod_qi_shoup_.get(), cpu_rns_base.qiHat_mod_qi_shoup(), size_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);

        qiHatInv_mod_qi_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_, stream);
        qiHatInv_mod_qi_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_, stream);
        cudaMemcpyAsync(qiHatInv_mod_qi_.get(), cpu_rns_base.QHatInvModq(), size_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(qiHatInv_mod_qi_shoup_.get(), cpu_rns_base.QHatInvModq_shoup(), size_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);

        qiInv_ = phantom::util::make_cuda_auto_ptr<double>(size_, stream);
        cudaMemcpyAsync(qiInv(), cpu_rns_base.inv(), size_ * sizeof(double),
                        cudaMemcpyHostToDevice, stream);
    }

    __global__ void decompose_array_uint64(uint64_t *dst, const cuDoubleComplex *src, const DModulus *modulus,
                                           const uint32_t sparse_poly_degree, const uint32_t sparse_ratio,
                                           const uint32_t coeff_mod_size) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < sparse_poly_degree * coeff_mod_size;
             tid += blockDim.x * gridDim.x) {
            size_t twr = tid / sparse_poly_degree;
            size_t coeff_id = tid % sparse_poly_degree;
            DModulus mod = modulus[twr];

            double coeffd;
            if (coeff_id < sparse_poly_degree >> 1) {
                coeffd = round(cuCreal(src[coeff_id]));
            } else {
                coeffd = round(cuCimag(src[coeff_id - (sparse_poly_degree >> 1)]));
            }
            bool is_negative = static_cast<bool>(signbit(coeffd));
            auto coeffu = static_cast<uint64_t>(fabs(coeffd));
            uint32_t index = tid * sparse_ratio;

            uint64_t temp = barrett_reduce_uint64_uint64(coeffu, mod.value(), mod.const_ratio()[1]);

            if (is_negative) {
                temp = mod.value() - temp;
            }

            dst[index] = temp;

            for (uint32_t i = 1; i < sparse_ratio; i++) {
                dst[index + i] = 0;
            }
        }
    }

    __global__ void decompose_array_uint128(uint64_t *dst, const cuDoubleComplex *src, const DModulus *modulus,
                                            const uint32_t sparse_poly_degree, const uint32_t sparse_ratio,
                                            const uint32_t coeff_mod_size) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < sparse_poly_degree * coeff_mod_size;
             tid += blockDim.x * gridDim.x) {
            size_t twr = tid / sparse_poly_degree;
            size_t coeff_id = tid % sparse_poly_degree;
            DModulus mod = modulus[twr];

            double coeffd;
            if (coeff_id < sparse_poly_degree >> 1) {
                coeffd = round(cuCreal(src[coeff_id]));
            } else {
                coeffd = round(cuCimag(src[coeff_id - (sparse_poly_degree >> 1)]));
            }
            bool is_negative = static_cast<bool>(signbit(coeffd));
            coeffd = fabs(coeffd);
            uint64_t coeffu[2] = {
                    static_cast<uint64_t>(fmod(coeffd, two_pow_64_dev)),
                    static_cast<uint64_t>(coeffd / two_pow_64_dev)
            };
            uint32_t index = tid * sparse_ratio;

            uint64_t temp = barrett_reduce_uint128_uint64({coeffu[1], coeffu[0]}, mod.value(), mod.const_ratio());

            if (is_negative) {
                temp = mod.value() - temp;
            }

            dst[index] = temp;

            for (uint32_t i = 1; i < sparse_ratio; i++) {
                dst[index + i] = 0;
            }
        }
    }

    __global__ void decompose_array_uint_slow_first_part(uint64_t *dst, const cuDoubleComplex *src,
                                                         const uint32_t sparse_poly_degree,
                                                         const uint32_t coeff_mod_size) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < sparse_poly_degree; tid += blockDim.x * gridDim.x) {
            double coeffd;
            if (tid < sparse_poly_degree >> 1)
                coeffd = round(cuCreal(src[tid]));
            else
                coeffd = round(cuCimag(src[tid - (sparse_poly_degree >> 1)]));

            size_t coeff_id = tid * (coeff_mod_size + 1);
            dst[coeff_id + coeff_mod_size] = static_cast<bool>(signbit(coeffd));
            coeffd = fabs(coeffd);
            for (uint32_t i = 0; i < coeff_mod_size; i++) {
                if (coeffd >= 1) {
                    dst[coeff_id + i] = static_cast<uint64_t>(fmod(coeffd, two_pow_64_dev));
                    coeffd /= two_pow_64_dev;
                } else {
                    dst[coeff_id + i] = 0;
                }
            }
        }
    }

    __global__ void decompose_array_uint_slow_second_part(uint64_t *dst, const uint64_t *src, const DModulus *modulus,
                                                          const uint32_t sparse_poly_degree,
                                                          const uint32_t sparse_ratio,
                                                          const uint32_t coeff_mod_size) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < sparse_poly_degree * coeff_mod_size;
             tid += blockDim.x * gridDim.x) {
            size_t twr = tid / sparse_poly_degree;
            size_t coeff_id = (tid % sparse_poly_degree) * (coeff_mod_size + 1);
            DModulus mod = modulus[twr];

            uint128_t temp = {src[coeff_id + coeff_mod_size - 1], 0};
            for (uint32_t i = coeff_mod_size - 1; i--;) {
                temp.lo = src[coeff_id + i];
                temp.hi = barrett_reduce_uint128_uint64(temp, mod.value(), mod.const_ratio());
            }
            // temp.hi holds the final reduction value

            // Save the result modulo i-th prime
            uint32_t index = tid * sparse_ratio;
            if (src[coeff_id + coeff_mod_size]) {
                temp.hi = mod.value() - temp.hi;
            }

            dst[index] = temp.hi;

            for (uint32_t i = 1; i < sparse_ratio; i++) {
                dst[index + i] = 0;
            }
        }
    }

    void DRNSBase::decompose_array(uint64_t *dst, const cuDoubleComplex *src, const uint32_t sparse_poly_degree,
                                   const uint32_t sparse_ratio, const uint32_t max_coeff_bit_count,
                                   const cudaStream_t &stream) const {
        uint64_t gridDimGlb = sparse_poly_degree * size() / blockDimGlb.x;
        if (max_coeff_bit_count <= 64) {
            decompose_array_uint64<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, src, base(), sparse_poly_degree, sparse_ratio,
                    size());
        } else if (max_coeff_bit_count <= 128) {
            decompose_array_uint128<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, src, base(), sparse_poly_degree, sparse_ratio,
                    size());
        } else {
            auto coeffu = make_cuda_auto_ptr<uint64_t>(sparse_poly_degree * (size() + 1), stream);
            decompose_array_uint_slow_first_part<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    coeffu.get(), src, sparse_poly_degree, size());
            decompose_array_uint_slow_second_part<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, coeffu.get(), base(), sparse_poly_degree, sparse_ratio, size());
        }
    }

    __global__ void compose_kernel(cuDoubleComplex *dst, uint64_t *temp_prod_array, uint64_t *acc_mod_array,
                                   const uint64_t *src, const uint32_t size, const DModulus *base_q,
                                   const uint64_t *base_prod, const uint64_t *punctured_prod_array,
                                   const uint64_t *inv_punctured_prod_mod_base_array,
                                   const uint64_t *inv_punctured_prod_mod_base_array_shoup,
                                   const uint64_t *upper_half_threshold, const double inv_scale,
                                   const uint32_t coeff_count,
                                   const uint32_t sparse_coeff_count, const uint32_t sparse_ratio) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < sparse_coeff_count; tid += blockDim.x * gridDim.x) {
            if (size > 1) {
                uint64_t prod;

                for (uint32_t i = 0; i < size; i++) {
                    // [a[j] * hat(q)_j^(-1)]_(q_j)
                    prod = multiply_and_reduce_shoup(src[tid * sparse_ratio + i * coeff_count],
                                                     inv_punctured_prod_mod_base_array[i],
                                                     inv_punctured_prod_mod_base_array_shoup[i], base_q[i].value());

                    // * hat(q)_j over ZZ
                    multiply_uint_uint64(punctured_prod_array + i * size, size, // operand1 and size
                                         prod, // operand2 with uint64_t
                                         temp_prod_array + tid * size); // result and size

                    // accumulation and mod Q over ZZ
                    add_uint_uint_mod(temp_prod_array + tid * size, acc_mod_array + tid * size, base_prod, size,
                                      acc_mod_array + tid * size);
                }
            } else {
                acc_mod_array[tid] = src[tid * sparse_ratio];
            }

            // Create floating-point representations of the multi-precision integer coefficients
            // Scaling instead incorporated above; this can help in cases
            // where otherwise pow(two_pow_64, j) would overflow due to very
            // large coeff_modulus_size and very large scale
            // res[i] = res_accum * inv_scale;
            double res = 0.0;
            double scaled_two_pow_64 = inv_scale;
            uint64_t diff;

            if (is_greater_than_or_equal_uint(acc_mod_array + tid * size, upper_half_threshold, size)) {
                for (uint32_t i = 0; i < size; i++, scaled_two_pow_64 *= two_pow_64_dev) {
                    if (acc_mod_array[tid * size + i] > base_prod[i]) {
                        diff = acc_mod_array[tid * size + i] - base_prod[i];
                        res += diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
                    } else {
                        diff = base_prod[i] - acc_mod_array[tid * size + i];
                        res -= diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
                    }
                }
            } else {
                for (size_t i = 0; i < size; i++, scaled_two_pow_64 *= two_pow_64_dev) {
                    diff = acc_mod_array[tid * size + i];
                    res += diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
                }
            }

            if (tid < sparse_coeff_count >> 1)
                dst[tid].x = res;
            else
                dst[tid - (sparse_coeff_count >> 1)].y = res;
        }
    }

    void DRNSBase::compose_array(cuDoubleComplex *dst, const uint64_t *src, const uint64_t *upper_half_threshold,
                                 const double inv_scale, const uint32_t coeff_count, const uint32_t sparse_coeff_count,
                                 const uint32_t sparse_ratio, const cudaStream_t &stream) const {
        if (!src) {
            throw invalid_argument("input array cannot be null");
        }

        uint32_t rns_poly_uint64_count = sparse_coeff_count * size();
        auto temp_prod_array = make_cuda_auto_ptr<uint64_t>(rns_poly_uint64_count, stream);
        auto acc_mod_array = make_cuda_auto_ptr<uint64_t>(rns_poly_uint64_count, stream);
        cudaMemsetAsync(acc_mod_array.get(), 0, rns_poly_uint64_count * sizeof(uint64_t), stream);

        uint64_t gridDimGlb = ceil(sparse_coeff_count / blockDimGlb.x);

        compose_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                dst, temp_prod_array.get(), acc_mod_array.get(), src, size(), base(),
                big_modulus(), big_qiHat(), QHatInvModq(), QHatInvModq_shoup(),
                upper_half_threshold, inv_scale, coeff_count, sparse_coeff_count, sparse_ratio);
    }
}

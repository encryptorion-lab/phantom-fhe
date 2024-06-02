#include "ntt.cuh"
#include "polymath.cuh"
#include "rns.cuh"
#include "rns_base.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

void DRNSBase::init(const RNSBase &cpu_rns_base, const std::shared_ptr<cuda_stream_wrapper> &stream_wrapper) {
    size_ = cpu_rns_base.size();

//    base_.acquire(allocate<DModulus>(global_pool(), size_));
    base_ = phantom::util::cuda_make_shared<DModulus>(size_, stream_wrapper);
    for (size_t idx = 0; idx < size_; idx++) {
        auto temp_modulus = *(cpu_rns_base.base() + idx);
        DModulus temp(temp_modulus.value(), temp_modulus.const_ratio().at(0), temp_modulus.const_ratio().at(1));
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(base() + idx, &temp, sizeof(DModulus),
                                           cudaMemcpyHostToDevice, stream_wrapper->get_stream()));
        cudaStreamSynchronize(stream_wrapper->get_stream());
    }

//    big_Q_.acquire(allocate<uint64_t>(global_pool(), size_));
    big_Q_ = phantom::util::cuda_make_shared<uint64_t>(size_, stream_wrapper);
    cudaMemcpyAsync(big_modulus(), cpu_rns_base.big_modulus(), size_ * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, stream_wrapper->get_stream());

//    big_qiHat_.acquire(allocate<uint64_t>(global_pool(), size_ * size_));
    big_qiHat_ = phantom::util::cuda_make_shared<uint64_t>(size_ * size_, stream_wrapper);
    cudaMemcpyAsync(big_qiHat(), cpu_rns_base.big_qiHat(), size_ * size_ * sizeof(std::uint64_t),
                    cudaMemcpyHostToDevice, stream_wrapper->get_stream());

//    qiHat_mod_qi_.acquire(allocate<uint64_t>(global_pool(), size_));
//    qiHat_mod_qi_shoup_.acquire(allocate<uint64_t>(global_pool(), size_));
    qiHat_mod_qi_ = phantom::util::cuda_make_shared<uint64_t>(size_, stream_wrapper);
    qiHat_mod_qi_shoup_ = phantom::util::cuda_make_shared<uint64_t>(size_, stream_wrapper);
    cudaMemcpyAsync(qiHat_mod_qi_.get(), cpu_rns_base.qiHat_mod_qi(), size_ * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, stream_wrapper->get_stream());
    cudaMemcpyAsync(qiHat_mod_qi_shoup_.get(), cpu_rns_base.qiHat_mod_qi_shoup(), size_ * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, stream_wrapper->get_stream());

//    qiHatInv_mod_qi_.acquire(allocate<uint64_t>(global_pool(), size_));
//    qiHatInv_mod_qi_shoup_.acquire(allocate<uint64_t>(global_pool(), size_));
    qiHatInv_mod_qi_ = phantom::util::cuda_make_shared<uint64_t>(size_, stream_wrapper);
    qiHatInv_mod_qi_shoup_ = phantom::util::cuda_make_shared<uint64_t>(size_, stream_wrapper);
    cudaMemcpyAsync(qiHatInv_mod_qi_.get(), cpu_rns_base.QHatInvModq(), size_ * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, stream_wrapper->get_stream());
    cudaMemcpyAsync(qiHatInv_mod_qi_shoup_.get(), cpu_rns_base.QHatInvModq_shoup(), size_ * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, stream_wrapper->get_stream());

//    qiInv_.acquire(allocate<double>(global_pool(), size_));
    qiInv_ = phantom::util::cuda_make_shared<double>(size_, stream_wrapper);
    cudaMemcpyAsync(qiInv(), cpu_rns_base.inv(), size_ * sizeof(double),
                    cudaMemcpyHostToDevice, stream_wrapper->get_stream());
    cudaStreamSynchronize(stream_wrapper->get_stream());
}

template<typename T>
// T = double or uint64_t
__global__ void decompose_uint64(std::uint64_t *dst, const T coefft, const bool is_negative, const DModulus *modulus,
                                 const uint32_t poly_degree, const uint32_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        auto coeffu = static_cast<uint64_t>(coefft);
        uint64_t temp = barrett_reduce_uint64_uint64(coeffu, mod.value(), mod.const_ratio()[1]);

        if (is_negative) {
            temp = mod.value() - temp;
        }

        dst[tid] = temp;
    }
}

__global__ void decompose_uint128(std::uint64_t *dst, const double coeffd, const bool is_negative,
                                  const DModulus *modulus, const uint32_t poly_degree, const uint32_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        uint64_t coeffu[2] = {
                static_cast<uint64_t>(fmod(coeffd, two_pow_64_dev)),
                static_cast<uint64_t>(coeffd / two_pow_64_dev)
        };

        uint64_t temp = barrett_reduce_uint128_uint64({coeffu[1], coeffu[0]}, mod.value(), mod.const_ratio());
        if (is_negative) {
            temp = mod.value() - temp;
        }
        dst[tid] = temp;
    }
}

__global__ void decompose_uint(uint64_t *dst, const uint64_t *coeffu, const bool is_negative, const DModulus *modulus,
                               const uint32_t poly_degree, const uint32_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = modulus[twr];

        uint128_t temp = {coeffu[coeff_mod_size - 1], 0};
        for (uint32_t i = coeff_mod_size - 1; i--;) {
            temp.lo = coeffu[i];
            temp.hi = barrett_reduce_uint128_uint64(temp, mod.value(), mod.const_ratio());
        }
        // temp.hi holds the final reduction value

        // Save the result modulo i-th prime
        if (is_negative) {
            temp.hi = mod.value() - temp.hi;
        }

        dst[tid] = temp.hi;
    }
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
                                                     const uint32_t sparse_poly_degree, const uint32_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < sparse_poly_degree; tid += blockDim.x * gridDim.x) {
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
                                                      const uint32_t sparse_poly_degree, const uint32_t sparse_ratio,
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

void DRNSBase::decompose(uint64_t *dst, const double value, const uint32_t poly_degree,
                         const uint32_t coeff_bit_count) const {
    double coeffd = round(value);
    bool is_negative = signbit(coeffd);
    coeffd = fabs(coeffd);
    uint64_t gridDimGlb = poly_degree * size() / blockDimGlb.x;

    if (coeff_bit_count <= 64) {
        decompose_uint64<<<gridDimGlb, blockDimGlb>>>(dst, coeffd, is_negative, base(), poly_degree, size());
    } else if (coeff_bit_count <= 128) {
        decompose_uint128<<<gridDimGlb, blockDimGlb>>>(dst, coeffd, is_negative, base(), poly_degree, size());
    } else {
        Pointer<uint64_t> coeffu;
        coeffu.acquire(allocate<uint64_t>(global_pool(), size()));
        for (uint32_t i = 0; i < size(); i++) {
            if (coeffd >= 1) {
                coeffu.get()[i] = static_cast<uint64_t>(fmod(coeffd, two_pow_64_dev));
                coeffd /= two_pow_64_dev;
            } else {
                coeffu.get()[i] = 0;
            }
        }

        decompose_uint<<<gridDimGlb, blockDimGlb>>>(dst, coeffu.get(), is_negative, base(), poly_degree, size());
    }
}

void DRNSBase::decompose(uint64_t *dst, const int64_t value, const uint32_t poly_degree,
                         const uint32_t coeff_bit_count) const {
    bool is_negative = signbit(value);
    uint64_t coeffu = fabs(value);
    uint64_t gridDimGlb = poly_degree * size() / blockDimGlb.x;
    decompose_uint64<<<gridDimGlb, blockDimGlb>>>(dst, coeffu, is_negative, base(), poly_degree, size());
}

void DRNSBase::decompose_array(uint64_t *dst, const cuDoubleComplex *src, const uint32_t sparse_poly_degree,
                               const uint32_t sparse_ratio, const uint32_t max_coeff_bit_count) const {
    uint64_t gridDimGlb = sparse_poly_degree * size() / blockDimGlb.x;
    if (max_coeff_bit_count <= 64) {
        decompose_array_uint64<<<gridDimGlb, blockDimGlb>>>(dst, src, base(), sparse_poly_degree, sparse_ratio, size());
    } else if (max_coeff_bit_count <= 128) {
        decompose_array_uint128<<<gridDimGlb, blockDimGlb>>>(dst, src, base(), sparse_poly_degree, sparse_ratio,
                                                             size());
    } else {
        Pointer<uint64_t> coeffu;
        coeffu.acquire(allocate<uint64_t>(global_pool(), sparse_poly_degree * (size() + 1)));
        decompose_array_uint_slow_first_part<<<gridDimGlb, blockDimGlb>>>(coeffu.get(), src, sparse_poly_degree,
                                                                          size());
        decompose_array_uint_slow_second_part<<<gridDimGlb, blockDimGlb>>>(dst, coeffu.get(), base(),
                                                                           sparse_poly_degree, sparse_ratio, size());
    }
}

__global__ void compose_kernel(cuDoubleComplex *dst, uint64_t *temp_prod_array, uint64_t *acc_mod_array,
                               const uint64_t *src, const uint32_t size, const DModulus *base_q,
                               const uint64_t *base_prod, const uint64_t *punctured_prod_array,
                               const uint64_t *inv_punctured_prod_mod_base_array,
                               const uint64_t *inv_punctured_prod_mod_base_array_shoup,
                               const uint64_t *upper_half_threshold, const double inv_scale, const uint32_t coeff_count,
                               const uint32_t sparse_coeff_count, const uint32_t sparse_ratio) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < sparse_coeff_count; tid += blockDim.x * gridDim.x) {
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
                             const uint32_t sparse_ratio) const {
    if (!src) {
        throw invalid_argument("input array cannot be null");
    }

    Pointer<uint64_t> temp_prod_array, acc_mod_array;
    uint32_t rns_poly_uint64_count = sparse_coeff_count * size();
    temp_prod_array.acquire(allocate<uint64_t>(global_pool(), rns_poly_uint64_count));
    acc_mod_array.acquire(allocate<uint64_t>(global_pool(), rns_poly_uint64_count));
    cudaMemset(acc_mod_array.get(), 0, rns_poly_uint64_count * sizeof(uint64_t));

    uint64_t gridDimGlb = ceil(sparse_coeff_count / blockDimGlb.x);

    compose_kernel<<<gridDimGlb, blockDimGlb>>>(dst, temp_prod_array.get(), acc_mod_array.get(), src, size(), base(),
                                                big_modulus(), big_qiHat(), QHatInvModq(), QHatInvModq_shoup(),
                                                upper_half_threshold, inv_scale, coeff_count, sparse_coeff_count,
                                                sparse_ratio);
}

/** multi-integer mod provided modulus
 @param[in]: value multi integer
 @param[in]: size, the number of uint64_t in the value array
 @param[in]: mod, the modulus
 @ret: result
 */
__forceinline__ __device__ uint64_t decompose_uint(const uint64_t *value, const size_t coeff_mod_size,
                                                   const DModulus modulus) {
    if (coeff_mod_size == 1) {
        // pi < t, which is impossible
        return barrett_reduce_uint64_uint64(*value, modulus.value(), modulus.const_ratio()[1]);
    } else if (coeff_mod_size == 2) {
        return barrett_reduce_uint128_uint64({value[1], value[0]}, modulus.value(), modulus.const_ratio());
    }

    uint128_t temp;
    temp.lo = 0;
    temp.hi = value[coeff_mod_size - 1];
    for (size_t i = coeff_mod_size - 1; i--;) {
        temp.lo = value[i];
        temp.hi = barrett_reduce_uint128_uint64(temp, modulus.value(), modulus.const_ratio());
    }
    return temp.hi;
}

__global__ void decompose_array_uint_kernel(uint64_t *dst, uint64_t *temp_data, const uint64_t *plain_data,
                                            const DModulus *modulus, const size_t poly_degree,
                                            const size_t coeff_mod_size, const uint64_t *plain_upper_half_increment,
                                            const uint64_t plain_upper_half_threshold) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree; tid += blockDim.x * gridDim.x) {
        uint64_t pt = plain_data[tid];

        if (pt >= plain_upper_half_threshold) {
            add_uint_uint64(plain_upper_half_increment, coeff_mod_size, pt, temp_data + tid * coeff_mod_size);
        } else {
            temp_data[tid * coeff_mod_size] = pt;
        }
        // TODO: optimize?
        for (size_t twr = 0; twr < coeff_mod_size; twr++) {
            dst[twr * poly_degree + tid] =
                    decompose_uint(temp_data + tid * coeff_mod_size, coeff_mod_size, modulus[twr]);
        }
    }
}

void DRNSBase::decompose_array(uint64_t *dst, const uint64_t *src, const DModulus *modulus, const size_t poly_degree,
                               const uint64_t *plain_upper_half_increment,
                               const uint64_t plain_upper_half_threshold) const {
    Pointer<uint64_t> temp;
    temp.acquire(allocate<uint64_t>(global_pool(), poly_degree * size()));
    cudaMemset(temp.get(), 0, poly_degree * size() * sizeof(uint64_t));

    uint64_t gridDimGlb = poly_degree * size() / blockDimGlb.x;
    decompose_array_uint_kernel<<<gridDimGlb, blockDimGlb>>>(dst, temp.get(), src, modulus, poly_degree, size(),
                                                             plain_upper_half_increment, plain_upper_half_threshold);
}

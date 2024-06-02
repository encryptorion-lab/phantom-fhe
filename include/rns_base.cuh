#pragma once

#include "util/rns.h"
#include "cuda_wrapper.cuh"

class DRNSBase {
private:
    std::size_t size_{};

    phantom::util::cuda_shared_ptr<DModulus> base_;
    phantom::util::cuda_shared_ptr<uint64_t> big_Q_;
    phantom::util::cuda_shared_ptr<uint64_t> big_qiHat_;
    phantom::util::cuda_shared_ptr<uint64_t> qiHat_mod_qi_;
    phantom::util::cuda_shared_ptr<uint64_t> qiHat_mod_qi_shoup_;
    phantom::util::cuda_shared_ptr<uint64_t> qiHatInv_mod_qi_;
    phantom::util::cuda_shared_ptr<uint64_t> qiHatInv_mod_qi_shoup_;
    phantom::util::cuda_shared_ptr<double> qiInv_;

public:
    DRNSBase() : size_(0) {}

    explicit DRNSBase(const phantom::util::RNSBase &cpu_rns_base,
                      const std::shared_ptr<phantom::util::cuda_stream_wrapper> &stream_wrapper) {
        init(cpu_rns_base, stream_wrapper);
    }

    void init(const phantom::util::RNSBase &cpu_rns_base,
              const std::shared_ptr<phantom::util::cuda_stream_wrapper> &stream_wrapper);

    [[nodiscard]] __host__ __device__ __forceinline__ std::size_t size() const noexcept { return size_; }

    [[nodiscard]] inline auto base() const { return base_.get(); }

    [[nodiscard]] inline auto big_modulus() const { return big_Q_.get(); }

    [[nodiscard]] inline auto big_qiHat() const { return big_qiHat_.get(); }

    [[nodiscard]] inline auto QHatModq() const { return qiHat_mod_qi_.get(); }

    [[nodiscard]] inline auto QHatModq_shoup() const { return qiHat_mod_qi_shoup_.get(); }

    [[nodiscard]] inline auto QHatInvModq() const { return qiHatInv_mod_qi_.get(); }

    [[nodiscard]] inline auto QHatInvModq_shoup() const { return qiHatInv_mod_qi_shoup_.get(); }

    [[nodiscard]] inline auto qiInv() const { return qiInv_.get(); }

    __host__ void decompose(uint64_t *dst, int64_t value, uint32_t coeff_count, uint32_t coeff_bit_count) const;

    __host__ void decompose(uint64_t *dst, double value, uint32_t coeff_count, uint32_t coeff_bit_count) const;

    __host__ void decompose_array(uint64_t *dst, const cuDoubleComplex *src, uint32_t sparse_coeff_count,
                                  uint32_t sparse_ratio, uint32_t max_coeff_bit_count) const;

    __host__ void decompose_array(uint64_t *dst, const uint64_t *src, const DModulus *modulus, size_t poly_degree,
                                  const uint64_t *plain_upper_half_increment,
                                  uint64_t plain_upper_half_threshold) const;

    __host__ void compose_array(cuDoubleComplex *dst, const uint64_t *src, const uint64_t *upper_half_threshold,
                                double inv_scale, uint32_t coeff_count, uint32_t sparse_coeff_count,
                                uint32_t sparse_ratio) const;
};

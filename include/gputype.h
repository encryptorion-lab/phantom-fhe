#pragma once

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include "uintmath.cuh"
#include "cuda_wrapper.cuh"

/** Pre-computation for coeff modulus
 * value: the modulus value
 * const_ratio_: 2^128/value, in 128-bit
 */
class DModulus {

private:

    uint64_t value_ = 0;
    uint64_t const_ratio_[2] = {0, 0}; // 0 corresponding low, 1 corresponding high

public:

    DModulus() = default;

    DModulus(const uint64_t value, const uint64_t ratio0, const uint64_t ratio1) :
            value_(value), const_ratio_{ratio0, ratio1} {}

    void set(const uint64_t value, const uint64_t const_ratio0, const uint64_t const_ratio1) {
        cudaMemcpyAsync(&value_, &value, sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(&(const_ratio_[0]), &const_ratio0, sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(&(const_ratio_[1]), &const_ratio1, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // Returns a const pointer to the value of the current Modulus.
    __device__ __host__ inline const uint64_t *data() const noexcept { return &value_; }

    __device__ __host__ inline uint64_t value() const { return value_; }

    __device__ __host__ inline auto &const_ratio() const { return const_ratio_; }
};

/** The GPU information for one RNS coeff
 * SID_: the cuda stream identifier
 * n_: poly degree
 * in_: input values
 * modulus_: the coeff modulus
 * twiddle: the forward NTT table
 * itwiddle: the inverse NTT table
 */

class DNTTTable {

private:

    uint64_t n_ = 0; // vector length for this NWT
    uint64_t size_ = 0; // coeff_modulus_size
    phantom::util::cuda_auto_ptr<DModulus> modulus_; // modulus for this NWT
    phantom::util::cuda_auto_ptr<uint64_t> twiddle_; // forward NTT table
    phantom::util::cuda_auto_ptr<uint64_t> twiddle_shoup_; // forward NTT table
    phantom::util::cuda_auto_ptr<uint64_t> itwiddle_; // inverse NTT table
    phantom::util::cuda_auto_ptr<uint64_t> itwiddle_shoup_; // inverse NTT table
    phantom::util::cuda_auto_ptr<uint64_t> n_inv_mod_q_; // n^(-1) modulo q
    phantom::util::cuda_auto_ptr<uint64_t> n_inv_mod_q_shoup_; // n^(-1) modulo q, shoup version

public:

    DNTTTable() = default;

    DNTTTable(const DNTTTable &source) = delete;

    DNTTTable(DNTTTable &&source) = delete;

    DNTTTable &operator=(const DNTTTable &source) = delete;

    DNTTTable &operator=(DNTTTable &&source) = delete;

    ~DNTTTable() = default;

    [[nodiscard]] uint64_t n() const { return n_; }

    [[nodiscard]] uint64_t size() const { return size_; }

    [[nodiscard]] DModulus *modulus() const { return modulus_.get(); }

    [[nodiscard]] uint64_t *twiddle() const { return twiddle_.get(); }

    [[nodiscard]] uint64_t *twiddle_shoup() const { return twiddle_shoup_.get(); }

    [[nodiscard]] uint64_t *itwiddle() const { return itwiddle_.get(); }

    [[nodiscard]] uint64_t *itwiddle_shoup() const { return itwiddle_shoup_.get(); }

    [[nodiscard]] uint64_t *n_inv_mod_q() const { return n_inv_mod_q_.get(); }

    [[nodiscard]] uint64_t *n_inv_mod_q_shoup() const { return n_inv_mod_q_shoup_.get(); }

    DNTTTable(const uint64_t n, const uint64_t size, const DModulus *modulus, const uint64_t *twiddle,
              const uint64_t *twiddle_shoup, const uint64_t *itwiddle, const uint64_t *itwiddle_shoup,
              const uint64_t *n_inv_mod_q, const uint64_t *n_inv_mod_q_shoup,
              const cudaStream_t &stream) : n_(n), size_(size) {
        modulus_ = phantom::util::make_cuda_auto_ptr<DModulus>(size, stream);
        twiddle_ = phantom::util::make_cuda_auto_ptr<uint64_t>(n * size, stream);
        twiddle_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(n * size, stream);
        itwiddle_ = phantom::util::make_cuda_auto_ptr<uint64_t>(n * size, stream);
        itwiddle_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(n * size, stream);
        n_inv_mod_q_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size, stream);
        n_inv_mod_q_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size, stream);
        cudaMemcpyAsync(modulus_.get(), modulus, size * sizeof(DModulus), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(twiddle_.get(), twiddle, n * size * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(twiddle_shoup_.get(), twiddle_shoup, n * size * sizeof(uint64_t), cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(itwiddle_.get(), itwiddle, n * size * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(itwiddle_shoup_.get(), itwiddle_shoup, n * size * sizeof(uint64_t), cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(n_inv_mod_q_.get(), n_inv_mod_q, size * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(n_inv_mod_q_shoup_.get(), n_inv_mod_q_shoup, size * sizeof(uint64_t), cudaMemcpyHostToDevice,
                        stream);
    }

    void init(const uint64_t n, const uint64_t size, const cudaStream_t &stream) {
        n_ = n;
        size_ = size;
        modulus_ = phantom::util::make_cuda_auto_ptr<DModulus>(size, stream);
        twiddle_ = phantom::util::make_cuda_auto_ptr<uint64_t>(n * size, stream);
        twiddle_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(n * size, stream);
        itwiddle_ = phantom::util::make_cuda_auto_ptr<uint64_t>(n * size, stream);
        itwiddle_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(n * size, stream);
        n_inv_mod_q_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size, stream);
        n_inv_mod_q_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size, stream);
    }

    void set(const DModulus *modulus, const uint64_t *twiddle, const uint64_t *twiddle_shoup, const uint64_t *itwiddle,
             const uint64_t *itwiddle_shoup, const uint64_t n_inv_mod_q, const uint64_t n_inv_mod_q_shoup,
             const uint64_t index, const cudaStream_t &stream) const {
        cudaMemcpyAsync(modulus_.get() + index, modulus, sizeof(DModulus), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(twiddle_.get() + index * n_, twiddle, n_ * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(twiddle_shoup_.get() + index * n_, twiddle_shoup, n_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(itwiddle_.get() + index * n_, itwiddle, n_ * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(itwiddle_shoup_.get() + index * n_, itwiddle_shoup, n_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(n_inv_mod_q_.get() + index, &n_inv_mod_q, sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(n_inv_mod_q_shoup_.get() + index, &n_inv_mod_q_shoup, sizeof(uint64_t), cudaMemcpyHostToDevice,
                        stream);
    }
};

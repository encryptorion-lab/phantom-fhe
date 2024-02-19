#pragma once

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include "mempool.cuh"
#include "uintmath.cuh"

/** Pre-computation for coeff modulus
 * value: the modulus value
 * const_ratio_: 2^128/value, in 128-bit
 */
typedef struct DModulus {
    uint64_t value_ = 0;
    uint64_t const_ratio_[2] = {0, 0}; // 0 corresponding low, 1 corresponding high

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
} DModulus;

/** The GPU information for one RNS coeff
 * SID_: the cuda stream identifier
 * n_: poly degree
 * in_: input values
 * modulus_: the coeff modulus
 * twiddle: the forward NTT table
 * itwiddle: the inverse NTT table
 */

typedef struct DNTTTable {
    uint64_t n_; // vector length for this NWT
    uint64_t size_; // coeff_modulus_size
    phantom::util::Pointer<DModulus> modulus_; // modulus for this NWT
    phantom::util::Pointer<uint64_t> twiddle_; // forward NTT table
    phantom::util::Pointer<uint64_t> twiddle_shoup_; // forward NTT table
    phantom::util::Pointer<uint64_t> itwiddle_; // inverse NTT table
    phantom::util::Pointer<uint64_t> itwiddle_shoup_; // inverse NTT table
    phantom::util::Pointer<uint64_t> n_inv_mod_q_; // n^(-1) modulo q
    phantom::util::Pointer<uint64_t> n_inv_mod_q_shoup_; // n^(-1) modulo q, shoup version

    DNTTTable() {
        n_ = 0;
        size_ = 0;
        modulus_ = phantom::util::Pointer<DModulus>();
        twiddle_ = phantom::util::Pointer<uint64_t>();
        twiddle_shoup_ = phantom::util::Pointer<uint64_t>();
        itwiddle_ = phantom::util::Pointer<uint64_t>();
        itwiddle_shoup_ = phantom::util::Pointer<uint64_t>();
        n_inv_mod_q_ = phantom::util::Pointer<uint64_t>();
        n_inv_mod_q_shoup_ = phantom::util::Pointer<uint64_t>();
    }

    DNTTTable(const DNTTTable &source) {
        n_ = source.n_;
        size_ = source.size_;
        modulus_.acquire(phantom::util::allocate<DModulus>(phantom::util::global_pool(), size_));
        twiddle_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n_ * size_));
        twiddle_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n_ * size_));
        itwiddle_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n_ * size_));
        itwiddle_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n_ * size_));
        n_inv_mod_q_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), size_));
        n_inv_mod_q_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), size_));
        cudaMemcpyAsync(modulus_.get(), source.modulus_.get(), size_ * sizeof(DModulus), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(twiddle_.get(), source.twiddle_.get(), n_ * size_ * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(twiddle_shoup_.get(), source.twiddle_shoup_.get(), n_ * size_ * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(itwiddle_.get(), source.itwiddle_.get(), n_ * size_ * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(itwiddle_shoup_.get(), source.itwiddle_shoup_.get(), n_ * size_ * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(n_inv_mod_q_.get(), source.n_inv_mod_q_.get(), size_ * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(n_inv_mod_q_shoup_.get(), source.n_inv_mod_q_shoup_.get(), size_ * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
    }

    DNTTTable(DNTTTable &&source) noexcept {
        n_ = source.n_;
        size_ = source.size_;
        modulus_.acquire(source.modulus_);
        twiddle_.acquire(source.twiddle_);
        twiddle_shoup_.acquire(source.twiddle_shoup_);
        itwiddle_.acquire(source.itwiddle_);
        itwiddle_shoup_.acquire(source.itwiddle_shoup_);
        n_inv_mod_q_.acquire(source.n_inv_mod_q_);
        n_inv_mod_q_shoup_.acquire(source.n_inv_mod_q_shoup_);
    }

    __host__ DNTTTable &operator=(const DNTTTable &source) {
        n_ = source.n_;
        size_ = source.size_;
        modulus_.acquire(phantom::util::allocate<DModulus>(phantom::util::global_pool(), size_));
        twiddle_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n_ * size_));
        twiddle_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n_ * size_));
        itwiddle_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n_ * size_));
        itwiddle_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n_ * size_));
        n_inv_mod_q_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), size_));
        n_inv_mod_q_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), size_));
        cudaMemcpyAsync(modulus_.get(), source.modulus_.get(), size_ * sizeof(DModulus), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(twiddle_.get(), source.twiddle_.get(), n_ * size_ * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(twiddle_shoup_.get(), source.twiddle_shoup_.get(), n_ * size_ * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(itwiddle_.get(), source.itwiddle_.get(), n_ * size_ * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(itwiddle_shoup_.get(), source.itwiddle_shoup_.get(), n_ * size_ * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(n_inv_mod_q_.get(), source.n_inv_mod_q_.get(), size_ * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(n_inv_mod_q_shoup_.get(), source.n_inv_mod_q_shoup_.get(), size_ * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        return *this;
    }

    __device__ __host__ DNTTTable &operator=(DNTTTable &&source) noexcept {
        n_ = source.n_;
        size_ = source.size_;
        modulus_.acquire(source.modulus_);
        twiddle_.acquire(source.twiddle_);
        twiddle_shoup_.acquire(source.twiddle_shoup_);
        itwiddle_.acquire(source.itwiddle_);
        itwiddle_shoup_.acquire(source.itwiddle_shoup_);
        n_inv_mod_q_.acquire(source.n_inv_mod_q_);
        n_inv_mod_q_shoup_.acquire(source.n_inv_mod_q_shoup_);
        return *this;
    }

    __device__ __host__ uint64_t n() const { return n_; }

    __device__ __host__ uint64_t size() const { return size_; }

    __device__ __host__ DModulus *modulus() const { return modulus_.get(); }

    __device__ __host__ uint64_t *twiddle() const { return twiddle_.get(); }

    __device__ __host__ uint64_t *twiddle_shoup() const { return twiddle_shoup_.get(); }

    __device__ __host__ uint64_t *itwiddle() const { return itwiddle_.get(); }

    __device__ __host__ uint64_t *itwiddle_shoup() const { return itwiddle_shoup_.get(); }

    uint64_t *n_inv_mod_q() const { return n_inv_mod_q_.get(); }

    uint64_t *n_inv_mod_q_shoup() const { return n_inv_mod_q_shoup_.get(); }

    DNTTTable(const uint64_t n, const uint64_t size, const DModulus *modulus, const uint64_t *twiddle,
              const uint64_t *twiddle_shoup, const uint64_t *itwiddle, const uint64_t *itwiddle_shoup,
              const uint64_t *n_inv_mod_q, const uint64_t *n_inv_mod_q_shoup) : n_(n), size_(size) {
        modulus_.acquire(phantom::util::allocate<DModulus>(phantom::util::global_pool(), size));
        twiddle_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n * size));
        twiddle_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n * size));
        itwiddle_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n * size));
        itwiddle_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n * size));
        n_inv_mod_q_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), size));
        n_inv_mod_q_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), size));
        cudaMemcpyAsync(modulus_.get(), modulus, size * sizeof(DModulus), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(twiddle_.get(), twiddle, n * size * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(twiddle_shoup_.get(), twiddle_shoup, n * size * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(itwiddle_.get(), itwiddle, n * size * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(itwiddle_shoup_.get(), itwiddle_shoup, n * size * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(n_inv_mod_q_.get(), n_inv_mod_q, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(n_inv_mod_q_shoup_.get(), n_inv_mod_q_shoup, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    void init(const uint64_t n, const uint64_t size) {
        n_ = n;
        size_ = size;
        modulus_.acquire(phantom::util::allocate<DModulus>(phantom::util::global_pool(), size));
        twiddle_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n * size));
        twiddle_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n * size));
        itwiddle_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n * size));
        itwiddle_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), n * size));
        n_inv_mod_q_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), size));
        n_inv_mod_q_shoup_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), size));
    }

    void set(const DModulus *modulus, const uint64_t *twiddle, const uint64_t *twiddle_shoup, const uint64_t *itwiddle,
             const uint64_t *itwiddle_shoup, const uint64_t n_inv_mod_q, const uint64_t n_inv_mod_q_shoup,
             const uint64_t index) const {
        cudaMemcpyAsync(modulus_.get() + index, modulus, sizeof(DModulus), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(twiddle_.get() + index * n_, twiddle, n_ * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(twiddle_shoup_.get() + index * n_, twiddle_shoup, n_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice);
        cudaMemcpyAsync(itwiddle_.get() + index * n_, itwiddle, n_ * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(itwiddle_shoup_.get() + index * n_, itwiddle_shoup, n_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice);
        cudaMemcpyAsync(n_inv_mod_q_.get() + index, &n_inv_mod_q, sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(n_inv_mod_q_shoup_.get() + index, &n_inv_mod_q_shoup, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    ~DNTTTable() = default;
} DNTTTable;

typedef struct DCKKSEncoderInfo {
    cudaStream_t SID_;
    uint32_t m_; // order of the multiplicative group
    uint32_t sparse_slots_ = 0;
    phantom::util::Pointer<cuDoubleComplex> in_; // input buffer, length must be n
    phantom::util::Pointer<cuDoubleComplex> twiddle_; // forward FFT table
    phantom::util::Pointer<uint32_t> mul_group_;

    DCKKSEncoderInfo() = default;

    DCKKSEncoderInfo &operator=(DCKKSEncoderInfo &&source) noexcept {
        SID_ = source.SID_;
        m_ = source.m_;
        sparse_slots_ = source.sparse_slots_;
        in_.acquire(source.in_);
        twiddle_.acquire(source.twiddle_);
        mul_group_.acquire(source.mul_group_);
        return *this;
    }

    explicit DCKKSEncoderInfo(const size_t coeff_count) {
        m_ = coeff_count << 1;
        const uint32_t slots = coeff_count >> 1; // n/2
        const uint32_t slots_half = slots >> 1;

        CUDA_CHECK(cudaStreamCreate(&SID_));
        in_.acquire(phantom::util::allocate<cuDoubleComplex>(phantom::util::global_pool(), slots));
        twiddle_.acquire(phantom::util::allocate<cuDoubleComplex>(phantom::util::global_pool(), m_));
        mul_group_.acquire(phantom::util::allocate<uint32_t>(phantom::util::global_pool(), slots_half));
    }

    __device__ __host__ cudaStream_t &SID() { return SID_; }

    __device__ __host__ uint32_t m() const { return m_; }

    __device__ __host__ uint32_t sparse_slots() const { return sparse_slots_; }

    __device__ __host__ cuDoubleComplex *in() { return in_.get(); }

    __device__ __host__ cuDoubleComplex *twiddle() { return twiddle_.get(); }

    __device__ __host__ uint32_t *mul_group() { return mul_group_.get(); }

    __device__ __host__ cuDoubleComplex *in() const { return in_.get(); }

    __device__ __host__ cuDoubleComplex *twiddle() const { return twiddle_.get(); }

    __device__ __host__ uint32_t *mul_group() const { return mul_group_.get(); }

    __device__ __host__ void set_sparse_slots(const uint32_t sparse_slots) { sparse_slots_ = sparse_slots; }
} DCKKSEncoderInfo;

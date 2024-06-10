#pragma once

#include "context.cuh"

namespace phantom::util {
    class ComplexRoots {
    private:
        static constexpr double PI_ = 3.1415926535897932384626433832795028842;

        // Contains 0~(n/8-1)-th powers of the n-th primitive root.
        cuDoubleComplex *roots_;

        std::size_t degree_of_roots_;
    public:
        explicit ComplexRoots(std::size_t degree_of_roots);

        [[nodiscard]] cuDoubleComplex get_root(std::size_t index) const;
    };

    inline cuDoubleComplex polar(const double &__rho, const double &__theta) {
        return make_cuDoubleComplex(__rho * cos(__theta), __rho * sin(__theta));
    }
}

class DCKKSEncoderInfo {

private:

    uint32_t m_{}; // order of the multiplicative group
    uint32_t sparse_slots_ = 0;
    phantom::util::cuda_auto_ptr<cuDoubleComplex> in_; // input buffer, length must be n
    phantom::util::cuda_auto_ptr<cuDoubleComplex> twiddle_; // forward FFT table
    phantom::util::cuda_auto_ptr<uint32_t> mul_group_;

public:

    explicit DCKKSEncoderInfo(const size_t coeff_count, const cudaStream_t &stream) {
        m_ = coeff_count << 1;
        const uint32_t slots = coeff_count >> 1; // n/2
        const uint32_t slots_half = slots >> 1;

        in_ = phantom::util::make_cuda_auto_ptr<cuDoubleComplex>(slots, stream);
        twiddle_ = phantom::util::make_cuda_auto_ptr<cuDoubleComplex>(m_, stream);
        mul_group_ = phantom::util::make_cuda_auto_ptr<uint32_t>(slots_half, stream);
    }

    DCKKSEncoderInfo(const DCKKSEncoderInfo &copy) = delete;

    DCKKSEncoderInfo(DCKKSEncoderInfo &&source) = delete;

    DCKKSEncoderInfo &operator=(const DCKKSEncoderInfo &copy) = delete;

    DCKKSEncoderInfo &operator=(DCKKSEncoderInfo &&source) = delete;

    ~DCKKSEncoderInfo() = default;

    [[nodiscard]] uint32_t m() const { return m_; }

    [[nodiscard]] uint32_t sparse_slots() const { return sparse_slots_; }

    cuDoubleComplex *in() { return in_.get(); }

    cuDoubleComplex *twiddle() { return twiddle_.get(); }

    uint32_t *mul_group() { return mul_group_.get(); }

    [[nodiscard]] cuDoubleComplex *in() const { return in_.get(); }

    [[nodiscard]] cuDoubleComplex *twiddle() const { return twiddle_.get(); }

    [[nodiscard]] uint32_t *mul_group() const { return mul_group_.get(); }

    void set_sparse_slots(const uint32_t sparse_slots) { sparse_slots_ = sparse_slots; }
};

void special_fft_forward(DCKKSEncoderInfo &gp, const cudaStream_t &stream);

void special_fft_backward(DCKKSEncoderInfo &gp, double scalar, const cudaStream_t &stream);

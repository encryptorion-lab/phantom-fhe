#pragma once

#include <cuComplex.h>

#include "context.cuh"
#include "fft.h"
#include "ntt.cuh"
#include "plaintext.h"
#include "rns.cuh"

class PhantomCKKSEncoder {

private:

    uint32_t slots_{};
    std::unique_ptr<phantom::util::ComplexRoots> complex_roots_;
    std::vector<cuDoubleComplex> root_powers_;
    std::vector<uint32_t> rotation_group_;
    std::unique_ptr<DCKKSEncoderInfo> gpu_ckks_msg_vec_;
    uint32_t first_chain_index_ = 1;

    void encode_internal(const PhantomContext &context,
                         const std::vector<cuDoubleComplex> &values,
                         size_t chain_index, double scale,
                         PhantomPlaintext &destination,
                         const cudaStream_t &stream);

    inline void encode_internal(const PhantomContext &context,
                                const std::vector<double> &values,
                                size_t chain_index, double scale,
                                PhantomPlaintext &destination,
                                const cudaStream_t &stream) {
        size_t values_size = values.size();
        std::vector<cuDoubleComplex> input(values_size);
        for (size_t i = 0; i < values_size; i++) {
            input[i] = make_cuDoubleComplex(values[i], 0.0);
        }
        encode_internal(context, input, chain_index, scale, destination, stream);
    }

    void decode_internal(const PhantomContext &context,
                         const PhantomPlaintext &plain,
                         std::vector<cuDoubleComplex> &destination,
                         const cudaStream_t &stream);

    inline void decode_internal(const PhantomContext &context,
                                const PhantomPlaintext &plain,
                                std::vector<double> &destination,
                                const cudaStream_t &stream) {
        std::vector<cuDoubleComplex> output;
        decode_internal(context, plain, output, stream);
        destination.resize(slots_);
        for (size_t i = 0; i < slots_; i++)
            destination[i] = output[i].x;
    }

public:

    explicit PhantomCKKSEncoder(const PhantomContext &context);

    PhantomCKKSEncoder(const PhantomCKKSEncoder &copy) = delete;

    PhantomCKKSEncoder(PhantomCKKSEncoder &&source) = delete;

    PhantomCKKSEncoder &operator=(const PhantomCKKSEncoder &assign) = delete;

    PhantomCKKSEncoder &operator=(PhantomCKKSEncoder &&assign) = delete;

    ~PhantomCKKSEncoder() = default;

    template<class T>
    inline void encode(const PhantomContext &context,
                       const std::vector<T> &values,
                       double scale,
                       PhantomPlaintext &destination,
                       size_t chain_index = 1) { // first chain index

        const auto &s = cudaStreamPerThread;
        destination.chain_index_ = 0;
        destination.resize(context.coeff_mod_size_, context.poly_degree_, s);
        encode_internal(context, values, chain_index, scale, destination, s);
    }

    template<class T>
    [[nodiscard]] inline auto encode(const PhantomContext &context, const std::vector<T> &values,
                                     double scale,
                                     size_t chain_index = 1) { // first chain index

        PhantomPlaintext destination;
        encode(context, values, scale, destination, chain_index);
        return destination;
    }

    template<class T>
    inline void decode(const PhantomContext &context,
                       const PhantomPlaintext &plain,
                       std::vector<T> &destination) {
        decode_internal(context, plain, destination, cudaStreamPerThread);
    }

    template<class T>
    [[nodiscard]] inline auto decode(const PhantomContext &context, const PhantomPlaintext &plain) {
        std::vector<T> destination;
        decode(context, plain, destination);
        return destination;
    }

    [[nodiscard]] inline std::size_t slot_count() const noexcept {
        return slots_;
    }

    auto &gpu_ckks_msg_vec() {
        return *gpu_ckks_msg_vec_;
    }
};

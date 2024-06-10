#pragma once

#include <cassert>

#include "context.cuh"
#include "polymath.cuh"

class PhantomPlaintext {

    friend class PhantomBatchEncoder;

    friend class PhantomCKKSEncoder;

    friend class PhantomSecretKey;

private:

    phantom::parms_id_type parms_id_ = phantom::parms_id_zero;
    std::size_t chain_index_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    size_t coeff_modulus_size_ = 0;
    double scale_ = 1.0;
    phantom::util::cuda_auto_ptr<uint64_t> data_;

public:

    PhantomPlaintext() = default;

    PhantomPlaintext(const PhantomPlaintext &) = default;

    PhantomPlaintext &operator=(const PhantomPlaintext &) = default;

    PhantomPlaintext(PhantomPlaintext &&) = default;

    PhantomPlaintext &operator=(PhantomPlaintext &&) = default;

    ~PhantomPlaintext() = default;

    void resize(const size_t coeff_modulus_size, const size_t poly_modulus_degree, const cudaStream_t &stream) {
        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(coeff_modulus_size * poly_modulus_degree, stream);

        coeff_modulus_size_ = coeff_modulus_size;
        poly_modulus_degree_ = poly_modulus_degree;
    }

    void set_chain_index(const size_t chain_index) {
        chain_index_ = chain_index;
    }

    [[nodiscard]] std::size_t coeff_count() const noexcept {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }

    [[nodiscard]] auto &parms_id() const noexcept {
        return parms_id_;
    }

    [[nodiscard]] auto &chain_index() const noexcept {
        return chain_index_;
    }

    [[nodiscard]] auto &scale() const noexcept {
        return scale_;
    }

    [[nodiscard]] auto data() const noexcept {
        return data_.get();
    }

    [[nodiscard]] auto &data_ptr() noexcept {
        return data_;
    }
};

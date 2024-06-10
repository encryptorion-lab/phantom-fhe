#pragma once

#include "context.cuh"
#include "polymath.cuh"

class PhantomCiphertext {

    friend class PhantomPublicKey;

    friend class PhantomSecretKey;

private:

    phantom::parms_id_type parms_id_ = phantom::parms_id_zero;

    // The index this ciphertext corresponding
    std::size_t chain_index_ = 0;

    // The number of poly in ciphertext
    std::size_t size_ = 0;

    // The poly degree
    std::size_t poly_modulus_degree_ = 0;

    // The coeff prime number
    std::size_t coeff_modulus_size_ = 0;

    // The scale this ciphertext corresponding to
    double scale_ = 1.0;

    // The correction factor for BGV decryption
    std::uint64_t correction_factor_ = 1;

    // the degree of the scaling factor for the encrypted message
    size_t noiseScaleDeg_ = 1;

    bool is_ntt_form_ = true;

    bool is_asymmetric_ = false;

    phantom::util::cuda_auto_ptr<uint64_t> data_;

public:

    PhantomCiphertext() = default;

    PhantomCiphertext(const PhantomCiphertext &) = default;

    PhantomCiphertext &operator=(const PhantomCiphertext &) = default;

    PhantomCiphertext(PhantomCiphertext &&) = default;

    PhantomCiphertext &operator=(PhantomCiphertext &&) = default;

    ~PhantomCiphertext() = default;

    /* Reset and malloc the ciphertext
     * @notice: when size is larger, the previous data is copied.
     */
    void resize(const PhantomContext &context, size_t chain_index, size_t size, const cudaStream_t &stream) {
        auto &context_data = context.get_context_data(chain_index);
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto coeff_modulus_size = coeff_modulus.size();
        auto poly_modulus_degree = parms.poly_modulus_degree();

        size_t old_size = size_ * coeff_modulus_size_ * poly_modulus_degree_;
        size_t new_size = size * coeff_modulus_size * poly_modulus_degree;

        if (new_size == 0) {
            data_.reset();
            return;
        }

        if (new_size != old_size) {
            auto prev_data(std::move(data_));
            data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size * coeff_modulus_size * poly_modulus_degree, stream);
            size_t copy_size = std::min(old_size, new_size);
            cudaMemcpyAsync(data_.get(), prev_data.get(), copy_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                            stream);
        }

        size_ = size;
        chain_index_ = chain_index;
        poly_modulus_degree_ = poly_modulus_degree;
        coeff_modulus_size_ = coeff_modulus_size;
    }

    void resize(size_t size, size_t coeff_modulus_size, size_t poly_modulus_degree, const cudaStream_t &stream) {
        size_t old_size = size_ * coeff_modulus_size_ * poly_modulus_degree_;
        size_t new_size = size * coeff_modulus_size * poly_modulus_degree;

        if (new_size == 0) {
            data_.reset();
            return;
        }

        if (new_size != old_size) {
            auto prev_data(std::move(data_));
            data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size * coeff_modulus_size * poly_modulus_degree, stream);
            size_t copy_size = std::min(old_size, new_size);
            cudaMemcpyAsync(data_.get(), prev_data.get(), copy_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                            stream);
        }

        size_ = size;
        coeff_modulus_size_ = coeff_modulus_size;
        poly_modulus_degree_ = poly_modulus_degree;
    }

    void SetNoiseScaleDeg(size_t noiseScaleDeg) {
        noiseScaleDeg_ = noiseScaleDeg;
    }

    void set_scale(double scale) {
        scale_ = scale;
    }

    void set_chain_index(std::size_t chain_index) {
        chain_index_ = chain_index;
    }

    void set_poly_modulus_degree(std::size_t poly_modulus_degree) {
        poly_modulus_degree_ = poly_modulus_degree;
    }

    void set_coeff_modulus_size(std::size_t coeff_modulus_size) {
        coeff_modulus_size_ = coeff_modulus_size;
    }

    void set_correction_factor(std::uint64_t correction_factor) {
        correction_factor_ = correction_factor;
    }

    void set_ntt_form(bool is_ntt_form) {
        is_ntt_form_ = is_ntt_form;
    }

    [[nodiscard]] auto &size() const noexcept {
        return size_;
    }

    [[nodiscard]] auto &GetNoiseScaleDeg() const {
        return noiseScaleDeg_;
    }

    [[nodiscard]] auto &is_ntt_form() const noexcept {
        return is_ntt_form_;
    }

    [[nodiscard]] bool is_asymmetric() const noexcept {
        return is_asymmetric_;
    }

    [[nodiscard]] auto &parms_id() const noexcept {
        return parms_id_;
    }

    [[nodiscard]] auto &chain_index() const noexcept {
        return chain_index_;
    }

    [[nodiscard]] auto &poly_modulus_degree() const noexcept {
        return poly_modulus_degree_;
    }

    [[nodiscard]] auto &coeff_modulus_size() const noexcept {
        return coeff_modulus_size_;
    }

    [[nodiscard]] auto &scale() const noexcept {
        return scale_;
    }

    [[nodiscard]] auto &correction_factor() const noexcept {
        return correction_factor_;
    }

    [[nodiscard]] auto data() const {
        return data_.get();
    }

    [[nodiscard]] auto &data_ptr() {
        return data_;
    }
};

#pragma once

#include "context.cuh"
#include "polymath.cuh"
#include "prng.cuh"

class PhantomCiphertext {

    friend class PhantomPublicKey;

    friend class PhantomSecretKey;

private:

    size_t chain_index_ = 0; // The index this ciphertext corresponding
    size_t size_ = 0; // The number of poly in ciphertext
    size_t poly_modulus_degree_ = 0; // The poly degree
    size_t coeff_modulus_size_ = 0; // The coeff prime number
    double scale_ = 1.0; // The scale this ciphertext corresponding to
    uint64_t correction_factor_ = 1; // The correction factor for BGV decryption
    size_t noiseScaleDeg_ = 1; // the degree of the scaling factor for the encrypted message
    bool is_ntt_form_ = true;
    bool is_asymmetric_ = false;
    phantom::util::cuda_auto_ptr<uint64_t> data_;
    std::vector<uint8_t> seed_; // only for symmetric encryption

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
            data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size * coeff_modulus_size * poly_modulus_degree,
                                                                stream);
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
            data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size * coeff_modulus_size * poly_modulus_degree,
                                                                stream);
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

    [[nodiscard]] auto &seed_ptr() {
        return seed_;
    }

    void save(std::ostream &stream) const {
        stream.write(reinterpret_cast<const char *>(&chain_index_), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char *>(&size_), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char *>(&poly_modulus_degree_), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char *>(&coeff_modulus_size_), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char *>(&scale_), sizeof(double));
        stream.write(reinterpret_cast<const char *>(&correction_factor_), sizeof(std::uint64_t));
        stream.write(reinterpret_cast<const char *>(&noiseScaleDeg_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&is_ntt_form_), sizeof(bool));
        stream.write(reinterpret_cast<const char *>(&is_asymmetric_), sizeof(bool));

        uint64_t *h_data;
        cudaMallocHost(&h_data, size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaMemcpy(h_data, data_.get(), size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        stream.write(reinterpret_cast<char *>(h_data),
                     size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaFreeHost(h_data);
    }

    void load(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&chain_index_), sizeof(std::size_t));
        stream.read(reinterpret_cast<char *>(&size_), sizeof(std::size_t));
        stream.read(reinterpret_cast<char *>(&poly_modulus_degree_), sizeof(std::size_t));
        stream.read(reinterpret_cast<char *>(&coeff_modulus_size_), sizeof(std::size_t));
        stream.read(reinterpret_cast<char *>(&scale_), sizeof(double));
        stream.read(reinterpret_cast<char *>(&correction_factor_), sizeof(std::uint64_t));
        stream.read(reinterpret_cast<char *>(&noiseScaleDeg_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&is_ntt_form_), sizeof(bool));
        stream.read(reinterpret_cast<char *>(&is_asymmetric_), sizeof(bool));

        uint64_t *h_data;
        cudaMallocHost(&h_data, size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        stream.read(reinterpret_cast<char *>(h_data),
                    size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_ * coeff_modulus_size_ * poly_modulus_degree_,
                                                            cudaStreamPerThread);
        cudaMemcpyAsync(data_.get(), h_data, size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
        cudaFreeHost(h_data);
        cudaStreamSynchronize(cudaStreamPerThread);
    }

    void save_symmetric(std::ostream &stream) const {
        if (is_asymmetric_)
            throw std::runtime_error("Asymmetric ciphertext does not have seed.");

        if (size_ != 2)
            throw std::runtime_error("This method is only for 2-polynomial ciphertext.");

        stream.write(reinterpret_cast<const char *>(&chain_index_), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char *>(&size_), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char *>(&poly_modulus_degree_), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char *>(&coeff_modulus_size_), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char *>(&scale_), sizeof(double));
        stream.write(reinterpret_cast<const char *>(&correction_factor_), sizeof(std::uint64_t));
        stream.write(reinterpret_cast<const char *>(&noiseScaleDeg_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&is_ntt_form_), sizeof(bool));
        stream.write(reinterpret_cast<const char *>(&is_asymmetric_), sizeof(bool));

        // Only save c0
        uint64_t *h_c0;
        cudaMallocHost(&h_c0, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaMemcpy(h_c0, data_.get(), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        stream.write(reinterpret_cast<char *>(h_c0), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaFreeHost(h_c0);

        // Save seed of a instead of c1
        stream.write(reinterpret_cast<const char *>(seed_.data()),
                     phantom::util::global_variables::prng_seed_byte_count);
    }

    void load_symmetric(const PhantomContext &context, std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&chain_index_), sizeof(std::size_t));
        stream.read(reinterpret_cast<char *>(&size_), sizeof(std::size_t));
        stream.read(reinterpret_cast<char *>(&poly_modulus_degree_), sizeof(std::size_t));
        stream.read(reinterpret_cast<char *>(&coeff_modulus_size_), sizeof(std::size_t));
        stream.read(reinterpret_cast<char *>(&scale_), sizeof(double));
        stream.read(reinterpret_cast<char *>(&correction_factor_), sizeof(std::uint64_t));
        stream.read(reinterpret_cast<char *>(&noiseScaleDeg_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&is_ntt_form_), sizeof(bool));
        stream.read(reinterpret_cast<char *>(&is_asymmetric_), sizeof(bool));

        if (is_asymmetric_)
            throw std::runtime_error("Asymmetric ciphertext does not have seed.");

        if (size_ != 2)
            throw std::runtime_error("This method is only for 2-polynomial ciphertext.");

        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(2 * coeff_modulus_size_ * poly_modulus_degree_,
                                                            cudaStreamPerThread);
        auto *d_c0 = data_.get();
        auto *d_c1 = data_.get() + coeff_modulus_size_ * poly_modulus_degree_;

        // Load c0 directly from stream
        uint64_t *h_c0;
        cudaMallocHost(&h_c0, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        stream.read(reinterpret_cast<char *>(h_c0), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaMemcpyAsync(d_c0, h_c0, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);

        // Load c1 by generating from seed
        seed_.resize(phantom::util::global_variables::prng_seed_byte_count);
        stream.read(reinterpret_cast<char *>(seed_.data()), phantom::util::global_variables::prng_seed_byte_count);

        auto d_seed = phantom::util::make_cuda_auto_ptr<uint8_t>(
                phantom::util::global_variables::prng_seed_byte_count, cudaStreamPerThread);
        cudaMemcpyAsync(d_seed.get(), seed_.data(), phantom::util::global_variables::prng_seed_byte_count,
                        cudaMemcpyHostToDevice, cudaStreamPerThread);

        // uniform random generator
        auto &first_context_data = context.get_context_data(context.get_first_index());
        auto &first_parms = first_context_data.parms();
        auto &first_coeff_modulus = first_parms.coeff_modulus();
        auto first_coeff_mod_size = first_coeff_modulus.size();

        if (first_coeff_mod_size != coeff_modulus_size_) {
            throw std::runtime_error("Only support ciphertext without modulus switching.");
        }

        auto base_rns = context.gpu_rns_tables().modulus();
        sample_uniform_poly_wrap(
                d_c1, d_seed.get(), base_rns, poly_modulus_degree_, coeff_modulus_size_, cudaStreamPerThread);

        if (!is_ntt_form_) {
            // Transform c1 to non-NTT form
            nwt_2d_radix8_backward_inplace(d_c1, context.gpu_rns_tables(), coeff_modulus_size_, 0, cudaStreamPerThread);
        }

        cudaStreamSynchronize(cudaStreamPerThread);

        // cleanup h_c0
        cudaFreeHost(h_c0);
    }
};

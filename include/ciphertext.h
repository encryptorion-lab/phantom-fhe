#pragma once

#include "context.cuh"
#include "polymath.cuh"

typedef struct PhantomCiphertext {
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

    // The data_
    phantom::util::Pointer<uint8_t> prng_seed_; // prng seed

    // multiple poly, include size_ * coeff_modulus_size_ * poly_modulus_degree_
    phantom::util::Pointer<uint64_t> data_;

    PhantomCiphertext() {
        chain_index_ = 0;
        size_ = 0;
        poly_modulus_degree_ = 0;
        coeff_modulus_size_ = 0;
        scale_ = 0;
        correction_factor_ = 1;
        noiseScaleDeg_ = 1;
        is_ntt_form_ = true;
        is_asymmetric_ = false;
        prng_seed_.acquire(phantom::util::allocate<uint8_t>(phantom::util::global_pool(),
                                                            phantom::util::global_variables::prng_seed_byte_count));
    }

    explicit PhantomCiphertext(const PhantomContext &context) {
        prng_seed_.acquire(phantom::util::allocate<uint8_t>(phantom::util::global_pool(),
                                                            phantom::util::global_variables::prng_seed_byte_count));
    }

    // copy constructor
    PhantomCiphertext(const PhantomCiphertext &copy) {
        parms_id_ = copy.parms_id_;
        chain_index_ = copy.chain_index_;
        size_ = copy.size_;
        poly_modulus_degree_ = copy.poly_modulus_degree_;
        coeff_modulus_size_ = copy.coeff_modulus_size_;
        scale_ = copy.scale_;
        correction_factor_ = copy.correction_factor_;
        noiseScaleDeg_ = copy.noiseScaleDeg_;
        is_ntt_form_ = copy.is_ntt_form_;
        is_asymmetric_ = copy.is_asymmetric_;
        if (copy.prng_seed() != nullptr) {
            prng_seed_.acquire(phantom::util::allocate<uint8_t>(phantom::util::global_pool(),
                                                                phantom::util::global_variables::prng_seed_byte_count));
            cudaMemcpy(prng_seed(), copy.prng_seed(),
                       phantom::util::global_variables::prng_seed_byte_count * sizeof(uint8_t),
                       cudaMemcpyDeviceToDevice);
        }
        if (copy.data() != nullptr) {
            data_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(),
                                                            size_ * coeff_modulus_size_ * poly_modulus_degree_));
            cudaMemcpy(data(), copy.data(),
                       size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                       cudaMemcpyDeviceToDevice);
        }
    }

    // copy assignment
    PhantomCiphertext &operator=(const PhantomCiphertext &copy) {
        if (this != &copy) {
            parms_id_ = copy.parms_id_;
            chain_index_ = copy.chain_index_;
            size_ = copy.size_;
            poly_modulus_degree_ = copy.poly_modulus_degree_;
            coeff_modulus_size_ = copy.coeff_modulus_size_;
            scale_ = copy.scale_;
            correction_factor_ = copy.correction_factor_;
            noiseScaleDeg_ = copy.noiseScaleDeg_;
            is_ntt_form_ = copy.is_ntt_form_;
            is_asymmetric_ = copy.is_asymmetric_;
            if (copy.prng_seed() != nullptr) {
                prng_seed_.acquire(phantom::util::allocate<uint8_t>(phantom::util::global_pool(),
                                                                    phantom::util::global_variables::prng_seed_byte_count));
                cudaMemcpy(prng_seed(), copy.prng_seed(),
                           phantom::util::global_variables::prng_seed_byte_count * sizeof(uint8_t),
                           cudaMemcpyDeviceToDevice);
            }
            if (copy.data() != nullptr) {
                data_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(),
                                                                size_ * coeff_modulus_size_ * poly_modulus_degree_ *
                                                                sizeof(uint64_t)));
                cudaMemcpy(data(), copy.data(),
                           size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                           cudaMemcpyDeviceToDevice);
            }
        }
        return *this;
    }

    // move constructor
    PhantomCiphertext(PhantomCiphertext &&move) noexcept {
        parms_id_ = move.parms_id_;
        chain_index_ = move.chain_index_;
        size_ = move.size_;
        poly_modulus_degree_ = move.poly_modulus_degree_;
        coeff_modulus_size_ = move.coeff_modulus_size_;
        scale_ = move.scale_;
        correction_factor_ = move.correction_factor_;
        noiseScaleDeg_ = move.correction_factor_;
        is_ntt_form_ = move.is_ntt_form_;
        is_asymmetric_ = move.is_asymmetric_;
        prng_seed_.acquire(move.prng_seed_);
        data_.acquire(move.data_);
    }

    // move assignment
    PhantomCiphertext &operator=(PhantomCiphertext &&move) {
        parms_id_ = move.parms_id_;
        chain_index_ = move.chain_index_;
        size_ = move.size_;
        poly_modulus_degree_ = move.poly_modulus_degree_;
        coeff_modulus_size_ = move.coeff_modulus_size_;
        scale_ = move.scale_;
        correction_factor_ = move.correction_factor_;
        noiseScaleDeg_ = move.correction_factor_;
        is_ntt_form_ = move.is_ntt_form_;
        is_asymmetric_ = move.is_asymmetric_;
        prng_seed_.acquire(move.prng_seed_);
        data_.acquire(move.data_);
        return *this;
    }

    inline void free() {
        parms_id_ = phantom::parms_id_zero;
        chain_index_ = 0;
        size_ = 0;
        poly_modulus_degree_ = 0;
        coeff_modulus_size_ = 0;
        scale_ = 0;
        correction_factor_ = 0;
        noiseScaleDeg_ = 0;
    }

    ~PhantomCiphertext() {
        free();
    }

    /** return the number of ploy in ciphertext
     */
    [[nodiscard]] __host__ __device__ __forceinline__ std::size_t size() const noexcept {
        return size_;
    }

    /* Reset and malloc the ciphertext
     * @notice: when size is larger, the previous data is copied.
     */
    void resize(const PhantomContext &context, size_t chain_index, size_t size) {
        auto &context_data = context.get_context_data(chain_index);
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto coeff_modulus_size = coeff_modulus.size();
        auto poly_modulus_degree = parms.poly_modulus_degree();

        size_t old_size = size_ * coeff_modulus_size_ * poly_modulus_degree_;
        size_t new_size = size * coeff_modulus_size * poly_modulus_degree;

        phantom::util::Pointer<uint64_t> prev_data;

        if (new_size > old_size) {
            prev_data.acquire(data_);
            data_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(),
                                                            size * poly_modulus_degree * coeff_modulus_size));
            PHANTOM_CHECK_CUDA(cudaMemcpy(data_.get(), prev_data.get(), old_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        } else if (new_size > 0 && new_size < old_size) {
            prev_data.acquire(data_);
            data_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(),
                                                            size * poly_modulus_degree * coeff_modulus_size));
            PHANTOM_CHECK_CUDA(cudaMemcpy(data_.get(), prev_data.get(), new_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        }

        size_ = size;
        chain_index_ = chain_index;
        poly_modulus_degree_ = poly_modulus_degree;
        coeff_modulus_size_ = coeff_modulus_size;
    }

    void resize(size_t size, size_t coeff_modulus_size, size_t poly_modulus_degree) {
        size_t old_size = size_ * coeff_modulus_size_ * poly_modulus_degree_;
        size_t new_size = size * coeff_modulus_size * poly_modulus_degree;
        phantom::util::Pointer<uint64_t> prev_data;
        prev_data.acquire(data_);
        data_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(),
                                                        size * coeff_modulus_size * poly_modulus_degree));

        if (new_size > old_size)
            PHANTOM_CHECK_CUDA(cudaMemcpy(data_.get(), prev_data.get(), old_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        if (new_size > 0 && new_size <= old_size)
            PHANTOM_CHECK_CUDA(cudaMemcpy(data_.get(), prev_data.get(), new_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

        size_ = size;
        coeff_modulus_size_ = coeff_modulus_size;
        poly_modulus_degree_ = poly_modulus_degree;
    }

    /**
    * Get the degree of the scaling factor for the encrypted message.
    */
    [[nodiscard]] size_t GetNoiseScaleDeg() const {
        return noiseScaleDeg_;
    }

    /**
    * Set the degree of the scaling factor for the encrypted message.
    */
    void SetNoiseScaleDeg(size_t noiseScaleDeg) {
        noiseScaleDeg_ = noiseScaleDeg;
    }

    /**
    Returns whether the ciphertext is in NTT form.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ bool is_ntt_form() const noexcept {
        return is_ntt_form_;
    }

    /**
    Returns whether the ciphertext is in NTT form.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ bool &is_ntt_form() noexcept {
        return is_ntt_form_;
    }

    /**
    Returns whether the ciphertext is encrypted using asymmetric encryption.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ bool &is_asymmetric() noexcept {
        return is_asymmetric_;
    }

    /**
    Returns a reference to parms_id.
    @see EncryptionParameters for more information about parms_id.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &parms_id() noexcept {
        return parms_id_;
    }

    /**
    Returns a reference to parms_id.
    @see EncryptionParameters for more information about parms_id.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &parms_id() const noexcept {
        return parms_id_;
    }

    /**
    Returns a const reference to chain_index.
    @see EncryptionParameters for more information about chain_index.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &chain_index() noexcept {
        return chain_index_;
    }

    /**
    Returns a const reference to chain_index.
    @see EncryptionParameters for more information about chain_index.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &chain_index() const noexcept {
        return chain_index_;
    }

    /**
    Returns a reference to the scale. This is only needed when using the
    CKKS encryption scheme. The user should have little or no reason to ever
    change the scale by hand.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &scale() noexcept {
        return scale_;
    }

    /**
    Returns a constant reference to the scale. This is only needed when
    using the CKKS encryption scheme.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &scale() const noexcept {
        return scale_;
    }

    void set_scale(double scale) {
        scale_ = scale;
    }

    /**
Returns a reference to the correction factor. This is only needed when using the BGV encryption scheme. The user
should have little or no reason to ever change the scale by hand.
*/
    [[nodiscard]] __host__ __device__ __forceinline__ std::uint64_t &correction_factor() noexcept {
        return correction_factor_;
    }

    /**
    Returns a constant reference to the correction factor. This is only needed when using the BGV encryption scheme.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ const std::uint64_t &correction_factor() const noexcept {
        return correction_factor_;
    }

    __host__ __device__ __forceinline__ uint8_t *prng_seed() const {
        return (uint8_t *) (prng_seed_.get());
    }

    __host__ __device__ __forceinline__ uint64_t *data() const {
        return (uint64_t *) (data_.get());
    }
} PhantomCiphertext;

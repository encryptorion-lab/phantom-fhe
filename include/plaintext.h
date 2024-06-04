#pragma once

#include <cassert>

#include "context.cuh"
#include "polymath.cuh"
#include "mempool.cuh"

typedef struct PhantomPlaintext {
    phantom::parms_id_type parms_id_ = phantom::parms_id_zero;
    // The index this ciphertext corresponding
    std::size_t chain_index_ = 0;
    // plaintext
    phantom::util::Pointer<uint64_t> data_;
    // poly_modulus_degree, i.e., N
    std::size_t poly_modulus_degree_ = 0;

    size_t coeff_modulus_size_ = 0;

    double scale_ = 1.0;

    PhantomPlaintext() = default;

    explicit PhantomPlaintext(const PhantomContext &context) {
        auto &context_data = context.get_context_data(0);
        auto &parms = context_data.parms();

        if (parms.scheme() == phantom::scheme_type::bfv || parms.scheme() == phantom::scheme_type::bgv)
            coeff_modulus_size_ = 1;
        else if (parms.scheme() == phantom::scheme_type::ckks)
            coeff_modulus_size_ = context.coeff_mod_size_;
        else
            throw std::invalid_argument("invalid FHE scheme.");

        if (coeff_modulus_size_ == 0) {
            throw std::invalid_argument("PhantomContext not inited yet.");
        }
        poly_modulus_degree_ = context.poly_degree_;
        // Malloc memory
        data_.acquire(
            phantom::util::allocate<uint64_t>(phantom::util::global_pool(), coeff_modulus_size_ * poly_modulus_degree_));
    }

    // copy constructor
    PhantomPlaintext(const PhantomPlaintext &copy) {
        parms_id_ = copy.parms_id_;
        chain_index_ = copy.chain_index_;
        poly_modulus_degree_ = copy.poly_modulus_degree_;
        coeff_modulus_size_ = copy.coeff_modulus_size_;
        scale_ = copy.scale_;
        data_.acquire(
            phantom::util::allocate<uint64_t>(phantom::util::global_pool(), coeff_modulus_size_ * poly_modulus_degree_));
        PHANTOM_CHECK_CUDA(
            cudaMemcpy(data_.get(), copy.data_.get(), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                cudaMemcpyDeviceToDevice));
    }

    // move constructor
    PhantomPlaintext(PhantomPlaintext &&source) noexcept {
        parms_id_ = source.parms_id_;
        chain_index_ = source.chain_index_;
        poly_modulus_degree_ = source.poly_modulus_degree_;
        coeff_modulus_size_ = source.coeff_modulus_size_;
        scale_ = source.scale_;
        data_.acquire(source.data_);
    }

    // copy assignment
    PhantomPlaintext &operator=(const PhantomPlaintext &copy) {
        if (this != &copy) {
            parms_id_ = copy.parms_id_;
            chain_index_ = copy.chain_index_;
            poly_modulus_degree_ = copy.poly_modulus_degree_;
            coeff_modulus_size_ = copy.coeff_modulus_size_;
            scale_ = copy.scale_;
            data_.acquire(
                phantom::util::allocate<uint64_t>(phantom::util::global_pool(), coeff_modulus_size_ * poly_modulus_degree_));
            PHANTOM_CHECK_CUDA(cudaMemcpy(data_.get(), copy.data_.get(),
                coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                cudaMemcpyDeviceToDevice));
        }
        return *this;
    }

    // move assignment
    PhantomPlaintext &operator=(PhantomPlaintext &&source) noexcept {
        if (this != (PhantomPlaintext *)&source) {
            parms_id_ = source.parms_id_;
            chain_index_ = source.chain_index_;
            poly_modulus_degree_ = source.poly_modulus_degree_;
            coeff_modulus_size_ = source.coeff_modulus_size_;
            scale_ = source.scale_;
            data_.acquire(source.data_);
        }
        return *this;
    }

    ~PhantomPlaintext() = default;

    /**
    Resizes the plaintext to have a given coefficient count. The plaintext
    is automatically reallocated if the new coefficient count does not fit in
    the current capacity.

    @param[in] coeff_count The number of coefficients in the plaintext polynomial
    @throws std::invalid_argument if coeff_count is negative
    @throws std::logic_error if the plaintext is NTT transformed
    */
    __host__ __forceinline__ void resize(const size_t coeff_modulus_size, const size_t poly_modulus_degree) {
        if (is_ntt_form()) {
#ifdef __CUDA_ARCH__
            assert(is_ntt_form());
#else
            throw std::logic_error("cannot reserve for an NTT transformed Plaintext");
#endif
        }
        coeff_modulus_size_ = coeff_modulus_size;
        poly_modulus_degree_ = poly_modulus_degree;
        data_.acquire(
            phantom::util::allocate<uint64_t>(phantom::util::global_pool(), coeff_modulus_size_ * poly_modulus_degree_));
    }

    /**
    Returns the coefficient count of the current plaintext polynomial.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ std::size_t coeff_count() const noexcept {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }

    /**
    Returns whether the plaintext is in NTT form.
    The chain_index must remain zero unless the
    plaintext polynomial is in NTT form.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ bool is_ntt_form() const noexcept {
        return (chain_index_ != 0);
    }

    /**
    Returns a reference to parms_id. The parms_id must remain zero unless the
    plaintext polynomial is in NTT form.
    @see EncryptionParameters for more information about parms_id.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &parms_id() const noexcept {
        return parms_id_;
    }

    [[nodiscard]] __host__ __device__ __forceinline__ auto &parms_id() noexcept {
        return parms_id_;
    }

    [[nodiscard]] __host__ __device__ __forceinline__ auto &chain_index() const noexcept {
        return chain_index_;
    }

    [[nodiscard]] __host__ __device__ __forceinline__ auto &chain_index() noexcept {
        return chain_index_;
    }

    /**
        Returns a reference to the scale. This is only needed when using the CKKS
        encryption scheme. The user should have little or no reason to ever change
        the scale by hand.
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

    /**
    Returns the significant coefficient count of the current plaintext polynomial.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ size_t significant_coeff_count() const {
        if (!poly_modulus_degree_) {
            return 0;
        }
        size_t ret = poly_modulus_degree_;
        uint64_t *value = data_.get() + poly_modulus_degree_ - 1;
        for (; (!(*value)) && ret; ret--) {
            value--;
        }
        return ret;
    }

    __host__ __device__ __forceinline__ uint64_t *data() const {
        return (uint64_t *)(data_.get());
    }
} PhantomPlaintext;

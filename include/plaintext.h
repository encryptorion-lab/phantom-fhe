#pragma once

#include <cassert>

#include "context.cuh"
#include "polymath.cuh"
#include "mempool.cuh"
#include "util/polycore.h"

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
        CUDA_CHECK(
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
            CUDA_CHECK(cudaMemcpy(data_.get(), copy.data_.get(),
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

    [[nodiscard]] std::string to_string() const {
        if (is_ntt_form())
            throw std::invalid_argument("cannot convert NTT transformed plaintext to string");
        return phantom::util::poly_to_hex_string(data(), coeff_modulus_size_ * poly_modulus_degree_, 1);
    }

    void save(std::ostream &stream) {
        auto old_except_mask = stream.exceptions();
        try {
            // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
            stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            stream.write(reinterpret_cast<const char *>(&parms_id_), sizeof(phantom::parms_id_type));
            uint64_t coeff_modulus_size = static_cast<uint64_t>(coeff_modulus_size_);
            uint64_t chain64 = static_cast<uint64_t>(chain_index_);
            uint64_t poly_modulus_degree = static_cast<uint64_t>(poly_modulus_degree_);
            stream.write(reinterpret_cast<const char *>(&coeff_modulus_size), sizeof(uint64_t));
            stream.write(reinterpret_cast<const char *>(&chain64), sizeof(uint64_t));
            stream.write(reinterpret_cast<const char *>(&poly_modulus_degree), sizeof(uint64_t));
            stream.write(reinterpret_cast<const char *>(&scale_), sizeof(double));

            uint64_t data_size = coeff_modulus_size_ * poly_modulus_degree_;
            stream.write(reinterpret_cast<const char *>(&data_size), sizeof(std::uint64_t));

            std::vector<uint64_t> temp_data;
            temp_data.resize(coeff_modulus_size_ * poly_modulus_degree_);
            CUDA_CHECK(cudaMemcpy(temp_data.data(), data_.get(),
                coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                cudaMemcpyDeviceToHost));
            stream.write(
                reinterpret_cast<const char *>(temp_data.data()),
                static_cast<std::streamsize>(phantom::util::mul_safe(data_size, sizeof(uint64_t))));
        }
        catch (const std::ios_base::failure &) {
            stream.exceptions(old_except_mask);
            throw std::runtime_error("I/O error");
        }
        catch (...) {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);
    }

    void load(const PhantomContext &context, std::istream &stream) {
        PhantomPlaintext new_data(context);

        auto old_except_mask = stream.exceptions();
        try {
            // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
            stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);

            phantom::parms_id_type parms_id{};
            stream.read(reinterpret_cast<char *>(&parms_id), sizeof(phantom::parms_id_type));

            uint64_t coeff_modulus64 = 0;
            stream.read(reinterpret_cast<char *>(&coeff_modulus64), sizeof(uint64_t));
            uint64_t chain64 = 0;
            stream.read(reinterpret_cast<char *>(&chain64), sizeof(uint64_t));
            uint64_t poly_degree64 = 0;
            stream.read(reinterpret_cast<char *>(&poly_degree64), sizeof(uint64_t));
            double scale = 0;
            stream.read(reinterpret_cast<char *>(&scale), sizeof(double));

            // Set the metadata
            new_data.parms_id_ = parms_id;
            new_data.coeff_modulus_size_ = static_cast<size_t>(coeff_modulus64);
            new_data.chain_index_ = static_cast<size_t>(chain64);
            new_data.poly_modulus_degree_ = static_cast<size_t>(poly_degree64);
            new_data.scale_ = scale;

            auto total_uint64_count =
                    phantom::util::mul_safe(new_data.poly_modulus_degree_, new_data.coeff_modulus_size_);

            new_data.data_.acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), total_uint64_count));

            uint64_t data_size;
            stream.read(reinterpret_cast<char *>(&data_size), sizeof(std::uint64_t));

            std::vector<uint64_t> temp_data;
            temp_data.resize(total_uint64_count);
            stream.read(
                reinterpret_cast<char *>(temp_data.data()),
                static_cast<std::streamsize>(phantom::util::mul_safe(total_uint64_count, sizeof(uint64_t))));

            CUDA_CHECK(cudaMemcpy(new_data.data_.get(), temp_data.data(), total_uint64_count * sizeof(uint64_t),
                cudaMemcpyHostToDevice));
        }
        catch (const std::ios_base::failure &) {
            stream.exceptions(old_except_mask);
            throw std::runtime_error("I/O error");
        }
        catch (...) {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);

        std::swap(*this, new_data);
    }

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

#pragma once

#include <cassert>

#include "context.cuh"
#include "polymath.cuh"

class PhantomPlaintext {

    friend class PhantomBatchEncoder;

    friend class PhantomCKKSEncoder;

    friend class PhantomSecretKey;

private:

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

    void save(std::ostream &stream) const {
        stream.write(reinterpret_cast<const char *>(&chain_index_), sizeof(chain_index_));
        stream.write(reinterpret_cast<const char *>(&poly_modulus_degree_), sizeof(poly_modulus_degree_));
        stream.write(reinterpret_cast<const char *>(&coeff_modulus_size_), sizeof(coeff_modulus_size_));
        stream.write(reinterpret_cast<const char *>(&scale_), sizeof(scale_));

        uint64_t *h_data;
        cudaMallocHost(&h_data, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaMemcpy(h_data, data_.get(), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        stream.write(reinterpret_cast<char *>(h_data), coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        cudaFreeHost(h_data);
    }

    void load(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&chain_index_), sizeof(chain_index_));
        stream.read(reinterpret_cast<char *>(&poly_modulus_degree_), sizeof(poly_modulus_degree_));
        stream.read(reinterpret_cast<char *>(&coeff_modulus_size_), sizeof(coeff_modulus_size_));
        stream.read(reinterpret_cast<char *>(&scale_), sizeof(scale_));

        uint64_t *h_data;
        cudaMallocHost(&h_data, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        stream.read(reinterpret_cast<char *>(h_data),
                    coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t));
        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(coeff_modulus_size_ * poly_modulus_degree_,
                                                            cudaStreamPerThread);
        cudaMemcpyAsync(data_.get(), h_data, coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
        cudaFreeHost(h_data);
    }
};

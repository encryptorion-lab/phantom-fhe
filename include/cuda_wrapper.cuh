#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "util.cuh"

namespace phantom::util {

    class cuda_stream_wrapper {
    public:
        cuda_stream_wrapper() {
            PHANTOM_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }

        ~cuda_stream_wrapper() {
            PHANTOM_CHECK_CUDA(cudaStreamDestroy(stream));
        }

        [[nodiscard]] auto &get_stream() const {
            return stream;
        }

    private:
        cudaStream_t stream{};
    };

    template<class T>
    class cuda_shared_ptr {

    private:
        T *ptr_ = nullptr;
        size_t n_ = 0;
        cudaStream_t cudaStream_ = nullptr;

    public:
        cuda_shared_ptr() = default;

        explicit cuda_shared_ptr(T *ptr, size_t n, const cudaStream_t &stream) {
            ptr_ = ptr;
            n_ = n;
            cudaStream_ = stream;
        }

        // copy constructor
        cuda_shared_ptr(const cuda_shared_ptr &obj) {
            PHANTOM_CHECK_CUDA(cudaMallocAsync(&this->ptr_, obj.n_ * sizeof(T), obj.cudaStream_));
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(this->ptr_, obj.ptr_, obj.n_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               obj.cudaStream_));
            this->n_ = obj.n_;
            this->cudaStream_ = obj.cudaStream_;
        }

        // copy assignment
        cuda_shared_ptr &operator=(const cuda_shared_ptr &obj) {
            if (this == &obj) {
                return *this;
            }

            reset();

            PHANTOM_CHECK_CUDA(cudaMallocAsync(&this->ptr_, obj.n_ * sizeof(T), obj.cudaStream_));
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(this->ptr_, obj.ptr_, obj.n_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               obj.cudaStream_));
            this->n_ = obj.n_;
            this->cudaStream_ = obj.cudaStream_;
            return *this;
        }

        // move constructor
        cuda_shared_ptr(cuda_shared_ptr &&dyingObj) noexcept {
            // share the underlying pointer
            this->ptr_ = dyingObj.ptr_;
            this->n_ = dyingObj.n_;
            this->cudaStream_ = dyingObj.cudaStream_;

            // reset the dying object
            dyingObj.ptr_ = nullptr;
            dyingObj.n_ = 0;
            dyingObj.cudaStream_ = nullptr;
        }

        // move assignment
        cuda_shared_ptr &operator=(cuda_shared_ptr &&dyingObj) noexcept {
            if (this == &dyingObj) {
                return *this;
            }

            reset();

            this->ptr_ = dyingObj.ptr_;
            this->n_ = dyingObj.n_;
            this->cudaStream_ = dyingObj.cudaStream_;

            // reset the dying object
            dyingObj.ptr_ = nullptr;
            dyingObj.n_ = 0;
            dyingObj.cudaStream_ = nullptr;

            return *this;
        }

        ~cuda_shared_ptr() // destructor
        {
            reset();
        }

        T *get() const {
            return this->ptr_;
        }

        T *operator->() const {
            return this->ptr_;
        }

        T &operator*() const {
            return this->ptr_;
        }

        [[nodiscard]] auto &get_n() const {
            return this->n_;
        }

        [[nodiscard]] auto &get_stream() const {
            return this->cudaStream_;
        }

        void reset() {
            if (ptr_ == nullptr) {
                return;
            }
            auto err = cudaFreeAsync(ptr_, cudaStream_);
            if (err != cudaSuccess) {
                std::cerr << "Error freeing " << n_ << " * " << sizeof(T) << " bytes at " << ptr_
                          << " on stream " << cudaStream_ << std::endl;
                std::cerr << "Error code: " << cudaGetErrorString(err) << std::endl;
            }
            n_ = 0;
            cudaStream_ = nullptr;
        }
    };

    template<class T>
    cuda_shared_ptr<T> cuda_make_shared(size_t n, const cudaStream_t &stream) {
        T *ptr;
        PHANTOM_CHECK_CUDA(cudaMallocAsync(&ptr, n * sizeof(T), stream));
        return cuda_shared_ptr<T>(ptr, n, stream);
    }
}

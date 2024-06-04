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

    public:
        cuda_shared_ptr() : ptr_(nullptr), refCount_(nullptr), cudaStream_(nullptr) {
        }

        cuda_shared_ptr(T *ptr, const cudaStream_t &stream) {
            ptr_ = ptr;
            refCount_ = new size_t(1);
            cudaStream_ = stream;
        }

        // copy constructor
        cuda_shared_ptr(const cuda_shared_ptr &obj) {
            this->ptr_ = obj.ptr_;
            this->refCount_ = obj.refCount_;
            this->cudaStream_ = obj.cudaStream_;
            if (obj.ptr_ != nullptr) {
                (*this->refCount_)++;
            }
        }

        // copy assignment
        cuda_shared_ptr &operator=(const cuda_shared_ptr &obj) {
            if (this == &obj) {
                return *this;
            }

            cleanup();

            this->ptr_ = obj.ptr_;
            this->refCount_ = obj.refCount_;
            this->cudaStream_ = obj.cudaStream_;
            if (obj.ptr_ != nullptr) {
                (*this->refCount_)++;
            }
            return *this;
        }

        // move constructor
        cuda_shared_ptr(cuda_shared_ptr &&dyingObj) noexcept {
            // share the underlying pointer
            this->ptr_ = dyingObj.ptr_;
            this->refCount_ = dyingObj.refCount_;
            this->cudaStream_ = dyingObj.cudaStream_;

            // reset the dying object
            dyingObj.ptr_ = nullptr;
            dyingObj.refCount_ = nullptr;
            dyingObj.cudaStream_ = nullptr;
        }

        // move assignment
        cuda_shared_ptr &operator=(cuda_shared_ptr &&dyingObj) noexcept {
            cleanup();

            this->ptr_ = dyingObj.ptr_;
            this->refCount_ = dyingObj.refCount_;
            this->cudaStream_ = dyingObj.cudaStream_;

            // reset the dying object
            dyingObj.ptr_ = nullptr;
            dyingObj.refCount_ = nullptr;
            dyingObj.cudaStream_ = nullptr;

            return *this;
        }

        ~cuda_shared_ptr() // destructor
        {
            cleanup();
        }

        [[nodiscard]] size_t get_count() const {
            return *this->refCount_;
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

    private:
        void cleanup() {
            if (ptr_ == nullptr || refCount_ == nullptr) {
                return;
            }

            (*refCount_)--;
            if (*refCount_ == 0) {
//                std::cout << "Freeing memory at " << ptr << std::endl; // debug
                PHANTOM_CHECK_CUDA(cudaFreeAsync(ptr_, cudaStream_));
                delete refCount_;
            }

            cudaStream_ = nullptr;
        }

        T *ptr_ = nullptr;
        size_t *refCount_ = nullptr;
        cudaStream_t cudaStream_ = nullptr;
    };

    template<class T>
    cuda_shared_ptr<T> cuda_make_shared(size_t n, const cudaStream_t &stream) {
        T *ptr;
        cudaMallocAsync(&ptr, n * sizeof(T), stream);
        return cuda_shared_ptr<T>(ptr, stream);
    }
}

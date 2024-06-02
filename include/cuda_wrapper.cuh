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

        cudaStream_t &get_stream() {
            return stream;
        }

    private:
        cudaStream_t stream{};
    };

    template<class T>
    class cuda_shared_ptr {

    public:
        cuda_shared_ptr() : ptr(nullptr), refCount(nullptr), stream_wrapper(nullptr) {
        }

        cuda_shared_ptr(T *ptr, const std::shared_ptr<cuda_stream_wrapper> &stream_wrapper)
                : ptr(ptr),
                  refCount(new size_t(1)),
                  stream_wrapper(stream_wrapper) {
        }

        // copy constructor
        cuda_shared_ptr(const cuda_shared_ptr &obj) {
            this->ptr = obj.ptr;
            this->refCount = obj.refCount;
            this->stream_wrapper = obj.stream_wrapper;
            if (obj.ptr != nullptr) {
                (*this->refCount)++;
            }
        }

        // copy assignment
        cuda_shared_ptr &operator=(const cuda_shared_ptr &obj) {
            cleanup();

            this->ptr = obj.ptr;
            this->refCount = obj.refCount;
            this->stream_wrapper = obj.stream_wrapper;
            if (obj.ptr != nullptr) {
                (*this->refCount)++;
            }
            return *this;
        }

        // move constructor
        cuda_shared_ptr(cuda_shared_ptr &&dyingObj) noexcept {
            // share the underlying pointer
            this->ptr = dyingObj.ptr;
            this->refCount = dyingObj.refCount;
            this->stream_wrapper = dyingObj.stream_wrapper;

            // reset the dying object
            dyingObj.ptr = nullptr;
            dyingObj.refCount = nullptr;
            dyingObj.stream_wrapper = nullptr;
        }

        // move assignment
        cuda_shared_ptr &operator=(cuda_shared_ptr &&dyingObj) noexcept {
            cleanup();

            this->ptr = dyingObj.ptr;
            this->refCount = dyingObj.refCount;
            this->stream_wrapper = dyingObj.stream_wrapper;

            // reset the dying object
            dyingObj.ptr = nullptr;
            dyingObj.refCount = nullptr;
            dyingObj.stream_wrapper = nullptr;

            return *this;
        }

        ~cuda_shared_ptr() // destructor
        {
            cleanup();
        }

        [[nodiscard]] size_t get_count() const {
            return *this->refCount;
        }

        T *get() const {
            return this->ptr;
        }

        T *operator->() const {
            return this->ptr;
        }

        T &operator*() const {
            return this->ptr;
        }

    private:
        void cleanup() {
            if (ptr == nullptr) {
                return;
            }
            (*refCount)--;
            if (*refCount == 0) {
//                std::cout << "Freeing memory at " << ptr << std::endl; // debug
                PHANTOM_CHECK_CUDA(cudaFreeAsync(ptr, stream_wrapper->get_stream()));
                cudaStreamSynchronize(stream_wrapper->get_stream());
                delete refCount;
            }
            stream_wrapper.reset();
        }

        T *ptr = nullptr;
        size_t *refCount = nullptr;
        std::shared_ptr<cuda_stream_wrapper> stream_wrapper = nullptr;
    };

    template<class T>
    cuda_shared_ptr<T> cuda_make_shared(size_t n, const std::shared_ptr<cuda_stream_wrapper> &stream_wrapper) {
        T *ptr;
        cudaMallocAsync(&ptr, n * sizeof(T), stream_wrapper->get_stream());
        return cuda_shared_ptr<T>(ptr, stream_wrapper);
    }
}

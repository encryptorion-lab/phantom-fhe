#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "cuda_wrapper.cuh"

using namespace phantom::util;

#define EPSINON 0.001

inline bool operator==(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) {
    return fabs(lhs.x - rhs.x) < EPSINON;
}

__global__ void scaling_kernel(cuDoubleComplex *data, int element_count, float scale) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (auto i = tid; i < element_count; i += stride) {
        data[tid].x *= scale;
        data[tid].y *= scale;
    }
}

int main() {
    cuda_stream_wrapper stream_wrapper;
    const auto &stream = stream_wrapper.get_stream();

    int dim = 1024;
    cufftHandle plan;
    cufftPlan1d(&plan, dim, CUFFT_Z2Z, 1);
    cufftSetStream(plan, stream);

    std::vector<cuDoubleComplex> h_in(dim, make_cuDoubleComplex(0, 0));
    for (int i = 3; i < dim; i++) {
        h_in[i] = make_cuDoubleComplex(1.0f / i, -2.0f / i);
    }

    for (int i = 0; i < dim; i++) {
        printf("%f + %fi, ", h_in[i].x, h_in[i].y);
    }
    printf("\n");

    auto d_data = make_cuda_auto_ptr<cuDoubleComplex>(dim, stream);
    cudaMemcpyAsync(d_data.get(), h_in.data(), dim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice,
                    stream);

    cufftExecZ2Z(plan, d_data.get(), d_data.get(), CUFFT_FORWARD);

    scaling_kernel<<<dim / 128, 128, 0, stream>>>(
            d_data.get(), dim, 1.f / dim);

    cufftExecZ2Z(plan, d_data.get(), d_data.get(), CUFFT_INVERSE);

    std::vector<cuDoubleComplex> h_out(dim);
    cudaMemcpyAsync(h_out.data(), d_data.get(), dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost,
                    stream);

    cudaStreamSynchronize(stream);

    for (int i = 0; i < dim; i++) {
        printf("%f + %fi, ", h_out[i].x, h_out[i].y);
    }
    printf("\n");

    cufftDestroy(plan);

    for (int i = 0; i < dim; i++) {
        if (!(h_in[i] == h_out[i])) {
            throw std::logic_error("Error");
        }
    }

    return 0;
}
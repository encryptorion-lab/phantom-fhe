#include "batchencoder.h"

using namespace std;
using namespace phantom;
using namespace phantom::util;

PhantomBatchEncoder::PhantomBatchEncoder(const PhantomContext &context) {
    const auto &s = phantom::util::global_variables::default_stream->get_stream();
    auto &context_data = context.get_context_data(0);
    auto &parms = context_data.parms();
    if (parms.scheme() != scheme_type::bfv && parms.scheme() != scheme_type::bgv) {
        throw std::invalid_argument("PhantomBatchEncoder only supports BFV/BGV scheme");
    }

    // Set the slot count
    auto poly_degree = parms.poly_modulus_degree();
    slots_ = poly_degree;

    // Populate matrix representation index map
    data_ = make_cuda_auto_ptr<uint64_t>(slots_, s);
    matrix_reps_index_map_ = make_cuda_auto_ptr<uint64_t>(slots_, s);
    populate_matrix_reps_index_map(s);
}

void PhantomBatchEncoder::populate_matrix_reps_index_map(const cudaStream_t &stream) const {
    vector<uint64_t> temp;
    int logn = phantom::arith::get_power_of_two(slots_);
    // Copy from the matrix to the value vectors
    size_t row_size = slots_ >> 1;
    size_t m = slots_ << 1;
    uint64_t gen = 5;
    uint64_t pos = 1;
    temp.resize(slots_);
    for (size_t i = 0; i < row_size; i++) {
        // Position in normal bit order
        uint64_t index1 = (pos - 1) >> 1;
        uint64_t index2 = (m - pos - 1) >> 1;

        // Set the bit-reversed locations
        temp[i] = (uint64_t) (arith::reverse_bits(index1, logn));
        temp[row_size | i] = static_cast<size_t>(arith::reverse_bits(index2, logn));

        // Next primitive root
        pos *= gen;
        pos &= (m - 1);
    }
    cudaMemcpyAsync(matrix_reps_index_map_.get(), temp.data(), sizeof(uint64_t) * slots_, cudaMemcpyHostToDevice,
                    stream);
}

__global__ void encode_gpu(uint64_t *out, uint64_t *in, size_t in_size, uint64_t *index_map, uint64_t mod,
                           size_t slots) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < slots; tid += blockDim.x * gridDim.x) {
        if (tid < in_size) {
            const uint64_t temp = in[tid];
            out[index_map[tid]] = temp + (temp >> 63) * mod;
        } else
            out[index_map[tid]] = 0;
    }
}

void PhantomBatchEncoder::encode(const PhantomContext &context, const std::vector<uint64_t> &values_matrix,
                                 PhantomPlaintext &destination, const phantom::util::cuda_stream_wrapper &stream_wrapper) const {
    const auto &s = stream_wrapper.get_stream();

    auto &context_data = context.get_context_data(0);
    auto &parms = context_data.parms();
    auto &plain_modulus = parms.plain_modulus();
    size_t values_matrix_size = values_matrix.size();
    if (values_matrix_size > slots_) {
        throw std::logic_error("values_matrix size is too large");
    }

    destination.coeff_modulus_size_ = 1;
    destination.poly_modulus_degree_ = context.poly_degree_;
    // Malloc memory
    destination.data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(
            destination.coeff_modulus_size_ * destination.poly_modulus_degree_, s);

    cudaMemcpyAsync(data_.get(), values_matrix.data(), values_matrix.size() * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, s);

    uint64_t gridDimGlb = ceil(slots_ / blockDimGlb.x);
    encode_gpu<<<gridDimGlb, blockDimGlb, 0, s>>>(
            destination.data(), data_.get(), values_matrix_size,
            matrix_reps_index_map_.get(), plain_modulus.value(), slots_);

    nwt_2d_radix8_backward_inplace(destination.data(), context.gpu_plain_tables(), 1, 0, s);
}

__global__ void decode_gpu(uint64_t *out, uint64_t *in, uint64_t *index_map, uint64_t slots) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < slots; tid += blockDim.x * gridDim.x) {
        out[tid] = in[index_map[tid]];
    }
}

void PhantomBatchEncoder::decode(const PhantomContext &context, const PhantomPlaintext &plain,
                                 std::vector<uint64_t> &destination, const phantom::util::cuda_stream_wrapper &stream_wrapper) const {
    const auto &s = stream_wrapper.get_stream();

    destination.resize(plain.poly_modulus_degree_);

    // Copy plain.data_
    auto plain_data_copy = make_cuda_auto_ptr<uint64_t>(slots_, s);
    cudaMemcpyAsync(plain_data_copy.get(), plain.data(), slots_ * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);

    nwt_2d_radix8_forward_inplace(plain_data_copy.get(), context.gpu_plain_tables(), 1, 0, s);

    auto out = make_cuda_auto_ptr<uint64_t>(slots_, s);
    uint64_t gridDimGlb = ceil(slots_ / blockDimGlb.x);
    decode_gpu<<<gridDimGlb, blockDimGlb, 0, s>>>(
            out.get(), plain_data_copy.get(), matrix_reps_index_map_.get(), slots_);

    cudaMemcpyAsync(destination.data(), out.get(), sizeof(uint64_t) * slots_, cudaMemcpyDeviceToHost, s);

    // explicit synchronization in case user wants to use the result immediately
    cudaStreamSynchronize(s);
}

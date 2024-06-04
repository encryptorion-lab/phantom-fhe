#include "batchencoder.h"

using namespace std;
using namespace phantom;
using namespace phantom::util;

PhantomBatchEncoder::PhantomBatchEncoder(const PhantomContext &context) {
    auto &context_data = context.get_context_data(0);
    auto &parms = context_data.parms();
    if (parms.scheme() != scheme_type::bfv && parms.scheme() != scheme_type::bgv) {
        throw std::invalid_argument("unsupported scheme");
    }

    // Set the slot count
    auto poly_degree = parms.poly_modulus_degree();
    slots_ = poly_degree;

    // Populate matrix representation index map
    data_.acquire(allocate<uint64_t>(global_pool(), slots_));
    matrix_reps_index_map_.acquire(allocate<uint64_t>(global_pool(), slots_));
    populate_matrix_reps_index_map();
}

void PhantomBatchEncoder::populate_matrix_reps_index_map() const {
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
        temp[i] = (uint64_t)(arith::reverse_bits(index1, logn));
        temp[row_size | i] = static_cast<size_t>(arith::reverse_bits(index2, logn));

        // Next primitive root
        pos *= gen;
        pos &= (m - 1);
    }
    PHANTOM_CHECK_CUDA(
            cudaMemcpy(matrix_reps_index_map_.get(), temp.data(), sizeof(uint64_t) * slots_, cudaMemcpyHostToDevice));
}

__global__ void encode_gpu(uint64_t *out, uint64_t *in, size_t in_size, uint64_t *index_map, uint64_t mod,
                           size_t slots) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < slots; tid += blockDim.x * gridDim.x) {
        if (tid < in_size) {
            const uint64_t temp = in[tid];
            out[index_map[tid]] = temp + (temp >> 63) * mod;
        }
        else
            out[index_map[tid]] = 0;
    }
}

// TODO: support <uint64_t> type
void PhantomBatchEncoder::encode(const PhantomContext &context, const std::vector<int64_t> &values_matrix,
                                 PhantomPlaintext &destination) {
    auto &context_data = context.get_context_data(0);
    auto &parms = context_data.parms();
    auto &plain_modulus = parms.plain_modulus();
    size_t values_matrix_size = values_matrix.size();
    if (values_matrix_size > slots_) {
        throw std::logic_error("values_matrix size is too large");
    }

    PHANTOM_CHECK_CUDA(cudaMemcpy(data_.get(), values_matrix.data(), values_matrix.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    uint64_t gridDimGlb = ceil(slots_ / blockDimGlb.x);
    encode_gpu<<<gridDimGlb, blockDimGlb>>>(destination.data(), data_.get(), values_matrix_size,
                                            matrix_reps_index_map_.get(), plain_modulus.value(), slots_);

    nwt_2d_radix8_backward_inplace(destination.data(), context.gpu_plain_tables(), 1, 0);
}

__global__ void decode_gpu(uint64_t *out, uint64_t *in, uint64_t *index_map, uint64_t slots) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < slots; tid += blockDim.x * gridDim.x) {
        out[tid] = in[index_map[tid]];
    }
}

void PhantomBatchEncoder::decode(const PhantomContext &context, const PhantomPlaintext &plain,
                                 std::vector<std::int64_t> &destination) const {
    if (plain.is_ntt_form()) {
        throw std::invalid_argument("plain cannot be in NTT form");
    }

    destination.resize(plain.poly_modulus_degree_);

    // Copy plain.data_
    Pointer<uint64_t> plain_data_copy;
    plain_data_copy.acquire(allocate<uint64_t>(global_pool(), slots_));
    PHANTOM_CHECK_CUDA(cudaMemcpy(plain_data_copy.get(), plain.data(), slots_ * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

    nwt_2d_radix8_forward_inplace(plain_data_copy.get(), context.gpu_plain_tables(), 1, 0);

    Pointer<uint64_t> out;
    out.acquire(allocate<uint64_t>(global_pool(), slots_));
    uint64_t gridDimGlb = ceil(slots_ / blockDimGlb.x);
    decode_gpu<<<gridDimGlb, blockDimGlb>>>(out.get(), plain_data_copy.get(), matrix_reps_index_map_.get(), slots_);

    PHANTOM_CHECK_CUDA(cudaMemcpy(destination.data(), out.get(), sizeof(uint64_t) * slots_, cudaMemcpyDeviceToHost));
}

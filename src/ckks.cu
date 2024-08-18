#include "ckks.h"
#include "fft.h"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

__global__ void bit_reverse_kernel(cuDoubleComplex *dst, cuDoubleComplex *src, uint64_t in_size,
                                   uint32_t log_n) {
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < in_size; tid += blockDim.x * gridDim.x) {
        dst[reverse_bits_uint32(tid, log_n)] = src[tid];
    }
}

PhantomCKKSEncoder::PhantomCKKSEncoder(const PhantomContext &context) {
    const auto &s = cudaStreamPerThread;

    auto &context_data = context.get_context_data(first_chain_index_);
    auto &parms = context_data.parms();
    std::size_t coeff_count = parms.poly_modulus_degree();

    if (parms.scheme() != scheme_type::ckks) {
        throw std::invalid_argument("unsupported scheme");
    }

    slots_ = coeff_count >> 1;
    uint32_t m = coeff_count << 1;
    uint32_t slots_half = slots_ >> 1;

    gpu_ckks_msg_vec_ = std::make_unique<DCKKSEncoderInfo>(coeff_count, s);

    // We need m powers of the primitive 2n-th root, m = 2n
    root_powers_.reserve(m);
    rotation_group_.reserve(slots_half);

    uint32_t gen = 5;
    uint32_t pos = 1; // Position in normal bit order
    for (size_t i = 0; i < slots_half; i++) {
        // Set the bit-reversed locations
        rotation_group_[i] = pos;

        // Next primitive root
        pos *= gen; // 5^i mod m
        pos &= (m - 1);
    }

    // Powers of the primitive 2n-th root have 4-fold symmetry
    if (m >= 8) {
        complex_roots_ = std::make_unique<util::ComplexRoots>(util::ComplexRoots(static_cast<size_t>(m)));
        for (size_t i = 0; i < m; i++) {
            root_powers_[i] = complex_roots_->get_root(i);
        }
    } else if (m == 4) {
        root_powers_[0] = {1, 0};
        root_powers_[1] = {0, 1};
        root_powers_[2] = {-1, 0};
        root_powers_[3] = {0, -1};
    }

    cudaMemcpyAsync(gpu_ckks_msg_vec_->twiddle(), root_powers_.data(), m * sizeof(cuDoubleComplex),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(gpu_ckks_msg_vec_->mul_group(), rotation_group_.data(), slots_half * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, s);
}

void PhantomCKKSEncoder::encode_internal(const PhantomContext &context, const std::vector<cuDoubleComplex> &values,
                                         size_t chain_index, double scale,
                                         PhantomPlaintext &destination, const cudaStream_t &stream) {
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context_data.gpu_rns_tool();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count = parms.poly_modulus_degree();
    size_t log_slot_count = arith::get_power_of_two(slots_);
    size_t values_size = values.size();

    if (values.empty()) {
        throw std::invalid_argument("Input vector is empty");
    } else if (values_size > slots_) {
        throw std::invalid_argument("Input vector exceeds max slots");
    }

    // Check that scale is positive and not too large
    if (scale <= 0 || (static_cast<int>(log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count())) {
        throw std::invalid_argument("scale out of bounds");
    }

    auto temp = make_cuda_auto_ptr<cuDoubleComplex>(values_size, stream);
    cudaMemcpyAsync(temp.get(), values.data(), values_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice,
                    stream);

    cudaMemsetAsync(gpu_ckks_msg_vec_->in(), 0, slots_ * sizeof(cuDoubleComplex), stream);

    size_t gridDimGlb = std::ceil((float) values_size / (float) blockDimGlb.x);
    bit_reverse_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            gpu_ckks_msg_vec_->in(), temp.get(), values_size, log_slot_count);

    double fix = scale / static_cast<double>(slots_);

    special_fft_backward(*gpu_ckks_msg_vec_, log_slot_count, fix, stream);

    // TODO: boundary check on GPU
    vector<cuDoubleComplex> temp2(slots_);
    cudaMemcpyAsync(temp2.data(), gpu_ckks_msg_vec_->in(), slots_ * sizeof(cuDoubleComplex),
                    cudaMemcpyDeviceToHost, stream);
    // explicit stream synchronize to avoid error
    cudaStreamSynchronize(stream);

    double max_coeff = 0;
    for (std::size_t i = 0; i < slots_; i++) {
        max_coeff = std::max(max_coeff, std::fabs(temp2[i].x));
    }
    for (std::size_t i = 0; i < slots_; i++) {
        max_coeff = std::max(max_coeff, std::fabs(temp2[i].y));
    }
    // Verify that the values are not too large to fit in coeff_modulus
    // Note that we have an extra + 1 for the sign bit
    // Don't compute logarithmis of numbers less than 1
    int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max(max_coeff, 1.0)))) + 1;

    if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count()) {
        throw std::invalid_argument("encoded values are too large");
    }

    rns_tool.base_Ql().decompose_array(destination.data(), gpu_ckks_msg_vec_->in(), coeff_count, max_coeff_bit_count,
                                       stream);

    nwt_2d_radix8_forward_inplace(destination.data(), context.gpu_rns_tables(), coeff_modulus_size, 0, stream);

    destination.chain_index_ = chain_index;
    destination.scale_ = scale;
}

void PhantomCKKSEncoder::decode_internal(const PhantomContext &context, const PhantomPlaintext &plain,
                                         std::vector<cuDoubleComplex> &destination, const cudaStream_t &stream) {
    auto &context_data = context.get_context_data(plain.chain_index_);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context_data.gpu_rns_tool();
    const size_t coeff_modulus_size = coeff_modulus.size();
    const size_t coeff_count = parms.poly_modulus_degree();
    size_t log_slot_count = arith::get_power_of_two(slots_);
    const size_t rns_poly_uint64_count = coeff_count * coeff_modulus_size;

    if (plain.scale() <= 0 ||
        (static_cast<int>(log2(plain.scale())) >= context_data.total_coeff_modulus_bit_count())) {
        throw std::invalid_argument("scale out of bounds");
    }

    auto upper_half_threshold = context_data.upper_half_threshold();
    int logn = arith::get_power_of_two(coeff_count);
    auto gpu_upper_half_threshold = make_cuda_auto_ptr<uint64_t>(upper_half_threshold.size(), stream);
    cudaMemcpyAsync(gpu_upper_half_threshold.get(), upper_half_threshold.data(),
                    upper_half_threshold.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

    cudaMemsetAsync(gpu_ckks_msg_vec_->in(), 0, slots_ * sizeof(cuDoubleComplex), stream);

    // Quick sanity check
    if ((logn < 0) || (coeff_count < POLY_MOD_DEGREE_MIN) || (coeff_count > POLY_MOD_DEGREE_MAX)) {
        throw std::logic_error("invalid parameters");
    }

    double inv_scale = double(1.0) / plain.scale();
    // Create mutable copy of input
    auto plain_copy = make_cuda_auto_ptr<uint64_t>(rns_poly_uint64_count, stream);
    cudaMemcpyAsync(plain_copy.get(), plain.data(), rns_poly_uint64_count * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                    stream);

    nwt_2d_radix8_backward_inplace(plain_copy.get(), context.gpu_rns_tables(), coeff_modulus_size, 0, stream);

    // CRT-compose the polynomial
    rns_tool.base_Ql().compose_array(gpu_ckks_msg_vec().in(), plain_copy.get(), gpu_upper_half_threshold.get(),
                                     inv_scale, coeff_count, stream);

    special_fft_forward(*gpu_ckks_msg_vec_, log_slot_count, stream);

    auto out = make_cuda_auto_ptr<cuDoubleComplex>(slots_, stream);
    size_t gridDimGlb = std::ceil((float) slots_ / (float) blockDimGlb.x);
    bit_reverse_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            out.get(), gpu_ckks_msg_vec_->in(), slots_, log_slot_count);

    destination.resize(slots_);
    cudaMemcpyAsync(destination.data(), out.get(), slots_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);

    // explicit synchronization in case user wants to use the result immediately
    cudaStreamSynchronize(stream);
}

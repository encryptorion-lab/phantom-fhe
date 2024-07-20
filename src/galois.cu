#include "galois.cuh"
#include "util/numth.h"

using namespace std;
using namespace phantom;
using namespace phantom::util;

__global__ void
apply_galois_ntt_permutation(uint64_t *dst, const uint64_t *src, const uint32_t *permutation_table, size_t poly_degree,
                             uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;

        dst[tid] = src[permutation_table[tid % poly_degree] + twr * poly_degree];
    }
}

__global__ void
apply_galois_permutation(uint64_t *dst, const uint64_t *src, const DModulus *modulus, const uint64_t *index_raws,
                         uint64_t poly_degree, uint64_t coeff_mod_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        uint64_t mod_value = modulus[twr].value();

        uint64_t idx = index_raws[tid % poly_degree];
        uint64_t res = src[tid];
        if (idx >= poly_degree) {
            // check if index_raw is odd or over times of coeff_count
            // for odd: x^k mod (x^n+1) = -x^(k-n)
            // for even: x^k mod (x^n+1) = x^(k-2n)
            int64_t non_zero = (res != 0);
            res = (mod_value - res) & static_cast<uint64_t>(-non_zero);
        }
        dst[(idx % poly_degree) + twr * poly_degree] = res;
    }
}

[[nodiscard]] std::uint32_t PhantomGaloisTool::get_elt_from_step(int step) const {
    auto n = static_cast<uint32_t>(coeff_count_);
    uint32_t m32 = n * 2;
    auto m = static_cast<uint64_t>(m32);

    if (step == 0) {
        return static_cast<uint32_t>(m - 1);
    } else {
        // Extract sign of steps. When steps is positive, the rotation
        // is to the left; when steps is negative, it is to the right.
        bool sign = step < 0;
        auto pos_step = static_cast<uint32_t>(abs(step));

        if (pos_step >= (n >> 1)) {
            throw invalid_argument("step count too large");
        }

        pos_step &= m32 - 1;
        if (sign) {
            step = static_cast<int>(n >> 1) - static_cast<int>(pos_step);
        } else {
            step = static_cast<int>(pos_step);
        }

        // Construct Galois element for row rotation
        auto gen = static_cast<uint64_t>(generator_);
        uint64_t galois_elt = 1;
        while (step--) {
            galois_elt *= gen;
            galois_elt &= m - 1;
        }
        return static_cast<uint32_t>(galois_elt);
    }
}

[[nodiscard]] std::vector<uint32_t> PhantomGaloisTool::get_elts_all() const {
    auto m = static_cast<uint32_t>(static_cast<uint64_t>(coeff_count_) << 1);
    vector<uint32_t> galois_elts{};

    // Generate Galois keys for m - 1 (X -> X^{m-1})
    // using for
    galois_elts.push_back(m - 1);

    // Generate Galois key for power of generator_ mod m (X -> X^{5^k}) and
    // for negative power of generator_ mod m (X -> X^{-5^k})
    uint64_t pos_power = generator_;
    uint64_t neg_power = 0;
    phantom::arith::try_invert_uint_mod(generator_, m, neg_power);
    for (int i = 0; i < coeff_count_power_ - 1; i++) {
        galois_elts.push_back(static_cast<uint32_t>(pos_power));
        pos_power *= pos_power;
        pos_power &= (m - 1);

        galois_elts.push_back(static_cast<uint32_t>(neg_power));
        neg_power *= neg_power;
        neg_power &= (m - 1);
    }

    return galois_elts;
}

void PhantomGaloisTool::apply_galois(uint64_t *operand, const DNTTTable &rns_table, size_t coeff_mod_size,
                                     size_t galois_elt_idx, uint64_t *result, const cudaStream_t &stream) {
    auto &index_raws = index_raw_tables_[galois_elt_idx];
    uint64_t gridDimGlb = coeff_count_ * coeff_mod_size / blockDimGlb.x;

    if (result == operand) {
        // use temporary memory to store the result to avoid permutation bug
        auto tmp_result = make_cuda_auto_ptr<uint64_t>(coeff_count_ * coeff_mod_size, stream);
        apply_galois_permutation<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                tmp_result.get(), operand, rns_table.modulus(), index_raws.get(), coeff_count_, coeff_mod_size);
        cudaMemcpyAsync(result, tmp_result.get(), coeff_count_ * coeff_mod_size * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        apply_galois_permutation<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                result, operand, rns_table.modulus(), index_raws.get(), coeff_count_, coeff_mod_size);
    }

}

void PhantomGaloisTool::apply_galois_ntt(uint64_t *operand, size_t coeff_mod_size, size_t galois_elt_idx,
                                         uint64_t *result, const cudaStream_t &stream) {
    auto table = permutation_tables_[galois_elt_idx].get();
    uint64_t gridDimGlb = coeff_count_ * coeff_mod_size / blockDimGlb.x;

    if (result == operand) {
        // use temporary memory to store the result to avoid permutation bug
        auto tmp_result = make_cuda_auto_ptr<uint64_t>(coeff_count_ * coeff_mod_size, stream);
        apply_galois_ntt_permutation<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                tmp_result.get(), operand, table, coeff_count_, coeff_mod_size);
        cudaMemcpyAsync(result, tmp_result.get(), coeff_count_ * coeff_mod_size * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        apply_galois_ntt_permutation<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                result, operand, table, coeff_count_, coeff_mod_size);
    }
}

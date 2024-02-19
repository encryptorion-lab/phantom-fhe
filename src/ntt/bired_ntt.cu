#include "butterfly.cuh"
#include "ntt.cuh"
#include "uintmodmath.cuh"

__global__ void forward_bired_ntt_kernel(uint64_t *inout, const uint64_t *W_n, const uint64_t *W_n1, const uint64_t *W_n2,
                                         const DModulus *modulus, size_t batch_size, size_t n) {

}

// __global__ void inverse_bired_ntt_kernel(uint64_t *inout, const uint64_t *twiddles, const uint64_t *twiddles_shoup,
//                                          const DModulus *modulus, size_t coeff_mod_size, size_t n) {}

// void forward_bired_ntt(uint64_t *inout, const uint64_t *twiddles, const uint64_t *twiddles_shoup,
//                        const DModulus *modulus, size_t dim, size_t coeff_modulus_size) {
//     const size_t per_block_memory = dim * sizeof(uint64_t);
//
//     forward_bired_ntt_kernel<<<coeff_modulus_size, dim / 2, per_block_memory>>>(inout, twiddles, twiddles_shoup, modulus,
//                                                                            coeff_modulus_size, dim);
// }

// void inverse_bired_ntt(uint64_t *inout, const uint64_t *itwiddles, const uint64_t *itwiddles_shoup,
//                        const DModulus *modulus, const uint64_t *scalar, const uint64_t *scalar_shoup, size_t dim,
//                        size_t coeff_modulus_size) {
//     const size_t per_block_memory = dim * sizeof(uint64_t);
//
//     inverse_bired_ntt_kernel<<<coeff_modulus_size, dim / 2, per_block_memory>>>(inout, itwiddles, itwiddles_shoup, modulus,
//                                                                            scalar, scalar_shoup, coeff_modulus_size,
//                                                                            dim);
// }

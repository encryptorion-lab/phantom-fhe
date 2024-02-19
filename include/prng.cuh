#ifndef _PRNG_
#define _PRNG_

#include <cstring>
#include <gputype.h>
#include <random>

/** Obtain the random from device
 * @param[out] buf The obtained random number, will be used as seed
 * @param[in] count The required length of the random
 */
void inline random_bytes(unsigned char *buf, size_t count) {
    std::random_device rd("/dev/urandom");
    size_t len = (count + 3) / 4;
    std::vector<uint32_t> temp;
    temp.reserve(len);
    while (len) {
        len--;
        temp[len] = rd();
    }
    CUDA_CHECK(cudaMemcpy(buf, temp.data(), count, cudaMemcpyHostToDevice));
}

__host__ __device__ inline uint32_t load_littleendian(const unsigned char *x) {
    return (uint32_t)(x[0]) | (((uint32_t)(x[1])) << 8) | (((uint32_t)(x[2])) << 16) | (((uint32_t)(x[3])) << 24);
}

__host__ __device__ inline uint32_t rotate(const uint32_t u, const int c) {
    return (u << c) | (u >> (32 - c));
}

/** Generate Pseudo random number with provided key and nonce
 * @param[out] out The generated random number
 * @param[in] outlen The required length of random number
 * @param[in] nonce The nonce used for PRNG
 * @param[in] key The key (seed) used for PRNG
 * @param[in] keylen The length of PRNG key
 */
void __device__ salsa20_gpu(uint32_t *out, const size_t outlen, const uint64_t nonce, const uint8_t *key,
                            const size_t keylen);

/** Generate a random ternary poly.
 * Notice: for x^i, the random is same for different coeff modulus.
 * @param[in] prng_seed The PRNG seed
 * @param[in] modulus
 * @param[in] poly_degree The degree of poly
 * @param[in] coeff_mod_size The number of coeff modulus
 */
__global__ void sample_ternary_poly(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                                    const uint64_t poly_degree, const uint64_t coeff_mod_size);

/** Generate a random uniform poly
 * Notice: generated random must be less than 0xFFFFFFFFFFFFFFFFULL - modulus - 1 to ensure uniform
 * @param[out] destination The buffer to stored the generated rand
 * @param[in] prng_seed The PRNG seed
 * @param[in] N The degree of poly
 * @param[in] total_size N * coeff_size
 * @param[in] coeff_index The index of the coeff modulus
 * @param[in] coeff_mod The corresponding coeff modulus
 */
__global__ void sample_uniform_poly(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                                    const uint64_t poly_degree, const uint64_t coeff_mod_size);

/** noise sampling has two methods:
 * 1. rounded Gaussian generation. supprots any std (max derivation = 6 * std), seal uses std::normal_distribution for processing.
 * 2. Centered Binomial Distribution. Only support std <= 3.2 (const global value). Seal defaults uses this.
 * Notice: in this version, we use Centered Binomial Distribution for noise.
 * Notice: for x^i, the random is same for different coeff modulus.
 * @param[in] prng_seed The PRNG seed
 * @param[in] modulus
 * @param[in] poly_degree The degree of poly
 * @param[in] coeff_mod_size The number of coeff modulus
 */
__global__ void sample_error_poly(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                                  const uint64_t poly_degree, const uint64_t coeff_mod_size);

#endif

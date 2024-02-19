#pragma once

// Bounds for bit-length of all coefficient moduli
constexpr int MOD_BIT_COUNT_MAX = 61;
constexpr int MOD_BIT_COUNT_MIN = 2;

// Bit-length of internally used coefficient moduli, e.g., auxiliary base in BFV
constexpr int INTERNAL_MOD_BIT_COUNT = 61;

// Bounds for bit-length of user-defined coefficient moduli
constexpr int USER_MOD_BIT_COUNT_MAX = 60;
constexpr int USER_MOD_BIT_COUNT_MIN = 2;

// Bounds for bit-length of the plaintext modulus
constexpr int PLAIN_MOD_BIT_COUNT_MAX = USER_MOD_BIT_COUNT_MAX;
constexpr int PLAIN_MOD_BIT_COUNT_MIN = USER_MOD_BIT_COUNT_MIN;

// Bounds for number of coefficient moduli (no hard requirement)
constexpr int COEFF_MOD_COUNT_MAX = 64;
constexpr int COEFF_MOD_COUNT_MIN = 1;

// Bounds for polynomial modulus degree (no hard requirement)
constexpr int POLY_MOD_DEGREE_MAX = 131072;
constexpr int POLY_MOD_DEGREE_MIN = 2;

// Upper bound on the size of a ciphertext (no hard requirement)
constexpr int CIPHERTEXT_SIZE_MAX = 16;
constexpr int CIPHERTEXT_SIZE_MIN = 2;

// How many pairs of modular integers can we multiply and accumulate in a 128-bit data type
#define MULTIPLY_ACCUMULATE_MOD_MAX (1 << (128 - (MOD_BIT_COUNT_MAX << 1)))

// CUDA support
#include <cuda_runtime_api.h>
#include <cuda.h>

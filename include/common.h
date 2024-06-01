#pragma once

//#include "util/defines.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
//#include <type_traits>
#include <utility>
#include <vector>
#include <stdio.h>

#define SWITCH_POINT 2048
#define MAX_THREAD_PER_BLOCK 1024
#define NTT_THREAD_PER_BLOCK 1024

namespace phantom {
    namespace util {
        // MUST dividible by poly_degree
        constexpr dim3 blockDimGlb(128);

        // ntt block threads, max = 2^16 * coeff_mod_size / (8*thread) as we do 8 pre-thread ntt
        constexpr dim3 gridDimNTT(4096);
        constexpr dim3 blockDimNTT(128);
        // radix-8 nwt, DO NOT change this variable !!!
        constexpr size_t per_thread_sample_size = 8;
        // per_block_pad can be 1, 2, 4, etc., per_block_pad * phase1_sample_size / 8 <= blockDim.x
        // per_block_pad = 4 seems to be most optimized
        // for n=4096, max pad = 8
        constexpr size_t per_block_pad = 4;

        constexpr double two_pow_64 = 18446744073709551616.0;

        constexpr int n_cuda_streams = 10;

        __device__ __constant__ constexpr double two_pow_64_dev = 18446744073709551616.0;

        __device__ __constant__ constexpr int bytes_per_uint64_dev = sizeof(std::uint64_t);

        __device__ __constant__ constexpr int bits_per_nibble_dev = 4;

        __device__ __constant__ constexpr int bits_per_byte_dev = 8;

        __device__ __constant__ constexpr int bits_per_uint64_dev = bytes_per_uint64_dev * bits_per_byte_dev;
    }
}

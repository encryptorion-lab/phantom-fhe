#include <nvbench/nvbench.cuh>

#include "phantom.h"
#include "util.cuh"

using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

template<size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
         size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void
load_data_to_shared_memory(uint64_t const *A, size_t lda, uint64_t const *B, size_t ldb,
                           uint64_t A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
                           uint64_t B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                           size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n, size_t k) {
    // Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS;
         ++load_idx) {
        size_t const A_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        uint64_t val = 0;
        if (A_row_idx < m && A_col_idx < k) {
            val = A[A_row_idx * lda + A_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.
        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
        A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = val;
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx = 0; load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) / NUM_THREADS;
         ++load_idx) {
        size_t const B_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X;
        size_t const B_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X;
        size_t const B_row_idx = thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx;
        size_t const B_col_idx = blockIdx.x * BLOCK_TILE_SIZE_X + B_thread_block_tile_col_idx;

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        uint64_t val = 0;
        if (B_row_idx < k && B_col_idx < n) {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.
        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
        B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }
}

template<size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
         size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_Y = 0U>
__device__ void load_data_to_shared_memory_transposed(
        uint64_t const *A, size_t lda, uint64_t const *B, size_t ldb,
        uint64_t A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y],
        uint64_t B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
        size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n, size_t k) {
    // Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS;
         ++load_idx) {
        size_t const A_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        uint64_t val = 0;
        if (A_row_idx < m && A_col_idx < k) {
            val = A[A_row_idx * lda + A_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.
        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
        A_thread_block_tile_transposed[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx] = val;
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx = 0; load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) / NUM_THREADS;
         ++load_idx) {
        size_t const B_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X;
        size_t const B_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X;
        size_t const B_row_idx = thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx;
        size_t const B_col_idx = blockIdx.x * BLOCK_TILE_SIZE_X + B_thread_block_tile_col_idx;

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        uint64_t val = 0;
        if (B_row_idx < k && B_col_idx < n) {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.
        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
        B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }
}

template<size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
         size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_Y = 0U, typename VECTOR_TYPE = int4>
__device__ void load_data_to_shared_memory_transposed_vectorized(
        uint64_t const *A, size_t lda, uint64_t const *B, size_t ldb,
        uint64_t A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y],
        uint64_t B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
        size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n, size_t k) {
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(uint64_t)};
    static_assert(sizeof(VECTOR_TYPE) % sizeof(uint64_t) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);

    // The skew size could affect the data alignment in shared memory when we use vectorized load.
    // We need to make sure the data alignment is correct.
    static_assert((BLOCK_TILE_SIZE_Y) * sizeof(uint64_t) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X) * sizeof(uint64_t) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y) * sizeof(uint64_t) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(uint64_t) % sizeof(VECTOR_TYPE) == 0U);

// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_Y * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS; ++load_idx) {
        size_t const A_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) /
                                                 VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) %
                                                 VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        int4 A_row_vector_vals{0, 0, 0, 0};
        if (A_row_idx < m && A_col_idx < k) {
            A_row_vector_vals = *reinterpret_cast<int4 const *>(&A[A_row_idx * lda + A_col_idx]);
        }
        if (A_col_idx + NUM_VECTOR_UNITS > k) {
            // Number of invalid elements in the last vector.
            size_t const num_invalid_elements{A_col_idx + NUM_VECTOR_UNITS - k};
            // Mask out the invalid elements.
            uint64_t *const A_row_vector_vals_ptr{reinterpret_cast<uint64_t *>(&A_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i) {
                A_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] = static_cast<uint64_t>(0);
            }
        }
        // If this is true, the following if can be removed.
        // static_assert(VECTORIZED_BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y %
        // NUM_THREADS ==
        //               0U);
        if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y && A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K) {
            for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i) {
                A_thread_block_tile_transposed[A_thread_block_tile_col_idx + i][A_thread_block_tile_row_idx] =
                        reinterpret_cast<uint64_t const *>(&A_row_vector_vals)[i];
            }
        }
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) / NUM_THREADS; ++load_idx) {
        size_t const B_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) /
                                                 VECTORIZED_BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) %
                                                 VECTORIZED_BLOCK_TILE_SIZE_X * NUM_VECTOR_UNITS};
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X + B_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        int4 B_row_vector_vals{0, 0, 0, 0};
        if (B_row_idx < k && B_col_idx < n) {
            B_row_vector_vals = *reinterpret_cast<int4 const *>(&B[B_row_idx * ldb + B_col_idx]);
        }
        if (B_col_idx + NUM_VECTOR_UNITS > n) {
            // Number of invalid elements in the last vector.
            size_t const num_invalid_elements{B_col_idx + NUM_VECTOR_UNITS - n};
            // Mask out the invalid elements.
            uint64_t *const B_row_vector_vals_ptr{reinterpret_cast<uint64_t *>(&B_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i) {
                B_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] = static_cast<uint64_t>(0);
            }
        }
        // If this is true, the following if can be removed.
        // static_assert(VECTORIZED_BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K %
        // NUM_THREADS ==
        //               0U);
        if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K && B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X) {
            *reinterpret_cast<int4 *>(&B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx]) =
                    B_row_vector_vals;
        }
    }
}

__global__ void gemm_v00(uint64_t *batched_C, size_t ldc, const uint64_t *batched_A, size_t lda,
                         const uint64_t *batched_B, size_t ldb, const DModulus *mod_chain, size_t m, size_t n,
                         size_t k) {
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute C[C_row_idx, C_col_idx] = A[C_row_idx, :] * B[:, C_col_idx].
    uint128_t sum = {0, 0};
    const uint64_t *A = batched_A + blockIdx.z * m * k;
    const uint64_t *B = batched_B + blockIdx.z * k * n;

    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        const uint64_t a = A[C_row_idx * lda + k_idx];
        const uint64_t b = B[k_idx * ldb + C_col_idx];
        sum.lo += a * b;
        sum.hi += __umul64hi(a, b);
    }
    uint64_t *C = batched_C + blockIdx.z * m * n;
    C[C_row_idx * ldc + C_col_idx] =
            barrett_reduce_uint128_uint64(sum, mod_chain[blockIdx.z].value(), mod_chain[blockIdx.z].const_ratio());
}

__global__ void gemm_v01(uint64_t *batched_C, size_t ldc, const uint64_t *batched_A, size_t lda,
                         const uint64_t *batched_B, size_t ldb, const DModulus *mod_chain, size_t m, size_t n,
                         size_t k) {
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute C[C_row_idx, C_col_idx] = A[C_row_idx, :] * B[:, C_col_idx].
    uint128_t sum = {0, 0};
    const uint64_t *A = batched_A + blockIdx.z * m * k;
    const uint64_t *B = batched_B + blockIdx.z * k * n;

    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        const uint64_t a = A[C_row_idx * lda + k_idx];
        const uint64_t b = B[k_idx * ldb + C_col_idx];
        sum.lo += a * b;
        sum.hi += __umul64hi(a, b);
    }
    uint64_t *C = batched_C + blockIdx.z * m * n;
    C[C_row_idx * ldc + C_col_idx] =
            barrett_reduce_uint128_uint64(sum, mod_chain[blockIdx.z].value(), mod_chain[blockIdx.z].const_ratio());
}

template<size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(uint64_t *batched_C, size_t ldc, const uint64_t *batched_A, size_t lda,
                         const uint64_t *batched_B, size_t ldb, const DModulus *mod_chain, size_t m, size_t n,
                         size_t k) {
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS = BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y;
    size_t const thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Compute the row and column of C that this thread is responsible for.
    size_t const C_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const C_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ uint64_t A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ uint64_t B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles = k / BLOCK_TILE_SIZE_K;

    // Each thread compute C[C_row_idx, C_col_idx] = A[C_row_idx, :] * B[:, C_col_idx].
    uint128_t sum = {0, 0};
    const uint64_t *A = batched_A + blockIdx.z * m * k;
    const uint64_t *B = batched_B + blockIdx.z * k * n;

    for (size_t thread_block_tile_idx = 0; thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx) {
        load_data_to_shared_memory<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS>(
                A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile, thread_block_tile_idx, thread_linear_idx, m,
                n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i = 0; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
            // Doing this results in 2 TOPS.
            // Suppose blockDim.x = blockDim.y = 32.
            // Effectively, for a warp, in one iteration, we read the value from
            // A_thread_block_tile at the same location on the shared memory
            // resulting in a broadcast, we also read 32 values that have no
            // bank conflicts from B_thread_block_tile. Even with that, all the
            // values have to be read from the shared memory and consequence is
            // the shared memory instruction runs very intensively just to
            // compute a small number of values using simple arithmetic
            // instructions, which is not efficient.
            const uint64_t a = A_thread_block_tile[threadIdx.y][k_i];
            const uint64_t b = B_thread_block_tile[k_i][threadIdx.x];
            sum.lo += a * b;
            sum.hi += __umul64hi(a, b);
        }
        __syncthreads();
    }
    uint64_t *C = batched_C + blockIdx.z * m * n;
    C[C_row_idx * ldc + C_col_idx] =
            barrett_reduce_uint128_uint64(sum, mod_chain[blockIdx.z].value(), mod_chain[blockIdx.z].const_ratio());
}

template<size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v03(uint64_t *batched_C, size_t ldc, const uint64_t *batched_A, size_t lda,
                         const uint64_t *batched_B, size_t ldb, const DModulus *mod_chain, size_t m, size_t n,
                         size_t k) {
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS = BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y;
    size_t const thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ uint64_t A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ uint64_t B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles = k / BLOCK_TILE_SIZE_K;

    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X]
    uint128_t C_thread_results[THREAD_TILE_SIZE_Y] = {0, 0};
    const uint64_t *A = batched_A + blockIdx.z * m * k;
    const uint64_t *B = batched_B + blockIdx.z * k * n;
    for (size_t thread_block_tile_idx = 0; thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx) {
        load_data_to_shared_memory<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS>(
                A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile, thread_block_tile_idx, thread_linear_idx, m,
                n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i = 0; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
            size_t const B_thread_block_tile_row_idx{k_i};
            // B_val is cached in the register to alleviate the pressure on the
            // shared memory access.
            uint64_t const B_val{
                    B_thread_block_tile[B_thread_block_tile_row_idx][thread_linear_idx % BLOCK_TILE_SIZE_X]};
#pragma unroll
            for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx) {
                size_t const A_thread_block_tile_row_idx{thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y +
                                                         thread_tile_row_idx};
                size_t const A_thread_block_tile_col_idx{k_i};
                uint64_t const A_val{A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx]};
                C_thread_results[thread_tile_row_idx].lo += A_val * B_val;
                C_thread_results[thread_tile_row_idx].hi += __umul64hi(A_val, B_val);
            }
        }
        __syncthreads();
    }

    // Write the results to DRAM.
    uint64_t *C = batched_C + blockIdx.z * m * n;
#pragma unroll
    for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx) {
        size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y + thread_tile_row_idx};
        size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X + thread_linear_idx % BLOCK_TILE_SIZE_X};
        if (C_row_idx < m && C_col_idx < n) {
            C[C_row_idx * ldc + C_col_idx] =
                    barrett_reduce_uint128_uint64(C_thread_results[thread_tile_row_idx], mod_chain[blockIdx.z].value(),
                                                  mod_chain[blockIdx.z].const_ratio());
        }
    }
}

template<size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t m, size_t n, size_t k>
__global__ void gemm_v04(uint64_t *batched_C, size_t ldc, const uint64_t *batched_A, size_t lda,
                         const uint64_t *batched_B, size_t ldb, const DModulus *mod_chain) {
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS = BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y;
    size_t const thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ uint64_t A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ uint64_t B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t constexpr num_thread_block_tiles = k / BLOCK_TILE_SIZE_K;

    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X]
    uint128_t C_thread_result = {0, 0};
    const uint64_t *A = batched_A + blockIdx.z * m * k;
    const uint64_t *B = batched_B + blockIdx.z * k * n;
#pragma unroll
    for (size_t thread_block_tile_idx = 0; thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx) {
#pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS;
             ++load_idx) {
            size_t const A_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
            size_t const A_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};
            static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
            A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] =
                    A[A_row_idx * lda + A_col_idx];
        }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
        for (size_t load_idx = 0; load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) / NUM_THREADS;
             ++load_idx) {
            size_t const B_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X;
            size_t const B_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X;
            size_t const B_row_idx = thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx;
            size_t const B_col_idx = blockIdx.x * BLOCK_TILE_SIZE_X + B_thread_block_tile_col_idx;
            static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
            B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] =
                    B[B_row_idx * ldb + B_col_idx];
        }
        __syncthreads();

#pragma unroll
        for (size_t k_i = 0; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
            // B_val is cached in the register to alleviate the pressure on the
            // shared memory access.
            uint64_t const B_val{B_thread_block_tile[k_i][thread_linear_idx % BLOCK_TILE_SIZE_X]};
            uint64_t const A_val{A_thread_block_tile[thread_linear_idx / BLOCK_TILE_SIZE_X][k_i]};
            C_thread_result.lo += A_val * B_val;
            C_thread_result.hi += __umul64hi(A_val, B_val);
        }
        __syncthreads();
    }

    // Write the results to DRAM.
    uint64_t *C = batched_C + blockIdx.z * m * n;
    size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + thread_linear_idx / BLOCK_TILE_SIZE_X};
    size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X + thread_linear_idx % BLOCK_TILE_SIZE_X};
    C[C_row_idx * ldc + C_col_idx] = barrett_reduce_uint128_uint64(C_thread_result, mod_chain[blockIdx.z].value(),
                                                                   mod_chain[blockIdx.z].const_ratio());
}

void launch_gemm_kernel_v00(uint64_t *batched_C, size_t ldc, const uint64_t *batched_A, size_t lda,
                            const uint64_t *batched_B, size_t ldb, const DModulus *mod_chain, size_t m, size_t n,
                            size_t k, size_t batch_size, cudaStream_t stream) {
    dim3 constexpr block_dim{32U, 4U, 1U};
    dim3 const grid_dim{static_cast<unsigned int>(m) / block_dim.x, static_cast<unsigned int>(n) / block_dim.y,
                        static_cast<unsigned int>(batch_size)};
    gemm_v00<<<grid_dim, block_dim, 0U, stream>>>(batched_C, ldc, batched_A, lda, batched_B, ldb, mod_chain, m, n, k);
    PHANTOM_CHECK_CUDA_LAST();
}

void launch_gemm_kernel_v01(uint64_t *batched_C, size_t ldc, const uint64_t *batched_A, size_t lda,
                            const uint64_t *batched_B, size_t ldb, const DModulus *mod_chain, size_t m, size_t n,
                            size_t k, size_t batch_size, cudaStream_t stream) {
    dim3 constexpr block_dim{32U, 4U, 1U};
    dim3 const grid_dim{static_cast<unsigned int>(n) / block_dim.x, static_cast<unsigned int>(m) / block_dim.y,
                        static_cast<unsigned int>(batch_size)};
    gemm_v01<<<grid_dim, block_dim, 0U, stream>>>(batched_C, ldc, batched_A, lda, batched_B, ldb, mod_chain, m, n, k);
    PHANTOM_CHECK_CUDA_LAST();
}

void launch_gemm_kernel_v02(uint64_t *batched_C, size_t ldc, const uint64_t *batched_A, size_t lda,
                            const uint64_t *batched_B, size_t ldb, const DModulus *mod_chain, size_t m, size_t n,
                            size_t k, size_t batch_size, cudaStream_t stream) {
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{16U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{8U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    dim3 const grid_dim{static_cast<unsigned int>(n) / block_dim.x, static_cast<unsigned int>(m) / block_dim.y,
                        static_cast<unsigned int>(batch_size)};
    size_t smem_size = sizeof(uint64_t) * BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K +
                       sizeof(uint64_t) * BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X;
    gemm_v02<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K><<<grid_dim, block_dim, smem_size, stream>>>(
            batched_C, ldc, batched_A, lda, batched_B, ldb, mod_chain, m, n, k);
    PHANTOM_CHECK_CUDA_LAST();
}

void launch_gemm_kernel_v03(uint64_t *batched_C, size_t ldc, const uint64_t *batched_A, size_t lda,
                            const uint64_t *batched_B, size_t ldb, const DModulus *mod_chain, size_t m, size_t n,
                            size_t k, size_t batch_size, cudaStream_t stream) {
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{16U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{8U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    // Each thread computes THREAD_TILE_SIZE_Y values of C.
    constexpr unsigned int THREAD_TILE_SIZE_Y{1U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{static_cast<unsigned int>(n) / BLOCK_TILE_SIZE_X,
                        static_cast<unsigned int>(m) / BLOCK_TILE_SIZE_Y, static_cast<unsigned int>(batch_size)};
    size_t smem_size = sizeof(uint64_t) * BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K +
                       sizeof(uint64_t) * BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X;
    gemm_v03<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y>
            <<<grid_dim, block_dim, smem_size, stream>>>(batched_C, ldc, batched_A, lda, batched_B, ldb, mod_chain, m,
                                                         n, k);
    PHANTOM_CHECK_CUDA_LAST();
}

template<size_t m, size_t n, size_t k>
void launch_gemm_kernel_v04(uint64_t *batched_C, size_t ldc, const uint64_t *batched_A, size_t lda,
                            const uint64_t *batched_B, size_t ldb, const DModulus *mod_chain, size_t batch_size,
                            cudaStream_t stream) {
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{16U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{8U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    // Each thread computes THREAD_TILE_SIZE_Y values of C.
    constexpr unsigned int NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    dim3 constexpr block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{static_cast<unsigned int>(n) / BLOCK_TILE_SIZE_X,
                        static_cast<unsigned int>(m) / BLOCK_TILE_SIZE_Y, static_cast<unsigned int>(batch_size)};
    size_t smem_size = sizeof(uint64_t) * BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K +
                       sizeof(uint64_t) * BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X;
    gemm_v04<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, m, n, k>
            <<<grid_dim, block_dim, smem_size, stream>>>(batched_C, ldc, batched_A, lda, batched_B, ldb, mod_chain);
    PHANTOM_CHECK_CUDA_LAST();
}

void bench_gemm(nvbench::state &state) {
    const auto version = state.get_int64("Version");
    const auto batch_size = state.get_int64("Batch Size");
    const auto dim = state.get_int64("Dimension");

    constexpr size_t m{256U};
    constexpr size_t k{256U};
    constexpr size_t n{256U};

    constexpr size_t lda{k};
    constexpr size_t ldb{n};
    constexpr size_t ldc{n};

    state.collect_dram_throughput();
    // state.collect_l1_hit_rates();
    // state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();

    // Provide throughput information:
    state.add_element_count(batch_size * m * n, "NumElements");
    state.add_global_memory_reads<nvbench::uint64_t>(batch_size * m * k + batch_size * k * n, "ReadDataSize");
    state.add_global_memory_writes<nvbench::uint64_t>(batch_size * m * n, "WriteDataSize");

    // generate modulus in host
    const auto h_modulus_chain = CoeffModulus::Create(dim, std::vector<int>(batch_size, 50));
    // copy modulus to device
    DModulus *d_modulus_chain;
    cudaMalloc(&d_modulus_chain, batch_size * sizeof(DModulus));
    for (size_t i = 0; i < batch_size; i++) {
        d_modulus_chain[i].set(h_modulus_chain[i].value(), h_modulus_chain[i].const_ratio()[0],
                               h_modulus_chain[i].const_ratio()[1]);
    }

    // create input and output
    uint64_t *h_batched_A = new uint64_t[batch_size * m * k];
    uint64_t *h_batched_A_shoup = new uint64_t[batch_size * m * k];
    for (size_t mod_idx = 0; mod_idx < batch_size; mod_idx++) {
        for (size_t i = 0; i < m * k; i++) {
            h_batched_A[mod_idx * m * k + i] = 1;
            h_batched_A_shoup[mod_idx * m * k + i] = compute_shoup(1, h_modulus_chain[mod_idx].value());
        }
    }

    uint64_t *h_batched_B = new uint64_t[batch_size * k * n];
    for (size_t i = 0; i < batch_size * k * n; i++) {
        h_batched_B[i] = 1;
    }

    uint64_t *h_batched_C = new uint64_t[batch_size * m * n];
    for (size_t i = 0; i < batch_size * m * n; i++) {
        h_batched_C[i] = 0;
    }

    uint64_t *d_batched_A;
    cudaMalloc(&d_batched_A, batch_size * m * k * sizeof(uint64_t));
    cudaMemcpy(d_batched_A, h_batched_A, batch_size * m * k * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t *d_batched_A_shoup;
    cudaMalloc(&d_batched_A_shoup, batch_size * m * k * sizeof(uint64_t));
    cudaMemcpy(d_batched_A_shoup, h_batched_A_shoup, batch_size * m * k * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t *d_batched_B;
    cudaMalloc(&d_batched_B, batch_size * k * n * sizeof(uint64_t));
    cudaMemcpy(d_batched_B, h_batched_B, batch_size * k * n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t *d_batched_C;
    cudaMalloc(&d_batched_C, batch_size * m * n * sizeof(uint64_t));

    if (version == 0) {
        state.exec([d_batched_C, &ldc, d_batched_A, &lda, d_batched_B, &ldb, d_modulus_chain, &m, &n, &k,
                    &batch_size](nvbench::launch &launch) {
            launch_gemm_kernel_v00(d_batched_C, ldc, d_batched_A, lda, d_batched_B, ldb, d_modulus_chain, m, n, k,
                                   batch_size, nullptr);
        });
    }
    else if (version == 1) {
        state.exec([d_batched_C, &ldc, d_batched_A, &lda, d_batched_B, &ldb, d_modulus_chain, &m, &n, &k,
                    &batch_size](nvbench::launch &launch) {
            launch_gemm_kernel_v01(d_batched_C, ldc, d_batched_A, lda, d_batched_B, ldb, d_modulus_chain, m, n, k,
                                   batch_size, nullptr);
        });
    }
    else if (version == 2) {
        state.exec([d_batched_C, &ldc, d_batched_A, &lda, d_batched_B, &ldb, d_modulus_chain, &m, &n, &k,
                    &batch_size](nvbench::launch &launch) {
            launch_gemm_kernel_v02(d_batched_C, ldc, d_batched_A, lda, d_batched_B, ldb, d_modulus_chain, m, n, k,
                                   batch_size, nullptr);
        });
    }
    else if (version == 3) {
        state.exec([d_batched_C, &ldc, d_batched_A, &lda, d_batched_B, &ldb, d_modulus_chain, &m, &n, &k,
                    &batch_size](nvbench::launch &launch) {
            launch_gemm_kernel_v03(d_batched_C, ldc, d_batched_A, lda, d_batched_B, ldb, d_modulus_chain, m, n, k,
                                   batch_size, nullptr);
        });
    }
    else if (version == 4) {
        state.exec([d_batched_C, &ldc, d_batched_A, &lda, d_batched_B, &ldb, d_modulus_chain, &m, &n, &k,
                    &batch_size](nvbench::launch &launch) {
            launch_gemm_kernel_v04<m, n, k>(d_batched_C, ldc, d_batched_A, lda, d_batched_B, ldb, d_modulus_chain,
                                            batch_size, nullptr);
        });
    }

    cudaMemcpy(h_batched_C, d_batched_C, batch_size * m * n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < batch_size * m * n; i++) {
        if (h_batched_C[i] != k) {
            std::cout << i << " " << h_batched_C[i] << std::endl;
            throw std::logic_error("Error");
        }
    }

    cudaFree(d_modulus_chain);
    cudaFree(d_batched_A);
    cudaFree(d_batched_A_shoup);
    cudaFree(d_batched_B);
    cudaFree(d_batched_C);
    delete[] h_batched_A;
    delete[] h_batched_A_shoup;
    delete[] h_batched_B;
    delete[] h_batched_C;
}

NVBENCH_BENCH(bench_gemm)
        .add_int64_axis("Version", {0, 1, 2, 3, 4, 5})
        .add_int64_axis("Batch Size", {30})
        .add_int64_power_of_two_axis("Dimension", {17})
        .set_timeout(10); // Limit to one second per measurement.

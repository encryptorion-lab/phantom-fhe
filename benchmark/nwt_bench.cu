#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>

#include "phantom.h"

using namespace phantom;

void fnwt_1d_bench(nvbench::state &state) {
    const auto batch_size = state.get_int64("Batch Size");
    const auto dim = state.get_int64("Dimension");

    state.collect_dram_throughput();
    // state.collect_l1_hit_rates();
    // state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();

    // Provide throughput information:
    state.add_element_count(batch_size * dim, "NumElements");
    state.add_global_memory_reads<nvbench::int64_t>(batch_size * dim, "ReadDataSize");
    state.add_global_memory_writes<nvbench::int64_t>(batch_size * dim, "WriteDataSize");

    uint64_t *inout;
    cudaMalloc(&inout, batch_size * dim * sizeof(uint64_t));
    DModulus *modulus;
    cudaMalloc(&modulus, batch_size * sizeof(DModulus));
    uint64_t *twiddles;
    cudaMalloc(&twiddles, batch_size * dim * sizeof(uint64_t));
    uint64_t *twiddles_shoup;
    cudaMalloc(&twiddles_shoup, batch_size * dim * sizeof(uint64_t));

    state.exec([&batch_size, &dim, inout, twiddles, twiddles_shoup, modulus](nvbench::launch &launch) {
            // fnwt_1d(inout, twiddles, twiddles_shoup, modulus, dim, batch_size, 0);
            fnwt_1d_opt(inout, twiddles, twiddles_shoup, modulus, dim, batch_size, 0);
        }
    );
}

NVBENCH_BENCH(fnwt_1d_bench)
    .add_int64_axis("Batch Size", {256 * 30})
    .add_int64_power_of_two_axis("Dimension", nvbench::range(8, 11, 1))
    .set_timeout(1); // Limit to one second per measurement.

void inwt_1d_bench(nvbench::state &state) {
    const auto batch_size = state.get_int64("Batch Size");
    const auto dim = state.get_int64("Dimension");

    state.collect_dram_throughput();
    // state.collect_l1_hit_rates();
    // state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();

    // Provide throughput information:
    state.add_element_count(batch_size * dim, "NumElements");
    state.add_global_memory_reads<nvbench::int64_t>(batch_size * dim, "ReadDataSize");
    state.add_global_memory_writes<nvbench::int64_t>(batch_size * dim, "WriteDataSize");

    uint64_t *inout;
    cudaMalloc(&inout, batch_size * dim * sizeof(uint64_t));
    DModulus *modulus;
    cudaMalloc(&modulus, batch_size * sizeof(DModulus));
    uint64_t *twiddles;
    cudaMalloc(&twiddles, batch_size * dim * sizeof(uint64_t));
    uint64_t *twiddles_shoup;
    cudaMalloc(&twiddles_shoup, batch_size * dim * sizeof(uint64_t));
    uint64_t *scalar, *scalar_shoup;
    cudaMalloc(&scalar, batch_size * sizeof(uint64_t));
    cudaMalloc(&scalar_shoup, batch_size * sizeof(uint64_t));

    state.exec([&batch_size, &dim, inout, twiddles, twiddles_shoup, modulus, scalar, scalar_shoup](nvbench::launch &launch) {
            inwt_1d(inout, twiddles, twiddles_shoup, modulus, scalar, scalar_shoup, dim, batch_size, 0);
        }
    );
}

// NVBENCH_BENCH(inwt_1d_bench)
//     .add_int64_axis("Batch Size", {1})
//     .add_int64_power_of_two_axis("Dimension", nvbench::range(8, 11, 1))
//     .set_timeout(1); // Limit to one second per measurement.

void fnwt_2d_bench(nvbench::state &state) {
    const auto batch_size = state.get_int64("Batch Size");
    const auto dim = state.get_int64("Dimension");

    state.collect_dram_throughput();
    // state.collect_l1_hit_rates();
    // state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();

    // Provide throughput information:
    state.add_element_count(batch_size * dim, "NumElements");
    state.add_global_memory_reads<nvbench::int64_t>(batch_size * dim, "ReadDataSize");
    state.add_global_memory_writes<nvbench::int64_t>(batch_size * dim, "WriteDataSize");

    uint64_t *inout;
    cudaMalloc(&inout, batch_size * dim * sizeof(uint64_t));

    DNTTTable ntt_tables;
    ntt_tables.init(dim, batch_size);

    state.exec([&batch_size, &dim, inout, ntt_tables](nvbench::launch &launch) {
            nwt_2d_radix8_forward_inplace(inout, ntt_tables, batch_size, 0);
        }
    );
}

NVBENCH_BENCH(fnwt_2d_bench)
    .add_int64_axis("Batch Size", {30})
    .add_int64_power_of_two_axis("Dimension", nvbench::range(16, 16, 1))
    .set_timeout(1); // Limit to one second per measurement.

void inwt_2d_bench(nvbench::state &state) {
    const auto batch_size = state.get_int64("Batch Size");
    const auto dim = state.get_int64("Dimension");

    state.collect_dram_throughput();
    // state.collect_l1_hit_rates();
    // state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();

    // Provide throughput information:
    state.add_element_count(batch_size * dim, "NumElements");
    state.add_global_memory_reads<nvbench::int64_t>(batch_size * dim, "ReadDataSize");
    state.add_global_memory_writes<nvbench::int64_t>(batch_size * dim, "WriteDataSize");

    uint64_t *inout;
    cudaMalloc(&inout, batch_size * dim * sizeof(uint64_t));

    DNTTTable ntt_tables;
    ntt_tables.init(dim, batch_size);

    state.exec([&batch_size, &dim, inout, ntt_tables](nvbench::launch &launch) {
            nwt_2d_radix8_backward_inplace(inout, ntt_tables, batch_size, 0);
        }
    );
}

// NVBENCH_BENCH(inwt_2d_bench)
//     .add_int64_axis("Batch Size", {1, 2, 4, 8, 16, 32})
//     .add_int64_power_of_two_axis("Dimension", nvbench::range(12, 17, 1))
//     .set_timeout(1); // Limit to one second per measurement.

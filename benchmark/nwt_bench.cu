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

    phantom::util::cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();

    auto inout = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto modulus = phantom::util::make_cuda_auto_ptr<DModulus>(batch_size, s);
    auto twiddles = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto twiddles_shoup = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(s));
    state.exec([&batch_size, &dim, &inout, &twiddles, &twiddles_shoup, &modulus, &s](nvbench::launch &launch) {
                   // fnwt_1d(inout, twiddles, twiddles_shoup, modulus, dim, batch_size, 0);
                   fnwt_1d_opt(inout.get(), twiddles.get(), twiddles_shoup.get(), modulus.get(), dim, batch_size, 0, s);
               }
    );
}

NVBENCH_BENCH(fnwt_1d_bench)
        .add_int64_axis("Batch Size", {1, 10, 100, 1000})
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

    phantom::util::cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();

    auto inout = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto modulus = phantom::util::make_cuda_auto_ptr<DModulus>(batch_size, s);
    auto twiddles = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto twiddles_shoup = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);

    auto scalar = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size, s);
    auto scalar_shoup = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size, s);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(s));
    state.exec([&batch_size, &dim, &inout, &twiddles, &twiddles_shoup, &modulus, &scalar, &scalar_shoup, &s](
                       nvbench::launch &launch) {
                   inwt_1d(inout.get(), twiddles.get(), twiddles_shoup.get(), modulus.get(), scalar.get(), scalar_shoup.get(), dim,
                           batch_size, 0, s);
               }
    );
}

NVBENCH_BENCH(inwt_1d_bench)
        .add_int64_axis("Batch Size", {1, 10, 100, 1000})
        .add_int64_power_of_two_axis("Dimension", nvbench::range(8, 11, 1))
        .set_timeout(1); // Limit to one second per measurement.

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

    phantom::util::cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();

    auto inout = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);

    DNTTTable ntt_tables;
    ntt_tables.init(dim, batch_size, s);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(s));
    state.exec([&batch_size, &dim, &inout, &ntt_tables, &s](nvbench::launch &launch) {
                   nwt_2d_radix8_forward_inplace(inout.get(), ntt_tables, batch_size, 0, s);
               }
    );
}

NVBENCH_BENCH(fnwt_2d_bench)
        .add_int64_axis("Batch Size", {1, 10, 100, 1000})
        .add_int64_power_of_two_axis("Dimension", nvbench::range(12, 17, 1))
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

    phantom::util::cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();

    auto inout = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);

    DNTTTable ntt_tables;
    ntt_tables.init(dim, batch_size, s);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(s));
    state.exec([&batch_size, &dim, &inout, &ntt_tables, &s](nvbench::launch &launch) {
                   nwt_2d_radix8_backward_inplace(inout.get(), ntt_tables, batch_size, 0, s);
               }
    );
}

NVBENCH_BENCH(inwt_2d_bench)
        .add_int64_axis("Batch Size", {1, 10, 100, 1000})
        .add_int64_power_of_two_axis("Dimension", nvbench::range(12, 17, 1))
        .set_timeout(1); // Limit to one second per measurement.

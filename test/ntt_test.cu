#include "phantom.h"

using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

void test_nwt_1d(size_t log_dim, size_t batch_size) {
    phantom::util::cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();
    size_t dim = 1 << log_dim;

    // generate modulus in host
    const auto h_modulus = CoeffModulus::Create(dim, std::vector<int>(batch_size, 50));
    // copy modulus to device
    auto modulus = phantom::util::make_cuda_auto_ptr<DModulus>(batch_size, s);
    for (size_t i = 0; i < batch_size; i++) {
        modulus.get()[i].set(h_modulus[i].value(), h_modulus[i].const_ratio()[0], h_modulus[i].const_ratio()[1]);
    }

    auto twiddles = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto twiddles_shoup = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto itwiddles = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto itwiddles_shoup = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    auto d_n_inv_mod_q = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size, s);
    auto d_n_inv_mod_q_shoup = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size, s);

    for (size_t i = 0; i < batch_size; i++) {
        // generate twiddles in host
        auto h_ntt_table = NTT(log_dim, h_modulus[i]);
        // copy twiddles to device
        cudaMemcpyAsync(twiddles.get() + i * dim, h_ntt_table.get_from_root_powers().data(),
                        dim * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(twiddles_shoup.get() + i * dim, h_ntt_table.get_from_root_powers_shoup().data(),
                        dim * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(itwiddles.get() + i * dim, h_ntt_table.get_from_inv_root_powers().data(),
                        dim * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(itwiddles_shoup.get() + i * dim, h_ntt_table.get_from_inv_root_powers_shoup().data(),
                        dim * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_n_inv_mod_q.get() + i, &h_ntt_table.inv_degree_modulo(), sizeof(uint64_t),
                        cudaMemcpyHostToDevice,
                        s);
        cudaMemcpyAsync(d_n_inv_mod_q_shoup.get() + i, &h_ntt_table.inv_degree_modulo_shoup(), sizeof(uint64_t),
                        cudaMemcpyHostToDevice, s);
    }

    // create input
    auto h_idata = std::make_unique<uint64_t[]>(batch_size * dim);
    for (size_t i = 0; i < batch_size * dim; i++) {
        h_idata.get()[i] = 1;
    }

    auto d_data = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    cudaMemcpyAsync(d_data.get(), h_idata.get(), batch_size * dim * sizeof(uint64_t), cudaMemcpyHostToDevice, s);

    // fnwt_1d(d_data, twiddles, twiddles_shoup, modulus, dim, batch_size, 0);
    fnwt_1d_opt(d_data.get(), twiddles.get(), twiddles_shoup.get(), modulus.get(), dim, batch_size, 0, s);
    inwt_1d_opt(d_data.get(), itwiddles.get(), itwiddles_shoup.get(), modulus.get(), d_n_inv_mod_q.get(),
                d_n_inv_mod_q_shoup.get(), dim, batch_size, 0, s);

    auto h_odata = std::make_unique<uint64_t[]>(batch_size * dim);
    cudaMemcpyAsync(h_odata.get(), d_data.get(), batch_size * dim * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    for (size_t i = 0; i < batch_size * dim; i++) {
        if (h_idata.get()[i] != h_odata.get()[i]) {
            std::cout << i << " " << h_idata.get()[i] << " != " << h_idata.get()[i] << std::endl;
            throw std::logic_error("Error");
        }
    }
}

void test_nwt_2d(size_t log_dim, size_t batch_size) {
    phantom::util::cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();

    size_t dim = 1 << log_dim;

    // generate modulus in host
    const auto h_modulus = CoeffModulus::Create(dim, std::vector<int>(batch_size, 50));
    // copy modulus to device
    auto modulus = phantom::util::make_cuda_auto_ptr<DModulus>(batch_size, s);

    for (size_t i = 0; i < batch_size; i++) {
        modulus.get()[i].set(h_modulus[i].value(), h_modulus[i].const_ratio()[0], h_modulus[i].const_ratio()[1]);
    }

    DNTTTable d_ntt_tables;
    d_ntt_tables.init(dim, batch_size, s);

    for (size_t i = 0; i < batch_size; i++) {
        // generate twiddles in host
        auto h_ntt_table = NTT(log_dim, h_modulus[i]);
        d_ntt_tables.set(&modulus.get()[i],
                         h_ntt_table.get_from_root_powers().data(),
                         h_ntt_table.get_from_root_powers_shoup().data(),
                         h_ntt_table.get_from_inv_root_powers().data(),
                         h_ntt_table.get_from_inv_root_powers_shoup().data(),
                         h_ntt_table.inv_degree_modulo(),
                         h_ntt_table.inv_degree_modulo_shoup(),
                         i, s);
    }

    // create input
    auto h_data = std::make_unique<uint64_t[]>(batch_size * dim);
    for (size_t i = 0; i < batch_size * dim; i++) {
        h_data[i] = 2;
    }

    auto d_data = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    cudaMemcpyAsync(d_data.get(), h_data.get(), batch_size * dim * sizeof(uint64_t), cudaMemcpyHostToDevice, s);

    nwt_2d_radix8_forward_inplace(d_data.get(), d_ntt_tables, batch_size, 0, s);
    nwt_2d_radix8_backward_inplace(d_data.get(), d_ntt_tables, batch_size, 0, s);

    cudaMemcpyAsync(h_data.get(), d_data.get(), batch_size * dim * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    for (size_t i = 0; i < batch_size * dim; i++) {
        if (h_data.get()[i] != 2) {
            std::cout << i << " " << h_data.get()[i] << std::endl;
            throw std::logic_error("Error");
        }
    }
}

int main() {
    // single batch
    test_nwt_1d(8, 1);
    test_nwt_1d(9, 1);
    test_nwt_1d(10, 1);
    test_nwt_1d(11, 1);

    test_nwt_2d(12, 1);
    test_nwt_2d(13, 1);
    test_nwt_2d(14, 1);
    test_nwt_2d(15, 1);
    test_nwt_2d(16, 1);
    test_nwt_2d(17, 1);

    // multiple batches
    test_nwt_1d(8, 10);
    test_nwt_1d(9, 10);
    test_nwt_1d(10, 10);
    test_nwt_1d(11, 10);

    test_nwt_2d(12, 10);
    test_nwt_2d(13, 10);
    test_nwt_2d(14, 10);
    test_nwt_2d(15, 10);
    test_nwt_2d(16, 10);
    test_nwt_2d(17, 10);
    return 0;
}

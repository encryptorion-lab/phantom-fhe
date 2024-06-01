#include "phantom.h"

using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

void test_nwt_1d(size_t log_dim, size_t batch_size) {
    size_t dim = 1 << log_dim;

    // generate modulus in host
    const auto h_modulus = CoeffModulus::Create(dim, std::vector<int>(batch_size, 50));
    // copy modulus to device
    DModulus *modulus;
    cudaMalloc(&modulus, batch_size * sizeof(DModulus));
    for (size_t i = 0; i < batch_size; i++) {
        modulus[i].set(h_modulus[i].value(), h_modulus[i].const_ratio()[0], h_modulus[i].const_ratio()[1]);
    }

    uint64_t *twiddles, *twiddles_shoup;
    cudaMalloc(&twiddles, batch_size * dim * sizeof(uint64_t));
    cudaMalloc(&twiddles_shoup, batch_size * dim * sizeof(uint64_t));
    uint64_t *itwiddles, *itwiddles_shoup;
    cudaMalloc(&itwiddles, batch_size * dim * sizeof(uint64_t));
    cudaMalloc(&itwiddles_shoup, batch_size * dim * sizeof(uint64_t));
    uint64_t *d_n_inv_mod_q, *d_n_inv_mod_q_shoup;
    cudaMalloc(&d_n_inv_mod_q, batch_size * sizeof(uint64_t));
    cudaMalloc(&d_n_inv_mod_q_shoup, batch_size * sizeof(uint64_t));

    for (size_t i = 0; i < batch_size; i++) {
        // generate twiddles in host
        auto h_ntt_table = NTT(log_dim, h_modulus[i]);
        // copy twiddles to device
        cudaMemcpy(twiddles + i * dim, h_ntt_table.get_from_root_powers().data(),
                   dim * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(twiddles_shoup + i * dim, h_ntt_table.get_from_root_powers_shoup().data(),
                   dim * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(itwiddles + i * dim, h_ntt_table.get_from_inv_root_powers().data(),
                   dim * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(itwiddles_shoup + i * dim, h_ntt_table.get_from_inv_root_powers_shoup().data(),
                   dim * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_inv_mod_q + i, &h_ntt_table.inv_degree_modulo(), sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_inv_mod_q_shoup + i, &h_ntt_table.inv_degree_modulo_shoup(), sizeof(uint64_t),
                   cudaMemcpyHostToDevice);
    }

    // create input
    uint64_t *h_idata = new uint64_t[batch_size * dim];
    for (size_t i = 0; i < batch_size * dim; i++) {
        h_idata[i] = 1;
    }

    uint64_t *d_data;
    cudaMalloc(&d_data, batch_size * dim * sizeof(uint64_t));
    cudaMemcpy(d_data, h_idata, batch_size * dim * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // fnwt_1d(d_data, twiddles, twiddles_shoup, modulus, dim, batch_size, 0);
    fnwt_1d_opt(d_data, twiddles, twiddles_shoup, modulus, dim, batch_size, 0);
    inwt_1d_opt(d_data, itwiddles, itwiddles_shoup, modulus, d_n_inv_mod_q, d_n_inv_mod_q_shoup, dim, batch_size, 0);

    uint64_t *h_odata = new uint64_t[batch_size * dim];
    cudaMemcpy(h_odata, d_data, batch_size * dim * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < batch_size * dim; i++) {
        if (h_idata[i] != h_odata[i]) {
            std::cout << i << " " << h_idata[i] << " != " << h_idata[i] << std::endl;
            throw std::logic_error("Error");
        }
    }
    cudaFree(modulus);
    cudaFree(twiddles);
    cudaFree(twiddles_shoup);
    cudaFree(itwiddles);
    cudaFree(itwiddles_shoup);
    cudaFree(d_n_inv_mod_q);
    cudaFree(d_n_inv_mod_q_shoup);
    cudaFree(d_data);
    delete[] h_idata;
    delete[] h_odata;
}

void test_nwt_2d(size_t log_dim, size_t batch_size) {
    size_t dim = 1 << log_dim;

    // generate modulus in host
    const auto h_modulus = CoeffModulus::Create(dim, std::vector<int>(batch_size, 50));
    // copy modulus to device
    DModulus *modulus;
    cudaMalloc(&modulus, batch_size * sizeof(DModulus));
    for (size_t i = 0; i < batch_size; i++) {
        modulus[i].set(h_modulus[i].value(), h_modulus[i].const_ratio()[0], h_modulus[i].const_ratio()[1]);
    }

    DNTTTable d_ntt_tables;
    d_ntt_tables.init(dim, batch_size);

    for (size_t i = 0; i < batch_size; i++) {
        // generate twiddles in host
        auto h_ntt_table = NTT(log_dim, h_modulus[i]);
        d_ntt_tables.set(&modulus[i],
                         h_ntt_table.get_from_root_powers().data(),
                         h_ntt_table.get_from_root_powers_shoup().data(),
                         h_ntt_table.get_from_inv_root_powers().data(),
                         h_ntt_table.get_from_inv_root_powers_shoup().data(),
                         h_ntt_table.inv_degree_modulo(),
                         h_ntt_table.inv_degree_modulo_shoup(),
                         i);
    }

    // create input
    uint64_t *h_data = new uint64_t[batch_size * dim];
    for (size_t i = 0; i < batch_size * dim; i++) {
        h_data[i] = 2;
    }

    uint64_t *d_data;
    cudaMalloc(&d_data, batch_size * dim * sizeof(uint64_t));
    cudaMemcpy(d_data, h_data, batch_size * dim * sizeof(uint64_t), cudaMemcpyHostToDevice);

    nwt_2d_radix8_forward_inplace(d_data, d_ntt_tables, batch_size, 0);
    nwt_2d_radix8_backward_inplace(d_data, d_ntt_tables, batch_size, 0);

    cudaMemcpy(h_data, d_data, batch_size * dim * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < batch_size * dim; i++) {
        if (h_data[i] != 2) {
            std::cout << i << " " << h_data[i] << std::endl;
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

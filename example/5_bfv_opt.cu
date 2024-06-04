#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>

#include "example.h"
#include "phantom.h"
#include "util.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

void example_bfv_encrypt_decrypt_hps() {
    std::cout << std::endl
            << "---------------Testing BFV sym Enc & Dec-----------------------" << std::endl;
    size_t poly_modulus_degree = 8192;
    EncryptionParameters parms(scheme_type::bfv);
    std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {35, 35, 35, 36}));
    parms.set_mul_tech(mul_tech_type::hps);
    auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
    parms.set_plain_modulus(plainModulus);

    PhantomContext context(parms);
    print_parameters(context);

    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);

    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);

    PhantomCiphertext cipher(context);

    PhantomBatchEncoder batchEncoder(context);
    size_t slot_count = batchEncoder.slots_;
    std::vector<int64_t> pod_matrix(slot_count, 0ULL);
    for (size_t idx = 0; idx < slot_count; idx++)
        pod_matrix[idx] = idx;

    PhantomPlaintext plain_matrix(context);
    batchEncoder.encode(context, pod_matrix, plain_matrix);

    secret_key.encrypt_symmetric(context, plain_matrix, cipher, false);
    auto noise_budget = secret_key.invariant_noise_budget(context, cipher);
    cout << "cipher noise budget is: " << noise_budget << endl;
    PhantomCiphertext cipher_copy(cipher);
    PhantomPlaintext plain(context);

    size_t times = 1;
    auto start = chrono::high_resolution_clock::now();
    for (size_t idx = 0; idx < times; idx++) {
        secret_key.decrypt(context, cipher_copy, plain);
    }
    auto finish = chrono::high_resolution_clock::now();
    auto microseconds = chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "bfv decrypt time is us: " << microseconds.count() / times << std::endl;

    secret_key.decrypt(context, cipher, plain);

    std::vector<int64_t> res;
    batchEncoder.decode(context, plain, res);

    cout << "res[0] is: " << res[0] << endl;
    cout << "res[1] is: " << res[1] << endl;

    bool correctness = true;
    for (size_t idx = 0; idx < slot_count; idx++)
        correctness &= res[idx] == pod_matrix[idx];
    if (!correctness)
        throw std::logic_error("Error in encrypt_symmetric & decrypt");
}

void example_bfv_encrypt_decrypt_hps_asym() {
    std::cout << std::endl
            << "------------Testing BFV asym Enc & Dec-------------" << std::endl;
    size_t poly_modulus_degree = 8192;
    EncryptionParameters parms(scheme_type::bfv);
    std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {35, 35, 35, 36}));
    parms.set_mul_tech(mul_tech_type::hps);
    auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
    parms.set_plain_modulus(plainModulus);

    PhantomContext context(parms);
    print_parameters(context);

    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);

    /*start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; i++)
        secret_key.gen_publickey(context, public_key);
    finish = chrono::high_resolution_clock::now();
    microseconds = chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "public key generation time is us: " << microseconds.count() / 100 << std::endl;
    */
    PhantomCiphertext cipher(context);

    PhantomBatchEncoder batchEncoder(context);
    size_t slot_count = batchEncoder.slots_;
    std::vector<int64_t> pod_matrix(slot_count, 0ULL);
    for (size_t idx = 0; idx < slot_count; idx++) {
        pod_matrix[idx] = rand() % parms.plain_modulus().value();
    }
    PhantomPlaintext plain_matrix(context);
    batchEncoder.encode(context, pod_matrix, plain_matrix);

    public_key.encrypt_asymmetric(context, plain_matrix, cipher, false);
    auto noise_budget = secret_key.invariant_noise_budget(context, cipher);
    cout << "cipher noise budget is: " << noise_budget << endl;
    PhantomPlaintext plain(context);
    secret_key.decrypt(context, cipher, plain);

    std::vector<int64_t> res;
    batchEncoder.decode(context, plain, res);

    bool correctness = true;
    for (size_t idx = 0; idx < slot_count; idx++)
        correctness &= res[idx] == pod_matrix[idx];
    if (!correctness)
        throw std::logic_error("Error in encrypt_asymmetric & decrypt");
}

void example_bfv_hybrid_key_switching() {
    std::cout << std::endl
            << "------------Testing BFV hybrid key-switching-------------" << std::endl;

    EncryptionParameters parms(scheme_type::bfv);
    size_t poly_modulus_degree = 8192;

    size_t alpha = 2;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                 {
                                                     50, 50,
                                                     51, 51
                                                 }));
    parms.set_special_modulus_size(alpha);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
    parms.set_plain_modulus(plainModulus);
    PhantomContext context(parms);

    print_parameters(context);

    PhantomSecretKey secret_key(parms);
    PhantomPublicKey public_key(context);
    PhantomCiphertext sym_cipher(context);
    PhantomCiphertext asym_cipher(context);
    PhantomBatchEncoder batchEncoder(context);
    PhantomPlaintext plain_matrix(context);
    PhantomPlaintext dec_plain(context);
    PhantomPlaintext dec_asym_plain(context);
    PhantomRelinKey relin_keys(context);

    std::vector<int64_t> dec_res;
    size_t slot_count = batchEncoder.slots_;
    std::vector<int64_t> pod_matrix(slot_count, 0ULL);
    for (size_t idx = 0; idx < slot_count; idx++)
        pod_matrix[idx] = idx;

    secret_key.gen_secretkey(context);
    secret_key.gen_relinkey(context, relin_keys);
    batchEncoder.encode(context, pod_matrix, plain_matrix);
    secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
    PhantomCiphertext sym_cipher_copy(sym_cipher);
    // PhantomCiphertext destination(context);
    multiply_inplace(context, sym_cipher, sym_cipher_copy);
    relinearize_inplace(context, sym_cipher, relin_keys);
    secret_key.decrypt(context, sym_cipher, dec_plain);
    batchEncoder.decode(context, dec_plain, dec_res);

    bool correctness = true;
    size_t threshold = std::log2(plainModulus.value());
    for (size_t idx = 0; idx < threshold; idx++)
        correctness &= dec_res[idx] == idx * idx;
    if (!correctness)
        throw std::logic_error("Error in mul symmetric");

    secret_key.gen_publickey(context, public_key);
    public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
    PhantomCiphertext asym_cipher_copy(asym_cipher);
    // PhantomCiphertext destination2(context);
    multiply_inplace(context, asym_cipher, asym_cipher_copy);
    relinearize_inplace(context, asym_cipher, relin_keys);
    secret_key.decrypt(context, asym_cipher, dec_asym_plain);

    batchEncoder.decode(context, dec_asym_plain, dec_res);

    for (size_t idx = 0; idx < threshold; idx++)
        correctness &= dec_res[idx] == idx * idx;
    if (!correctness)
        throw std::logic_error("Error in mul asymmetric");
}

void bfv_multiply_correctness(mul_tech_type mul_tech) {
    EncryptionParameters parms(scheme_type::bfv);
    size_t poly_modulus_degree = 16384;
    const std::vector<int> modulus_bits = {50, 50, 50, 50, 50, 50, 50, 51};
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, modulus_bits));
    parms.set_mul_tech(mul_tech);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
    parms.set_plain_modulus(plainModulus);
    PhantomContext context(parms);

    //    print_parameters(context);

    PhantomSecretKey secret_key(parms);
    PhantomPublicKey public_key(context);
    PhantomCiphertext sym_cipher(context);
    PhantomCiphertext asym_cipher(context);
    PhantomBatchEncoder batchEncoder(context);
    PhantomPlaintext plain_matrix(context);
    PhantomPlaintext dec_plain(context);
    PhantomPlaintext dec_asym_plain(context);
    PhantomRelinKey relin_keys(context);

    secret_key.gen_secretkey(context);
    secret_key.gen_relinkey(context, relin_keys);

    std::vector<int64_t> dec_res;
    size_t slot_count = batchEncoder.slots_;
    std::vector<int64_t> pod_matrix(slot_count, 0ULL);
    for (size_t idx = 0; idx < slot_count; idx++)
        pod_matrix[idx] = idx;

    batchEncoder.encode(context, pod_matrix, plain_matrix);
    secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);

    PhantomCiphertext sym_cipher_copy(sym_cipher);
    multiply_and_relin_inplace(context, sym_cipher_copy, sym_cipher, relin_keys);
    multiply_and_relin_inplace(context, sym_cipher_copy, sym_cipher, relin_keys);
    multiply_and_relin_inplace(context, sym_cipher_copy, sym_cipher, relin_keys);
    // multiply_and_relin_inplace(context, sym_cipher_copy, sym_cipher, relin_keys);
    // multiply_and_relin_inplace(context, sym_cipher_copy, sym_cipher, relin_keys);
    // multiply_and_relin_inplace(context, sym_cipher_copy, sym_cipher, relin_keys);
    // multiply_and_relin_inplace(context, sym_cipher_copy, sym_cipher, relin_keys);

    secret_key.decrypt(context, sym_cipher_copy, dec_plain);
    batchEncoder.decode(context, dec_plain, dec_res);

    bool correctness = true;
    for (size_t idx = 0; idx < 5; idx++)
        correctness &= dec_res[idx] == idx * idx * idx * idx;
    if (!correctness) {
        std::cout << "dec_res[0]: " << dec_res[0] << std::endl;
        throw std::logic_error("Error in mul symmetric");
    }

    secret_key.gen_publickey(context, public_key);
    public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
    PhantomCiphertext asym_cipher_copy(asym_cipher);
    multiply_and_relin_inplace(context, asym_cipher_copy, asym_cipher, relin_keys);
    multiply_and_relin_inplace(context, asym_cipher_copy, asym_cipher, relin_keys);
    multiply_and_relin_inplace(context, asym_cipher_copy, asym_cipher, relin_keys);
    multiply_and_relin_inplace(context, asym_cipher_copy, asym_cipher, relin_keys);
    multiply_and_relin_inplace(context, asym_cipher_copy, asym_cipher, relin_keys);
    multiply_and_relin_inplace(context, asym_cipher_copy, asym_cipher, relin_keys);
    multiply_and_relin_inplace(context, asym_cipher_copy, asym_cipher, relin_keys);

    secret_key.decrypt(context, asym_cipher_copy, dec_asym_plain);

    batchEncoder.decode(context, dec_asym_plain, dec_res);

    for (size_t idx = 0; idx < 5; idx++)
        correctness &= dec_res[idx] == idx * idx * idx * idx * idx * idx * idx * idx;
    if (!correctness) {
        std::cout << "dec_res[0]: " << dec_res[0] << std::endl;
        throw std::logic_error("Error in mul asymmetric");
    }
}

void example_bfv_multiply_correctness() {
    std::cout << "-------------bfv_multiply_correctness BEHZ------------------------" << std::endl;
    bfv_multiply_correctness(mul_tech_type::behz);

    std::cout << "-------------bfv_multiply_correctness HPS-------------------------" << std::endl;
    bfv_multiply_correctness(mul_tech_type::hps);

    std::cout << "-------------bfv_multiply_correctness HPSoverQ--------------------" << std::endl;
    bfv_multiply_correctness(mul_tech_type::hps_overq);

    std::cout << "-------------bfv_multiply_correctness HPSoverQLeveled-------------" << std::endl;
    bfv_multiply_correctness(mul_tech_type::hps_overq_leveled);
}

static void
bfv_multiply_bench(mul_tech_type mul_tech, size_t poly_modulus_degree, const std::vector<int> &modulus_bits,
                   size_t alpha) {
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, modulus_bits));
    parms.set_mul_tech(mul_tech);
    parms.set_special_modulus_size(alpha);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
    parms.set_plain_modulus(plainModulus);
    PhantomContext context(parms);

    //    print_parameters(context);

    PhantomSecretKey secret_key(parms);
    PhantomPublicKey public_key(context);
    PhantomCiphertext sym_cipher(context);
    PhantomCiphertext asym_cipher(context);
    PhantomBatchEncoder batchEncoder(context);
    PhantomPlaintext plain_matrix(context);
    PhantomPlaintext dec_plain(context);
    PhantomPlaintext dec_asym_plain(context);
    PhantomRelinKey relin_keys(context);

    std::vector<int64_t> dec_res;
    size_t slot_count = batchEncoder.slots_;
    std::vector<int64_t> pod_matrix(slot_count, 0ULL);
    for (size_t idx = 0; idx < slot_count; idx++)
        pod_matrix[idx] = idx;

    secret_key.gen_secretkey(context);

    if (context.using_keyswitching()) {
        secret_key.gen_relinkey(context, relin_keys);
    }
    batchEncoder.encode(context, pod_matrix, plain_matrix);
    secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);

    constexpr size_t n_tests = 100;

    size_t max_mult_depth = 20;

    //    if (poly_modulus_degree == 8192) {
    //        max_mult_depth = 2;
    //    } else if (poly_modulus_degree == 16384) {
    //        max_mult_depth = 6;
    //    } else if (poly_modulus_degree == 32768) {
    //        max_mult_depth = 13;
    //    }

    PhantomCiphertext sym_cipher_copy(sym_cipher);

    for (size_t mult_depth = 0; mult_depth < max_mult_depth; mult_depth++) {
        CUDATimer timer("mult&relin");
        for (size_t idx = 0; idx < n_tests; idx++) {
            PhantomCiphertext sym_cipher_copy2(sym_cipher_copy);
            timer.start();
            multiply_and_relin_inplace(context, sym_cipher_copy2, sym_cipher, relin_keys);
            timer.stop();
        }
        multiply_and_relin_inplace(context, sym_cipher_copy, sym_cipher, relin_keys);
    }

    CUDATimer timer_dec("decrypt");
    for (size_t idx = 0; idx < n_tests; idx++) {
        timer_dec.start();
        secret_key.decrypt(context, sym_cipher_copy, dec_plain);
        timer_dec.stop();
    }

    batchEncoder.decode(context, dec_plain, dec_res);

    bool correctness = true;
    for (size_t idx = 0; idx < 2; idx++) {
        int64_t result = idx;
        for (size_t mult_depth = 0; mult_depth < max_mult_depth; mult_depth++)
            result *= idx;
        correctness &= dec_res[idx] == result;
    }
    if (!correctness) {
        std::cout << "dec_res[0]: " << dec_res[0] << std::endl;
        throw std::logic_error("Error in mul symmetric");
    }
}

void example_bfv_multiply_benchmark() {
    struct param {
        size_t n;
        std::vector<int> modulus_bits;
        size_t alpha = 1;
    };

    std::vector<param> params;

    // For comparison with OpenFHE
    //    params.push_back({8192, std::vector<int>(3, 60)});
    //    params.push_back({16384, std::vector<int>(6, 60)});
    //    params.push_back({16384, std::vector<int>(6, 60), 2});
    //    params.push_back({32768, std::vector<int>(12, 60)});
    //    params.push_back({32768, std::vector<int>(12, 60), 3});

    // For comparison with Turkey22 on 3060Ti
    //    params.emplace_back(4096, std::vector<int>{36, 36, 37});
    //    params.push_back({8192, std::vector<int>{43, 43, 44, 44, 44}});
    //    params.push_back({16384, std::vector<int>{48, 48, 48, 49, 49, 49, 49, 49, 49}});
    //    params.push_back({32768, std::vector<int>{55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56}});

    // make graph of leveled
    params.push_back({32768, std::vector<int>(14, 60)});

    for (auto &param: params) {
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "poly_modulus_degree: " << param.n << std::endl;
        std::cout << "modulus_bits: [";
        for (auto &coeff: param.modulus_bits)
            std::cout << coeff << ", ";
        std::cout << "]" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "alpha: " << param.alpha << std::endl;

        std::cout << "-------------BEHZ------------------------" << std::endl;
        bfv_multiply_bench(mul_tech_type::behz, param.n, param.modulus_bits, param.alpha);

        std::cout << "-------------HPS-------------------------" << std::endl;
        bfv_multiply_bench(mul_tech_type::hps, param.n, param.modulus_bits, param.alpha);

        std::cout << "-------------HPSoverQ--------------------" << std::endl;
        bfv_multiply_bench(mul_tech_type::hps_overq, param.n, param.modulus_bits, param.alpha);

        std::cout << "-------------HPSoverQLeveled-------------" << std::endl;
        bfv_multiply_bench(mul_tech_type::hps_overq_leveled, param.n, param.modulus_bits, param.alpha);
    }
}

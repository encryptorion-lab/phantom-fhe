#include "bench_utils.cuh"
#include "util.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

void bgv_performance_test(EncryptionParameters &parms) {
    PhantomContext context(parms);
    print_parameters(context);
    cout << endl;

    auto &context_data = context.get_context_data(context.get_first_index());
    auto &first_parms = context_data.parms();
    auto &plain_modulus = first_parms.plain_modulus();

    cuda_stream_wrapper stream;

    print_timer_banner();

    auto count = 100;

    {
        CUDATimer timer("gen_secretkey", *global_variables::default_stream);
        for (auto i = 0; i < count; i++) {
            timer.start();
            PhantomSecretKey secret_key(context);
            timer.stop();
        }
    }

    PhantomSecretKey secret_key(context);

    {
        CUDATimer timer("gen_publickey", *global_variables::default_stream);
        for (auto i = 0; i < count; i++) {
            timer.start();
            PhantomPublicKey public_key = secret_key.gen_publickey(context);
            timer.stop();
        }
    }

    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    // Generate relinearization keys
    {
        CUDATimer timer("gen_relinkey", *global_variables::default_stream);
        for (auto i = 0; i < count; i++) {
            timer.start();
            PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
            timer.stop();
        }
    }

    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    /*
    Generate Galois keys. In larger examples the Galois keys can use a lot of
    memory, which can be a problem in constrained systems. The user should
    try some larger runs of the test and observe their effect on the
    memory pool allocation size. The key generation can also take a long time,
    as can be observed from the print-out.
    */
    PhantomGaloisKey gal_keys = secret_key.create_galois_keys(context);

    PhantomBatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();
    random_device rd;

    PhantomPlaintext plain;

    /*
    Populate a vector of values to batch.
    */
    vector<uint64_t> pod_vector;
    for (size_t i = 0; i < slot_count; i++) {
        pod_vector.push_back(static_cast<int64_t>(plain_modulus.reduce(rd())));
    }

    /*
    [Batching]
    There is nothing unusual here. We batch our random plaintext matrix
    into the polynomial. Note how the plaintext we create is of the exactly
    right size so unnecessary reallocations are avoided.
    */
    {
        CUDATimer timer("encode", stream);
        for (auto i = 0; i < count; i++) {
            timer.start();
            batch_encoder.encode(context, pod_vector, plain, stream);
            timer.stop();
        }
    }

    /*
    [Unbatching]
    We unbatch what we just batched.
    */
    vector<uint64_t> pod_vector2(slot_count);
    {
        CUDATimer timer("decode", stream);
        for (auto i = 0; i < count; i++) {
            timer.start();
            batch_encoder.decode(context, plain, pod_vector2, stream);
            timer.stop();
        }
    }

    if (pod_vector2 != pod_vector) {
        throw runtime_error("Batch/unbatch failed. Something is wrong.");
    }

    /*
    [Encryption]
    We make sure our ciphertext is already allocated and large enough
    to hold the encryption with these encryption parameters. We encrypt
    our random batched matrix here.
    */
    PhantomCiphertext encrypted;
    {
        CUDATimer timer("encrypt_asymmetric", stream);
        for (auto i = 0; i < count; i++) {
            timer.start();
            public_key.encrypt_asymmetric(context, plain, encrypted, stream);
            timer.stop();
        }
    }

    /*
    [Decryption]
    We decrypt what we just encrypted.
    */
    {
        CUDATimer timer("decrypt", stream);
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.decrypt(context, encrypted, plain, stream);
            timer.stop();
        }
    }

    // homomorphic operations

    PhantomPlaintext plain1;
    PhantomPlaintext plain2;
    PhantomCiphertext encrypted1;
    batch_encoder.encode(context, vector<uint64_t>(slot_count, 1), plain1, stream);
    public_key.encrypt_asymmetric(context, plain1, encrypted1, stream);
    PhantomCiphertext encrypted2;
    batch_encoder.encode(context, vector<uint64_t>(slot_count, 1), plain2, stream);
    public_key.encrypt_asymmetric(context, plain2, encrypted2, stream);

    /*
    [Add]
    We create two ciphertexts and perform a few additions with them.
    */
    {
        CUDATimer timer("add_inplace", stream);
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            timer.start();
            add_inplace(context, tmp_ct, encrypted2, stream);
            timer.stop();
        }
    }

    /*
    [Add Plain]
    */
    {
        CUDATimer timer("add_plain", stream);
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            timer.start();
            add_plain_inplace(context, tmp_ct, plain, stream);
            timer.stop();
        }
    }

    /*
    [Multiply]
    We multiply two ciphertexts. Since the size of the result will be 3,
    and will overwrite the first argument, we reserve first enough memory
    to avoid reallocating during multiplication.
    */
    {
        CUDATimer timer("multiply", stream);
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            timer.start();
            multiply_inplace(context, tmp_ct, encrypted2, stream);
            relinearize_inplace(context, tmp_ct, relin_keys, stream);
            timer.stop();
        }
    }

    /*
    [Multiply Plain]
    We multiply a ciphertext with a random plaintext. Recall that
    multiply_plain does not change the size of the ciphertext, so we use
    encrypted2 here.
    */
    {
        CUDATimer timer("multiply_plain", stream);
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            timer.start();
            multiply_plain_inplace(context, tmp_ct, plain, stream);
            timer.stop();
        }
    }

    /*
    [Rotate Rows One Step]
    We rotate matrix rows by one step left and measure the time.
    */
    {
        CUDATimer timer("rotate_rows_inplace_one_step", stream);
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            timer.start();
            rotate_rows_inplace(context, tmp_ct, 1, gal_keys, stream);
            timer.stop();
        }
    }
}

int main() {
    print_example_banner("BGV Performance Test with Degrees: 4096, 8192, 16384, 32768, and 65536");
    auto scheme = scheme_type::bgv;

    std::vector<int> galois_steps = {1};

    // 2 ^ 12

    {
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 12);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {35, 35,

                                                                           38}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }

    // 2 ^ 13

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 13);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {54, 54, 54,

                                                                           56}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 13);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {36, 36, 36, 36, 36,

                                                                           38}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 13);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {36, 36, 36, 36,

                                                                           37, 37}));
        parms.set_special_modulus_size(2);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }

    // 2 ^ 14

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 14);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {54, 54, 54, 54, 54, 54, 54,

                                                                           60}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 14);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,

                                                                           42}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 14);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {36, 36, 36, 36, 36, 36, 36, 36,

                                                                           37, 37, 38, 38}));
        parms.set_special_modulus_size(4);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }

    // 2 ^ 15

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 15);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(
                CoeffModulus::Create(poly_modulus_degree, {55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,

                                                           56}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 15);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(
                CoeffModulus::Create(poly_modulus_degree, {36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
                                                           36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,

                                                           53}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 15);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                     {36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,

                                                      38, 38, 38, 38, 38, 38, 38, 39}));
        parms.set_special_modulus_size(8);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }

    // 2 ^ 16

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 16);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                     {55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56,
                                                      56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,

                                                      56}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 16);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                     {36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37,
                                                      37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                                                      37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,

                                                      48}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 16);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                     {36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
                                                      36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,

                                                      39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40}));
        parms.set_special_modulus_size(16);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bgv_performance_test(parms);
    }

    /*

    // 2 ^ 17

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 17);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                        56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,

                        58}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 17);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
                        38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,

                        40}));
        parms.set_special_modulus_size(1);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bfv_bgv_performance_test(parms);
    }

    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 17);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                        37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,

                        37, 37, 37, 37, 37, 37, 37, 37,
                        38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
                        38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38}));
        parms.set_special_modulus_size(32);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        bfv_bgv_performance_test(parms);
    }
     */

    return 0;
}

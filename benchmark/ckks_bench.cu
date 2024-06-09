#include "bench_utils.cuh"
#include "util.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

void ckks_performance_test(EncryptionParameters &parms, double scale) {
    PhantomContext context(parms);
    print_parameters(context);
    cout << endl;

    auto &context_data = context.get_context_data(context.get_first_index());
    auto &first_parms = context_data.parms();
    size_t poly_modulus_degree = first_parms.poly_modulus_degree();

    const auto &stream = context.get_cuda_stream(0);

    print_timer_banner();

    auto count = 100;

    PhantomSecretKey secret_key;
    {
        CUDATimer timer("gen_secretkey");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.gen_secretkey(context);
            timer.stop();
        }
    }

    PhantomPublicKey public_key;
    {
        CUDATimer timer("gen_publickey");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.gen_publickey(context, public_key);
            timer.stop();
        }
    }

    PhantomRelinKey relin_keys;
    PhantomGaloisKey gal_keys;
    // Generate relinearization keys
    {
        CUDATimer timer("gen_relinkey");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.gen_relinkey(context, relin_keys);
            timer.stop();
        }
    }

    secret_key.create_galois_keys(context, gal_keys);

    PhantomCKKSEncoder ckks_encoder(context);

    /*
    Populate a vector of floating-point values to batch.
    */
    std::vector<cuDoubleComplex> pod_vector;
    random_device rd;
    pod_vector.reserve(ckks_encoder.slot_count());
    for (size_t i = 0; i < ckks_encoder.slot_count(); i++)
        pod_vector.push_back(make_cuDoubleComplex(double(1.01), double(1.01)));
    std::vector<cuDoubleComplex> pod_vector2;
    std::vector<cuDoubleComplex> pod_vector3;
    pod_vector3.resize(ckks_encoder.slot_count());
    std::vector<cuDoubleComplex> pod_vector4;
    pod_vector4.resize(ckks_encoder.slot_count());

    /*
    [Encoding]
    For scale we use the square root of the last coeff_modulus prime
    from parms.
    */
    PhantomPlaintext plain;
    {
        CUDATimer timer("encode");
        for (auto i = 0; i < count; i++) {
            timer.start();
            ckks_encoder.encode(context, pod_vector, scale, plain);
            timer.stop();
        }
    }

    /*
    [Decoding]
    */
    {
        CUDATimer timer("decode");
        for (auto i = 0; i < count; i++) {
            timer.start();
            ckks_encoder.decode(context, plain, pod_vector2);
            timer.stop();
        }
    }

    /*
    [Encryption]
    */
    PhantomCiphertext encrypted;
    {
        CUDATimer timer("encrypt_asymmetric");
        for (auto i = 0; i < count; i++) {
            timer.start();
            public_key.encrypt_asymmetric(context, plain, encrypted);
            timer.stop();
        }
    }

    /*
    [Decryption]
    */
    PhantomPlaintext plain2;
    {
        CUDATimer timer("decrypt");
        for (auto i = 0; i < count; i++) {
            timer.start();
            secret_key.decrypt(context, encrypted, plain2);
            timer.stop();
        }
    }

    // homomorphic operations

    PhantomCiphertext encrypted1;
    for (size_t j = 0; j < ckks_encoder.slot_count(); j++)
        pod_vector3[j] = make_cuDoubleComplex(double(1), double(0));
    ckks_encoder.encode(context, pod_vector3, scale, plain);
    public_key.encrypt_asymmetric(context, plain, encrypted1);

    PhantomCiphertext encrypted2;
    for (size_t j = 0; j < ckks_encoder.slot_count(); j++)
        pod_vector4[j] = make_cuDoubleComplex(double(1), double(0));
    ckks_encoder.encode(context, pod_vector4, scale, plain2);
    public_key.encrypt_asymmetric(context, plain2, encrypted2);

    /*
    [Add]
    */
    {
        CUDATimer timer("add");
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            timer.start();
            add_inplace(context, tmp_ct, encrypted2);
            timer.stop();
        }
    }

    /*
    [Add Plain]
    */
    {
        CUDATimer timer("add_plain");
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            timer.start();
            add_plain_inplace(context, tmp_ct, plain);
            timer.stop();
        }
    }

    /*
    [Multiply]
    */
    {
        CUDATimer timer("multiply");
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            timer.start(stream);
            multiply_inplace(context, tmp_ct, encrypted2, stream);
            relinearize_inplace(context, tmp_ct, relin_keys, stream);
            timer.stop(stream);
        }
    }

    /*
    [Multiply Plain]
    */
    {
        CUDATimer timer("multiply_plain");
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            timer.start();
            multiply_plain_inplace(context, tmp_ct, plain);
            timer.stop();
        }
    }

    /*
    [Rescale]
    */
    {
        CUDATimer timer("rescale_to_next");
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            multiply_inplace(context, tmp_ct, encrypted2);
            relinearize_inplace(context, tmp_ct, relin_keys);
            timer.start();
            rescale_to_next_inplace(context, tmp_ct);
            timer.stop();
        }
    }

    /*
    [Rotate Vector]
    */
    {
        CUDATimer timer("rotate_vector_one_step");
        for (auto i = 0; i < count; i++) {
            PhantomCiphertext tmp_ct(encrypted1);
            timer.start();
            rotate_vector_inplace(context, tmp_ct, 1, gal_keys);
            timer.stop();
        }
    }
}

int main() {
    print_example_banner("CKKS Performance Test with Degrees: 4096, 8192, 16384, 32768, and 65536");

    std::vector<int> galois_steps = {1};

    // 2 ^ 13

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 13;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 60}));
        parms.set_special_modulus_size(1);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 13;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
        parms.set_special_modulus_size(1);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    // 2 ^ 14

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 14;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 60}));
        parms.set_special_modulus_size(1);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 14;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 60, 60}));
        parms.set_special_modulus_size(2);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    // 2 ^ 15

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 15;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                                           40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
        parms.set_special_modulus_size(1);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 15;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60}));
        parms.set_special_modulus_size(2);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 15;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60}));
        parms.set_special_modulus_size(3);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 15;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                     {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60}));
        parms.set_special_modulus_size(4);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    // 2^ 16

    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(
                CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                           40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                           40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
        parms.set_special_modulus_size(1);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(
                CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                           40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                           40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60}));
        parms.set_special_modulus_size(2);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(
                CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                           40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                           40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60}));
        parms.set_special_modulus_size(3);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60}));
        parms.set_special_modulus_size(4);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60, 60}));
        parms.set_special_modulus_size(5);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 1 << 16;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_galois_elts(get_elts_from_steps(galois_steps, poly_modulus_degree));
        parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60, 60, 60}));
        parms.set_special_modulus_size(6);
        double scale = pow(2.0, 40);
        ckks_performance_test(parms, scale);
    }

    return 0;
}

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>
#include "example.h"
#include "phantom.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

void example_bgv_enc(EncryptionParameters &parms, PhantomContext &context) {
    std::cout << "Example: BGV Basics" << std::endl;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomBatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();
    size_t row_size = slot_count / 2;
    cout << "Plaintext matrix row size: " << row_size << endl;

    /*
    Here we create the following matrix:
        [ 1,  2,  3,  4,  0,  0, ...,  0 ]
        [ 0,  0,  0,  0,  0,  0, ...,  0 ]
    */
    std::vector<uint64_t> pod_matrix(slot_count, 0ULL);
    for (size_t i = 0; i < slot_count; i++) {
        pod_matrix[i] = rand() % parms.plain_modulus().value();
        // pod_matrix[i] = i;
    }
    cout << "Input vector: " << endl;
    print_vector(pod_matrix, 3, 7);

    PhantomPlaintext x_plain;

    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    batch_encoder.encode(context, pod_matrix, x_plain);

    // Decode check
    vector<uint64_t> result;
    batch_encoder.decode(context, x_plain, result);
    cout << "We can immediately decode this plaintext to check the correctness." << endl;
    print_vector(result, 3, 7);
    bool correctness = true;
    for (size_t idx = 0; idx < slot_count; idx++)
        correctness &= result[idx] == pod_matrix[idx];
    if (!correctness)
        throw std::logic_error("Error in encode/decode");
    result.clear();

    // Symmetric encryption check
    PhantomCiphertext x_symmetric_cipher;
    cout << "BGV symmetric test begin, encrypting ......" << endl;
    secret_key.encrypt_symmetric(context, x_plain, x_symmetric_cipher);
    PhantomPlaintext x_symmetric_plain;
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, x_symmetric_cipher, x_symmetric_plain);
    cout << "Decode the decrypted plaintext." << endl;
    batch_encoder.decode(context, x_symmetric_plain, result);
    print_vector(result, 3, 7);
    for (size_t idx = 0; idx < slot_count; idx++)
        correctness &= result[idx] == pod_matrix[idx];
    if (!correctness)
        throw std::logic_error("Error in symmetric encryption");
    result.clear();

    // Asymmetric encryption check
    cout << "BGV asymmetric test begin, encrypting ......" << endl;
    PhantomCiphertext x_asymmetric_cipher;
    public_key.encrypt_asymmetric(context, x_plain, x_asymmetric_cipher);
    PhantomPlaintext x_asymmetric_plain;
    // BECAREFUL FOR THE MULTIPLICATIVE LEVEL!!!
    // cout << "We drop the ciphertext for some level, and Decrypting ......" << endl;
    // mod_switch_to_inplace(context, x_asymmetric_cipher, 3);
    secret_key.decrypt(context, x_asymmetric_cipher, x_asymmetric_plain);

    batch_encoder.decode(context, x_asymmetric_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    for (size_t idx = 0; idx < slot_count; idx++)
        correctness &= result[idx] == pod_matrix[idx];
    if (!correctness)
        throw std::logic_error("Error in asymmetric encryption");
    result.clear();
}

void example_bgv_add(EncryptionParameters &parms, PhantomContext &context) {
    std::cout << "Example: BGV HomAdd/HomSub test" << std::endl;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomBatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();
    size_t row_size = slot_count / 2;
    cout << "Plaintext matrix row size: " << row_size << endl;

    /*
    Here we create the following matrix:
        [ 1,  2,  3,  4,  0,  0, ...,  0 ]
        [ 0,  0,  0,  0,  0,  0, ...,  0 ]
    */
    std::vector<uint64_t> input1(slot_count, 0ULL);
    std::vector<uint64_t> input2(slot_count, 0ULL);
    for (size_t i = 0; i < slot_count; i++) {
        input1[i] = rand() % parms.plain_modulus().value();
        input2[i] = rand() % parms.plain_modulus().value();
    }
    cout << "Input vector: " << endl;
    print_vector(input1, 3, 7);
    print_vector(input2, 3, 7);

    PhantomPlaintext x_plain, y_plain;
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    batch_encoder.encode(context, input1, x_plain);
    batch_encoder.encode(context, input2, y_plain);

    PhantomCiphertext x_cipher, y_cipher;
    cout << "BGV HomAdd/Sub test begin, encrypting ......" << endl;
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);

    cout << "Homomorphic adding ......" << endl;
    add_inplace(context, y_cipher, x_cipher);
    add_inplace(context, x_cipher, y_cipher);

    PhantomPlaintext x_plus_y_plain;
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, x_cipher, x_plus_y_plain);

    vector<uint64_t> result;
    batch_encoder.decode(context, x_plus_y_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    bool correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= (result[i] == (input1[i] + input1[i] + input2[i]) % parms.plain_modulus().value());
    }
    if (!correctness)
        throw std::logic_error("Error in Asymmetric HomAdd");
    result.clear();

    cout << "Homomorphic subtracting ......" << endl;

    sub_inplace(context, y_cipher, x_cipher, true);
    sub_inplace(context, x_cipher, y_cipher, false);

    PhantomPlaintext x_minus_y_plain;
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, x_cipher, x_minus_y_plain);

    batch_encoder.decode(context, x_minus_y_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= (result[i] == ((input1[i] + input2[i]) % parms.plain_modulus().value()));
    }
    if (!correctness)
        throw std::logic_error("Error in Asymmetric HomSub");
    result.clear();

    cout << "Homomorphic add many ......" << endl;
    vector<vector<uint64_t>> input;
    vector<PhantomCiphertext> ciphers;
    uint64_t input_vector_size = 20;
    input.resize(input_vector_size);
    ciphers.reserve(input_vector_size);
    for (size_t i = 0; i < input_vector_size; i++) {
        input[i].reserve(slot_count);
        for (size_t j = 0; j < slot_count; j++) {
            input[i].push_back(rand() % parms.plain_modulus().value());
        }

        cout << "Input vector " << i << " : length = " << slot_count << endl;
        print_vector(input[i], 3, 7);

        PhantomPlaintext plain;
        batch_encoder.encode(context, input[i], plain);

        PhantomCiphertext cipher;
        public_key.encrypt_asymmetric(context, plain, cipher);

        ciphers.push_back(cipher);
    }

    PhantomCiphertext sum_cipher;
    add_many(context, ciphers, sum_cipher);
    // add(context, ciphers[0], ciphers[1], sum_cipher);

    PhantomPlaintext sum_plain;
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, sum_cipher, sum_plain);
    batch_encoder.decode(context, sum_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    correctness = true;
    vector<int64_t> expected_res;

    for (size_t j = 0; j < slot_count; j++) {
        int64_t sum = 0;
        for (size_t i = 0; i < input_vector_size; i++) {
            sum = (sum + input[i][j]) % parms.plain_modulus().value();
        }
        expected_res.push_back(sum);
        correctness &= result[j] == sum;
    }
    cout << "Expected plaintext: " << endl;
    print_vector(expected_res, 3, 7);
    if (!correctness)
        throw std::logic_error("Error in Asymmetric HomAddMany");
    result.clear();
}

void example_bgv_add_plain(EncryptionParameters &parms, PhantomContext &context) {
    std::cout << "Example: BGV HomAddPlain/HomSubPlain test" << std::endl;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomBatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();
    size_t row_size = slot_count / 2;
    cout << "Plaintext matrix row size: " << row_size << endl;

    /*
    Here we create the following matrix:
        [ 1,  2,  3,  4,  0,  0, ...,  0 ]
        [ 0,  0,  0,  0,  0,  0, ...,  0 ]
    */
    std::vector<uint64_t> input1(slot_count, 0ULL);
    std::vector<uint64_t> input2(slot_count, 0ULL);
    for (size_t i = 0; i < slot_count; i++) {
        input1[i] = rand() % parms.plain_modulus().value();
        input2[i] = rand() % parms.plain_modulus().value();
        // pod_matrix[i] = i;
    }
    cout << "Input vector: " << endl;
    print_vector(input1, 3, 7);
    print_vector(input2, 3, 7);

    //
    PhantomPlaintext x_plain, y_plain;
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    batch_encoder.encode(context, input1, x_plain);
    batch_encoder.encode(context, input2, y_plain);

    PhantomCiphertext x_cipher;
    cout << "BGV HomAddPlain/SubPlain test begin, encrypting ......" << endl;
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    vector<uint64_t> result;
    bool correctness = true;
    cout << "Homomorphic adding ......" << endl;
    add_plain_inplace(context, x_cipher, y_plain);

    PhantomPlaintext x_plus_y_plain;
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, x_cipher, x_plus_y_plain);

    batch_encoder.decode(context, x_plus_y_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);

    for (size_t i = 0; i < slot_count; i++) {
        correctness &= (result[i] == (input1[i] + input2[i]) % parms.plain_modulus().value());
    }
    if (!correctness)
        throw std::logic_error("Error in asymmetric HomAddPlain");
    result.clear();

    cout << "Homomorphic subtracting ......" << endl;

    sub_plain_inplace(context, x_cipher, y_plain);

    PhantomPlaintext x_minus_y_plain;
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, x_cipher, x_minus_y_plain);

    batch_encoder.decode(context, x_minus_y_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    correctness = true;
    cout << "plain_modulus : " << parms.plain_modulus().value() << endl;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= (result[i] == input1[i]);
    }
    if (!correctness)
        throw std::logic_error("Error in asymmetric HomSubPlain");
    result.clear();
}

void example_bgv_mul(EncryptionParameters &parms, PhantomContext &context) {
    std::cout << "Example: BGV HomMul test" << std::endl;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomBatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();
    size_t row_size = slot_count / 2;
    cout << "Plaintext matrix row size: " << row_size << endl;

    /*
    Here we create the following matrix:
        [ 1,  2,  3,  4,  0,  0, ...,  0 ]
        [ 0,  0,  0,  0,  0,  0, ...,  0 ]
    */
    std::vector<uint64_t> input1(slot_count);
    std::vector<uint64_t> input2(slot_count);
    for (size_t i = 0; i < slot_count; i++) {
        input1[i] = rand() % parms.plain_modulus().value();
        input2[i] = rand() % parms.plain_modulus().value();
    }
    cout << "Input vector: " << endl;
    print_vector(input1, 3, 7);
    print_vector(input2, 3, 7);

    PhantomPlaintext xy_plain;

    PhantomPlaintext x_plain = batch_encoder.encode(context, input1);
    PhantomPlaintext y_plain = batch_encoder.encode(context, input2);

    PhantomCiphertext x_cipher;
    PhantomCiphertext y_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);

    cout << "Compute and relinearize x*y." << endl;
    multiply_inplace(context, x_cipher, y_cipher);
    relinearize_inplace(context, x_cipher, relin_keys);
    mod_switch_to_next_inplace(context, x_cipher);

    secret_key.decrypt(context, x_cipher, xy_plain);

    vector<uint64_t> result;
    result = batch_encoder.decode(context, xy_plain);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= result[i] == (input1[i] * input2[i] % parms.plain_modulus().value());
    }
    if (!correctness)
        throw std::logic_error("Error in Homomorphic multiplication");

    std::cout << "Example: BGV HomSqr test" << std::endl;

    cout << "Message vector: " << endl;
    print_vector(input1, 3, 7);

    PhantomPlaintext xx_plain;

    batch_encoder.encode(context, input1, x_plain);
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    cout << "Compute and relinearize x^2." << endl;
    multiply_inplace(context, x_cipher, x_cipher);
    relinearize_inplace(context, x_cipher, relin_keys);
    mod_switch_to_next_inplace(context, x_cipher);

    secret_key.decrypt(context, x_cipher, xx_plain);

    batch_encoder.decode(context, xx_plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= result[i] == (input1[i] * input1[i] % parms.plain_modulus().value());
    }
    if (!correctness)
        throw std::logic_error("Error in Homomorphic squaring");
    result.clear();
    input1.clear();
}

void examples_bgv() {
    srand(time(NULL));

    EncryptionParameters parms(scheme_type::bgv);

    /*
    As an example, we evaluate the degree 8 polynomial

        x^8

    over an encrypted x over integers 1, 2, 3, 4. The coefficients of the
    polynomial can be considered as plaintext inputs, as we will see below. The
    computation is done modulo the plain_modulus 1032193.

    Computing over encrypted data in the BGV scheme is similar to that in BFV.
    The purpose of this example is mainly to explain the differences between BFV
    and BGV in terms of ciphertext coefficient modulus selection and noise control.

    Most of the following code are repeated from "BFV basics" and "encoders" examples.
    */

    /*
    Note that scheme_type is now "bgv".
    */
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    /*
    We can certainly use BFVDefault coeff_modulus. In later parts of this example,
    we will demonstrate how to choose coeff_modulus that is more useful in BGV.
    */
    size_t alpha = 2;
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {35, 35, 35, 35, 40, 40}));
    parms.set_special_modulus_size(alpha);
    //    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
    PhantomContext context(parms);

    print_parameters(context);
    cout << endl;

    example_bgv_enc(parms, context);
    example_bgv_add(parms, context);
    example_bgv_add_plain(parms, context);
    example_bgv_mul(parms, context);

    cout << endl;
};

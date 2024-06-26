#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
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
using namespace phantom::util;

void example_ckks_enc(PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS Encode/Decode complex vector" << std::endl;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> input(slot_count);
    double rand_real;
    double rand_imag;
    // srand(time(0));
    for (size_t i = 0; i < slot_count; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        input[i] = make_cuDoubleComplex(rand_real, rand_imag);
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);

    PhantomPlaintext x_plain;
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    encoder.encode(context, input, scale, x_plain, 1);

    bool correctness = true;

    // Decode check
    vector<cuDoubleComplex> result;
    encoder.decode(context, x_plain, result);
    cout << "We can immediately decode this plaintext to check the correctness." << endl;
    print_vector(result, 3, 7);
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= result[i] == input[i];
    }
    if (!correctness)
        throw std::logic_error("encode/decode complex vector error");
    result.clear();


    // double vector test
    std::cout << "Example: CKKS Encode/Decode double vector" << std::endl;
    vector<double> input_double(slot_count);
    // srand(time(0));
    for (size_t i = 0; i < slot_count; i++) {
        input_double[i] = (double) rand() / RAND_MAX;
    }
    cout << "Input vector: " << endl;
    print_vector(input_double, 3, 7);

    PhantomPlaintext pt;
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    encoder.encode(context, input_double, scale, pt, 1);

    correctness = true;

    // Decode check
    vector<double> result_double;
    encoder.decode(context, pt, result_double);
    cout << "We can immediately decode this plaintext to check the correctness." << endl;
    print_vector(result_double, 3, 7);
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= compare_double(result_double[i], input_double[i]);
    }
    if (!correctness)
        throw std::logic_error("encode/decode double vector error");
    result.clear();

    // Symmetric encryption check
    PhantomCiphertext x_symmetric_cipher;
    cout << "CKKS symmetric test begin, encrypting ......" << endl;
    secret_key.encrypt_symmetric(context, x_plain, x_symmetric_cipher);
    PhantomPlaintext x_symmetric_plain;
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, x_symmetric_cipher, x_symmetric_plain);
    cout << "Decode the decrypted plaintext." << endl;
    encoder.decode(context, x_symmetric_plain, result);
    // print_vector(result, 3, 7);
    correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= result[i] == input[i];
    }
    if (!correctness)
        throw std::logic_error("Symmetric encryption error");
    result.clear();

    // Asymmetric encryption check
    cout << "CKKS asymmetric test begin, encrypting ......" << endl;
    PhantomCiphertext x_asymmetric_cipher;
    public_key.encrypt_asymmetric(context, x_plain, x_asymmetric_cipher);
    PhantomPlaintext x_asymmetric_plain;
    // BECAREFUL FOR THE MULTIPLICATIVE LEVEL!!!
    // cout << "We drop the ciphertext for some level, and Decrypting ......" << endl;
    // mod_switch_to_inplace(context, x_asymmetric_cipher, 3);
    secret_key.decrypt(context, x_asymmetric_cipher, x_asymmetric_plain);

    encoder.decode(context, x_asymmetric_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= result[i] == input[i];
    }
    if (!correctness)
        throw std::logic_error("Asymmetric encryption error");
    result.clear();
}

void example_ckks_add(PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS Add" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> input1, input2, result;
    size_t msg_size1 = slot_count;
    size_t msg_size2 = slot_count;
    input1.reserve(msg_size1);
    input2.reserve(msg_size2);
    double rand_real, rand_imag;
    srand(time(0));
    for (size_t i = 0; i < msg_size1; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        input1.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    for (size_t i = 0; i < msg_size2; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        input2.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    cout << "Input vector 1: length = " << msg_size1 << endl;
    print_vector(input1, 3, 7);
    cout << "Input vector 2: length = " << msg_size2 << endl;
    print_vector(input2, 3, 7);

    PhantomPlaintext x_plain, y_plain;
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    encoder.encode(context, input1, scale, x_plain);
    encoder.encode(context, input2, scale, y_plain);

    PhantomCiphertext x_sym_cipher, y_sym_cipher;
    cout << "CKKS symmetric HomAdd/Sub test begin, encrypting ......" << endl;
    secret_key.encrypt_symmetric(context, x_plain, x_sym_cipher);
    secret_key.encrypt_symmetric(context, y_plain, y_sym_cipher);

    cout << "Homomorphic adding ......" << endl;
    add_inplace(context, x_sym_cipher, y_sym_cipher);

    PhantomPlaintext x_plus_y_sym_plain;
    cout << "Decrypting ......" << endl;

    secret_key.decrypt(context, x_sym_cipher, x_plus_y_sym_plain);

    encoder.decode(context, x_plus_y_sym_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    bool correctness = true;
    for (size_t i = 0; i < max(msg_size1, msg_size2); i++) {
        if (i >= msg_size1)
            correctness &= result[i] == input2[i];
        else if (i >= msg_size2)
            correctness &= result[i] == input1[i];
        else
            correctness &= result[i] == cuCadd(input1[i], input2[i]);
    }
    if (!correctness)
        throw std::logic_error("Symmetric HomAdd error");
    result.clear();

    cout << "Homomorphic subtracting ......" << endl;
    sub_inplace(context, x_sym_cipher, y_sym_cipher);

    PhantomPlaintext x_minus_y_sym_plain;
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, x_sym_cipher, x_minus_y_sym_plain);
    encoder.decode(context, x_minus_y_sym_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    correctness = true;
    for (size_t i = 0; i < max(msg_size1, msg_size2); i++) {
        correctness &= result[i] == input1[i];
    }
    if (!correctness)
        throw std::logic_error("Symmetric HomSub error");
    result.clear();

    PhantomCiphertext x_asym_cipher, y_asym_cipher;
    cout << "CKKS asymmetric HomAdd/Sub test begin, encrypting ......" << endl;
    public_key.encrypt_asymmetric(context, x_plain, x_asym_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_asym_cipher);

    cout << "Homomorphic adding ......" << endl;
    add_inplace(context, y_asym_cipher, x_asym_cipher);

    PhantomPlaintext x_plus_y_asym_plain;
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, y_asym_cipher, x_plus_y_asym_plain);
    encoder.decode(context, x_plus_y_asym_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    correctness = true;
    for (size_t i = 0; i < max(msg_size1, msg_size2); i++) {
        if (i >= msg_size1)
            correctness &= result[i] == input2[i];
        else if (i >= msg_size2)
            correctness &= result[i] == input1[i];
        else
            correctness &= result[i] == cuCadd(input1[i], input2[i]);
    }
    if (!correctness)
        throw std::logic_error("Asymmetric HomAdd error");
    result.clear();

    cout << "Homomorphic subtracting ......" << endl;
    sub_inplace(context, x_asym_cipher, y_asym_cipher, true);

    PhantomPlaintext x_minus_y_asym_plain;
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, x_asym_cipher, x_minus_y_asym_plain);
    encoder.decode(context, x_minus_y_asym_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    correctness = true;
    for (size_t i = 0; i < max(msg_size1, msg_size2); i++) {
        correctness &= result[i] == input2[i];
    }
    if (!correctness)
        throw std::logic_error("Asymmetric HomSub error");
    result.clear();

    cout << "Homomorphic add many ......" << endl;
    vector<vector<cuDoubleComplex>> input;
    vector<PhantomCiphertext> ciphers;
    uint64_t input_vector_size = 20;
    input.resize(input_vector_size);
    ciphers.reserve(input_vector_size);
    for (size_t i = 0; i < input_vector_size; i++) {
        size_t msg_size = slot_count;
        input[i].reserve(msg_size);
        double rand_real, rand_imag;
        for (size_t j = 0; j < msg_size1; j++) {
            rand_real = (double) rand() / RAND_MAX;
            rand_imag = (double) rand() / RAND_MAX;
            input[i].push_back(make_cuDoubleComplex(rand_real, rand_imag));
        }

        cout << "Input vector " << i << " : length = " << msg_size << endl;
        print_vector(input[i], 3, 7);

        PhantomPlaintext plain;
        encoder.encode(context, input[i], scale, plain);

        PhantomCiphertext asym_cipher;
        public_key.encrypt_asymmetric(context, plain, asym_cipher);

        ciphers.push_back(asym_cipher);
    }

    // PhantomCiphertext sum_cipher(context);
    // add_many(context, ciphers, sum_cipher);
    //
    // PhantomPlaintext sum_plain(context);
    // cout << "Decrypting ......" << endl;
    // secret_key.decrypt(context, sum_cipher, sum_plain);
    // encoder.decode(context, sum_plain, result);
    // cout << "Decode the decrypted plaintext." << endl;
    // print_vector(result, 3, 7);
    // correctness = true;
    // vector<cuDoubleComplex> expected_res;
    // expected_res.reserve(max(msg_size1, msg_size2));
    //
    // for (size_t k = 0; k < max(msg_size1, msg_size2); k++) {
    //     cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    //     for (size_t i = 0; i < input_vector_size; i++) {
    //         sum = cuCadd(sum, input[i][k]);
    //     }
    //     expected_res.push_back(sum);
    //     correctness &= result[k] == sum;
    // }
    // cout << "Expected plaintext: " << endl;
    // print_vector(expected_res, 3, 7);
    // if (!correctness)
    //     throw std::logic_error("Asymmetric HomAddMany error");
    // result.clear();
}

/** sym test ckks cipher mul a full sloted plaintext
 *  asym test ckks cipher mul only one non-zero slot
 */
void example_ckks_mul_plain(PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS cipher multiply plain vector" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> msg_vec, const_vec, result;
    double rand_real, rand_imag;

    size_t msg_size = slot_count;
    size_t const_size = slot_count;

    cout << "------------- Symmetric case ---------------" << endl;

    msg_vec.reserve(msg_size);
    for (size_t i = 0; i < msg_size; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        msg_vec.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(msg_vec, 3, 7);

    const_vec.reserve(const_size);
    for (size_t i = 0; i < const_size; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        const_vec.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Constant vector: " << endl;
    print_vector(const_vec, 3, 7);

    PhantomPlaintext plain, const_plain;
    // All messages should be with the same length.
    // CKKS encoder can zero-pad messages to the [encoding length]
    // the [encoding length] is determined by the first encoded message
    // the encoder will round up its length to the nearest pow-of-2.
    // if this pow-of-2 is less then slot_count, then sparse
    //    message encoding applied automatically.
    // So, always make sure the longest message is encoded first.
    encoder.encode(context, msg_vec, scale, plain);
    encoder.encode(context, const_vec, scale, const_plain);

    PhantomCiphertext sym_cipher;
    secret_key.encrypt_symmetric(context, plain, sym_cipher);
    multiply_plain_inplace(context, sym_cipher, const_plain);

    secret_key.decrypt(context, sym_cipher, plain);
    encoder.decode(context, plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < msg_size; i++) {
        correctness &= result[i] == cuCmul(msg_vec[i], const_vec[i]);
        if (!correctness) {
            cout << result[i].x << " + I * " << result[i].y << endl;
            cout << cuCmul(msg_vec[i], const_vec[i]).x << " + I * " << cuCmul(msg_vec[i], const_vec[i]).y << endl;
        }
    }
    if (!correctness)
        throw std::logic_error("Symmetric cipher multiply plain vector error");
    result.clear();
    msg_vec.clear();
    const_vec.clear();

    cout << "------------- Asymmetric case ---------------" << endl;
    msg_size >>= 2;
    msg_vec.reserve(msg_size);
    for (size_t i = 0; i < msg_size; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        msg_vec.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(msg_vec, 3, 7);

    const_vec.reserve(128);
    // This time, the length of const_vec is less than msg_vec,
    // however, CKKS encoder will automatically zero-padding const_vec
    // to the same length with msg_vec
    for (size_t i = 0; i < 128; i++) {
        if (i == 2) {
            rand_real = (double) rand() / RAND_MAX;
            rand_imag = (double) rand() / RAND_MAX;
            const_vec.push_back(make_cuDoubleComplex(rand_real, rand_imag));
        } else
            const_vec.push_back(make_cuDoubleComplex(0.0, 0.0));
    }
    cout << "Constant vector: " << endl;
    print_vector(const_vec, 3, 7);

    // reset the length of encoder
    encoder.reset_sparse_slots();
    encoder.encode(context, msg_vec, scale, plain);
    encoder.encode(context, const_vec, scale, const_plain);

    PhantomCiphertext asym_cipher;
    public_key.encrypt_asymmetric(context, plain, asym_cipher);
    multiply_plain_inplace(context, asym_cipher, const_plain);

    secret_key.decrypt(context, asym_cipher, plain);
    encoder.decode(context, plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    correctness = true;
    for (size_t i = 0; i < msg_size; i++) {
        if (i == 2)
            correctness &= result[i] == cuCmul(msg_vec[i], const_vec[i]);
        else
            correctness &= result[i] == make_cuDoubleComplex(0.0, 0.0);
    }
    if (!correctness)
        throw std::logic_error("Asymmetric cipher multiply plain vector error");
    result.clear();
    msg_vec.clear();
    const_vec.clear();
}

void example_ckks_mul(PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS HomMul test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, y_msg;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    size_t y_size = slot_count;
    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    y_msg.reserve(y_size);
    for (size_t i = 0; i < y_size; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        y_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(y_msg, 3, 7);

    PhantomPlaintext x_plain;
    PhantomPlaintext y_plain;
    PhantomPlaintext xy_plain;

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher;
    PhantomCiphertext y_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);
    cout << "Compute x*y*x." << endl;
    PhantomCiphertext xy_cipher = multiply(context, x_cipher, y_cipher);
    relinearize_inplace(context, xy_cipher, relin_keys);
    rescale_to_next_inplace(context, xy_cipher);
    cout << "    + Scale of x*y after rescale: " << log2(xy_cipher.scale()) << " bits" << endl;
    xy_cipher.set_scale(scale);
    mod_switch_to_next_inplace(context, x_cipher);
    cout << "    + Scale of x: " << log2(x_cipher.scale()) << " bits" << endl;
    PhantomCiphertext x2y_cipher = multiply(context, xy_cipher, x_cipher);
    relinearize_inplace(context, x2y_cipher, relin_keys);
    rescale_to_next_inplace(context, x2y_cipher);
    PhantomPlaintext x2y_plain = secret_key.decrypt(context, x2y_cipher);
    auto result = encoder.decode<cuDoubleComplex>(context, x2y_plain);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < x_size; i++) {
        correctness &= result[i] == cuCmul(x_msg[i], cuCmul(x_msg[i], y_msg[i]));
    }
    if (!correctness)
        throw std::logic_error("Homomorphic multiplication error");
    result.clear();
    x_msg.clear();
    y_msg.clear();
}

void example_ckks_rotation(PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS HomRot test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

    int step = 3;

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, result;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    PhantomPlaintext x_plain;
    PhantomPlaintext x_rot_plain;

    encoder.encode(context, x_msg, scale, x_plain);

    PhantomCiphertext x_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    cout << "Compute, rot vector x." << endl;
    rotate_vector_inplace(context, x_cipher, step, galois_keys);

    secret_key.decrypt(context, x_cipher, x_rot_plain);

    encoder.decode(context, x_rot_plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < x_size; i++) {
        correctness &= result[i] == x_msg[(i + step) % x_size];
    }
    if (!correctness)
        throw std::logic_error("Homomorphic rotation error");
    result.clear();
    x_msg.clear();

    std::cout << "Example: CKKS HomConj test" << std::endl;

    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    PhantomPlaintext x_conj_plain;

    encoder.encode(context, x_msg, scale, x_plain);
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    cout << "Compute, conjugate vector x." << endl;
    complex_conjugate_inplace(context, x_cipher, galois_keys);

    secret_key.decrypt(context, x_cipher, x_conj_plain);

    encoder.decode(context, x_conj_plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    correctness = true;
    for (size_t i = 0; i < x_size; i++) {
        correctness &= result[i] == make_cuDoubleComplex(x_msg[i].x, -x_msg[i].y);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic conjugate error");
    result.clear();
    x_msg.clear();
}

void examples_ckks() {
    srand(time(NULL));
    /*
    We saw in `2_encoders.cpp' that multiplication in CKKS causes scales
    in ciphertexts to grow. The scale of any ciphertext must not get too close
    to the total size of coeff_modulus, or else the ciphertext simply runs out of
    room to store the scaled-up plaintext. The CKKS scheme provides a `rescale'
    functionality that can reduce the scale, and stabilize the scale expansion.

    Rescaling is a kind of modulus switch operation (recall `3_levels.cpp').
    As modulus switching, it removes the last of the primes from coeff_modulus,
    but as a side-effect it scales down the ciphertext by the removed prime.
    Usually we want to have perfect control over how the scales are changed,
    which is why for the CKKS scheme it is more common to use carefully selected
    primes for the coeff_modulus.

    More precisely, suppose that the scale in a CKKS ciphertext is S, and the
    last prime in the current coeff_modulus (for the ciphertext) is P. Rescaling
    to the next level changes the scale to S/P, and removes the prime P from the
    coeff_modulus, as usual in modulus switching. The number of primes limits
    how many rescalings can be done, and thus limits the multiplicative depth of
    the computation.

    It is possible to choose the initial scale freely. One good strategy can be
    to is to set the initial scale S and primes P_i in the coeff_modulus to be
    very close to each other. If ciphertexts have scale S before multiplication,
    they have scale S^2 after multiplication, and S^2/P_i after rescaling. If all
    P_i are close to S, then S^2/P_i is close to S again. This way we stabilize the
    scales to be close to S throughout the computation. Generally, for a circuit
    of depth D, we need to rescale D times, i.e., we need to be able to remove D
    primes from the coefficient modulus. Once we have only one prime left in the
    coeff_modulus, the remaining prime must be larger than S by a few bits to
    preserve the pre-decimal-point value of the plaintext.

    Therefore, a generally good strategy is to choose parameters for the CKKS
    scheme as follows:

        (1) Choose a 60-bit prime as the first prime in coeff_modulus. This will
            give the highest precision when decrypting;
        (2) Choose another 60-bit prime as the last element of coeff_modulus, as
            this will be used as the special prime and should be as large as the
            largest of the other primes;
        (3) Choose the intermediate primes to be close to each other.

    We use CoeffModulus::Create to generate primes of the appropriate size. Note
    that our coeff_modulus is 200 bits total, which is below the bound for our
    poly_modulus_degree: CoeffModulus::MaxBitCount(8192) returns 218.
    */
    std::vector v_alpha = {1, 2, 3, 4, 15};
    for (auto alpha: v_alpha) {
        EncryptionParameters parms(scheme_type::ckks);

        size_t poly_modulus_degree = 1 << 15;
        double scale = pow(2.0, 40);
        switch (alpha) {
            case 1:
                parms.set_poly_modulus_degree(poly_modulus_degree);
                parms.set_coeff_modulus(
                        CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                                   40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
                break;
            case 2:
                parms.set_poly_modulus_degree(poly_modulus_degree);
                parms.set_coeff_modulus(CoeffModulus::Create(
                        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60}));
                parms.set_special_modulus_size(alpha);
                break;
            case 3:
                parms.set_poly_modulus_degree(poly_modulus_degree);
                parms.set_coeff_modulus(CoeffModulus::Create(
                        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60}));
                parms.set_special_modulus_size(alpha);
                break;
            case 4:
                parms.set_poly_modulus_degree(poly_modulus_degree);
                parms.set_coeff_modulus(CoeffModulus::Create(
                        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60}));
                // hybrid key-switching
                parms.set_special_modulus_size(alpha);
                break;
            case 15:
                poly_modulus_degree = 1 << 16;
                parms.set_poly_modulus_degree(poly_modulus_degree);
                parms.set_coeff_modulus(CoeffModulus::Create(
                        poly_modulus_degree,
                        {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                         50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                         50, 50, 50, 50, 50, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
                parms.set_special_modulus_size(alpha);
                scale = pow(2.0, 50);
                break;
            default:
                throw std::invalid_argument("unsupported alpha params");
        }

        /*
        We choose the initial scale to be 2^40. At the last level, this leaves us
        60-40=20 bits of precision before the decimal point, and enough (roughly
        10-20 bits) of precision after the decimal point. Since our intermediate
        primes are 40 bits (in fact, they are very close to 2^40), we can achieve
        scale stabilization as described above.
        */

        PhantomContext context(parms);
        print_parameters(context);
        cout << endl;

        example_ckks_enc(context, scale);
        example_ckks_add(context, scale);
        example_ckks_mul_plain(context, scale);
        example_ckks_mul(context, scale);
        example_ckks_rotation(context, scale);
    }
    cout << endl;
}

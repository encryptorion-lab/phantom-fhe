#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "example.h"
#include "phantom.h"
#include "util.cuh"

using namespace std;
using namespace phantom;

void example_ckks_enc(EncryptionParameters &parms, PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS Basics" << std::endl;

    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> input;
    size_t msg_size = slot_count;
    input.reserve(msg_size);
    double rand_real;
    double rand_imag;
    // srand(time(0));
    for (size_t i = 0; i < msg_size; i++) {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        input.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);

    //
    PhantomPlaintext x_plain(context);
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    encoder.encode(context, input, scale, x_plain);

    bool correctness = true;

    // Decode check
    vector<cuDoubleComplex> result;
    encoder.decode(context, x_plain, result);
    cout << "We can immediately decode this plaintext to check the correctness." << endl;
    print_vector(result, 3, 7);
    for (size_t i = 0; i < msg_size; i++) {
        correctness &= result[i] == input[i];
    }
    if (!correctness)
        throw std::logic_error("encode/decode error");
    result.clear();

    // Symmetric encryption check
    PhantomCiphertext x_symmetric_cipher(context);
    cout << "CKKS symmetric test begin, encrypting ......" << endl;
    secret_key.encrypt_symmetric(context, x_plain, x_symmetric_cipher, false);
    PhantomPlaintext x_symmetric_plain(context);
    cout << "Decrypting ......" << endl;
    secret_key.decrypt(context, x_symmetric_cipher, x_symmetric_plain);
    cout << "Decode the decrypted plaintext." << endl;
    encoder.decode(context, x_symmetric_plain, result);
    // print_vector(result, 3, 7);
    correctness = true;
    for (size_t i = 0; i < msg_size; i++) {
        correctness &= result[i] == input[i];
    }
    if (!correctness)
        throw std::logic_error("Symmetric encryption error");
    result.clear();

    // Asymmetric encryption check
    cout << "CKKS asymmetric test begin, encrypting ......" << endl;
    PhantomCiphertext x_asymmetric_cipher(context);
    public_key.encrypt_asymmetric(context, x_plain, x_asymmetric_cipher, false);
    PhantomPlaintext x_asymmetric_plain(context);
    // BECAREFUL FOR THE MULTIPLICATIVE LEVEL!!!
    // cout << "We drop the ciphertext for some level, and Decrypting ......" << endl;
    // mod_switch_to_inplace(context, x_asymmetric_cipher, 3);
    secret_key.decrypt(context, x_asymmetric_cipher, x_asymmetric_plain);

    encoder.decode(context, x_asymmetric_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    correctness = true;
    for (size_t i = 0; i < msg_size; i++) {
        correctness &= result[i] == input[i];
    }
    if (!correctness)
        throw std::logic_error("Asymmetric encryption error");
    result.clear();
}

void example_ckks_add(EncryptionParameters &parms, PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS evaluation" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);
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
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        input1.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    for (size_t i = 0; i < msg_size2; i++) {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        input2.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    cout << "Input vector 1: length = " << msg_size1 << endl;
    print_vector(input1, 3, 7);
    cout << "Input vector 2: length = " << msg_size2 << endl;
    print_vector(input2, 3, 7);

    PhantomPlaintext x_plain(context), y_plain(context);
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    encoder.encode(context, input1, scale, x_plain);
    encoder.encode(context, input2, scale, y_plain);

    PhantomCiphertext x_sym_cipher(context), y_sym_cipher(context);
    cout << "CKKS symmetric HomAdd/Sub test begin, encrypting ......" << endl;
    secret_key.encrypt_symmetric(context, x_plain, x_sym_cipher, false);
    secret_key.encrypt_symmetric(context, y_plain, y_sym_cipher, false);

    cout << "Homomorphic adding ......" << endl;
    add(context, x_sym_cipher, y_sym_cipher, x_sym_cipher); // x = x + y

    PhantomPlaintext x_plus_y_sym_plain(context);
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
    sub(context, x_sym_cipher, y_sym_cipher, x_sym_cipher); // x = x - y

    PhantomPlaintext x_minus_y_sym_plain(context);
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

    PhantomCiphertext x_asym_cipher(context), y_asym_cipher(context);
    cout << "CKKS asymmetric HomAdd/Sub test begin, encrypting ......" << endl;
    public_key.encrypt_asymmetric(context, x_plain, x_asym_cipher, false);
    public_key.encrypt_asymmetric(context, y_plain, y_asym_cipher, false);

    cout << "Homomorphic adding ......" << endl;
    add(context, x_asym_cipher, y_asym_cipher, y_asym_cipher);

    PhantomPlaintext x_plus_y_asym_plain(context);
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
    sub(context, y_asym_cipher, x_asym_cipher, x_asym_cipher);

    PhantomPlaintext x_minus_y_asym_plain(context);
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
            rand_real = (double)rand() / RAND_MAX;
            rand_imag = (double)rand() / RAND_MAX;
            input[i].push_back(make_cuDoubleComplex(rand_real, rand_imag));
        }

        cout << "Input vector " << i << " : length = " << msg_size << endl;
        print_vector(input[i], 3, 7);

        PhantomPlaintext plain(context);
        encoder.encode(context, input[i], scale, plain);

        PhantomCiphertext asym_cipher(context);
        public_key.encrypt_asymmetric(context, plain, asym_cipher, false);

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
void example_ckks_mul_plain(EncryptionParameters &parms, PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS cipher multiply plain vector" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);
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
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        msg_vec.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(msg_vec, 3, 7);

    const_vec.reserve(const_size);
    for (size_t i = 0; i < const_size; i++) {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        const_vec.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Constant vector: " << endl;
    print_vector(const_vec, 3, 7);

    PhantomPlaintext plain(context), const_plain(context);
    // All messages should be with the same length.
    // CKKS encoder can zero-pad messages to the [encoding length]
    // the [encoding length] is determined by the first encoded message
    // the encoder will round up its length to the nearest pow-of-2.
    // if this pow-of-2 is less then slot_count, then sparse
    //    message encoding applied automatically.
    // So, always make sure the longest message is encoded first.
    encoder.encode(context, msg_vec, scale, plain);
    encoder.encode(context, const_vec, scale, const_plain);

    PhantomCiphertext sym_cipher(context);
    secret_key.encrypt_symmetric(context, plain, sym_cipher, false);
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
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
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
            rand_real = (double)rand() / RAND_MAX;
            rand_imag = (double)rand() / RAND_MAX;
            const_vec.push_back(make_cuDoubleComplex(rand_real, rand_imag));
        }
        else
            const_vec.push_back(make_cuDoubleComplex(0.0, 0.0));
    }
    cout << "Constant vector: " << endl;
    print_vector(const_vec, 3, 7);

    // reset the length of encoder
    encoder.reset_sparse_slots();
    encoder.encode(context, msg_vec, scale, plain);
    encoder.encode(context, const_vec, scale, const_plain);

    PhantomCiphertext asym_cipher(context);
    public_key.encrypt_asymmetric(context, plain, asym_cipher, false);
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

void example_ckks_mul(EncryptionParameters &parms, PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS HomMul test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);
    PhantomRelinKey relin_keys(context);
    secret_key.gen_relinkey(context, relin_keys);

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, y_msg, result;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    size_t y_size = slot_count;
    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++) {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    y_msg.reserve(y_size);
    for (size_t i = 0; i < y_size; i++) {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        y_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(y_msg, 3, 7);

    PhantomPlaintext x_plain(context);
    PhantomPlaintext y_plain(context);
    PhantomPlaintext xy_plain(context);

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher(context);
    PhantomCiphertext y_cipher(context);

    public_key.encrypt_asymmetric(context, x_plain, x_cipher, false);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher, false);

    cout << "Compute, relinearize, and rescale x*y." << endl;
    multiply_inplace(context, x_cipher, y_cipher);
    relinearize_inplace(context, x_cipher, relin_keys);
    rescale_to_next_inplace(context, x_cipher);
    cout << "    + Scale of x*y after rescale: " << log2(x_cipher.scale()) << " bits" << endl;

    secret_key.decrypt(context, x_cipher, xy_plain);

    encoder.decode(context, xy_plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < x_size; i++) {
        correctness &= result[i] == cuCmul(x_msg[i], y_msg[i]);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic multiplication error");
    result.clear();
    x_msg.clear();
    y_msg.clear();
}

void example_ckks_rotation(EncryptionParameters &parms, PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS HomRot test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);
    PhantomGaloisKey galois_keys(context);
    secret_key.create_galois_keys(context, galois_keys);

    int step = 3;

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, result;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++) {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    PhantomPlaintext x_plain(context);
    PhantomPlaintext x_rot_plain(context);

    encoder.encode(context, x_msg, scale, x_plain);

    PhantomCiphertext x_cipher(context);

    public_key.encrypt_asymmetric(context, x_plain, x_cipher, false);

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
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    PhantomPlaintext x_conj_plain(context);

    encoder.encode(context, x_msg, scale, x_plain);
    public_key.encrypt_asymmetric(context, x_plain, x_cipher, false);

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

void example_ckks_basics(EncryptionParameters &parms, PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS Basics" << std::endl;

    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);
    PhantomRelinKey relin_keys(context);
    secret_key.gen_relinkey(context, relin_keys);
    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> input;
    double rand_real, rand_imag;
    size_t size = slot_count;
    input.reserve(size);
    for (size_t i = 0; i < size; i++) {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        input.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);

    // cout << "Evaluating polynomial PI*x^3 + 0.4x + 1 ..." << endl;

    /*
    We create plaintexts for PI, 0.4, and 1 using an overload of CKKSEncoder::encode
    that encodes the given floating-point value to every slot in the vector.
    */
    PhantomPlaintext plain_coeff3(context), plain_coeff1(context), plain_coeff0(context);
    encoder.encode(context, 3.14159265, scale, plain_coeff3);
    encoder.encode(context, 0.4, scale, plain_coeff1);
    encoder.encode(context, 1.0, scale, plain_coeff0);

    PhantomPlaintext x_plain(context);
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    encoder.encode(context, input, scale, x_plain);
    PhantomCiphertext x1_encrypted(context);
    public_key.encrypt_asymmetric(context, x_plain, x1_encrypted, false);

    /*
    To compute x^3 we first compute x^2 and relinearize. However, the scale has
    now grown to 2^80.
    */
    PhantomCiphertext x3_encrypted(context);
    print_line(__LINE__);
    cout << "Compute x^2 and relinearize:" << endl;
    x3_encrypted = x1_encrypted;
    multiply_inplace(context, x3_encrypted, x3_encrypted);
    relinearize_inplace(context, x3_encrypted, relin_keys);
    cout << "    + Scale of x^2 before rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;

    /*
    Now rescale; in addition to a modulus switch, the scale is reduced down by
    a factor equal to the prime that was switched away (40-bit prime). Hence, the
    new scale should be close to 2^40. Note, however, that the scale is not equal
    to 2^40: this is because the 40-bit prime is only close to 2^40.
    */
    print_line(__LINE__);
    cout << "Rescale x^2." << endl;
    rescale_to_next_inplace(context, x3_encrypted);
    cout << "    + Scale of x^2 after rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;

    /*
    Now x3_encrypted is at a different level than x1_encrypted, which prevents us
    from multiplying them to compute x^3. We could simply switch x1_encrypted to
    the next parameters in the modulus switching chain. However, since we still
    need to multiply the x^3 term with PI (plain_coeff3), we instead compute PI*x
    first and multiply that with x^2 to obtain PI*x^3. To this end, we compute
    PI*x and rescale it back from scale 2^80 to something close to 2^40.
    */
    print_line(__LINE__);
    cout << "Compute and rescale PI*x." << endl;
    PhantomCiphertext x1_encrypted_coeff3(context);
    x1_encrypted_coeff3 = x1_encrypted;
    multiply_plain_inplace(context, x1_encrypted_coeff3, plain_coeff3);
    cout << "    + Scale of PI*x before rescale: " << log2(x1_encrypted_coeff3.scale()) << " bits" << endl;
    rescale_to_next_inplace(context, x1_encrypted_coeff3);
    cout << "    + Scale of PI*x after rescale: " << log2(x1_encrypted_coeff3.scale()) << " bits" << endl;

    /*
    Since x3_encrypted and x1_encrypted_coeff3 have the same exact scale and use
    the same encryption parameters, we can multiply them together. We write the
    result to x3_encrypted, relinearize, and rescale. Note that again the scale
    is something close to 2^40, but not exactly 2^40 due to yet another scaling
    by a prime. We are down to the last level in the modulus switching chain.
    */
    print_line(__LINE__);
    cout << "Compute, relinearize, and rescale (PI*x)*x^2." << endl;
    multiply_inplace(context, x3_encrypted, x1_encrypted_coeff3);
    relinearize_inplace(context, x3_encrypted, relin_keys);
    cout << "    + Scale of PI*x^3 before rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;
    rescale_to_next_inplace(context, x3_encrypted);
    cout << "    + Scale of PI*x^3 after rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;

    // PhantomPlaintext plain_result(context);
    // print_line(__LINE__);
    // cout << "Decrypt and decode PI*x^3." << endl;
    // cout << "    + Expected result:" << endl;
    // vector<cuDoubleComplex> true_result;
    // for (size_t i = 0; i < input.size(); i++)
    // {
    //     auto x = input[i];
    //     auto x3 = cuCmul(cuCmul(x, x), make_cuDoubleComplex(x.x * 3.14159265, x.y * 3.14159265));
    //     true_result.push_back(x3);
    // }
    // print_vector(true_result, 3, 7);

    // cout << "===================== decrypt ============================" << endl;
    // secret_key.decrypt(context, x3_encrypted, plain_result);
    // vector<cuDoubleComplex> result;
    // encoder.decode(context, plain_result, result);
    // print_vector(result, 3, 7);
    // bool correctness = true;
    // for (size_t i = 0; i < size; i++)
    // {
    //     correctness &= result[i] == true_result[i];
    // }
    // if (correctness)
    //     cout << "    + Computed result ...... Correct." << endl
    //          << endl
    //          << "---------------------------------------" << endl;
    // if (!correctness)
    //     cout << "    + Computed result ...... inCorrect!!!!!!!!!!!!" << endl
    //          << endl
    //          << "---------------------------------------" << endl;

    /*
    Next we compute the degree one term. All this requires is one multiply_plain
    with plain_coeff1. We overwrite x1_encrypted with the result.
    */
    print_line(__LINE__);
    cout << "Compute and rescale 0.4*x." << endl;
    multiply_plain_inplace(context, x1_encrypted, plain_coeff1);
    cout << "    + Scale of 0.4*x before rescale: " << log2(x1_encrypted.scale()) << " bits" << endl;
    rescale_to_next_inplace(context, x1_encrypted);
    cout << "    + Scale of 0.4*x after rescale: " << log2(x1_encrypted.scale()) << " bits" << endl;

    /*
    Now we would hope to compute the sum of all three terms. However, there is
    a serious problem: the encryption parameters used by all three terms are
    different due to modulus switching from rescaling.

    Encrypted addition and subtraction require that the scales of the inputs are
    the same, and also that the encryption parameters (parms_id) match. If there
    is a mismatch, Evaluator will throw an exception.
    */
    cout << endl;
    print_line(__LINE__);
    cout << "Parameters used by all three terms are different." << endl;
    cout << "    + Modulus chain index for x3_encrypted: " << x3_encrypted.chain_index() << endl;
    cout << "    + Modulus chain index for x1_encrypted: " << x1_encrypted.chain_index() << endl;
    cout << "    + Modulus chain index for plain_coeff0: " << plain_coeff0.chain_index() << endl;
    cout << endl;

    /*
    Let us carefully consider what the scales are at this point. We denote the
    primes in coeff_modulus as P_0, P_1, P_2, P_3, in this order. P_3 is used as
    the special modulus and is not involved in rescalings. After the computations
    above the scales in ciphertexts are:

        - Product x^2 has scale 2^80 and is at level 2;
        - Product PI*x has scale 2^80 and is at level 2;
        - We rescaled both down to scale 2^80/P_2 and level 1;
        - Product PI*x^3 has scale (2^80/P_2)^2;
        - We rescaled it down to scale (2^80/P_2)^2/P_1 and level 0;
        - Product 0.4*x has scale 2^80;
        - We rescaled it down to scale 2^80/P_2 and level 1;
        - The contant term 1 has scale 2^40 and is at level 2.

    Although the scales of all three terms are approximately 2^40, their exact
    values are different, hence they cannot be added together.
    */
    print_line(__LINE__);
    cout << "The exact scales of all three terms are different:" << endl;
    ios old_fmt(nullptr);
    old_fmt.copyfmt(cout);
    cout << fixed << setprecision(10);
    cout << "    + Exact scale in PI*x^3: " << x3_encrypted.scale() << endl;
    cout << "    + Exact scale in  0.4*x: " << x1_encrypted.scale() << endl;
    cout << "    + Exact scale in      1: " << plain_coeff0.scale() << endl;
    cout << endl;
    cout.copyfmt(old_fmt);

    print_line(__LINE__);
    cout << "Normalize scales to scale." << endl;
    x3_encrypted.scale() = scale;
    x1_encrypted.scale() = scale;

    /*
    We still have a problem with mismatching encryption parameters. This is easy
    to fix by using traditional modulus switching (no rescaling). CKKS supports
    modulus switching just like the BFV scheme, allowing us to switch away parts
    of the coefficient modulus when it is simply not needed.
    */
    print_line(__LINE__);
    cout << "Normalize encryption parameters to the lowest level." << endl;
    auto last_chain_index = x3_encrypted.chain_index();
    cout << endl << x3_encrypted.chain_index() << endl;
    mod_switch_to_inplace(context, x3_encrypted, last_chain_index);
    mod_switch_to_inplace(context, x1_encrypted, last_chain_index);
    mod_switch_to_inplace(context, plain_coeff0, last_chain_index);

    /*
    All three ciphertexts are now compatible and can be added.
    */
    print_line(__LINE__);
    cout << "Compute PI*x^3 + 0.4*x + 1." << endl;
    PhantomCiphertext encrypted_result(context);
    add(context, x3_encrypted, x1_encrypted, encrypted_result);
    add_plain_inplace(context, encrypted_result, plain_coeff0);

    /*
    First print the true result.
    */
    PhantomPlaintext plain_result(context);
    print_line(__LINE__);
    cout << "Decrypt and decode PI*x^3 + 0.4x + 1." << endl;
    cout << "    + Expected result:" << endl;
    vector<cuDoubleComplex> true_result;
    for (size_t i = 0; i < input.size(); i++) {
        auto x = input[i];
        auto x3 = cuCmul(cuCmul(x, x), make_cuDoubleComplex(x.x * 3.14159265, x.y * 3.14159265));
        auto x1 = make_cuDoubleComplex(x.x * 0.4, x.y * 0.4);
        true_result.push_back(make_cuDoubleComplex(x3.x + x1.x + 1, x3.y + x1.y));
    }
    print_vector(true_result, 3, 7);

    /*
    Decrypt, decode, and print the result.
    */
    cout << "===================== decrypt ============================" << endl;
    secret_key.decrypt(context, encrypted_result, plain_result);
    vector<cuDoubleComplex> result;
    encoder.decode(context, plain_result, result);
    print_vector(result, 3, 7);
    bool correctness = true;
    for (size_t i = 0; i < size; i++) {
        correctness &= result[i] == true_result[i];
    }
    if (!correctness)
        throw std::logic_error("CKKS basics error");
    result.clear();
}

void example_ckks_stress_test(EncryptionParameters &parms, PhantomContext &context, const double &scale) {
    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);
    PhantomRelinKey relin_keys(context);
    secret_key.gen_relinkey(context, relin_keys);
    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();

    vector<cuDoubleComplex> input;
    double rand_real, rand_imag;
    size_t size = slot_count;
    input.reserve(size);
    for (size_t i = 0; i < size; i++) {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        input.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    // cout << "Input vector: " << endl;
    // print_vector(input, 3, 7);

    // cout << "Evaluating polynomial PI*x^3 + 0.4x + 1 ..." << endl;

    /*
    We create plaintexts for PI, 0.4, and 1 using an overload of CKKSEncoder::encode
    that encodes the given floating-point value to every slot in the vector.
    */
    PhantomPlaintext plain_coeff3(context), plain_coeff1(context), plain_coeff0(context);
    encoder.encode(context, 3.14159265, scale, plain_coeff3);
    encoder.encode(context, 0.4, scale, plain_coeff1);
    encoder.encode(context, 1.0, scale, plain_coeff0);

    PhantomPlaintext x_plain(context);
    // print_line(__LINE__);
    // cout << "Encode input vectors." << endl;
    encoder.encode(context, input, scale, x_plain);
    PhantomCiphertext x1_encrypted(context);
    public_key.encrypt_asymmetric(context, x_plain, x1_encrypted, false);

    /*
    To compute x^3 we first compute x^2 and relinearize. However, the scale has
    now grown to 2^80.
    */
    PhantomCiphertext x3_encrypted(context);
    // print_line(__LINE__);
    // cout << "Compute x^2 and relinearize:" << endl;
    x3_encrypted = x1_encrypted;
    multiply_inplace(context, x3_encrypted, x3_encrypted);
    relinearize_inplace(context, x3_encrypted, relin_keys);
    // cout << "    + Scale of x^2 before rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;

    /*
    Now rescale; in addition to a modulus switch, the scale is reduced down by
    a factor equal to the prime that was switched away (40-bit prime). Hence, the
    new scale should be close to 2^40. Note, however, that the scale is not equal
    to 2^40: this is because the 40-bit prime is only close to 2^40.
    */
    // print_line(__LINE__);
    // cout << "Rescale x^2." << endl;
    rescale_to_next_inplace(context, x3_encrypted);
    // cout << "    + Scale of x^2 after rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;

    /*
    Now x3_encrypted is at a different level than x1_encrypted, which prevents us
    from multiplying them to compute x^3. We could simply switch x1_encrypted to
    the next parameters in the modulus switching chain. However, since we still
    need to multiply the x^3 term with PI (plain_coeff3), we instead compute PI*x
    first and multiply that with x^2 to obtain PI*x^3. To this end, we compute
    PI*x and rescale it back from scale 2^80 to something close to 2^40.
    */
    // print_line(__LINE__);
    // cout << "Compute and rescale PI*x." << endl;
    PhantomCiphertext x1_encrypted_coeff3(context);
    x1_encrypted_coeff3 = x1_encrypted;
    multiply_plain_inplace(context, x1_encrypted_coeff3, plain_coeff3);
    // cout << "    + Scale of PI*x before rescale: " << log2(x1_encrypted_coeff3.scale()) << " bits" << endl;
    rescale_to_next_inplace(context, x1_encrypted_coeff3);
    // cout << "    + Scale of PI*x after rescale: " << log2(x1_encrypted_coeff3.scale()) << " bits" << endl;

    /*
    Since x3_encrypted and x1_encrypted_coeff3 have the same exact scale and use
    the same encryption parameters, we can multiply them together. We write the
    result to x3_encrypted, relinearize, and rescale. Note that again the scale
    is something close to 2^40, but not exactly 2^40 due to yet another scaling
    by a prime. We are down to the last level in the modulus switching chain.
    */
    // print_line(__LINE__);
    // cout << "Compute, relinearize, and rescale (PI*x)*x^2." << endl;
    multiply_inplace(context, x3_encrypted, x1_encrypted_coeff3);
    relinearize_inplace(context, x3_encrypted, relin_keys);
    // cout << "    + Scale of PI*x^3 before rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;
    rescale_to_next_inplace(context, x3_encrypted);
    // cout << "    + Scale of PI*x^3 after rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;

    /*
    Next we compute the degree one term. All this requires is one multiply_plain
    with plain_coeff1. We overwrite x1_encrypted with the result.
    */
    // print_line(__LINE__);
    // cout << "Compute and rescale 0.4*x." << endl;
    multiply_plain_inplace(context, x1_encrypted, plain_coeff1);
    // cout << "    + Scale of 0.4*x before rescale: " << log2(x1_encrypted.scale()) << " bits" << endl;
    rescale_to_next_inplace(context, x1_encrypted);
    // cout << "    + Scale of 0.4*x after rescale: " << log2(x1_encrypted.scale()) << " bits" << endl;

    /*
    Now we would hope to compute the sum of all three terms. However, there is
    a serious problem: the encryption parameters used by all three terms are
    different due to modulus switching from rescaling.

    Encrypted addition and subtraction require that the scales of the inputs are
    the same, and also that the encryption parameters (parms_id) match. If there
    is a mismatch, Evaluator will throw an exception.
    */
    // cout << endl;
    // print_line(__LINE__);
    // cout << "Parameters used by all three terms are different." << endl;
    // cout << "    + Modulus chain index for x3_encrypted: "
    //      << x3_encrypted.chain_index() << endl;
    // cout << "    + Modulus chain index for x1_encrypted: "
    //      << x1_encrypted.chain_index() << endl;
    // cout << "    + Modulus chain index for plain_coeff0: "
    //      << plain_coeff0.chain_index() << endl;
    // cout << endl;

    /*
    Let us carefully consider what the scales are at this point. We denote the
    primes in coeff_modulus as P_0, P_1, P_2, P_3, in this order. P_3 is used as
    the special modulus and is not involved in rescalings. After the computations
    above the scales in ciphertexts are:

        - Product x^2 has scale 2^80 and is at level 2;
        - Product PI*x has scale 2^80 and is at level 2;
        - We rescaled both down to scale 2^80/P_2 and level 1;
        - Product PI*x^3 has scale (2^80/P_2)^2;
        - We rescaled it down to scale (2^80/P_2)^2/P_1 and level 0;
        - Product 0.4*x has scale 2^80;
        - We rescaled it down to scale 2^80/P_2 and level 1;
        - The contant term 1 has scale 2^40 and is at level 2.

    Although the scales of all three terms are approximately 2^40, their exact
    values are different, hence they cannot be added together.
    */
    // print_line(__LINE__);
    // cout << "The exact scales of all three terms are different:" << endl;
    // ios old_fmt(nullptr);
    // old_fmt.copyfmt(cout);
    // cout << fixed << setprecision(10);
    // cout << "    + Exact scale in PI*x^3: " << x3_encrypted.scale() << endl;
    // cout << "    + Exact scale in  0.4*x: " << x1_encrypted.scale() << endl;
    // cout << "    + Exact scale in      1: " << plain_coeff0.scale() << endl;
    // cout << endl;
    // cout.copyfmt(old_fmt);

    x3_encrypted.scale() = scale;
    x1_encrypted.scale() = scale;

    /*
    We still have a problem with mismatching encryption parameters. This is easy
    to fix by using traditional modulus switching (no rescaling). CKKS supports
    modulus switching just like the BFV scheme, allowing us to switch away parts
    of the coefficient modulus when it is simply not needed.
    */
    // print_line(__LINE__);
    // cout << "Normalize encryption parameters to the lowest level." << endl;
    auto last_chain_index = x3_encrypted.chain_index();
    // cout << endl
    //      << x3_encrypted.chain_index() << endl;
    mod_switch_to_inplace(context, x1_encrypted, last_chain_index);
    mod_switch_to_inplace(context, plain_coeff0, last_chain_index);

    /*
    All three ciphertexts are now compatible and can be added.
    */
    // print_line(__LINE__);
    // cout << "Compute PI*x^3 + 0.4*x + 1." << endl;
    PhantomCiphertext encrypted_result(context);
    add(context, x3_encrypted, x1_encrypted, encrypted_result);
    add_plain_inplace(context, encrypted_result, plain_coeff0);

    /*
    First print the true result.
    */
    PhantomPlaintext plain_result(context);
    // print_line(__LINE__);
    // cout << "Decrypt and decode PI*x^3 + 0.4x + 1." << endl;
    // cout << "    + Expected result:" << endl;
    vector<cuDoubleComplex> true_result;
    for (size_t i = 0; i < input.size(); i++) {
        auto x = input[i];
        auto x3 = cuCmul(cuCmul(x, x), make_cuDoubleComplex(x.x * 3.14159265, x.y * 3.14159265));
        auto x1 = make_cuDoubleComplex(x.x * 0.4, x.y * 0.4);
        true_result.push_back(make_cuDoubleComplex(x3.x + x1.x + 1, x3.y + x1.y));
    }
    // print_vector(true_result, 3, 7);

    /*
    Decrypt, decode, and print the result.
    */
    // cout << "===================== decrypt ============================" << endl;
    secret_key.decrypt(context, encrypted_result, plain_result);
    vector<cuDoubleComplex> result;
    encoder.decode(context, plain_result, result);
    // print_vector(result, 3, 7);
    bool correctness = true;
    for (size_t i = 0; i < size; i++)
        correctness &= result[i] == true_result[i];
    if (!correctness)
        throw std::logic_error("Stress test error");
    result.clear();
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
    std::vector v_alpha = {15};
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
                return;
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

        example_ckks_enc(parms, context, scale);
        example_ckks_add(parms, context, scale);
        example_ckks_mul_plain(parms, context, scale);
        example_ckks_mul(parms, context, scale);
        example_ckks_rotation(parms, context, scale);
        //        example_ckks_basics(parms, context, scale);
        //        for (auto i = 0; i < 1000; i++) {
        //            example_ckks_stress_test(parms, context, scale);
        //        }
    }
    cout << endl;
}

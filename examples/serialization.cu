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

#define EPSINON 0.001

inline bool operator==(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) {
    return fabs(lhs.x - rhs.x) < EPSINON;
}

inline bool compare_double(const double &lhs, const double &rhs) {
    return fabs(lhs - rhs) < EPSINON;
}

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

    /************************************** test plaintext save/load **************************************************/
    cout << "Save plaintext to file." << endl;
    ofstream outfile_pt("/tmp/pt.txt", ofstream::binary);
    pt.save(outfile_pt);
    outfile_pt.close();

    cout << "Load plaintext from file." << endl;
    ifstream infile_pt("/tmp/pt.txt", ifstream::binary);
    PhantomPlaintext pt_load;
    pt_load.load(infile_pt);
    infile_pt.close();

    encoder.decode(context, pt, result_double);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result_double, 3, 7);
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= compare_double(result_double[i], input_double[i]);
    }
    if (!correctness)
        throw std::logic_error("save/load plaintext error");

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

    /******************************** test symmetric ciphertext save/load *********************************************/
    cout << "Save symmetric ciphertext to file." << endl;
    ofstream outfile_sym("/tmp/x_symmetric_cipher_seed.txt", ofstream::binary);
    x_symmetric_cipher.save_symmetric(outfile_sym);
    outfile_sym.close();

    cout << "Load symmetric ciphertext from file." << endl;
    ifstream infile_sym("/tmp/x_symmetric_cipher_seed.txt", ifstream::binary);
    PhantomCiphertext x_symmetric_cipher_load;
    x_symmetric_cipher_load.load_symmetric(context, infile_sym);
    infile_sym.close();

    secret_key.decrypt(context, x_symmetric_cipher_load, x_symmetric_plain);
    encoder.decode(context, x_symmetric_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= result[i] == input[i];
    }
    if (!correctness)
        throw std::logic_error("save/load symmetric ciphertext error");

    // Asymmetric encryption check
    cout << "CKKS asymmetric test begin, encrypting ......" << endl;
    PhantomCiphertext x_asymmetric_cipher;
    public_key.encrypt_asymmetric(context, x_plain, x_asymmetric_cipher);
    PhantomPlaintext x_asymmetric_plain;
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

    /******************************* test asymmetric ciphertext save/load *********************************************/
    cout << "Save asymmetric ciphertext to file." << endl;
    ofstream outfile("/tmp/x_asymmetric_cipher.txt", ofstream::binary);
    x_asymmetric_cipher.save(outfile);
    outfile.close();

    cout << "Load asymmetric ciphertext from file." << endl;
    ifstream infile("/tmp/x_asymmetric_cipher.txt", ifstream::binary);
    PhantomCiphertext x_asymmetric_cipher_load;
    x_asymmetric_cipher_load.load(infile);
    infile.close();

    secret_key.decrypt(context, x_asymmetric_cipher_load, x_asymmetric_plain);
    encoder.decode(context, x_asymmetric_plain, result);
    cout << "Decode the decrypted plaintext." << endl;
    print_vector(result, 3, 7);
    correctness = true;
    for (size_t i = 0; i < slot_count; i++) {
        correctness &= result[i] == input[i];
    }
    if (!correctness)
        throw std::logic_error("save/load asymmetric ciphertext error");
}

void example_bfv_enc_sym() {
    std::cout << std::endl << "Testing BFV sym Enc & Dec" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);

        PhantomContext context(parms);
        print_parameters(context);

        PhantomSecretKey secret_key(context);

        PhantomCiphertext cipher;

        PhantomBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        PhantomPlaintext plain_matrix = batchEncoder.encode(context, pod_matrix);

        secret_key.encrypt_symmetric(context, plain_matrix, cipher);

        PhantomPlaintext plain = secret_key.decrypt(context, cipher);

        std::vector<uint64_t> res;
        batchEncoder.decode(context, plain, res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= res[idx] == pod_matrix[idx];
        if (!correctness)
            throw std::logic_error("Error in encrypt_symmetric & decrypt");

        /******************************** test symmetric ciphertext save/load *****************************************/
        cout << "Save symmetric ciphertext to file." << endl;
        ofstream outfile_sym("/tmp/x_symmetric_cipher_seed.txt", ofstream::binary);
        cipher.save_symmetric(outfile_sym);
        outfile_sym.close();

        cout << "Load symmetric ciphertext from file." << endl;
        ifstream infile_sym("/tmp/x_symmetric_cipher_seed.txt", ifstream::binary);
        PhantomCiphertext cipher_load;
        cipher_load.load_symmetric(context, infile_sym);
        infile_sym.close();

        secret_key.decrypt(context, cipher_load, plain);
        batchEncoder.decode(context, plain, res);
        cout << "Decode the decrypted plaintext." << endl;
        correctness = true;
        for (size_t i = 0; i < slot_count; i++) {
            correctness &= res[i] == pod_matrix[i];
        }
        if (!correctness)
            throw std::logic_error("save/load symmetric ciphertext error");
    }
}

void example_bfv_enc_asym() {
    std::cout << std::endl << "Testing BFV asym Enc & Dec" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);

        PhantomContext context(parms);
        print_parameters(context);

        PhantomSecretKey secret_key(context);

        /******************************************* test secret key save/load ****************************************/
        cout << "Save secret key to file." << endl;
        ofstream outfile_secretkey("/tmp/secret_key.txt", ofstream::binary);
        secret_key.save(outfile_secretkey);
        outfile_secretkey.close();

        cout << "Load secret key from file." << endl;
        ifstream infile_secretkey("/tmp/secret_key.txt", ifstream::binary);
        PhantomSecretKey secret_key_load;
        secret_key_load.load(infile_secretkey);
        infile_secretkey.close();

        PhantomPublicKey public_key = secret_key_load.gen_publickey(context);

        /******************************************* test public key save/load ****************************************/
        cout << "Save public key to file." << endl;
        ofstream outfile_pubkey("/tmp/public_key.txt", ofstream::binary);
        public_key.save(outfile_pubkey);
        outfile_pubkey.close();

        cout << "Load public key from file." << endl;
        ifstream infile_pubkey("/tmp/public_key.txt", ifstream::binary);
        PhantomPublicKey public_key_load;
        public_key_load.load(infile_pubkey);
        infile_pubkey.close();

        PhantomCiphertext cipher;

        PhantomBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        PhantomPlaintext plain_matrix = batchEncoder.encode(context, pod_matrix);

        public_key_load.encrypt_asymmetric(context, plain_matrix, cipher);

        PhantomPlaintext plain = secret_key.decrypt(context, cipher);

        std::vector<uint64_t> res;
        batchEncoder.decode(context, plain, res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= res[idx] == pod_matrix[idx];
        if (!correctness)
            throw std::logic_error("Error in encrypt_asymmetric & decrypt");

        /******************************** test asymmetric ciphertext save/load ****************************************/
        cout << "Save asymmetric ciphertext to file." << endl;
        ofstream outfile_sym("/tmp/x_asymmetric_cipher_seed.txt", ofstream::binary);
        cipher.save(outfile_sym);
        outfile_sym.close();

        cout << "Load asymmetric ciphertext from file." << endl;
        ifstream infile_sym("/tmp/x_asymmetric_cipher_seed.txt", ifstream::binary);
        PhantomCiphertext cipher_load;
        cipher_load.load(infile_sym);
        infile_sym.close();

        secret_key.decrypt(context, cipher_load, plain);
        batchEncoder.decode(context, plain, res);
        cout << "Decode the decrypted plaintext." << endl;
        correctness = true;
        for (size_t i = 0; i < slot_count; i++) {
            correctness &= res[i] == pod_matrix[i];
        }
        if (!correctness)
            throw std::logic_error("save/load asymmetric ciphertext error");
    }
}

void example_bfv_mul() {
    std::cout << std::endl << "Testing BFV mul" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);

        PhantomContext context(parms);
        print_parameters(context);

        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomRelinKey relin_key = secret_key.gen_relinkey(context);

        /******************************************** test relin key save/load ****************************************/
        cout << "Save relin key to file." << endl;
        ofstream outfile_relinkey("/tmp/relin_key.txt", ofstream::binary);
        relin_key.save(outfile_relinkey);
        outfile_relinkey.close();

        cout << "Load relin key from file." << endl;
        ifstream infile_relinkey("/tmp/relin_key.txt", ifstream::binary);
        PhantomRelinKey relin_key_load;
        relin_key_load.load(infile_relinkey);
        infile_relinkey.close();

        PhantomBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> input(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            input[idx] = idx;

        PhantomPlaintext pt = batchEncoder.encode(context, input);
        PhantomCiphertext ct = public_key.encrypt_asymmetric(context, pt);
        multiply_and_relin_inplace(context, ct, ct, relin_key_load);
        PhantomPlaintext plain = secret_key.decrypt(context, ct);

        std::vector<uint64_t> res;
        batchEncoder.decode(context, plain, res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= res[idx] == (input[idx] * input[idx]) % plainModulus.value();
        if (!correctness)
            throw std::logic_error("Error in BFV mul");
    }
}

void example_bfv_rotate() {
    std::cout << std::endl << "Testing BFV rotate" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);

        PhantomContext context(parms);
        print_parameters(context);

        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomGaloisKey galois_key = secret_key.create_galois_keys(context);

        /******************************************* test galois key save/load ****************************************/
        cout << "Save galois key to file." << endl;
        ofstream outfile_galoiskey("/tmp/galois_key.txt", ofstream::binary);
        galois_key.save(outfile_galoiskey);
        outfile_galoiskey.close();

        cout << "Load galois key from file." << endl;
        ifstream infile_galoiskey("/tmp/galois_key.txt", ifstream::binary);
        PhantomGaloisKey galois_key_load;
        galois_key_load.load(infile_galoiskey);
        infile_galoiskey.close();

        PhantomBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> input(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            input[idx] = idx;

        PhantomPlaintext pt = batchEncoder.encode(context, input);
        PhantomCiphertext ct = public_key.encrypt_asymmetric(context, pt);
        rotate_inplace(context, ct, 0, galois_key_load);
        PhantomPlaintext pt_dec = secret_key.decrypt(context, ct);
        std::vector<uint64_t> output = batchEncoder.decode(context, pt_dec);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count / 2; idx++)
            correctness &= output[idx + slot_count / 2] == idx;
        for (size_t idx = slot_count / 2; idx < slot_count; idx++)
            correctness &= output[idx - slot_count / 2] == idx;
        if (!correctness)
            throw std::logic_error("Error in rotate column asymmetric");
    }
}

int main() {
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
    }

    example_bfv_enc_sym();
    example_bfv_enc_asym();
    example_bfv_mul();
    example_bfv_rotate();
    return 0;
}

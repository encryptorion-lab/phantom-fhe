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

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

void example_bfv_basics() {
    std::cout << "Example: BFV Basics" << std::endl;

    /*
    In this example, we demonstrate performing simple computations (a polynomial
    evaluation) on encrypted integers using the BFV encryption scheme.

    The first task is to set up an instance of the EncryptionParameters class.
    It is critical to understand how the different parameters behave, how they
    affect the encryption scheme, performance, and the security level. There are
    three encryption parameters that are necessary to set:

        - poly_modulus_degree (degree of polynomial modulus);
        - coeff_modulus ([ciphertext] coefficient modulus);
        - plain_modulus (plaintext modulus; only for the BFV scheme).

    The BFV scheme cannot perform arbitrary computations on encrypted data.
    Instead, each ciphertext has a specific quantity called the `invariant noise
    budget' -- or `noise budget' for short -- measured in bits. The noise budget
    in a freshly encrypted ciphertext (initial noise budget) is determined by
    the encryption parameters. Homomorphic operations consume the noise budget
    at a rate also determined by the encryption parameters. In BFV the two basic
    operations allowed on encrypted data are additions and multiplications, of
    which additions can generally be thought of as being nearly free in terms of
    noise budget consumption compared to multiplications. Since noise budget
    consumption compounds in sequential multiplications, the most significant
    factor in choosing appropriate encryption parameters is the multiplicative
    depth of the arithmetic circuit that the user wants to evaluate on encrypted
    data. Once the noise budget of a ciphertext reaches zero it becomes too
    corrupted to be decrypted. Thus, it is essential to choose the parameters to
    be large enough to support the desired computation; otherwise the result is
    impossible to make sense of even with the secret key.
    */
    EncryptionParameters parms(scheme_type::bfv);

    /*
    The first parameter we set is the degree of the `polynomial modulus'. This
    must be a positive power of 2, representing the degree of a power-of-two
    cyclotomic polynomial; it is not necessary to understand what this means.

    Larger poly_modulus_degree makes ciphertext sizes larger and all operations
    slower, but enables more complicated encrypted computations. Recommended
    values are 1024, 2048, 4096, 8192, 16384, 32768, but it is also possible
    to go beyond this range.

    In this example we use a relatively small polynomial modulus. Anything
    smaller than this will enable only very restricted encrypted computations.
    */
    size_t poly_modulus_degree = 32768;
    parms.set_poly_modulus_degree(poly_modulus_degree);

    /*
    Next we set the [ciphertext] `coefficient modulus' (coeff_modulus). This
    parameter is a large integer, which is a product of distinct prime numbers,
    each up to 60 bits in size. It is represented as a vector of these prime
    numbers, each represented by an instance of the Modulus class. The
    bit-length of coeff_modulus means the sum of the bit-lengths of its prime
    factors.

    A larger coeff_modulus implies a larger noise budget, hence more encrypted
    computation capabilities. However, an upper bound for the total bit-length
    of the coeff_modulus is determined by the poly_modulus_degree, as follows:

        +----------------------------------------------------+
        | poly_modulus_degree | max coeff_modulus bit-length |
        +---------------------+------------------------------+
        | 1024                | 27                           |
        | 2048                | 54                           |
        | 4096                | 109                          |
        | 8192                | 218                          |
        | 16384               | 438                          |
        | 32768               | 881                          |
        | 65536               | 1792                         |
        +---------------------+------------------------------+

    These numbers can also be found in seal/util/hestdparms.h encoded
    in the function SEAL_HE_STD_PARMS_128_TC, and can also be obtained from the
    function

        CoeffModulus::MaxBitCount(poly_modulus_degree).

    For example, if poly_modulus_degree is 4096, the coeff_modulus could consist
    of three 36-bit primes (108 bits).

    Microsoft SEAL comes with helper functions for selecting the coeff_modulus.
    For new users the easiest way is to simply use

        CoeffModulus::BFVDefault(poly_modulus_degree),

    which returns std::vector<Modulus> consisting of a generally good choice
    for the given poly_modulus_degree.
    */
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));

    /*
    The plaintext modulus can be any positive integer, even though here we take
    it to be a power of two. In fact, in many cases one might instead want it
    to be a prime number; we will see this in later examples. The plaintext
    modulus determines the size of the plaintext data type and the consumption
    of noise budget in multiplications. Thus, it is essential to try to keep the
    plaintext data type as small as possible for best performance. The noise
    budget in a freshly encrypted ciphertext is

        ~ log2(coeff_modulus/plain_modulus) (bits)

    and the noise budget consumption in a homomorphic multiplication is of the
    form log2(plain_modulus) + (other terms).

    The plaintext modulus is specific to the BFV scheme, and cannot be set when
    using the CKKS scheme.
    */
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
    // parms.set_plain_modulus(1024);

    /*
    Now that all parameters are set, we are ready to construct a SEALContext
    object. This is a heavy class that checks the validity and properties of the
    parameters we just set.
    */
    PhantomContext context(parms);

    /*
    Print the parameters that we have chosen.
    */
    std::cout << __LINE__ << std::endl;
    cout << "Set encryption parameters and print" << endl;
    cout << endl;

    cout << "~~~~~~ A naive way to calculate 4(x^2+1)(x+1)^2. ~~~~~~" << endl;

    /*
    The encryption schemes in Microsoft SEAL are public key encryption schemes.
    For users unfamiliar with this terminology, a public key encryption scheme
    has a separate public key for encrypting data, and a separate secret key for
    decrypting data. This way multiple parties can encrypt data using the same
    shared public key, but only the proper recipient of the data can decrypt it
    with the secret key.

    We are now ready to generate the secret and public keys. For this purpose
    we need an instance of the KeyGenerator class. Constructing a KeyGenerator
    automatically generates a secret key. We can then create as many public
    keys for it as we want using KeyGenerator::create_public_key.

    Note that KeyGenerator::create_public_key has another overload that takes
    no parameters and returns a Serializable<PublicKey> object. We will discuss
    this in `6_serialization.cpp'.
    */
    PhantomSecretKey secret_key(context);

    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomCiphertext cipher;

    // encode
    PhantomBatchEncoder batchEncoder(context);
    size_t slot_count = batchEncoder.slot_count();
    size_t row_size = slot_count / 2;
    std::vector<uint64_t> pod_matrix(slot_count, 0);
    pod_matrix[0] = 0ULL;
    pod_matrix[1] = 1ULL;
    pod_matrix[2] = 2ULL;
    pod_matrix[3] = 3ULL;
    pod_matrix[row_size] = 4ULL;
    pod_matrix[row_size + 1] = 5ULL;
    pod_matrix[row_size + 2] = 6ULL;
    pod_matrix[row_size + 3] = 7ULL;
    PhantomPlaintext plain_matrix;
    batchEncoder.encode(context, pod_matrix, plain_matrix);
    secret_key.encrypt_symmetric(context, plain_matrix, cipher);
    public_key.encrypt_asymmetric(context, plain_matrix, cipher);

    /*
    To be able to encrypt we need to construct an instance of Encryptor. Note
    that the Encryptor only requires the public key, as expected. It is also
    possible to use Microsoft SEAL in secret-key mode by providing the Encryptor
    the secret key instead. We will discuss this in `6_serialization.cpp'.

    Encryptor encryptor(context, public_key);


    Computations on the ciphertexts are performed with the Evaluator class. In
    a real use-case the Evaluator would not be constructed by the same party
    that holds the secret key.

    Evaluator evaluator(context);


    We will of course want to decrypt our results to verify that everything worked,
    so we need to also construct an instance of Decryptor. Note that the Decryptor
    requires the secret key.

    Decryptor decryptor(context, secret_key);


    As an example, we evaluate the degree 4 polynomial

        4x^4 + 8x^3 + 8x^2 + 8x + 4

    over an encrypted x = 6. The coefficients of the polynomial can be considered
    as plaintext inputs, as we will see below. The computation is done modulo the
    plain_modulus 1024.

    While this examples is simple and easy to understand, it does not have much
    practical value. In later examples we will demonstrate how to compute more
    efficiently on encrypted integers and real or complex numbers.

    Plaintexts in the BFV scheme are polynomials of degree less than the degree
    of the polynomial modulus, and coefficients integers modulo the plaintext
    modulus. For readers with background in ring theory, the plaintext space is
    the polynomial quotient ring Z_T[X]/(X^N + 1), where N is poly_modulus_degree
    and T is plain_modulus.

    To get started, we create a plaintext containing the constant 6. For the
    plaintext element we use a constructor that takes the desired polynomial as
    a string with coefficients represented as hexadecimal numbers.

    print_line(__LINE__);
    int x = 6;
    Plaintext x_plain(to_string(x));
    cout << "Express x = " + to_string(x) + " as a plaintext polynomial 0x" + x_plain.to_string() + "." << endl;

    We then encrypt the plaintext, producing a ciphertext. We note that the
    Encryptor::encrypt function has another overload that takes as input only
    a plaintext and returns a Serializable<Ciphertext> object. We will discuss
    this in `6_serialization.cpp'.

    print_line(__LINE__);
    Ciphertext x_encrypted;
    cout << "Encrypt x_plain to x_encrypted." << endl;
    encryptor.encrypt(x_plain, x_encrypted);

    In Microsoft SEAL, a valid ciphertext consists of two or more polynomials
    whose coefficients are integers modulo the product of the primes in the
    coeff_modulus. The number of polynomials in a ciphertext is called its `size'
    and is given by Ciphertext::size(). A freshly encrypted ciphertext always
    has size 2.

    cout << "    + size of freshly encrypted x: " << x_encrypted.size() << endl;

    There is plenty of noise budget left in this freshly encrypted ciphertext.

    cout << "    + noise budget in freshly encrypted x: " << decryptor.invariant_noise_budget(x_encrypted) << " bits"
         << endl;

    We decrypt the ciphertext and print the resulting plaintext in order to
    demonstrate correctness of the encryption.

    Plaintext x_decrypted;
    cout << "    + decryption of x_encrypted: ";
    decryptor.decrypt(x_encrypted, x_decrypted);
    cout << "0x" << x_decrypted.to_string() << " ...... Correct." << endl;

    When using Microsoft SEAL, it is typically advantageous to compute in a way
    that minimizes the longest chain of sequential multiplications. In other
    words, encrypted computations are best evaluated in a way that minimizes
    the multiplicative depth of the computation, because the total noise budget
    consumption is proportional to the multiplicative depth. For example, for
    our example computation it is advantageous to factorize the polynomial as

        4x^4 + 8x^3 + 8x^2 + 8x + 4 = 4(x + 1)^2 * (x^2 + 1)

    to obtain a simple depth 2 representation. Thus, we compute (x + 1)^2 and
    (x^2 + 1) separately, before multiplying them, and multiplying by 4.

    First, we compute x^2 and add a plaintext "1". We can clearly see from the
    print-out that multiplication has consumed a lot of noise budget. The user
    can vary the plain_modulus parameter to see its effect on the rate of noise
    budget consumption.

    print_line(__LINE__);
    cout << "Compute x_sq_plus_one (x^2+1)." << endl;
    Ciphertext x_sq_plus_one;
    evaluator.square(x_encrypted, x_sq_plus_one);
    Plaintext plain_one("1");
    evaluator.add_plain_inplace(x_sq_plus_one, plain_one);

    Encrypted multiplication results in the output ciphertext growing in size.
    More precisely, if the input ciphertexts have size M and N, then the output
    ciphertext after homomorphic multiplication will have size M+N-1. In this
    case we perform a squaring, and observe both size growth and noise budget
    consumption.

    cout << "    + size of x_sq_plus_one: " << x_sq_plus_one.size() << endl;
    cout << "    + noise budget in x_sq_plus_one: " << decryptor.invariant_noise_budget(x_sq_plus_one) << " bits"
         << endl;

    Even though the size has grown, decryption works as usual as long as noise
    budget has not reached 0.

    Plaintext decrypted_result;
    cout << "    + decryption of x_sq_plus_one: ";
    decryptor.decrypt(x_sq_plus_one, decrypted_result);
    cout << "0x" << decrypted_result.to_string() << " ...... Correct." << endl;

    Next, we compute (x + 1)^2.

    print_line(__LINE__);
    cout << "Compute x_plus_one_sq ((x+1)^2)." << endl;
    Ciphertext x_plus_one_sq;
    evaluator.add_plain(x_encrypted, plain_one, x_plus_one_sq);
    evaluator.square_inplace(x_plus_one_sq);
    cout << "    + size of x_plus_one_sq: " << x_plus_one_sq.size() << endl;
    cout << "    + noise budget in x_plus_one_sq: " << decryptor.invariant_noise_budget(x_plus_one_sq) << " bits"
         << endl;
    cout << "    + decryption of x_plus_one_sq: ";
    decryptor.decrypt(x_plus_one_sq, decrypted_result);
    cout << "0x" << decrypted_result.to_string() << " ...... Correct." << endl;


    Finally, we multiply (x^2 + 1) * (x + 1)^2 * 4.

    print_line(__LINE__);
    cout << "Compute encrypted_result (4(x^2+1)(x+1)^2)." << endl;
    Ciphertext encrypted_result;
    Plaintext plain_four("4");
    evaluator.multiply_plain_inplace(x_sq_plus_one, plain_four);
    evaluator.multiply(x_sq_plus_one, x_plus_one_sq, encrypted_result);
    cout << "    + size of encrypted_result: " << encrypted_result.size() << endl;
    cout << "    + noise budget in encrypted_result: " << decryptor.invariant_noise_budget(encrypted_result) << " bits"
         << endl;
    cout << "NOTE: Decryption can be incorrect if noise budget is zero." << endl;

    cout << endl;
    cout << "~~~~~~ A better way to calculate 4(x^2+1)(x+1)^2. ~~~~~~" << endl;

    Noise budget has reached 0, which means that decryption cannot be expected
    to give the correct result. This is because both ciphertexts x_sq_plus_one
    and x_plus_one_sq consist of 3 polynomials due to the previous squaring
    operations, and homomorphic operations on large ciphertexts consume much more
    noise budget than computations on small ciphertexts. Computing on smaller
    ciphertexts is also computationally significantly cheaper.

    `Relinearization' is an operation that reduces the size of a ciphertext after
    multiplication back to the initial size, 2. Thus, relinearizing one or both
    input ciphertexts before the next multiplication can have a huge positive
    impact on both noise growth and performance, even though relinearization has
    a significant computational cost itself. It is only possible to relinearize
    size 3 ciphertexts down to size 2, so often the user would want to relinearize
    after each multiplication to keep the ciphertext sizes at 2.

    Relinearization requires special `relinearization keys', which can be thought
    of as a kind of public key. Relinearization keys can easily be created with
    the KeyGenerator.

    Relinearization is used similarly in both the BFV and the CKKS schemes, but
    in this example we continue using BFV. We repeat our computation from before,
    but this time relinearize after every multiplication.

    print_line(__LINE__);
    cout << "Generate relinearization keys." << endl;
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    We now repeat the computation relinearizing after each multiplication.

    print_line(__LINE__);
    cout << "Compute and relinearize x_squared (x^2)," << endl;
    cout << string(13, ' ') << "then compute x_sq_plus_one (x^2+1)" << endl;
    Ciphertext x_squared;
    evaluator.square(x_encrypted, x_squared);
    cout << "    + size of x_squared: " << x_squared.size() << endl;
    evaluator.relinearize_inplace(x_squared, relin_keys);
    cout << "    + size of x_squared (after relinearization): " << x_squared.size() << endl;
    evaluator.add_plain(x_squared, plain_one, x_sq_plus_one);
    cout << "    + noise budget in x_sq_plus_one: " << decryptor.invariant_noise_budget(x_sq_plus_one) << " bits"
         << endl;
    cout << "    + decryption of x_sq_plus_one: ";
    decryptor.decrypt(x_sq_plus_one, decrypted_result);
    cout << "0x" << decrypted_result.to_string() << " ...... Correct." << endl;

    print_line(__LINE__);
    Ciphertext x_plus_one;
    cout << "Compute x_plus_one (x+1)," << endl;
    cout << string(13, ' ') << "then compute and relinearize x_plus_one_sq ((x+1)^2)." << endl;
    evaluator.add_plain(x_encrypted, plain_one, x_plus_one);
    evaluator.square(x_plus_one, x_plus_one_sq);
    cout << "    + size of x_plus_one_sq: " << x_plus_one_sq.size() << endl;
    evaluator.relinearize_inplace(x_plus_one_sq, relin_keys);
    cout << "    + noise budget in x_plus_one_sq: " << decryptor.invariant_noise_budget(x_plus_one_sq) << " bits"
         << endl;
    cout << "    + decryption of x_plus_one_sq: ";
    decryptor.decrypt(x_plus_one_sq, decrypted_result);
    cout << "0x" << decrypted_result.to_string() << " ...... Correct." << endl;

    print_line(__LINE__);
    cout << "Compute and relinearize encrypted_result (4(x^2+1)(x+1)^2)." << endl;
    evaluator.multiply_plain_inplace(x_sq_plus_one, plain_four);
    evaluator.multiply(x_sq_plus_one, x_plus_one_sq, encrypted_result);
    cout << "    + size of encrypted_result: " << encrypted_result.size() << endl;
    evaluator.relinearize_inplace(encrypted_result, relin_keys);
    cout << "    + size of encrypted_result (after relinearization): " << encrypted_result.size() << endl;
    cout << "    + noise budget in encrypted_result: " << decryptor.invariant_noise_budget(encrypted_result) << " bits"
         << endl;

    cout << endl;
    cout << "NOTE: Notice the increase in remaining noise budget." << endl;

    Relinearization clearly improved our noise consumption. We have still plenty
    of noise budget left, so we can expect the correct answer when decrypting.

    print_line(__LINE__);
    cout << "Decrypt encrypted_result (4(x^2+1)(x+1)^2)." << endl;
    decryptor.decrypt(encrypted_result, decrypted_result);
    cout << "    + decryption of 4(x^2+1)(x+1)^2 = 0x" << decrypted_result.to_string() << " ...... Correct." << endl;
    cout << endl;

    For x=6, 4(x^2+1)(x+1)^2 = 7252. Since the plaintext modulus is set to 1024,
    this result is computed in integers modulo 1024. Therefore the expected output
    should be 7252 % 1024 == 84, or 0x54 in hexadecimal.


    Sometimes we create customized encryption parameters which turn out to be invalid.
    Microsoft SEAL can interpret the reason why parameters are considered invalid.
    Here we simply reduce the polynomial modulus degree to make the parameters not
    compliant with the HomomorphicEncryption.org security standard.

    print_line(__LINE__);
    cout << "An example of invalid parameters" << endl;
    parms.set_poly_modulus_degree(2048);
    context = SEALContext(parms);
    print_parameters(context);
    cout << "Parameter validation (failed): " << context.parameter_error_message() << endl << endl;

    This information is helpful to fix invalid encryption parameters.
    */
}

void example_bfv_batch_unbatch() {
    std::cout << std::endl << "Testing BFV encode & decode" << std::endl;
    for (size_t poly_modulus_degree = 8192; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);

        PhantomContext context(parms);
        print_parameters(context);

        PhantomBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        PhantomPlaintext plain_matrix;
        batchEncoder.encode(context, pod_matrix, plain_matrix);

        std::vector<uint64_t> res;
        batchEncoder.decode(context, plain_matrix, res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= res[idx] == pod_matrix[idx];
        if (!correctness)
            throw std::logic_error("Error in encode & decode");
    }
}

void example_bfv_encrypt_decrypt() {
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

        PhantomPublicKey public_key = secret_key.gen_publickey(context);

        PhantomCiphertext cipher;

        PhantomBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        PhantomPlaintext plain_matrix;
        batchEncoder.encode(context, pod_matrix, plain_matrix);

        secret_key.encrypt_symmetric(context, plain_matrix, cipher);
        auto noise_budget = secret_key.invariant_noise_budget(context, cipher);
        cout << "cipher noise budget is: " << noise_budget << endl;
        PhantomPlaintext plain;

        secret_key.decrypt(context, cipher, plain);

        std::vector<uint64_t> res;
        batchEncoder.decode(context, plain, res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= res[idx] == pod_matrix[idx];
        if (!correctness)
            throw std::logic_error("Error in encrypt_symmetric & decrypt");
    }
}

void example_bfv_encrypt_decrypt_asym() {
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
        PhantomPublicKey public_key = secret_key.gen_publickey(context);

        /*start = chrono::high_resolution_clock::now();
        for (size_t i = 0; i < 100; i++)
            secret_key.gen_publickey(context, public_key);
        finish = chrono::high_resolution_clock::now();
        microseconds = chrono::duration_cast<std::chrono::microseconds>(finish - start);
        std::cout << "public key generation time is us: " << microseconds.count() / 100 << std::endl;
        */
        PhantomCiphertext cipher;

        PhantomBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++) {
            pod_matrix[idx] = rand() % parms.plain_modulus().value();
        }
        PhantomPlaintext plain_matrix;
        batchEncoder.encode(context, pod_matrix, plain_matrix);

        public_key.encrypt_asymmetric(context, plain_matrix, cipher);
        auto noise_budget = secret_key.invariant_noise_budget(context, cipher);
        cout << "cipher noise budget is: " << noise_budget << endl;
        PhantomPlaintext plain;
        secret_key.decrypt(context, cipher, plain);

        std::vector<uint64_t> res;
        batchEncoder.decode(context, plain, res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= res[idx] == pod_matrix[idx];
        if (!correctness)
            throw std::logic_error("Error in encrypt_asymmetric & decrypt");
    }
}

void example_bfv_add() {
    std::cout << std::endl << "Testing BFV (sym, asym) cipher Add" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomCiphertext sym_cipher;
        PhantomCiphertext asym_cipher;
        PhantomBatchEncoder batchEncoder(context);
        PhantomPlaintext plain_matrix;
        PhantomPlaintext dec_plain;
        PhantomPlaintext dec_asym_plain;

        std::vector<uint64_t> dec_res;
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        batchEncoder.encode(context, pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);
        PhantomCiphertext sym_cipher_copy(sym_cipher);
        PhantomCiphertext destination;
        destination = sym_cipher;
        add_inplace(context, destination, sym_cipher_copy);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(context, dec_plain, dec_res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= dec_res[idx] == 2 * pod_matrix[idx];
        if (!correctness)
            throw std::logic_error("Error in add symmetric");

        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
        PhantomCiphertext asym_cipher_copy(asym_cipher);
        PhantomCiphertext destination2;
        destination2 = asym_cipher;
        add_inplace(context, destination2, asym_cipher_copy);
        secret_key.decrypt(context, destination2, dec_asym_plain);

        batchEncoder.decode(context, dec_asym_plain, dec_res);

        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= dec_res[idx] == 2 * pod_matrix[idx];
        if (!correctness)
            throw std::logic_error("Error in add asymmetric");
    }
}

void example_bfv_sub() {
    std::cout << std::endl << "Testing BFV (sym, asym) cipher sub" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomCiphertext sym_cipher;
        PhantomCiphertext asym_cipher;
        PhantomBatchEncoder batchEncoder(context);
        PhantomPlaintext plain_matrix;
        PhantomPlaintext dec_plain;
        PhantomPlaintext dec_asym_plain;
        std::vector<uint64_t> dec_res;
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        batchEncoder.encode(context, pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);
        PhantomCiphertext sym_cipher_copy;
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher_copy);
        PhantomCiphertext destination;
        destination = sym_cipher;
        sub_inplace(context, destination, sym_cipher_copy);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(context, dec_plain, dec_res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= dec_res[idx] == 0;
        if (!correctness)
            throw std::logic_error("Error in sub symmetric");

        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
        PhantomCiphertext asym_cipher_copy;
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher_copy);
        PhantomCiphertext destination2;
        destination2 = asym_cipher;
        sub_inplace(context, destination2, asym_cipher_copy);
        secret_key.decrypt(context, destination2, dec_asym_plain);

        batchEncoder.decode(context, dec_asym_plain, dec_res);

        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= dec_res[idx] == 0;
        if (!correctness)
            throw std::logic_error("Error in sub asymmetric");
    }
}

void example_bfv_mul() {
    std::cout << std::endl << "Testing BFV (sym, asym) cipher mul" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768; poly_modulus_degree <<= 1) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);
        PhantomContext context(parms);

        print_parameters(context);

        PhantomSecretKey secret_key(context);
        PhantomCiphertext sym_cipher;
        PhantomCiphertext asym_cipher;
        PhantomBatchEncoder batchEncoder(context);
        PhantomPlaintext plain_matrix;
        PhantomPlaintext dec_plain;
        PhantomPlaintext dec_asym_plain;

        std::vector<uint64_t> dec_res;
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx % plainModulus.value();

        PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
        batchEncoder.encode(context, pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);
        PhantomCiphertext sym_cipher_copy(sym_cipher);
        // PhantomCiphertext destination(context);
        multiply_inplace(context, sym_cipher, sym_cipher_copy);
        relinearize_inplace(context, sym_cipher, relin_keys);
        secret_key.decrypt(context, sym_cipher, dec_plain);
        batchEncoder.decode(context, dec_plain, dec_res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= dec_res[idx] == idx * idx % plainModulus.value();
        if (!correctness)
            throw std::logic_error("Error in mul symmetric");

        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
        PhantomCiphertext asym_cipher_copy(asym_cipher);
        // PhantomCiphertext destination2(context);
        multiply_inplace(context, asym_cipher, asym_cipher_copy);
        relinearize_inplace(context, asym_cipher, relin_keys);
        secret_key.decrypt(context, asym_cipher, dec_asym_plain);

        batchEncoder.decode(context, dec_asym_plain, dec_res);

        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= dec_res[idx] == idx * idx % plainModulus.value();
        if (!correctness)
            throw std::logic_error("Error in mul asymmetric");
    }
}

void example_bfv_square() {
    std::cout << std::endl << "Testing BFV (sym, asym) cipher square" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);
        parms.set_mul_tech(mul_tech_type::behz);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomCiphertext sym_cipher;
        PhantomCiphertext asym_cipher;
        PhantomBatchEncoder batchEncoder(context);
        PhantomPlaintext plain_matrix;
        PhantomPlaintext dec_plain;
        PhantomPlaintext dec_asym_plain;

        std::vector<uint64_t> dec_res;
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
        batchEncoder.encode(context, pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);
        // PhantomCiphertext destination(context);
        multiply_inplace(context, sym_cipher, sym_cipher);
        relinearize_inplace(context, sym_cipher, relin_keys);
        secret_key.decrypt(context, sym_cipher, dec_plain);
        batchEncoder.decode(context, dec_plain, dec_res);

        bool correctness = true;
        size_t threshold = std::log2(plainModulus.value());
        for (size_t idx = 0; idx < threshold; idx++)
            correctness &= dec_res[idx] == idx * idx;
        if (!correctness)
            throw std::logic_error("Error in square symmetric");

        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
        multiply_inplace(context, asym_cipher, asym_cipher);
        relinearize_inplace(context, asym_cipher, relin_keys);
        secret_key.decrypt(context, asym_cipher, dec_asym_plain);

        batchEncoder.decode(context, dec_asym_plain, dec_res);

        for (size_t idx = 0; idx < threshold; idx++)
            correctness &= dec_res[idx] == idx * idx;
        if (!correctness)
            throw std::logic_error("Error in square asymmetric");
    }
}

void example_bfv_add_plain() {
    std::cout << std::endl << "Testing BFV (sym, asym) cipher Add Plain" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomCiphertext sym_cipher;
        PhantomCiphertext asym_cipher;
        PhantomBatchEncoder batchEncoder(context);
        PhantomPlaintext plain_matrix;
        PhantomPlaintext dec_plain;
        PhantomPlaintext dec_asym_plain;

        std::vector<uint64_t> dec_res;
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        batchEncoder.encode(context, pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);
        add_plain_inplace(context, sym_cipher, plain_matrix);
        secret_key.decrypt(context, sym_cipher, dec_plain);
        batchEncoder.decode(context, dec_plain, dec_res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= dec_res[idx] == 2 * pod_matrix[idx];
        if (!correctness)
            throw std::logic_error("Error in add_plain symmetric");

        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
        add_plain_inplace(context, asym_cipher, plain_matrix);
        secret_key.decrypt(context, asym_cipher, dec_asym_plain);

        batchEncoder.decode(context, dec_asym_plain, dec_res);

        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= dec_res[idx] == 2 * pod_matrix[idx];
        if (!correctness)
            throw std::logic_error("Error in add_plain asymmetric");
    }
}

void example_bfv_sub_plain() {
    std::cout << std::endl << "Testing BFV (sym, asym) cipher sub Plain" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomCiphertext sym_cipher;
        PhantomCiphertext asym_cipher;
        PhantomBatchEncoder batchEncoder(context);
        PhantomPlaintext plain_matrix;
        PhantomPlaintext dec_plain;
        PhantomPlaintext dec_asym_plain;

        std::vector<uint64_t> dec_res;
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        batchEncoder.encode(context, pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);
        sub_plain_inplace(context, sym_cipher, plain_matrix);
        secret_key.decrypt(context, sym_cipher, dec_plain);
        batchEncoder.decode(context, dec_plain, dec_res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= dec_res[idx] == 0;
        if (!correctness)
            throw std::logic_error("Error in sub_plain symmetric");

        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
        sub_plain_inplace(context, asym_cipher, plain_matrix);
        secret_key.decrypt(context, asym_cipher, dec_asym_plain);

        batchEncoder.decode(context, dec_asym_plain, dec_res);

        for (size_t idx = 0; idx < slot_count; idx++)
            correctness &= dec_res[idx] == 0;
        if (!correctness)
            throw std::logic_error("Error in sub_plain asymmetric");
    }
}

// plain has more than one slot set
void example_bfv_mul_many_plain() {
    std::cout << std::endl << "Testing BFV (sym, asym) cipher mul many Plain" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);

        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomCiphertext sym_cipher;
        PhantomCiphertext asym_cipher;
        PhantomBatchEncoder batchEncoder(context);
        PhantomPlaintext plain_matrix;
        PhantomPlaintext dec_plain;
        PhantomPlaintext dec_asym_plain;

        std::vector<uint64_t> dec_res;
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        batchEncoder.encode(context, pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);
        multiply_plain_inplace(context, sym_cipher, plain_matrix);
        secret_key.decrypt(context, sym_cipher, dec_plain);
        batchEncoder.decode(context, dec_plain, dec_res);

        bool correctness = true;
        size_t threshold = std::log2(plainModulus.value());
        for (size_t idx = 0; idx < threshold; idx++)
            correctness &= dec_res[idx] == idx * idx;
        if (!correctness)
            throw std::logic_error("Error in mul_many_plain symmetric");

        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
        multiply_plain_inplace(context, asym_cipher, plain_matrix);
        secret_key.decrypt(context, asym_cipher, dec_asym_plain);

        batchEncoder.decode(context, dec_asym_plain, dec_res);

        for (size_t idx = 0; idx < threshold; idx++)
            correctness &= dec_res[idx] == idx * idx;
        if (!correctness)
            throw std::logic_error("Error in mul_many_plain asymmetric");
    }
}

// plain has more than one slot set
void example_bfv_mul_one_plain() {
    std::cout << std::endl << "Testing BFV (sym, asym) cipher mul one Plain" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomCiphertext sym_cipher;
        PhantomCiphertext asym_cipher;
        PhantomBatchEncoder batchEncoder(context);
        PhantomPlaintext plain_matrix;
        PhantomPlaintext dec_plain;
        PhantomPlaintext dec_asym_plain;

        std::vector<uint64_t> dec_res;
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        batchEncoder.encode(context, pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);

        for (size_t idx = 0; idx < slot_count; idx++) {
            pod_matrix[idx] = 0;
        }
        pod_matrix[49] = 49;
        batchEncoder.encode(context, pod_matrix, plain_matrix);
        multiply_plain_inplace(context, sym_cipher, plain_matrix);
        secret_key.decrypt(context, sym_cipher, dec_plain);
        batchEncoder.decode(context, dec_plain, dec_res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count && idx != 49; idx++)
            correctness &= dec_res[idx] == 0;
        correctness &= dec_res[49] == 49 * 49;
        if (!correctness)
            throw std::logic_error("Error in mul_one_plain symmetric");

        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        batchEncoder.encode(context, pod_matrix, plain_matrix);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
        for (size_t idx = 0; idx < slot_count; idx++) {
            pod_matrix[idx] = 0;
        }
        pod_matrix[49] = 49;
        batchEncoder.encode(context, pod_matrix, plain_matrix);
        multiply_plain_inplace(context, asym_cipher, plain_matrix);
        secret_key.decrypt(context, asym_cipher, dec_asym_plain);

        batchEncoder.decode(context, dec_asym_plain, dec_res);

        for (size_t idx = 0; idx < slot_count && idx != 49; idx++)
            correctness &= dec_res[idx] == 0;
        correctness &= dec_res[49] == 49 * 49;
        if (!correctness)
            throw std::logic_error("Error in mul_one_plain asymmetric");
    }
}

uint64_t pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    for (uint64_t i = 0; i < exp; i++) {
        res = (res * base) % mod;
    }
    return res;
}

// plain has more than one slot set
void example_bfv_rotate_column() {
    std::cout << std::endl << "Testing BFV (sym, asym) cipher rotate column" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomCiphertext sym_cipher;
        PhantomCiphertext asym_cipher;
        PhantomBatchEncoder batchEncoder(context);
        PhantomPlaintext plain_matrix;
        PhantomPlaintext dec_plain;
        PhantomPlaintext dec_asym_plain;

        std::vector<uint64_t> dec_res;
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        PhantomGaloisKey galois_key = secret_key.create_galois_keys(context);
        batchEncoder.encode(context, pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);
        rotate_columns_inplace(context, sym_cipher, galois_key);
        secret_key.decrypt(context, sym_cipher, dec_plain);
        batchEncoder.decode(context, dec_plain, dec_res);

        bool correctness = true;
        for (size_t idx = 0; idx < slot_count / 2; idx++)
            correctness &= dec_res[idx + slot_count / 2] == idx;
        for (size_t idx = slot_count / 2; idx < slot_count; idx++)
            correctness &= dec_res[idx - slot_count / 2] == idx;
        if (!correctness)
            throw std::logic_error("Error in rotate column symmetric");

        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
        rotate_columns_inplace(context, asym_cipher, galois_key);
        secret_key.decrypt(context, asym_cipher, dec_asym_plain);

        batchEncoder.decode(context, dec_asym_plain, dec_res);

        for (size_t idx = 0; idx < slot_count / 2; idx++)
            correctness &= dec_res[idx + slot_count / 2] == idx;
        for (size_t idx = slot_count / 2; idx < slot_count; idx++)
            correctness &= dec_res[idx - slot_count / 2] == idx;
        if (!correctness)
            throw std::logic_error("Error in rotate column asymmetric");
    }
}

// plain has more than one slot set
void example_bfv_rotate_row() {
    std::cout << std::endl << "Testing BFV (sym, asym) cipher rotate row" << std::endl;
    for (size_t poly_modulus_degree = 4096; poly_modulus_degree <= 32768;
         poly_modulus_degree = poly_modulus_degree * 2) {
        EncryptionParameters parms(scheme_type::bfv);
        std::cout << "testing poly_modulus_degree = " << poly_modulus_degree << std::endl;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);
        PhantomContext context(parms);
        PhantomSecretKey secret_key(context);
        PhantomCiphertext sym_cipher;
        PhantomCiphertext asym_cipher;
        PhantomBatchEncoder batchEncoder(context);
        PhantomPlaintext plain_matrix;
        PhantomPlaintext dec_plain;
        PhantomPlaintext dec_asym_plain;

        std::vector<uint64_t> dec_res;
        size_t slot_count = batchEncoder.slot_count();
        std::vector<uint64_t> pod_matrix(slot_count, 0);
        for (size_t idx = 0; idx < slot_count; idx++)
            pod_matrix[idx] = idx;

        int step = -1;

        PhantomGaloisKey galois_key = secret_key.create_galois_keys(context);
        batchEncoder.encode(context, pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);
        rotate_rows_inplace(context, sym_cipher, step, galois_key);
        secret_key.decrypt(context, sym_cipher, dec_plain);
        batchEncoder.decode(context, dec_plain, dec_res);

        bool correctness = true;
        for (size_t idx = 0; idx < 500; idx++) {
            if (step < 0)
                correctness &= dec_res[static_cast<size_t>(static_cast<int>(idx) - step)] == idx;
            if (step > 0)
                correctness &= dec_res[idx] == idx + step;
        }
        if (!correctness)
            throw std::logic_error("Error in rotate row symmetric");

        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
        rotate_rows_inplace(context, asym_cipher, step, galois_key);
        secret_key.decrypt(context, asym_cipher, dec_asym_plain);
        batchEncoder.decode(context, dec_asym_plain, dec_res);

        for (size_t idx = 0; idx < 500; idx++) {
            if (step < 0)
                correctness &= dec_res[static_cast<size_t>(static_cast<int>(idx) - step)] == idx;
            if (step > 0)
                correctness &= dec_res[idx] == idx + step;
        }
        if (!correctness)
            throw std::logic_error("Error in rotate row asymmetric");
    }
}

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

    PhantomSecretKey secret_key(context);

    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomCiphertext cipher;

    PhantomBatchEncoder batchEncoder(context);
    size_t slot_count = batchEncoder.slot_count();
    std::vector<uint64_t> pod_matrix(slot_count, 0);
    for (size_t idx = 0; idx < slot_count; idx++)
        pod_matrix[idx] = idx;

    PhantomPlaintext plain_matrix;
    batchEncoder.encode(context, pod_matrix, plain_matrix);

    secret_key.encrypt_symmetric(context, plain_matrix, cipher);
    auto noise_budget = secret_key.invariant_noise_budget(context, cipher);
    cout << "cipher noise budget is: " << noise_budget << endl;
    PhantomCiphertext cipher_copy(cipher);
    PhantomPlaintext plain;

    size_t times = 1;
    auto start = chrono::high_resolution_clock::now();
    for (size_t idx = 0; idx < times; idx++) {
        secret_key.decrypt(context, cipher_copy, plain);
    }
    auto finish = chrono::high_resolution_clock::now();
    auto microseconds = chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "bfv decrypt time is us: " << microseconds.count() / times << std::endl;

    secret_key.decrypt(context, cipher, plain);

    std::vector<uint64_t> res;
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

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    /*start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; i++)
        secret_key.gen_publickey(context, public_key);
    finish = chrono::high_resolution_clock::now();
    microseconds = chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "public key generation time is us: " << microseconds.count() / 100 << std::endl;
    */
    PhantomCiphertext cipher;

    PhantomBatchEncoder batchEncoder(context);
    size_t slot_count = batchEncoder.slot_count();
    std::vector<uint64_t> pod_matrix(slot_count, 0);
    for (size_t idx = 0; idx < slot_count; idx++) {
        pod_matrix[idx] = rand() % parms.plain_modulus().value();
    }
    PhantomPlaintext plain_matrix;
    batchEncoder.encode(context, pod_matrix, plain_matrix);

    public_key.encrypt_asymmetric(context, plain_matrix, cipher);
    auto noise_budget = secret_key.invariant_noise_budget(context, cipher);
    cout << "cipher noise budget is: " << noise_budget << endl;
    PhantomPlaintext plain;
    secret_key.decrypt(context, cipher, plain);

    std::vector<uint64_t> res;
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

    PhantomSecretKey secret_key(context);
    PhantomCiphertext sym_cipher;
    PhantomCiphertext asym_cipher;
    PhantomBatchEncoder batchEncoder(context);
    PhantomPlaintext plain_matrix;
    PhantomPlaintext dec_plain;
    PhantomPlaintext dec_asym_plain;

    std::vector<uint64_t> dec_res;
    size_t slot_count = batchEncoder.slot_count();
    std::vector<uint64_t> pod_matrix(slot_count, 0);
    for (size_t idx = 0; idx < slot_count; idx++)
        pod_matrix[idx] = idx;

    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    batchEncoder.encode(context, pod_matrix, plain_matrix);
    secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);
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

    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
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

    PhantomSecretKey secret_key(context);
    PhantomCiphertext sym_cipher;
    PhantomCiphertext asym_cipher;
    PhantomBatchEncoder batchEncoder(context);
    PhantomPlaintext plain_matrix;
    PhantomPlaintext dec_plain;
    PhantomPlaintext dec_asym_plain;

    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    std::vector<uint64_t> dec_res;
    size_t slot_count = batchEncoder.slot_count();
    std::vector<uint64_t> pod_matrix(slot_count, 0);
    for (size_t idx = 0; idx < slot_count; idx++)
        pod_matrix[idx] = idx;

    batchEncoder.encode(context, pod_matrix, plain_matrix);
    secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);

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

    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher);
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

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key;
    PhantomCiphertext sym_cipher;
    PhantomCiphertext asym_cipher;
    PhantomBatchEncoder batchEncoder(context);
    PhantomPlaintext plain_matrix;
    PhantomPlaintext dec_plain;
    PhantomPlaintext dec_asym_plain;

    std::vector<uint64_t> dec_res;
    size_t slot_count = batchEncoder.slot_count();
    std::vector<uint64_t> pod_matrix(slot_count, 0);
    for (size_t idx = 0; idx < slot_count; idx++)
        pod_matrix[idx] = idx;

    PhantomRelinKey relin_keys;

    if (context.using_keyswitching()) {
        relin_keys = secret_key.gen_relinkey(context);
    }
    batchEncoder.encode(context, pod_matrix, plain_matrix);
    secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher);

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
        CUDATimer timer("mult&relin", *global_variables::default_stream);
        for (size_t idx = 0; idx < n_tests; idx++) {
            PhantomCiphertext sym_cipher_copy2(sym_cipher_copy);
            timer.start();
            multiply_and_relin_inplace(context, sym_cipher_copy2, sym_cipher, relin_keys);
            timer.stop();
        }
        multiply_and_relin_inplace(context, sym_cipher_copy, sym_cipher, relin_keys);
    }

    CUDATimer timer_dec("decrypt", *global_variables::default_stream);
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

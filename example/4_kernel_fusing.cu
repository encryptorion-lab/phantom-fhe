#include "example.h"
#include "util.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

void bgv_moddown_test(EncryptionParameters &parms) {
    PhantomContext context(parms);
    print_parameters(context);
    cout << endl;

    auto &context_data = context.get_context_data(context.get_first_index());
    auto &first_parms = context_data.parms();
    auto &plain_modulus = first_parms.plain_modulus();
    size_t poly_modulus_degree = first_parms.poly_modulus_degree();

    print_timer_banner();

    PhantomSecretKey secret_key(context);

    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    /*
    Generate Galois keys. In larger examples the Galois keys can use a lot of
    memory, which can be a problem in constrained systems. The user should
    try some larger runs of the test and observe their effect on the
    memory pool allocation size. The key generation can also take a long time,
    as can be observed from the print-out.
    */
    // PhantomGaloisKey gal_keys(context);
    // secret_key.create_galois_keys(context, gal_keys);

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
    batch_encoder.encode(context, pod_vector, plain);

    /*
    [Unbatching]
    We unbatch what we just batched.
    */
    vector<uint64_t> pod_vector2(slot_count);
    batch_encoder.decode(context, plain, pod_vector2);

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
    public_key.encrypt_asymmetric(context, plain, encrypted);

    /*
    [Decryption]
    We decrypt what we just encrypted.
    */
    secret_key.decrypt(context, encrypted, plain);

    // homomorphic operations

    PhantomPlaintext plain1;
    PhantomPlaintext plain2;
    PhantomCiphertext encrypted1;
    batch_encoder.encode(context, vector<uint64_t>(slot_count, 1), plain1);
    public_key.encrypt_asymmetric(context, plain1, encrypted1);
    PhantomCiphertext encrypted2;
    batch_encoder.encode(context, vector<uint64_t>(slot_count, 1), plain2);
    public_key.encrypt_asymmetric(context, plain2, encrypted2);

    /*
    [Add]
    We create two ciphertexts and perform a few additions with them.
    */
    // PhantomCiphertext tmp_ct(encrypted1);
    // add_inplace(context, tmp_ct, encrypted2);

    /*
    [Add Plain]
    */
    // PhantomCiphertext tmp_ct(encrypted1);
    // add_plain_inplace(context, tmp_ct, plain);

    /*
    [Multiply]
    We multiply two ciphertexts. Since the size of the result will be 3,
    and will overwrite the first argument, we reserve first enough memory
    to avoid reallocating during multiplication.
    */
    PhantomCiphertext tmp_ct(encrypted1);
    multiply_inplace(context, tmp_ct, encrypted2);
    relinearize_inplace(context, tmp_ct, relin_keys);

    /*
    [Multiply Plain]
    We multiply a ciphertext with a random plaintext. Recall that
    multiply_plain does not change the size of the ciphertext, so we use
    encrypted2 here.
    */
    // PhantomCiphertext tmp_ct(encrypted1);
    // multiply_plain_inplace(context, tmp_ct, plain);

    /*
    [Rotate Rows One Step]
    We rotate matrix rows by one step left and measure the time.
    */
    // PhantomCiphertext tmp_ct(encrypted1);
    // rotate_rows_inplace(context, tmp_ct, 1, gal_keys);
}

void bgv_moddown_test_main(scheme_type scheme, mul_tech_type mul_tech = mul_tech_type::none) {
    print_example_banner("BFV Performance Test with Degrees: 4096, 8192, 16384, 32768, and 65536");

    if (scheme != scheme_type::bfv && scheme != scheme_type::bgv) {
        throw std::invalid_argument("This function is only for BFV and BGV.");
    }

    if (scheme == scheme_type::bfv && mul_tech == mul_tech_type::none) {
        throw std::invalid_argument("For BFV benchmarking, mul tech must be set.");
    }

    // 2 ^ 12

    {
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 12);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {35, 35,

                                                                           38}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 13);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {36, 36, 36, 36, 36,

                                                                           38}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 13);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {36, 36, 36, 36,

                                                                           37, 37}));
        parms.set_special_modulus_size(2);
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 14);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,

                                                                           42}));
        parms.set_special_modulus_size(1);
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
    }
    {
        cout << endl;
        EncryptionParameters parms(scheme);
        size_t poly_modulus_degree = (1 << 14);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {36, 36, 36, 36, 36, 36, 36, 36,

                                                                           37, 37, 38, 38}));
        parms.set_special_modulus_size(4);
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv)
            parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
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
        if (scheme == scheme_type::bfv) parms.set_mul_tech(mul_tech);
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        bgv_moddown_test(parms);
    }
     */
}

/*
Prints a sub-menu to select the performance test.
*/
void example_kernel_fusing() {
    print_example_banner("Example: Performance Test");
    // bgv_moddown_test_main(phantom::scheme_type::bfv, phantom::mul_tech_type::behz);
    // bgv_moddown_test_main(phantom::scheme_type::bfv, phantom::mul_tech_type::hps);
    bgv_moddown_test_main(scheme_type::bgv);
}

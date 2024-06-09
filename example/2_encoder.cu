#include "example.h"
#include "ckks.h"
#include <chrono>

using namespace std;
using namespace phantom;
using namespace phantom::arith;

void example_ckks_encoder(size_t n, size_t scale_two_pow, size_t repeat) {
    print_example_banner("Example: Encoders / CKKS Encoder\n");

    /*
    [CKKSEncoder] (For CKKS scheme only)

    In this example we demonstrate the Cheon-Kim-Kim-Song (CKKS) scheme for
    computing on encrypted real or complex numbers. We start by creating
    encryption parameters for the CKKS scheme. There are two important
    differences compared to the BFV scheme:

        (1) CKKS does not use the plain_modulus encryption parameter;
        (2) Selecting the coeff_modulus in a specific way can be very important
            when using the CKKS scheme. We will explain this further in the file
            `ckks_basics.cpp'. In this example we use CoeffModulus::Create to
            generate 5 40-bit prime numbers.
    */
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = n;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    if (n == 32768)
        parms.set_coeff_modulus(
                CoeffModulus::Create(poly_modulus_degree, {
                        60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
    else
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    //  parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

    PhantomContext context(parms);
    print_parameters(context);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> input;
    vector<cuDoubleComplex> result;
    input.reserve(slot_count);
    result.reserve(slot_count);
    double rand_real, rand_imag;
    for (size_t i = 0; i < slot_count; i++) {
        rand_real = (double) rand() / RAND_MAX;
        rand_imag = (double) rand() / RAND_MAX;
        input.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    PhantomPlaintext plain;

    double scale = pow(2.0, scale_two_pow);
    print_line(__LINE__);
    cout << "Begin encoding with scale: 2^" << scale_two_pow << endl;

    encoder.encode(context, input, scale, plain);
//    encoder.decode(context, plain, result);

    bool fail_flag = true;
    for (std::size_t i = 0; i < input.size(); i++) {
        fail_flag &= complex_equ(input[i], result[i]);
    }
    if (fail_flag)
        cout << endl << "Correct" << endl;
}

void test_ckks_encoder() {
    size_t n = 4096;
    size_t scale_two_pow = 40;
    for (; n <= 32768; n <<= 1) {
        example_ckks_encoder(n, scale_two_pow, 1);
    }
}

void example_encoders() {
    test_ckks_encoder();
}

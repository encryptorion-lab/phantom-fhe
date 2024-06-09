#include "polymath.cuh"
#include "scalingvariant.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;

// Multiply plain by scalar coeff_div_plaintext and reposition if in upper-half.
// Result gets added into the c_0 term of ciphertext (c_0,c_1).
void multiply_add_plain_with_scaling_variant(const PhantomContext &context, const PhantomPlaintext &plain,
                                             size_t chain_index, PhantomCiphertext &cipher,
                                             const cudaStream_t &stream) {
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &rns_tool = context_data.gpu_rns_tool();
    auto poly_degree = parms.poly_modulus_degree(); // = N
    auto &coeff_modulus = parms.coeff_modulus(); // coeff modulus
    auto coeff_mod_size = coeff_modulus.size();
    uint64_t t = parms.plain_modulus().value();

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    bfv_add_timesQ_overt_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            cipher.data(), plain.data(),
            rns_tool.negQl_mod_t(),
            rns_tool.negQl_mod_t_shoup(),
            rns_tool.tInv_mod_q(),
            rns_tool.tInv_mod_q_shoup(),
            context.gpu_rns_tables().modulus(),
            t, poly_degree, coeff_mod_size);

    cipher.set_chain_index(chain_index);
    cipher.set_poly_modulus_degree(poly_degree);
    cipher.set_coeff_modulus_size(coeff_mod_size);
}

void multiply_sub_plain_with_scaling_variant(const PhantomContext &context, const PhantomPlaintext &plain,
                                             size_t chain_index, PhantomCiphertext &cipher,
                                             const cudaStream_t &stream) {
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &rns_tool = context_data.gpu_rns_tool();
    auto poly_degree = parms.poly_modulus_degree(); // = N
    auto &coeff_modulus = parms.coeff_modulus(); // coeff modulus
    auto coeff_mod_size = coeff_modulus.size();
    uint64_t t = parms.plain_modulus().value();

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    bfv_sub_timesQ_overt_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            cipher.data(), plain.data(),
            rns_tool.negQl_mod_t(),
            rns_tool.negQl_mod_t_shoup(),
            rns_tool.tInv_mod_q(),
            rns_tool.tInv_mod_q_shoup(),
            context.gpu_rns_tables().modulus(),
            t, poly_degree, coeff_mod_size);

    cipher.set_chain_index(chain_index);
    cipher.set_poly_modulus_degree(poly_degree);
    cipher.set_coeff_modulus_size(coeff_mod_size);
}

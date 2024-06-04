#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include "phantom.h"

using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

void modup_bench(nvbench::state& state) {
    const auto dropped_levels = state.get_int64("Dropped Levels");
    state.collect_dram_throughput();

    EncryptionParameters parms(scheme_type::ckks);

    // size_t poly_modulus_degree = 1 << 16;
    // parms.set_poly_modulus_degree(poly_modulus_degree);
    // parms.set_coeff_modulus(CoeffModulus::Create(
    //         poly_modulus_degree, {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    //                               50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    //                               50, 50, 50, 50, 50, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
    // parms.set_special_modulus_size(15);
    // double scale = pow(2.0, 50);

    size_t poly_modulus_degree = 1 << 15;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {
            60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
            50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
            60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60
        }));
    parms.set_special_modulus_size(15);
    double scale = pow(2.0, 50);

    PhantomContext context(parms);

    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);
    PhantomRelinKey relin_keys(context);
    secret_key.gen_relinkey(context, relin_keys);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    std::vector<double> x_msg(slot_count, 1);
    std::vector<double> y_msg(slot_count, 1);

    PhantomPlaintext x_plain(context);
    PhantomPlaintext y_plain(context);

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher(context);
    PhantomCiphertext y_cipher(context);

    public_key.encrypt_asymmetric(context, x_plain, x_cipher, false);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher, false);

    for (int i = 0; i < dropped_levels; i++) {
        multiply_inplace(context, x_cipher, y_cipher);
        relinearize_inplace(context, x_cipher, relin_keys);
        rescale_to_next_inplace(context, x_cipher);
        x_cipher.set_scale(scale);
        mod_switch_to_next_inplace(context, y_cipher);
    }

    multiply_inplace(context, x_cipher, y_cipher);
    // relinearize_inplace(context, x_cipher, relin_keys);
    auto& context_data = context.get_context_data(x_cipher.chain_index());
    size_t decomp_modulus_size = context_data.parms().coeff_modulus().size();
    auto& key_context_data = context.get_context_data(0);
    auto& key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto& key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();
    uint64_t* cks = x_cipher.data() + 2 * decomp_modulus_size * n;
    auto& rns_tool = context.get_context_data(x_cipher.chain_index()).gpu_rns_tool();

    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    // size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;

    // auto size_Ql_n = size_Ql * n;
    // auto size_QP_n = size_QP * n;
    auto size_QlP_n = size_QlP * n;

    // Prepare key
    auto& key_vector = relin_keys.public_keys_;
    auto key_poly_num = key_vector[0].pk_.size_;

    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    // mod up

    Pointer<uint64_t> t_mod_up;
    t_mod_up.acquire(allocate<uint64_t>(global_pool(), beta * size_QlP_n));

    state.exec([&rns_tool, &t_mod_up, cks, &context, &scheme](nvbench::launch& launch) {
        rns_tool.modup(t_mod_up.get(), cks, context.gpu_rns_tables(), scheme);
    });
}

NVBENCH_BENCH(modup_bench)
        .add_int64_axis("Dropped Levels", nvbench::range(0, 14, 1))
        .set_timeout(1); // Limit to one second per measurement.

void keyswitch_bench(nvbench::state& state) {
    const auto dropped_levels = state.get_int64("Dropped Levels");
    state.collect_dram_throughput();

    EncryptionParameters parms(scheme_type::ckks);

    // size_t poly_modulus_degree = 1 << 16;
    // parms.set_poly_modulus_degree(poly_modulus_degree);
    // parms.set_coeff_modulus(CoeffModulus::Create(
    //         poly_modulus_degree, {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    //                               50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    //                               50, 50, 50, 50, 50, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
    // parms.set_special_modulus_size(15);
    // double scale = pow(2.0, 50);

    size_t poly_modulus_degree = 1 << 15;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {
            60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
            50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
            60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60
        }));
    parms.set_special_modulus_size(15);
    double scale = pow(2.0, 50);

    PhantomContext context(parms);

    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);
    PhantomRelinKey relin_keys(context);
    secret_key.gen_relinkey(context, relin_keys);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    std::vector<double> x_msg(slot_count, 1);
    std::vector<double> y_msg(slot_count, 1);

    PhantomPlaintext x_plain(context);
    PhantomPlaintext y_plain(context);

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher(context);
    PhantomCiphertext y_cipher(context);

    public_key.encrypt_asymmetric(context, x_plain, x_cipher, false);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher, false);

    for (int i = 0; i < dropped_levels; i++) {
        multiply_inplace(context, x_cipher, y_cipher);
        relinearize_inplace(context, x_cipher, relin_keys);
        rescale_to_next_inplace(context, x_cipher);
        x_cipher.set_scale(scale);
        mod_switch_to_next_inplace(context, y_cipher);
    }

    multiply_inplace(context, x_cipher, y_cipher);
    // relinearize_inplace(context, x_cipher, relin_keys);
    auto& context_data = context.get_context_data(x_cipher.chain_index());
    size_t decomp_modulus_size = context_data.parms().coeff_modulus().size();
    auto& key_context_data = context.get_context_data(0);
    auto& key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto& key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();
    uint64_t* cks = x_cipher.data() + 2 * decomp_modulus_size * n;
    auto& rns_tool = context.get_context_data(x_cipher.chain_index()).gpu_rns_tool();

    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    // size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;

    // auto size_Ql_n = size_Ql * n;
    // auto size_QP_n = size_QP * n;
    auto size_QlP_n = size_QlP * n;

    // Prepare key
    auto& key_vector = relin_keys.public_keys_;
    auto key_poly_num = key_vector[0].pk_.size_;

    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    // mod up
    Pointer<uint64_t> t_mod_up;
    t_mod_up.acquire(allocate<uint64_t>(global_pool(), beta * size_QlP_n));

    rns_tool.modup(t_mod_up.get(), cks, context.gpu_rns_tables(), scheme);

    // key switch
    Pointer<uint64_t> cx;
    cx.acquire(allocate<uint64_t>(global_pool(), 2 * size_QlP_n));

    auto reduction_threshold =
            (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;

    state.exec([&cx, &t_mod_up, &relin_keys, &rns_tool, &modulus_QP, &reduction_threshold](nvbench::launch& launch) {
        key_switch_inner_prod(cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr_.get(), rns_tool, modulus_QP,
                              reduction_threshold);
    });
}

NVBENCH_BENCH(keyswitch_bench)
        .add_int64_axis("Dropped Levels", nvbench::range(0, 14, 1))
        .set_timeout(1); // Limit to one second per measurement.

void moddown_bench(nvbench::state& state) {
    const auto dropped_levels = state.get_int64("Dropped Levels");
    state.collect_dram_throughput();

    EncryptionParameters parms(scheme_type::ckks);

    // size_t poly_modulus_degree = 1 << 16;
    // parms.set_poly_modulus_degree(poly_modulus_degree);
    // parms.set_coeff_modulus(CoeffModulus::Create(
    //         poly_modulus_degree, {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    //                               50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    //                               50, 50, 50, 50, 50, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
    // parms.set_special_modulus_size(15);
    // double scale = pow(2.0, 50);

    size_t poly_modulus_degree = 1 << 15;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {
            60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
            50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
            60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60
        }));
    parms.set_special_modulus_size(15);
    double scale = pow(2.0, 50);

    PhantomContext context(parms);

    PhantomSecretKey secret_key(parms);
    secret_key.gen_secretkey(context);
    PhantomPublicKey public_key(context);
    secret_key.gen_publickey(context, public_key);
    PhantomRelinKey relin_keys(context);
    secret_key.gen_relinkey(context, relin_keys);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    std::vector<double> x_msg(slot_count, 1);
    std::vector<double> y_msg(slot_count, 1);

    PhantomPlaintext x_plain(context);
    PhantomPlaintext y_plain(context);

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher(context);
    PhantomCiphertext y_cipher(context);

    public_key.encrypt_asymmetric(context, x_plain, x_cipher, false);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher, false);

    for (int i = 0; i < dropped_levels; i++) {
        multiply_inplace(context, x_cipher, y_cipher);
        relinearize_inplace(context, x_cipher, relin_keys);
        rescale_to_next_inplace(context, x_cipher);
        x_cipher.set_scale(scale);
        mod_switch_to_next_inplace(context, y_cipher);
    }

    multiply_inplace(context, x_cipher, y_cipher);
    // relinearize_inplace(context, x_cipher, relin_keys);
    auto& context_data = context.get_context_data(x_cipher.chain_index());
    size_t decomp_modulus_size = context_data.parms().coeff_modulus().size();
    auto& key_context_data = context.get_context_data(0);
    auto& key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto& key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();
    uint64_t* cks = x_cipher.data() + 2 * decomp_modulus_size * n;
    auto& rns_tool = context.get_context_data(x_cipher.chain_index()).gpu_rns_tool();

    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    // size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;

    // auto size_Ql_n = size_Ql * n;
    // auto size_QP_n = size_QP * n;
    auto size_QlP_n = size_QlP * n;

    // Prepare key
    auto& key_vector = relin_keys.public_keys_;
    auto key_poly_num = key_vector[0].pk_.size_;

    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    // mod up
    Pointer<uint64_t> t_mod_up;
    t_mod_up.acquire(allocate<uint64_t>(global_pool(), beta * size_QlP_n));

    rns_tool.modup(t_mod_up.get(), cks, context.gpu_rns_tables(), scheme);

    // key switch
    Pointer<uint64_t> cx;
    cx.acquire(allocate<uint64_t>(global_pool(), 2 * size_QlP_n));

    auto reduction_threshold =
            (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;
    key_switch_inner_prod(cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr_.get(), rns_tool, modulus_QP,
                          reduction_threshold);

    // mod down
    auto cx_i = cx.get() + 0 * size_QlP_n;

    state.exec([&rns_tool, cx_i, &context, &scheme](nvbench::launch& launch) {
        rns_tool.moddown_from_NTT(cx_i, cx_i, context.gpu_rns_tables(), scheme);
    });
}

NVBENCH_BENCH(moddown_bench)
        .add_int64_axis("Dropped Levels", nvbench::range(0, 14, 1))
        .set_timeout(1); // Limit to one second per measurement.

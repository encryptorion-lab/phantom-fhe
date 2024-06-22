#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include "phantom.h"

using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

void modup_bench(nvbench::state &state) {
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

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    std::vector<double> x_msg(slot_count, 1);
    std::vector<double> y_msg(slot_count, 1);

    PhantomPlaintext x_plain;
    PhantomPlaintext y_plain;

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher;
    PhantomCiphertext y_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);

    for (int i = 0; i < dropped_levels; i++) {
        multiply_inplace(context, x_cipher, y_cipher);
        relinearize_inplace(context, x_cipher, relin_keys);
        mod_switch_to_next_inplace(context, x_cipher);
        x_cipher.set_scale(scale);
        mod_switch_to_next_inplace(context, y_cipher);
    }

    multiply_inplace(context, x_cipher, y_cipher);
    // relinearize_inplace(context, x_cipher, relin_keys);
    auto &context_data = context.get_context_data(x_cipher.chain_index());
    size_t decomp_modulus_size = context_data.parms().coeff_modulus().size();
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto &key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();
    uint64_t *cks = x_cipher.data() + 2 * decomp_modulus_size * n;
    auto &rns_tool = context.get_context_data(x_cipher.chain_index()).gpu_rns_tool();

    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    // size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;

    // auto size_Ql_n = size_Ql * n;
    // auto size_QP_n = size_QP * n;
    auto size_QlP_n = size_QlP * n;

    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    cuda_stream_wrapper stream;
    const auto &s = stream.get_stream();

    // mod up
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(s));
    state.exec([&rns_tool, &t_mod_up, cks, &context, &scheme, &s](nvbench::launch &launch) {
        rns_tool.modup(t_mod_up.get(), cks, context.gpu_rns_tables(), scheme, s);
    });
}

NVBENCH_BENCH(modup_bench)
        .add_int64_axis("Dropped Levels", nvbench::range(0, 14, 1))
        .set_timeout(1); // Limit to one second per measurement.

void keyswitch_bench(nvbench::state &state) {
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

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    std::vector<double> x_msg(slot_count, 1);
    std::vector<double> y_msg(slot_count, 1);

    PhantomPlaintext x_plain;
    PhantomPlaintext y_plain;

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher;
    PhantomCiphertext y_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);

    for (int i = 0; i < dropped_levels; i++) {
        multiply_inplace(context, x_cipher, y_cipher);
        relinearize_inplace(context, x_cipher, relin_keys);
        mod_switch_to_next_inplace(context, x_cipher);
        x_cipher.set_scale(scale);
        mod_switch_to_next_inplace(context, y_cipher);
    }

    multiply_inplace(context, x_cipher, y_cipher);
    // relinearize_inplace(context, x_cipher, relin_keys);
    auto &context_data = context.get_context_data(x_cipher.chain_index());
    size_t decomp_modulus_size = context_data.parms().coeff_modulus().size();
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto &key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();
    uint64_t *cks = x_cipher.data() + 2 * decomp_modulus_size * n;
    auto &rns_tool = context.get_context_data(x_cipher.chain_index()).gpu_rns_tool();

    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    // size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;

    // auto size_Ql_n = size_Ql * n;
    // auto size_QP_n = size_QP * n;
    auto size_QlP_n = size_QlP * n;

    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    cuda_stream_wrapper stream;
    const auto &s = stream.get_stream();

    // mod up
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);

    rns_tool.modup(t_mod_up.get(), cks, context.gpu_rns_tables(), scheme, s);

    // key switch
    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);

    auto reduction_threshold =
            (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;

    state.set_cuda_stream(nvbench::make_cuda_stream_view(s));
    state.exec([&cx, &t_mod_up, &relin_keys, &rns_tool, &modulus_QP, &reduction_threshold, &s](
            nvbench::launch &launch) {
        key_switch_inner_prod(cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr(), rns_tool, modulus_QP,
                              reduction_threshold, s);
    });
}

NVBENCH_BENCH(keyswitch_bench)
        .add_int64_axis("Dropped Levels", nvbench::range(0, 14, 1))
        .set_timeout(1); // Limit to one second per measurement.

void moddown_bench(nvbench::state &state) {
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

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    std::vector<double> x_msg(slot_count, 1);
    std::vector<double> y_msg(slot_count, 1);

    PhantomPlaintext x_plain;
    PhantomPlaintext y_plain;

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher;
    PhantomCiphertext y_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);

    for (int i = 0; i < dropped_levels; i++) {
        multiply_inplace(context, x_cipher, y_cipher);
        relinearize_inplace(context, x_cipher, relin_keys);
        mod_switch_to_next_inplace(context, x_cipher);
        x_cipher.set_scale(scale);
        mod_switch_to_next_inplace(context, y_cipher);
    }

    multiply_inplace(context, x_cipher, y_cipher);
    // relinearize_inplace(context, x_cipher, relin_keys);
    auto &context_data = context.get_context_data(x_cipher.chain_index());
    size_t decomp_modulus_size = context_data.parms().coeff_modulus().size();
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto &key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();
    uint64_t *cks = x_cipher.data() + 2 * decomp_modulus_size * n;
    auto &rns_tool = context.get_context_data(x_cipher.chain_index()).gpu_rns_tool();

    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    // size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;

    // auto size_Ql_n = size_Ql * n;
    // auto size_QP_n = size_QP * n;
    auto size_QlP_n = size_QlP * n;

    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    cuda_stream_wrapper stream;
    const auto &s = stream.get_stream();

    // mod up
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
    rns_tool.modup(t_mod_up.get(), cks, context.gpu_rns_tables(), scheme, s);

    // key switch
    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);

    auto reduction_threshold =
            (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;

    key_switch_inner_prod(cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr(), rns_tool, modulus_QP,
                          reduction_threshold, s);

    // mod down
    auto cx_i = cx.get() + 0 * size_QlP_n;

    state.set_cuda_stream(nvbench::make_cuda_stream_view(s));
    state.exec([&rns_tool, cx_i, &context, &scheme, &s](nvbench::launch &launch) {
        rns_tool.moddown_from_NTT(cx_i, cx_i, context.gpu_rns_tables(), scheme, s);
    });
}

NVBENCH_BENCH(moddown_bench)
        .add_int64_axis("Dropped Levels", nvbench::range(0, 14, 1))
        .set_timeout(1); // Limit to one second per measurement.

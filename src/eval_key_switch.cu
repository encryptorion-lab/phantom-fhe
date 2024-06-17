#include "evaluate.cuh"
#include "ntt.cuh"
#include "polymath.cuh"
#include "rns.cuh"
#include "rns_bconv.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

__global__ void key_switch_inner_prod_c2_and_evk(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                                                 const DModulus *modulus, size_t n, size_t size_QP, size_t size_QP_n,
                                                 size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                                                 size_t beta, size_t reduction_threshold) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size_QlP_n; tid += blockDim.x * gridDim.x) {
        size_t nid = tid / n;
        size_t twr = (nid >= size_Ql ? size_Q + (nid - size_Ql) : nid);
        // base_rns = {q0, q1, ..., qj, p}
        DModulus mod = modulus[twr];
        uint64_t evk_id = (tid % n) + twr * n;
        uint64_t c2_id = (tid % n) + nid * n;

        uint128_t prod0, prod1;
        uint128_t acc0, acc1;

        // ct^x = ( <RNS-Decomp(c*_2), evk_b> , <RNS-Decomp(c*_2), evk_a>
        // evk[key_index][rns]
        //
        // RNS-Decomp(c*_2)[key_index + rns_indx * twr] =
        //           ( {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod q0,
        //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod q1,
        //             ...
        //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod qj,
        //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod p, )
        //
        // decomp_mod_size = number of evks

        // evk[0]_a
        acc0 = multiply_uint64_uint64(c2[c2_id], evks[0][evk_id]);
        // evk[0]_b
        acc1 = multiply_uint64_uint64(c2[c2_id], evks[0][evk_id + size_QP_n]);

        for (uint64_t i = 1; i < beta; i++) {
            if (i && reduction_threshold == 0) {
                acc0.lo = barrett_reduce_uint128_uint64(acc0, mod.value(), mod.const_ratio());
                acc0.hi = 0;

                acc1.lo = barrett_reduce_uint128_uint64(acc1, mod.value(), mod.const_ratio());
                acc1.hi = 0;
            }

            prod0 = multiply_uint64_uint64(c2[c2_id + i * size_QlP_n], evks[i][evk_id]);
            add_uint128_uint128(acc0, prod0, acc0);

            prod1 = multiply_uint64_uint64(c2[c2_id + i * size_QlP_n], evks[i][evk_id + size_QP_n]);
            add_uint128_uint128(acc1, prod1, acc1);
        }

        uint64_t res0 = barrett_reduce_uint128_uint64(acc0, mod.value(), mod.const_ratio());
        dst[tid] = res0;

        uint64_t res1 = barrett_reduce_uint128_uint64(acc1, mod.value(), mod.const_ratio());
        dst[tid + size_QlP_n] = res1;
    }
}

void
key_switch_inner_prod(uint64_t *p_cx, const uint64_t *p_t_mod_up, const uint64_t *const *rlk, const DRNSTool &rns_tool,
                      const DModulus *modulus_QP, size_t reduction_threshold, const cudaStream_t &stream) {

    const size_t size_QP = rns_tool.size_QP();
    const size_t size_P = rns_tool.size_P();
    const size_t size_Q = size_QP - size_P;

    const size_t size_Ql = rns_tool.base_Ql().size();
    const size_t size_QlP = size_Ql + size_P;

    const size_t n = rns_tool.n();
    const auto size_QP_n = size_QP * n;
    const auto size_QlP_n = size_QlP * n;

    const size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(
            p_cx, p_t_mod_up, rlk, modulus_QP, n, size_QP, size_QP_n, size_QlP, size_QlP_n, size_Q, size_Ql, beta,
            reduction_threshold);
}

// cks refers to cipher to be key-switched
void keyswitch_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
                       const PhantomRelinKey &relin_keys, bool is_relin, const cudaStream_t &stream) {
    const auto &s = stream;

    // Extract encryption parameters.
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto &key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();

    // HPS and HPSOverQ does not drop modulus
    uint32_t levelsDropped;

    if (scheme == scheme_type::bfv) {
        levelsDropped = 0;
        if (mul_tech == mul_tech_type::hps_overq_leveled) {
            size_t depth = encrypted.GetNoiseScaleDeg();
            bool isKeySwitch = !is_relin;
            bool is_Asymmetric = encrypted.is_asymmetric();
            size_t levels = depth - 1;
            auto dcrtBits = static_cast<double>(context.get_context_data(1).gpu_rns_tool().qMSB());

            // how many levels to drop
            levelsDropped = FindLevelsToDrop(context, levels, dcrtBits, isKeySwitch, is_Asymmetric);
        }
    } else if (scheme == scheme_type::bgv || scheme == scheme_type::ckks) {
        levelsDropped = encrypted.chain_index() - 1;
    } else {
        throw invalid_argument("unsupported scheme in keyswitch_inplace");
    }

    auto &rns_tool = context.get_context_data(1 + levelsDropped).gpu_rns_tool();

    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;

    auto size_Ql_n = size_Ql * n;
    // auto size_QP_n = size_QP * n;
    auto size_QlP_n = size_QlP * n;

    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        auto t_cks = phantom::util::make_cuda_auto_ptr<uint64_t>(size_Q * n, s);
        cudaMemcpyAsync(t_cks.get(), c2, size_Q * n * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, s);
        rns_tool.scaleAndRound_HPS_Q_Ql(c2, t_cks.get(), s);
    }

    // mod up
    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
    rns_tool.modup(t_mod_up.get(), c2, context.gpu_rns_tables(), scheme, s);

    // key switch
    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);
    auto reduction_threshold =
            (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;
    key_switch_inner_prod(cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr(), rns_tool, modulus_QP,
                          reduction_threshold, s);

    // mod down
    for (size_t i = 0; i < 2; i++) {
        auto cx_i = cx.get() + i * size_QlP_n;
        rns_tool.moddown_from_NTT(cx_i, cx_i, context.gpu_rns_tables(), scheme, s);
    }

    for (size_t i = 0; i < 2; i++) {
        auto cx_i = cx.get() + i * size_QlP_n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
            auto ct_i = encrypted.data() + i * size_Q * n;
            auto t_cx = make_cuda_auto_ptr<uint64_t>(size_Q * n, s);
            rns_tool.ExpandCRTBasis_Ql_Q(t_cx.get(), cx_i, s);
            add_to_ct_kernel<<<(size_Q * n) / blockDimGlb.x, blockDimGlb, 0, s>>>(
                    ct_i, t_cx.get(), rns_tool.base_Q().base(), n, size_Q);
        } else {
            auto ct_i = encrypted.data() + i * size_Ql_n;
            add_to_ct_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
                    ct_i, cx_i, rns_tool.base_Ql().base(), n, size_Ql);
        }
    }
}

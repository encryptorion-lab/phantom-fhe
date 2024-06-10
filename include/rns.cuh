#pragma once

#include "gputype.h"
#include "rns_base.cuh"
#include "rns_bconv.cuh"
#include "cuda_wrapper.cuh"

#include "util/encryptionparams.h"
#include "util/rns.h"
#include "util/modulus.h"

namespace phantom {

    class DRNSTool {

    private:
        mul_tech_type mul_tech_ = mul_tech_type::behz;
        std::size_t n_ = 0;
        std::size_t size_Q_ = 0;
        std::size_t size_P_ = 0;
        std::size_t size_QP_ = 0;

        arith::RNSBase h_base_Ql_;

        arith::DRNSBase base_;
        arith::DRNSBase base_Q_;
        arith::DRNSBase base_Ql_;
        arith::DRNSBase base_QlP_;
        // q[last]^(-1) mod q[i] for i = 0..last-1
        util::cuda_auto_ptr<uint64_t> q_last_mod_q_;
        util::cuda_auto_ptr<uint64_t> q_last_mod_q_shoup_;
        util::cuda_auto_ptr<uint64_t> inv_q_last_mod_q_;
        util::cuda_auto_ptr<uint64_t> inv_q_last_mod_q_shoup_;

        // hybrid key-switching
        util::cuda_auto_ptr<uint64_t> bigP_mod_q_;
        util::cuda_auto_ptr<uint64_t> bigP_mod_q_shoup_;
        util::cuda_auto_ptr<uint64_t> bigPInv_mod_q_;
        util::cuda_auto_ptr<uint64_t> bigPInv_mod_q_shoup_;
        util::cuda_auto_ptr<uint64_t> partQlHatInv_mod_Ql_concat_;
        util::cuda_auto_ptr<uint64_t> partQlHatInv_mod_Ql_concat_shoup_;
        std::vector<DBaseConverter> v_base_part_Ql_to_compl_part_QlP_conv_{};
        DBaseConverter base_P_to_Ql_conv_;

        // plain modulus related (BFV/BGV)
        DModulus t_;
        uint64_t q_last_mod_t_ = 1;
        uint64_t inv_q_last_mod_t_ = 1;
        uint64_t inv_q_last_mod_t_shoup_ = 1;
        // Base converter: q --> t
        DBaseConverter base_q_to_t_conv_;

        // BGV correction factor
        util::cuda_auto_ptr<uint64_t> pjInv_mod_q_;
        util::cuda_auto_ptr<uint64_t> pjInv_mod_q_shoup_;
        util::cuda_auto_ptr<uint64_t> pjInv_mod_t_;
        util::cuda_auto_ptr<uint64_t> pjInv_mod_t_shoup_;
        uint64_t bigPInv_mod_t_{};
        uint64_t bigPInv_mod_t_shoup_{};
        DBaseConverter base_P_to_t_conv_;

        // BFV enc/add/sub
        uint64_t negQl_mod_t_{}; // Ql mod t
        uint64_t negQl_mod_t_shoup_{}; // Ql mod t
        util::cuda_auto_ptr<uint64_t> tInv_mod_q_; // t^(-1) mod q
        util::cuda_auto_ptr<uint64_t> tInv_mod_q_shoup_; // t^(-1) mod q

        // BFV BEHZ
        arith::DRNSBase base_B_;
        arith::DRNSBase base_Bsk_;
        arith::DRNSBase base_Bsk_m_tilde_;
        arith::DRNSBase base_t_gamma_;
        DModulus m_tilde_;
        DModulus m_sk_;
        DModulus gamma_;
        DNTTTable gpu_Bsk_tables_;
        util::cuda_auto_ptr<uint64_t> tModBsk_;
        util::cuda_auto_ptr<uint64_t> tModBsk_shoup_;
        // Base converter: q --> B_sk
        DBaseConverter base_q_to_Bsk_conv_;
        // Base converter: q --> {m_tilde}
        DBaseConverter base_q_to_m_tilde_conv_;
        // Base converter: B --> q
        DBaseConverter base_B_to_q_conv_;
        // Base converter: B --> {m_sk}
        DBaseConverter base_B_to_m_sk_conv_;
        // Base converter: q --> {t, gamma}
        DBaseConverter base_q_to_t_gamma_conv_;
        // prod(q)^(-1) mod Bsk
        util::cuda_auto_ptr<uint64_t> inv_prod_q_mod_Bsk_;
        util::cuda_auto_ptr<uint64_t> inv_prod_q_mod_Bsk_shoup_;
        // prod(q)^(-1) mod m_tilde
        uint64_t neg_inv_prod_q_mod_m_tilde_{};
        uint64_t neg_inv_prod_q_mod_m_tilde_shoup_{};
        // prod(B)^(-1) mod m_sk
        uint64_t inv_prod_B_mod_m_sk_{};
        uint64_t inv_prod_B_mod_m_sk_shoup_{};
        // gamma^(-1) mod t
        uint64_t inv_gamma_mod_t_{};
        uint64_t inv_gamma_mod_t_shoup_{};
        // prod(B) mod q
        util::cuda_auto_ptr<uint64_t> prod_B_mod_q_;
        // m_tilde_QHatInvModq
        util::cuda_auto_ptr<uint64_t> m_tilde_QHatInvModq_;
        util::cuda_auto_ptr<uint64_t> m_tilde_QHatInvModq_shoup_;
        // m_tilde^(-1) mod Bsk
        util::cuda_auto_ptr<uint64_t> inv_m_tilde_mod_Bsk_;
        util::cuda_auto_ptr<uint64_t> inv_m_tilde_mod_Bsk_shoup_;
        // prod(q) mod Bsk
        util::cuda_auto_ptr<uint64_t> prod_q_mod_Bsk_;
        // -prod(q)^(-1) mod {t, gamma}
        util::cuda_auto_ptr<uint64_t> neg_inv_q_mod_t_gamma_;
        util::cuda_auto_ptr<uint64_t> neg_inv_q_mod_t_gamma_shoup_;
        // prod({t, gamma}) mod q
        util::cuda_auto_ptr<uint64_t> prod_t_gamma_mod_q_;
        util::cuda_auto_ptr<uint64_t> prod_t_gamma_mod_q_shoup_;

        // BFV HPS
        // decrypt scale&round
        size_t qMSB_ = 0;
        size_t sizeQMSB_ = 0;
        size_t tMSB_ = 0;
        util::cuda_auto_ptr<uint64_t> t_QHatInv_mod_q_div_q_mod_t_;
        util::cuda_auto_ptr<uint64_t> t_QHatInv_mod_q_div_q_mod_t_shoup_;
        util::cuda_auto_ptr<double> t_QHatInv_mod_q_div_q_frac_;
        util::cuda_auto_ptr<uint64_t> t_QHatInv_mod_q_B_div_q_mod_t_;
        util::cuda_auto_ptr<uint64_t> t_QHatInv_mod_q_B_div_q_mod_t_shoup_;
        util::cuda_auto_ptr<double> t_QHatInv_mod_q_B_div_q_frac_;
        // multiply
        arith::DRNSBase base_Rl_;
        arith::DRNSBase base_QlRl_;
        arith::DRNSBase base_QlDrop_;
        DNTTTable gpu_QlRl_tables_;
        DBaseConverter base_Ql_to_Rl_conv_;
        DBaseConverter base_Rl_to_Ql_conv_;
        DBaseConverter base_Q_to_Rl_conv_;
        DBaseConverter base_Ql_to_QlDrop_conv_;
        util::cuda_auto_ptr<double> tRSHatInvModsDivsFrac_;
        util::cuda_auto_ptr<uint64_t> tRSHatInvModsDivsModr_;
        util::cuda_auto_ptr<uint64_t> tRSHatInvModsDivsModr_shoup_;
        util::cuda_auto_ptr<double> tQlSlHatInvModsDivsFrac_;
        util::cuda_auto_ptr<uint64_t> tQlSlHatInvModsDivsModq_;
        util::cuda_auto_ptr<uint64_t> tQlSlHatInvModsDivsModq_shoup_;
        util::cuda_auto_ptr<double> QlQHatInvModqDivqFrac_;
        util::cuda_auto_ptr<uint64_t> QlQHatInvModqDivqModq_;
        util::cuda_auto_ptr<uint64_t> QlQHatInvModqDivqModq_shoup_;

    public:
        explicit DRNSTool(size_t n, size_t size_P, const arith::RNSBase &base_Ql,
                          const std::vector<arith::Modulus> &modulus_QP,
                          const arith::Modulus &t, mul_tech_type mul_tech, cudaStream_t const &stream);

        DRNSTool() = default;

        void modup(uint64_t *dst, const uint64_t *cks, const DNTTTable &ntt_tables,
                   const scheme_type &scheme, const cudaStream_t &stream) const;

        void moddown(uint64_t *ct_i, uint64_t *cx_i, const DNTTTable &ntt_tables,
                     const scheme_type &scheme, const cudaStream_t &stream) const;

        void moddown_from_NTT(uint64_t *ct_i, uint64_t *cx_i, const DNTTTable &ntt_tables,
                              const scheme_type &scheme, const cudaStream_t &stream) const;

        void behz_decrypt_scale_and_round(uint64_t *src, uint64_t *temp, const DNTTTable &rns_table,
                                          uint64_t temp_mod_size, uint64_t poly_modulus_degree, uint64_t *dst,
                                          const cudaStream_t &stream) const;

        void hps_decrypt_scale_and_round(uint64_t *dst, const uint64_t *src,
                                         const cudaStream_t &stream) const;

        void scaleAndRound_HPS_QR_R(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const;

        void scaleAndRound_HPS_QlRl_Ql(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const;

        void scaleAndRound_HPS_Q_Ql(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const;

        void ExpandCRTBasis_Ql_Q(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const;

        void ExpandCRTBasis_Ql_Q_add_to_ct(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const;

        void divide_and_round_q_last(const uint64_t *src, size_t cipher_size, uint64_t *dst,
                                     const cudaStream_t &stream) const;

        void divide_and_round_q_last_ntt(uint64_t *src, size_t cipher_size, const DNTTTable &rns_tables,
                                         uint64_t *dst, const cudaStream_t &stream) const;

        void mod_t_and_divide_q_last_ntt(uint64_t *src, size_t cipher_size, const DNTTTable &rns_tables,
                                         uint64_t *dst, const cudaStream_t &stream) const;

        void decrypt_mod_t(uint64_t *dst, const uint64_t *src, const uint64_t poly_degree,
                           const cudaStream_t &stream) const;

        // BEHZ step 1: Convert from base q to base Bsk U {m_tilde}
        void fastbconv_m_tilde(uint64_t *dst, uint64_t *src, const cudaStream_t &stream) const;

        // BEHZ step 2: Reduce q-overflows in with Montgomery reduction, switching base to Bsk
        void sm_mrq(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const;

        // BEHZ step 7: divide by q and floor, producing a result in base Bsk
        void fast_floor(uint64_t *input_base_q, uint64_t *input_base_Bsk, uint64_t *out_base_Bsk,
                        const cudaStream_t &stream) const;

        // BEHZ step 8: use Shenoy-Kumaresan method to convert the result to base q
        void fastbconv_sk(uint64_t *input_base_Bsk, uint64_t *out_base_q, const cudaStream_t &stream) const;

        [[nodiscard]] auto *inv_prod_q_mod_Bsk() const { return inv_prod_q_mod_Bsk_.get(); }

        [[nodiscard]] auto *inv_prod_q_mod_Bsk_shoup() const { return inv_prod_q_mod_Bsk_shoup_.get(); }

        [[nodiscard]] uint64_t *prod_B_mod_q() const { return (uint64_t *) (prod_B_mod_q_.get()); }

        [[nodiscard]] auto *m_tilde_QHatInvModq() const { return m_tilde_QHatInvModq_.get(); }

        [[nodiscard]] auto *m_tilde_QHatInvModq_shoup() const { return m_tilde_QHatInvModq_shoup_.get(); }

        [[nodiscard]] auto *inv_m_tilde_mod_Bsk() const { return inv_m_tilde_mod_Bsk_.get(); }

        [[nodiscard]] auto *inv_m_tilde_mod_Bsk_shoup() const { return inv_m_tilde_mod_Bsk_shoup_.get(); }

        [[nodiscard]] uint64_t *prod_q_mod_Bsk() const { return (uint64_t *) (prod_q_mod_Bsk_.get()); }

        [[nodiscard]] auto *neg_inv_q_mod_t_gamma() const { return neg_inv_q_mod_t_gamma_.get(); }

        [[nodiscard]] auto *neg_inv_q_mod_t_gamma_shoup() const { return neg_inv_q_mod_t_gamma_shoup_.get(); }

        [[nodiscard]] auto *prod_t_gamma_mod_q() const { return prod_t_gamma_mod_q_.get(); }

        [[nodiscard]] auto *prod_t_gamma_mod_q_shoup() const { return prod_t_gamma_mod_q_shoup_.get(); }

        [[nodiscard]] auto *q_last_mod_q() const { return q_last_mod_q_.get(); }

        [[nodiscard]] auto *q_last_mod_q_shoup() const { return q_last_mod_q_shoup_.get(); }

        [[nodiscard]] auto *inv_q_last_mod_q() const { return inv_q_last_mod_q_.get(); }

        [[nodiscard]] auto *inv_q_last_mod_q_shoup() const { return inv_q_last_mod_q_shoup_.get(); }

        // hybrid key-switching

        [[nodiscard]] auto *bigP_mod_q() const noexcept { return bigP_mod_q_.get(); }

        [[nodiscard]] auto *bigP_mod_q_shoup() const noexcept { return bigP_mod_q_shoup_.get(); }

        [[nodiscard]] auto *bigPInv_mod_q() const noexcept { return bigPInv_mod_q_.get(); }

        [[nodiscard]] auto *bigPInv_mod_q_shoup() const noexcept { return bigPInv_mod_q_shoup_.get(); }

        [[nodiscard]] auto *pjInv_mod_q() const noexcept { return pjInv_mod_q_.get(); }

        [[nodiscard]] auto *pjInv_mod_q_shoup() const noexcept { return pjInv_mod_q_shoup_.get(); }

        [[nodiscard]] auto *pjInv_mod_t() const noexcept { return pjInv_mod_t_.get(); }

        [[nodiscard]] auto *pjInv_mod_t_shoup() const noexcept { return pjInv_mod_t_shoup_.get(); }

        [[nodiscard]] auto &v_base_part_Ql_to_compl_part_QlP_conv() const noexcept {
            return v_base_part_Ql_to_compl_part_QlP_conv_;
        }

        [[nodiscard]] auto &base_part_Ql_to_compl_part_QlP_conv(std::size_t index) const noexcept {
            return v_base_part_Ql_to_compl_part_QlP_conv_.at(index);
        }

        [[nodiscard]] auto &base_P_to_Ql_conv() const noexcept { return base_P_to_Ql_conv_; }

        [[nodiscard]] auto &base_P_to_t_conv() const noexcept { return base_P_to_t_conv_; }

        // HPS

        // decrypt scale and round

        [[nodiscard]] auto *t_QHatInv_mod_q_div_q_mod_t() const noexcept {
            return t_QHatInv_mod_q_div_q_mod_t_.get();
        }

        [[nodiscard]] auto *t_QHatInv_mod_q_div_q_mod_t_shoup() const noexcept {
            return t_QHatInv_mod_q_div_q_mod_t_shoup_.get();
        }

        [[nodiscard]] double *t_QHatInv_mod_q_div_q_frac() const noexcept {
            return t_QHatInv_mod_q_div_q_frac_.get();
        }

        [[nodiscard]] auto *t_QHatInv_mod_q_B_div_q_mod_t() const noexcept {
            return t_QHatInv_mod_q_B_div_q_mod_t_.get();
        }

        [[nodiscard]] auto *t_QHatInv_mod_q_B_div_q_mod_t_shoup() const noexcept {
            return t_QHatInv_mod_q_B_div_q_mod_t_shoup_.get();
        }

        [[nodiscard]] double *t_QHatInv_mod_q_B_div_q_frac() const noexcept {
            return (double *) (t_QHatInv_mod_q_B_div_q_frac_.get());
        }

        // multiply scale and round

        [[nodiscard]] auto &gpu_QlRl_tables() { return gpu_QlRl_tables_; }

        [[nodiscard]] double *tRSHatInvModsDivsFrac() const noexcept {
            return (double *) (tRSHatInvModsDivsFrac_.get());
        }

        [[nodiscard]] auto *tRSHatInvModsDivsModr() const noexcept { return tRSHatInvModsDivsModr_.get(); }

        [[nodiscard]] auto *tRSHatInvModsDivsModr_shoup() const noexcept {
            return tRSHatInvModsDivsModr_shoup_.get();
        }

        [[nodiscard]] double *tQlSlHatInvModsDivsFrac() const noexcept {
            return (double *) (tQlSlHatInvModsDivsFrac_.get());
        }

        [[nodiscard]] auto *tQlSlHatInvModsDivsModq() const noexcept { return tQlSlHatInvModsDivsModq_.get(); }

        [[nodiscard]] auto *tQlSlHatInvModsDivsModq_shoup() const noexcept {
            return tQlSlHatInvModsDivsModq_shoup_.get();
        }

        [[nodiscard]] double *QlQHatInvModqDivqFrac() const noexcept {
            return (double *) (QlQHatInvModqDivqFrac_.get());
        }

        [[nodiscard]] auto *QlQHatInvModqDivqModq() const noexcept { return QlQHatInvModqDivqModq_.get(); }

        [[nodiscard]] auto *QlQHatInvModqDivqModq_shoup() const noexcept {
            return QlQHatInvModqDivqModq_shoup_.get();
        }

        [[nodiscard]] auto &host_base_Ql() const { return h_base_Ql_; }

        [[nodiscard]] auto &base_Ql() const { return base_Ql_; }

        [[nodiscard]] auto &base_QlP() const { return base_QlP_; }

        [[nodiscard]] auto &base_Q() const { return base_Q_; }

        [[nodiscard]] auto &base_Rl() const { return base_Rl_; }

        [[nodiscard]] auto &base_QlRl() const { return base_QlRl_; }

        [[nodiscard]] auto &base_Bsk() const { return base_Bsk_; }

        [[nodiscard]] auto &base_Bsk_m_tilde() const { return base_Bsk_m_tilde_; }

        [[nodiscard]] auto negQl_mod_t() const { return negQl_mod_t_; }

        [[nodiscard]] auto negQl_mod_t_shoup() const { return negQl_mod_t_shoup_; }

        [[nodiscard]] auto tInv_mod_q() const { return tInv_mod_q_.get(); }

        [[nodiscard]] auto tInv_mod_q_shoup() const { return tInv_mod_q_shoup_.get(); }

        [[nodiscard]] auto &gpu_Bsk_tables() const { return gpu_Bsk_tables_; }

        [[nodiscard]] auto &gpu_QlRl_tables() const { return gpu_QlRl_tables_; }

        [[nodiscard]] auto tModBsk() const { return tModBsk_.get(); }

        [[nodiscard]] auto tModBsk_shoup() const { return tModBsk_shoup_.get(); }

        [[nodiscard]] auto size_QP() const { return size_QP_; }

        [[nodiscard]] auto size_P() const { return size_P_; }

        [[nodiscard]] auto n() const { return n_; }

        [[nodiscard]] auto qMSB() const { return qMSB_; }

        [[nodiscard]] auto mul_tech() const { return mul_tech_; }

        [[nodiscard]] auto &base_Ql_to_Rl_conv() const { return base_Ql_to_Rl_conv_; }

        [[nodiscard]] auto &base_Rl_to_Ql_conv() const { return base_Rl_to_Ql_conv_; }

        [[nodiscard]] auto &base_Q_to_Rl_conv() const { return base_Q_to_Rl_conv_; }

        [[nodiscard]] auto &q_last_mod_t() const { return q_last_mod_t_; }

        [[nodiscard]] auto &inv_q_last_mod_t() const { return inv_q_last_mod_t_; }

        [[nodiscard]] auto &inv_q_last_mod_t_shoup() const { return inv_q_last_mod_t_shoup_; }

    };

} // namespace phantom

#pragma once

class DBaseConverter {

private:

    phantom::arith::DRNSBase ibase_;
    phantom::arith::DRNSBase obase_;

    phantom::util::cuda_auto_ptr<uint64_t> qiHat_mod_pj_;
    phantom::util::cuda_auto_ptr<uint64_t> alpha_Q_mod_pj_;
    phantom::util::cuda_auto_ptr<uint64_t> negPQHatInvModq_;
    phantom::util::cuda_auto_ptr<uint64_t> negPQHatInvModq_shoup_;
    phantom::util::cuda_auto_ptr<uint64_t> QInvModp_;
    phantom::util::cuda_auto_ptr<uint64_t> PModq_;
    phantom::util::cuda_auto_ptr<uint64_t> PModq_shoup_;

public:

    DBaseConverter() = default;

    explicit DBaseConverter(phantom::arith::BaseConverter &cpu_base_converter, const cudaStream_t &stream) {
        init(cpu_base_converter, stream);
    }

    void init(phantom::arith::BaseConverter &cpu_base_converter, const cudaStream_t &stream) {
        ibase_.init(cpu_base_converter.ibase(), stream);
        obase_.init(cpu_base_converter.obase(), stream);

        qiHat_mod_pj_ = phantom::util::make_cuda_auto_ptr<uint64_t>(obase_.size() * ibase_.size(), stream);
        for (size_t idx = 0; idx < obase_.size(); idx++)
            cudaMemcpyAsync(qiHat_mod_pj_.get() + idx * ibase_.size(), cpu_base_converter.QHatModp(idx),
                            ibase_.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream);

        alpha_Q_mod_pj_ = phantom::util::make_cuda_auto_ptr<uint64_t>((ibase_.size() + 1) * obase_.size(), stream);
        for (size_t idx = 0; idx < ibase_.size() + 1; idx++)
            cudaMemcpyAsync(alpha_Q_mod_pj_.get() + idx * obase_.size(), cpu_base_converter.alphaQModp(idx),
                            obase_.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream);

        negPQHatInvModq_ = phantom::util::make_cuda_auto_ptr<uint64_t>(ibase_.size(), stream);
        cudaMemcpyAsync(negPQHatInvModq_.get(), cpu_base_converter.negPQHatInvModq(),
                        ibase_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

        negPQHatInvModq_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(ibase_.size(), stream);
        cudaMemcpyAsync(negPQHatInvModq_shoup_.get(), cpu_base_converter.negPQHatInvModq_shoup(),
                        ibase_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

        QInvModp_ = phantom::util::make_cuda_auto_ptr<uint64_t>(obase_.size() * ibase_.size(), stream);
        for (size_t idx = 0; idx < obase_.size(); idx++)
            cudaMemcpyAsync(QInvModp_.get() + idx * ibase_.size(), cpu_base_converter.QInvModp(idx),
                            ibase_.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream);

        PModq_ = phantom::util::make_cuda_auto_ptr<uint64_t>(ibase_.size(), stream);
        cudaMemcpyAsync(PModq_.get(), cpu_base_converter.PModq(), ibase_.size() * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);

        PModq_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(ibase_.size(), stream);
        cudaMemcpyAsync(PModq_shoup_.get(), cpu_base_converter.PModq_shoup(),
                        ibase_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    }

    void bConv_BEHZ(uint64_t *dst, const uint64_t *src, size_t n, const cudaStream_t &stream) const;

    void bConv_BEHZ_var1(uint64_t *dst, const uint64_t *src, size_t n, const cudaStream_t &stream) const;

    void bConv_HPS(uint64_t *dst, const uint64_t *src, size_t n, const cudaStream_t &stream) const;

    void exact_convert_array(uint64_t *dst, const uint64_t *src, uint64_t poly_degree, const cudaStream_t &stream) const;

    __host__ inline auto &ibase() const { return ibase_; }

    __host__ inline auto &obase() const { return obase_; }

    __host__ inline uint64_t *QHatModp() const { return qiHat_mod_pj_.get(); }

    __host__ inline uint64_t *alpha_Q_mod_pj() const { return alpha_Q_mod_pj_.get(); }

    __host__ inline uint64_t *negPQHatInvModq() const { return negPQHatInvModq_.get(); }

    __host__ inline uint64_t *negPQHatInvModq_shoup() const { return negPQHatInvModq_shoup_.get(); }

    __host__ inline uint64_t *QInvModp() const { return QInvModp_.get(); }

    __host__ inline uint64_t *PModq() const { return PModq_.get(); }

    __host__ inline uint64_t *PModq_shoup() const { return PModq_shoup_.get(); }
};

__global__ void bconv_mult_kernel(uint64_t *dst, const uint64_t *src, const uint64_t *scale,
                                  const uint64_t *scale_shoup, const DModulus *base, uint64_t base_size, uint64_t n);

__global__ void bconv_mult_unroll2_kernel(uint64_t *dst, const uint64_t *src, const uint64_t *scale,
                                          const uint64_t *scale_shoup, const DModulus *base, uint64_t base_size,
                                          uint64_t n);

__global__ void bconv_mult_unroll4_kernel(uint64_t *dst, const uint64_t *src, const uint64_t *scale,
                                          const uint64_t *scale_shoup, const DModulus *base, uint64_t base_size,
                                          uint64_t n);

__global__ void bconv_matmul_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *QHatModp,
                                    const DModulus *ibase, uint64_t ibase_size, const DModulus *obase,
                                    uint64_t obase_size, uint64_t n);

__global__ void bconv_matmul_unroll2_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *QHatModp,
                                            const DModulus *ibase, uint64_t ibase_size, const DModulus *obase,
                                            uint64_t obase_size, uint64_t n);

__global__ void bconv_matmul_unroll4_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *QHatModp,
                                            const DModulus *ibase, uint64_t ibase_size, const DModulus *obase,
                                            uint64_t obase_size, uint64_t n);

__forceinline__ __device__ auto base_convert_acc(const uint64_t *ptr, const uint64_t *QHatModp,
                                                 size_t out_prime_idx, size_t degree, size_t ibase_size,
                                                 size_t degree_idx) {
    phantom::arith::uint128_t accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        phantom::arith::uint128_t out;

        uint64_t op1 = ptr[i * degree + degree_idx];
        out = phantom::arith::multiply_uint64_uint64(op1, op2);
        add_uint128_uint128(out, accum, accum);
    }
    return accum;
}

__forceinline__ __device__ auto base_convert_acc_unroll2(const uint64_t *ptr, const uint64_t *QHatModp,
                                                         size_t out_prime_idx, size_t degree, size_t ibase_size,
                                                         size_t degree_idx) {
    phantom::arith::uint128_t2 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        phantom::arith::uint128_t2 out{};

        uint64_t op1_x, op1_y;
        phantom::arith::ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        out.x = phantom::arith::multiply_uint64_uint64(op1_x, op2);
        add_uint128_uint128(out.x, accum.x, accum.x);
        out.y = phantom::arith::multiply_uint64_uint64(op1_y, op2);
        add_uint128_uint128(out.y, accum.y, accum.y);
    }
    return accum;
}

__forceinline__ __device__ auto base_convert_acc_unroll4(const uint64_t *ptr, const uint64_t *QHatModp,
                                                         size_t out_prime_idx, size_t degree, size_t ibase_size,
                                                         size_t degree_idx) {
    phantom::arith::uint128_t4 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        phantom::arith::uint128_t4 out{};

        uint64_t op1_x, op1_y;
        phantom::arith::ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        out.x = phantom::arith::multiply_uint64_uint64(op1_x, op2);
        add_uint128_uint128(out.x, accum.x, accum.x);
        out.y = phantom::arith::multiply_uint64_uint64(op1_y, op2);
        add_uint128_uint128(out.y, accum.y, accum.y);

        uint64_t op1_z, op1_w;
        phantom::arith::ld_two_uint64(op1_z, op1_w, ptr + i * degree + degree_idx + 2);
        out.z = phantom::arith::multiply_uint64_uint64(op1_z, op2);
        add_uint128_uint128(out.z, accum.z, accum.z);
        out.w = phantom::arith::multiply_uint64_uint64(op1_w, op2);
        add_uint128_uint128(out.w, accum.w, accum.w);
    }
    return accum;
}

__forceinline__ __device__ double_t base_convert_acc_frac(const uint64_t *ptr, const double *qiInv, size_t degree,
                                                          size_t ibase_size, size_t degree_idx) {
    double_t accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const double op2 = qiInv[i];

        const uint64_t op1 = ptr[i * degree + degree_idx];
        accum += static_cast<double>(op1) * op2;
    }
    return accum;
}

__forceinline__ __device__ auto base_convert_acc_frac_unroll2(const uint64_t *ptr, const double *qiInv,
                                                              size_t degree, size_t ibase_size,
                                                              size_t degree_idx) {
    phantom::arith::double_t2 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const double op2 = qiInv[i];

        uint64_t op1_x, op1_y;
        phantom::arith::ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        accum.x += static_cast<double>(op1_x) * op2;
        accum.y += static_cast<double>(op1_y) * op2;
    }
    return accum;
}

__forceinline__ __device__ auto base_convert_acc_frac_unroll4(const uint64_t *ptr, const double *qiInv,
                                                              size_t degree, size_t ibase_size,
                                                              size_t degree_idx) {
    phantom::arith::double_t4 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const double op2 = qiInv[i];

        uint64_t op1_x, op1_y;
        phantom::arith::ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        accum.x += static_cast<double>(op1_x) * op2;
        accum.y += static_cast<double>(op1_y) * op2;

        uint64_t op1_z, op1_w;
        phantom::arith::ld_two_uint64(op1_z, op1_w, ptr + i * degree + degree_idx + 2);
        accum.z += static_cast<double>(op1_z) * op2;
        accum.w += static_cast<double>(op1_w) * op2;
    }
    return accum;
}

__global__ void add_to_ct_kernel(uint64_t *ct, const uint64_t *cx, const DModulus *modulus, size_t n, size_t size_Ql);

#pragma once

#include "context.cuh"

#include "plaintext.h"
#include "ciphertext.h"
#include "prng.cuh"

class PhantomSecretKey;

class PhantomPublicKey;

class PhantomRelinKey;

class PhantomGaloisKey;

class PhantomPublicKey
{
    friend class PhantomSecretKey;

    friend class PhantomRelinKey;

private:
    bool gen_flag_ = false;
    PhantomCiphertext pk_;
    phantom::util::cuda_auto_ptr<uint8_t> prng_seed_a_; // for compress pk

    /** Encrypt zero using the public key, internal function, no modulus switch here.
     * @param[in] context PhantomContext
     * @param[inout] cipher The generated ciphertext
     * @param[in] chain_index The id of the corresponding context data
     * @param[in] is_ntt_form Whether the ciphertext should be in NTT form
     */
    void encrypt_zero_asymmetric_internal_internal(const PhantomContext& context, PhantomCiphertext& cipher,
                                                   size_t chain_index,
                                                   bool is_ntt_form, const cudaStream_t& stream) const;

    /** Encrypt zero using the public key, and perform the model switch is necessary
     * @brief pk [pk0, pk1], ternary variable u, cbd (gauss) noise e0, e1, return [pk0*u+e0, pk1*u+e1]
     * @param[in] context PhantomContext
     * @param[inout] cipher The generated ciphertext
     * @param[in] chain_index The id of the corresponding context data
     */
    void encrypt_zero_asymmetric_internal(const PhantomContext& context, PhantomCiphertext& cipher,
                                          size_t chain_index, const cudaStream_t& stream) const;

public:
    PhantomPublicKey() = default;

    PhantomPublicKey(const PhantomPublicKey&) = delete;

    PhantomPublicKey& operator=(const PhantomPublicKey&) = delete;

    PhantomPublicKey(PhantomPublicKey&&) = default;

    PhantomPublicKey& operator=(PhantomPublicKey&&) = default;

    ~PhantomPublicKey() = default;

    /** asymmetric encryption.
     * @brief: asymmetric encryption requires modulus switching.
     * @param[in] context PhantomContext
     * @param[in] plain The data to be encrypted
     * @param[out] cipher The generated ciphertext
     */
    void encrypt_asymmetric(const PhantomContext& context, const PhantomPlaintext& plain, PhantomCiphertext& cipher);

    // for python wrapper

    inline PhantomCiphertext encrypt_asymmetric(const PhantomContext& context, const PhantomPlaintext& plain)
    {
        PhantomCiphertext cipher;
        encrypt_asymmetric(context, plain, cipher);
        return cipher;
    }

    inline PhantomCiphertext encrypt_zero_asymmetric(const PhantomContext& context)
    {
        const auto& s = cudaStreamPerThread;
        PhantomCiphertext cipher;
        encrypt_zero_asymmetric_internal(context, cipher, context.get_first_index(), s);
        return cipher;
    }

    void save(std::ostream& stream) const
    {
        if (!gen_flag_)
            throw std::invalid_argument("PhantomPublicKey has not been generated");
        pk_.save(stream);
    }

    void load(std::istream& stream)
    {
        pk_.load(stream);
        gen_flag_ = true;
    }
};

/** PhantomRelinKey contains the relinear key in RNS and NTT form
 * gen_flag denotes whether the secret key has been generated.
 */
class PhantomRelinKey
{
    friend class PhantomSecretKey;

private:
    bool gen_flag_ = false;
    std::vector<PhantomPublicKey> public_keys_;
    phantom::util::cuda_auto_ptr<uint64_t*> public_keys_ptr_;

public:
    PhantomRelinKey() = default;

    PhantomRelinKey(const PhantomRelinKey&) = delete;

    PhantomRelinKey& operator=(const PhantomRelinKey&) = delete;

    PhantomRelinKey(PhantomRelinKey&&) = default;

    PhantomRelinKey& operator=(PhantomRelinKey&&) = default;

    ~PhantomRelinKey() = default;

    [[nodiscard]] inline auto public_keys_ptr() const
    {
        return public_keys_ptr_.get();
    }

    void save(std::ostream& stream) const
    {
        if (!gen_flag_)
            throw std::invalid_argument("PhantomRelinKey has not been generated");

        const size_t dnum = public_keys_.size();
        stream.write(reinterpret_cast<const char*>(&dnum), sizeof(std::size_t));

        for (const auto& pk : public_keys_)
        {
            pk.save(stream);
        }
    }

    void load(std::istream& stream)
    {
        size_t dnum;
        stream.read(reinterpret_cast<char*>(&dnum), sizeof(std::size_t));
        public_keys_.resize(dnum);
        for (auto& pk : public_keys_)
        {
            pk.load(stream);
        }

        std::vector<uint64_t*> pk_ptr(dnum);
        for (size_t i = 0; i < dnum; i++)
            pk_ptr[i] = public_keys_[i].pk_.data();
        public_keys_ptr_ = phantom::util::make_cuda_auto_ptr<uint64_t*>(dnum, cudaStreamPerThread);
        cudaMemcpyAsync(public_keys_ptr_.get(), pk_ptr.data(), sizeof(uint64_t*) * dnum,
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
        cudaStreamSynchronize(cudaStreamPerThread);

        gen_flag_ = true;
    }
};

/** PhantomGaloisKey stores Galois keys.
 * gen_flag denotes whether the Galois key has been generated.
 */
class PhantomGaloisKey
{
    friend class PhantomSecretKey;

private:
    bool gen_flag_ = false;
    std::vector<PhantomRelinKey> relin_keys_;

public:
    PhantomGaloisKey() = default;

    PhantomGaloisKey(const PhantomGaloisKey&) = delete;

    PhantomGaloisKey& operator=(const PhantomGaloisKey&) = delete;

    PhantomGaloisKey(PhantomGaloisKey&&) = default;

    PhantomGaloisKey& operator=(PhantomGaloisKey&&) = default;

    ~PhantomGaloisKey() = default;

    [[nodiscard]] auto& get_relin_keys(size_t index) const
    {
        return relin_keys_.at(index);
    }

    void save(std::ostream& stream) const
    {
        if (!gen_flag_)
            throw std::invalid_argument("PhantomGaloisKey has not been generated");

        const size_t rlk_num = relin_keys_.size();
        stream.write(reinterpret_cast<const char*>(&rlk_num), sizeof(std::size_t));

        for (const auto& rlk : relin_keys_)
        {
            rlk.save(stream);
        }
    }

    void load(std::istream& stream)
    {
        size_t rlk_num;
        stream.read(reinterpret_cast<char*>(&rlk_num), sizeof(std::size_t));
        relin_keys_.resize(rlk_num);
        for (auto& rlk : relin_keys_)
        {
            rlk.load(stream);
        }

        gen_flag_ = true;
    }
};

/** PhantomSecretKey contains the secret key in RNS and NTT form
 * gen_flag denotes whether the secret key has been generated.
 * Always at chain index 0
 */
class PhantomSecretKey
{
private:
    bool gen_flag_ = false;
    size_t sk_max_power_ = 0; // the max power of secret key
    size_t poly_modulus_degree_ = 0;
    size_t coeff_modulus_size_ = 0;

    phantom::util::cuda_auto_ptr<uint64_t> secret_key_array_; // the powers of secret key

    /** Generate the powers of secret key
     * @param[in] context PhantomContext
     * @param[in] max_power the mox power of secret key
     * @param[out] secret_key_array
     */
    void compute_secret_key_array(const PhantomContext& context, size_t max_power, const cudaStream_t& stream);

    [[nodiscard]] inline auto secret_key_array() const
    {
        return secret_key_array_.get();
    }

    void gen_secretkey(const PhantomContext& context);

    /** Encrypt zero using the secret key, the ciphertext is in NTT form
     * @param[in] context PhantomContext
     * @param[inout] cipher The generated ciphertext
     * @param[in] chain_index The index of the context data
     * @param[in] is_ntt_form Whether the ciphertext needs to be in NTT form
     */
    void encrypt_zero_symmetric(const PhantomContext& context, PhantomCiphertext& cipher, const uint8_t* prng_seed_a,
                                size_t chain_index, bool is_ntt_form, const cudaStream_t& stream) const;

    /** Generate one public key for this secret key
     * Return PhantomPublicKey
     * @param[in] context PhantomContext
     * @param[inout] relin_key The generated relinear key
     * @throws std::invalid_argument if secret key or relinear key has not been inited
     */
    void generate_one_kswitch_key(const PhantomContext& context, uint64_t* new_key, PhantomRelinKey& relin_key,
                                  const cudaStream_t& stream) const;

    void
    bfv_decrypt(const PhantomContext& context, const PhantomCiphertext& encrypted, PhantomPlaintext& destination,
                const cudaStream_t& stream);

    void
    ckks_decrypt(const PhantomContext& context, const PhantomCiphertext& encrypted, PhantomPlaintext& destination,
                 const cudaStream_t& stream);

    void
    bgv_decrypt(const PhantomContext& context, const PhantomCiphertext& encrypted, PhantomPlaintext& destination,
                const cudaStream_t& stream);

public:
    PhantomSecretKey() = default;

    explicit inline PhantomSecretKey(const PhantomContext& context)
    {
        gen_secretkey(context);
    }

    PhantomSecretKey(const PhantomSecretKey&) = delete;

    PhantomSecretKey& operator=(const PhantomSecretKey&) = delete;

    PhantomSecretKey(PhantomSecretKey&&) = default;

    PhantomSecretKey& operator=(PhantomSecretKey&&) = default;

    ~PhantomSecretKey() = default;

    [[nodiscard]] PhantomPublicKey gen_publickey(const PhantomContext& context) const;

    [[nodiscard]] PhantomRelinKey gen_relinkey(const PhantomContext& context);

    [[nodiscard]] PhantomGaloisKey create_galois_keys(const PhantomContext& context) const;

    /** Symmetric encryption, the plaintext and ciphertext are in NTT form
     * @param[in] context PhantomContext
     * @param[in] plain The data to be encrypted
     * @param[out] cipher The generated ciphertext
     */
    void
    encrypt_symmetric(const PhantomContext& context, const PhantomPlaintext& plain, PhantomCiphertext& cipher) const;

    /** decryption
     * @param[in] context PhantomContext
     * @param[in] cipher The ciphertext to be decrypted
     * @param[out] plain The plaintext
     */
    void decrypt(const PhantomContext& context, const PhantomCiphertext& cipher, PhantomPlaintext& plain);

    // for python wrapper

    [[nodiscard]] inline PhantomCiphertext
    encrypt_symmetric(const PhantomContext& context, const PhantomPlaintext& plain) const
    {
        PhantomCiphertext cipher;
        encrypt_symmetric(context, plain, cipher);
        return cipher;
    }

    [[nodiscard]] inline PhantomPlaintext
    decrypt(const PhantomContext& context, const PhantomCiphertext& cipher)
    {
        PhantomPlaintext plain;
        decrypt(context, cipher, plain);
        return plain;
    }

    /**
    Computes the invariant noise budget (in bits) of a ciphertext. The
    invariant noise budget measures the amount of room there is for the noise
    to grow while ensuring correct decryptions. This function works only with
    the BFV scheme.
    * @param[in] context PhantomContext
    * @param[in] cipher The ciphertext to be decrypted
    */
    [[nodiscard]] int invariant_noise_budget(const PhantomContext& context, const PhantomCiphertext& cipher);

    void save(std::ostream& stream) const
    {
        if (!gen_flag_)
            throw std::invalid_argument("PhantomSecretKey has not been generated");

        stream.write(reinterpret_cast<const char*>(&sk_max_power_), sizeof(size_t));
        stream.write(reinterpret_cast<const char*>(&poly_modulus_degree_), sizeof(size_t));
        stream.write(reinterpret_cast<const char*>(&coeff_modulus_size_), sizeof(size_t));

        uint64_t* h_data;
        cudaMallocHost(&h_data, sk_max_power_ * poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t));
        cudaMemcpy(h_data, secret_key_array_.get(),
                   sk_max_power_ * poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        stream.write(reinterpret_cast<char*>(h_data),
                     sk_max_power_ * poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t));
        cudaFreeHost(h_data);
    }

    void load(std::istream& stream)
    {
        stream.read(reinterpret_cast<char*>(&sk_max_power_), sizeof(size_t));
        stream.read(reinterpret_cast<char*>(&poly_modulus_degree_), sizeof(size_t));
        stream.read(reinterpret_cast<char*>(&coeff_modulus_size_), sizeof(size_t));

        uint64_t* h_data;
        cudaMallocHost(&h_data, sk_max_power_ * poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t));
        stream.read(reinterpret_cast<char*>(h_data),
                    sk_max_power_ * poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t));
        secret_key_array_ = phantom::util::make_cuda_auto_ptr<uint64_t>(
            sk_max_power_ * poly_modulus_degree_ * coeff_modulus_size_,
            cudaStreamPerThread);
        cudaMemcpyAsync(secret_key_array_.get(), h_data,
                        sk_max_power_ * poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);

        cudaStreamSynchronize(cudaStreamPerThread);

        // cleanup h_data
        cudaFreeHost(h_data);

        gen_flag_ = true;
    }
};

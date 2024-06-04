#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>
#include "encryptionparams.h"
#include "modulus.h"
#include "ntt.h"
#include "uintarithsmallmod.h"

namespace phantom::arith {
    class RNSBase {

    private:
        bool initialize();

        // total number of small modulus in this base
        std::size_t size_;
        // vector of small modulus in this base
        std::vector<Modulus> mod_;
        // product of all small modulus in this base, stored in 1d vector
        std::vector<std::uint64_t> prod_mod_;
        // product of all small modulus's hat in this base, stored in 2d vector
        std::vector<std::uint64_t> prod_hat_;
        // vector of qiHat mod qi
        std::vector<uint64_t> hat_mod_;
        std::vector<uint64_t> hat_mod_shoup_;
        // vector of qiHatInv mod qi
        std::vector<uint64_t> hatInv_mod_;
        std::vector<uint64_t> hatInv_mod_shoup_;
        // vector of 1.0 / qi
        std::vector<double> inv_;

    public:
        RNSBase() : size_(0) {}

        // Construct the RNSBase from the parm, calculate
        // 1. the product of all coeff (big_Q_)
        // 2. the product of all coeff except myself (big_qiHat_)
        // 3. the inverse of the above product mod myself (qiHatInv_mod_qi_)
        explicit RNSBase(const std::vector<Modulus> &rnsbase);

        void init(const std::vector<Modulus> &rnsbase);

        // Copy from the copy RNSBase
        RNSBase(const RNSBase &copy);

        void init(const RNSBase &copy);

        // Move from the source
        RNSBase(RNSBase &&source) = default;

        RNSBase &operator=(const RNSBase &assign) = delete;

        // Get the index coeff modulus
        [[nodiscard]] inline const Modulus &operator[](std::size_t index) const {
            if (index >= size_) {
                throw std::out_of_range("index is out of range");
            }
            return mod_[index];
        }

        // Returns the number of coeff modulus
        [[nodiscard]] std::size_t size() const noexcept { return size_; }

        // Returns whether the specified modulus exists in the Q_
        [[nodiscard]] bool contains(const Modulus &value) const noexcept;

        // Return whether my Q_ is the subset of the provided superbase.Q_
        [[nodiscard]] bool is_subbase_of(const RNSBase &superbase) const noexcept;

        // Return whether the provided superbase.Q_ is a subset of my Q_
        [[nodiscard]] inline bool is_superbase_of(const RNSBase &subbase) const noexcept {
            return subbase.is_subbase_of(*this);
        }

        // A faster check compared with is_subbase_of
        [[maybe_unused]] [[nodiscard]] bool is_proper_subbase_of(const RNSBase &superbase) const noexcept {
            return (size_ < superbase.size_) && is_subbase_of(superbase);
        }

        // A faster check compared with is_superbase_of
        [[maybe_unused]] [[nodiscard]] bool is_proper_superbase_of(const RNSBase &subbase) const noexcept {
            return (size_ > subbase.size_) && !is_subbase_of(subbase);
        }

        // Add a modulus to my RNSBase
        [[nodiscard]] RNSBase extend(const Modulus &value) const;

        // Add other RNSBase to my RNSBase
        [[nodiscard]] RNSBase extend(const RNSBase &other) const;

        // Delete the last coeff and re-generate the RNSBase
        [[nodiscard]] RNSBase drop() const;

        // Delete the specified Modulus and re-generate the RNSBase
        [[nodiscard]] RNSBase drop(const Modulus &value) const;

        // Delete the specified Moduli and re-generate the RNSBase
        [[nodiscard]] RNSBase drop(const std::vector<Modulus> &values) const;

        // The CRT decompose, i.e., value % each modulus (Q_[0], Q_[1], ...)
        void decompose(std::uint64_t *value) const;

        // When the poly degree is count, perform the CRT in one invocation
        void decompose_array(std::uint64_t *value, std::size_t count) const;

        // CRT compose
        void compose(std::uint64_t *value) const;

        // When the poly degree is count, perform the CRT compose in one invocation
        void compose_array(std::uint64_t *value, std::size_t count) const;

        [[nodiscard]] const Modulus *base() const noexcept { return mod_.data(); }

        [[nodiscard]] const std::uint64_t *big_modulus() const noexcept { return prod_mod_.data(); }

        [[nodiscard]] const std::uint64_t *big_qiHat() const noexcept { return prod_hat_.data(); }

        [[nodiscard]] const uint64_t *qiHat_mod_qi() const noexcept { return hat_mod_.data(); }

        [[nodiscard]] const uint64_t *qiHat_mod_qi_shoup() const noexcept { return hat_mod_shoup_.data(); }

        [[nodiscard]] const uint64_t *QHatInvModq() const noexcept { return hatInv_mod_.data(); }

        [[nodiscard]] const uint64_t *QHatInvModq_shoup() const noexcept { return hatInv_mod_shoup_.data(); }

        [[nodiscard]] const double *inv() const noexcept { return inv_.data(); }

    };

    class BaseConverter {

    private:
        void initialize();

        RNSBase ibase_;
        RNSBase obase_;
        std::vector<std::vector<std::uint64_t>> QHatModp_;
        std::vector<std::vector<std::uint64_t>> alphaQModp_;
        std::vector<uint64_t> negPQHatInvModq_;
        std::vector<uint64_t> negPQHatInvModq_shoup_;
        std::vector<std::vector<std::uint64_t>> QInvModp_;
        std::vector<uint64_t> PModq_;
        std::vector<uint64_t> PModq_shoup_;

    public:
        BaseConverter(const RNSBase &ibase, const RNSBase &obase) : ibase_(std::move(ibase)), obase_(std::move(obase)) {
            initialize();
        }

        BaseConverter(BaseConverter &&source) = delete;

        BaseConverter &operator=(const BaseConverter &assign) = delete;

        BaseConverter &operator=(BaseConverter &&assign) = delete;

        size_t ibase_size() const noexcept { return ibase_.size(); }

        size_t obase_size() const noexcept { return obase_.size(); }

        const RNSBase &ibase() const noexcept { return ibase_; }

        const RNSBase &obase() const noexcept { return obase_; }

        uint64_t *QHatModp(size_t index) {
            if (index >= obase_size())
                throw std::out_of_range("QHatModp index is out of range");

            return QHatModp_[index].data();
        }

        uint64_t *alphaQModp(size_t index) {
            if (index >= ibase_size() + 1)
                throw std::out_of_range("alphaQModp index is out of range");

            return alphaQModp_[index].data();
        }

        auto *negPQHatInvModq() { return negPQHatInvModq_.data(); }

        auto *negPQHatInvModq_shoup() { return negPQHatInvModq_shoup_.data(); }

        uint64_t *QInvModp(size_t index) {
            if (index >= obase_size())
                throw std::out_of_range("QInvModp index is out of range");

            return QInvModp_[index].data();
        }

        auto *PModq() { return PModq_.data(); }

        auto *PModq_shoup() { return PModq_shoup_.data(); }

    };
} // namespace phantom::util

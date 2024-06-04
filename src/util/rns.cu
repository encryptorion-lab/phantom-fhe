// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <cstdio>
#include "util/common.h"
#include "util/numth.h"
#include "util/rns.h"
#include "util/uintarithmod.h"
#include "util/uintarithsmallmod.h"

using namespace std;

namespace phantom::arith {
    // Construct the RNSBase from the parm, calcuate
    // 1. the product of all coeff (big_Q_)
    // 2. the product of all coeff except myself (big_qiHat_)
    // 3. the inverse of the above product mod myself (qiHatInv_mod_qi_)
    RNSBase::RNSBase(const vector<Modulus> &rnsbase) : size_(rnsbase.size()) {
        if (!size_) {
            throw invalid_argument("rnsbase cannot be empty");
        }

        for (size_t i = 0; i < rnsbase.size(); i++) {
            // The base elements cannot be zero
            if (rnsbase[i].is_zero()) {
                throw invalid_argument("rnsbase is invalid");
            }

            for (size_t j = 0; j < i; j++) {
                // The base must be coprime
                if (!are_coprime(rnsbase[i].value(), rnsbase[j].value())) {
                    throw invalid_argument("rnsbase is invalid");
                }
            }
        }

        // Base is good; now copy it over to rnsbase_
        mod_.resize(size_);
        copy_n(rnsbase.cbegin(), size_, mod_.data());

        // Initialize CRT data
        if (!initialize())
            throw invalid_argument("rnsbase is invalid");
    }

    void RNSBase::init(const vector<Modulus> &rnsbase) {
        if (size_ != 0)
            throw logic_error("RNSBase is already initialized");
        size_ = rnsbase.size();
        if (!size_) {
            throw invalid_argument("rnsbase cannot be empty");
        }

        for (size_t i = 0; i < rnsbase.size(); i++) {
            // The base elements cannot be zero
            if (rnsbase[i].is_zero()) {
                throw invalid_argument("rnsbase is invalid");
            }

            for (size_t j = 0; j < i; j++) {
                // The base must be coprime
                if (!are_coprime(rnsbase[i].value(), rnsbase[j].value())) {
                    throw invalid_argument("rnsbase is invalid");
                }
            }
        }

        // Base is good; now copy it over to rnsbase_
        mod_.resize(size_);
        copy_n(rnsbase.cbegin(), size_, mod_.data());

        // Initialize CRT data
        if (!initialize())
            throw invalid_argument("rnsbase is invalid");
    }

    RNSBase::RNSBase(const RNSBase &copy) : size_(copy.size_) {
        // Copy over the base
        mod_.resize(size_);
        copy_n(copy.mod_.data(), size_, mod_.data());

        // Copy over CRT data
        prod_mod_.resize(size_);
        set_uint(copy.prod_mod_.data(), size_, prod_mod_.data());

        prod_hat_.resize(size_ * size_);
        set_uint(copy.prod_hat_.data(), size_ * size_, prod_hat_.data());

        hat_mod_.resize(size_);
        copy_n(copy.hat_mod_.data(), size_, hat_mod_.data());

        hat_mod_shoup_.resize(size_);
        copy_n(copy.hat_mod_shoup_.data(), size_, hat_mod_shoup_.data());

        hatInv_mod_.resize(size_);
        copy_n(copy.hatInv_mod_.data(), size_, hatInv_mod_.data());

        hatInv_mod_shoup_.resize(size_);
        copy_n(copy.hatInv_mod_shoup_.data(), size_, hatInv_mod_shoup_.data());

        inv_.resize(size_);
        copy_n(copy.inv_.data(), size_, inv_.data());
    }

    void RNSBase::init(const RNSBase &copy) {
        if (size_ != 0)
            throw logic_error("RNSBase is already initialized");
        size_ = copy.size();
        // Copy over the base
        mod_.resize(size_);
        copy_n(copy.mod_.data(), size_, mod_.data());

        // Copy over CRT data
        prod_mod_.resize(size_);
        set_uint(copy.prod_mod_.data(), size_, prod_mod_.data());

        prod_hat_.resize(size_ * size_);
        set_uint(copy.prod_hat_.data(), size_ * size_, prod_hat_.data());

        hat_mod_.resize(size_);
        copy_n(copy.hat_mod_.data(), size_, hat_mod_.data());

        hat_mod_shoup_.resize(size_);
        copy_n(copy.hat_mod_shoup_.data(), size_, hat_mod_shoup_.data());

        hatInv_mod_.resize(size_);
        copy_n(copy.hatInv_mod_.data(), size_, hatInv_mod_.data());

        hatInv_mod_shoup_.resize(size_);
        copy_n(copy.hatInv_mod_shoup_.data(), size_, hatInv_mod_shoup_.data());

        inv_.resize(size_);
        copy_n(copy.inv_.data(), size_, inv_.data());
    }

    bool RNSBase::contains(const Modulus &value) const noexcept {
        bool result = false;

        for (size_t i = 0; i < size_; i++)
            result = result || (mod_[i] == value);
        return result;
    }

    bool RNSBase::is_subbase_of(const RNSBase &superbase) const noexcept {
        bool result = true;
        for (size_t i = 0; i < size_; i++)
            result = result && superbase.contains(mod_[i]);
        return result;
    }

    RNSBase RNSBase::extend(const Modulus &value) const {
        if (value.is_zero()) {
            throw invalid_argument("value cannot be zero");
        }

        for (size_t i = 0; i < size_; i++) {
            if (!are_coprime(mod_[i].value(), value.value())) {
                throw logic_error("cannot extend by given value");
            }
        }

        // Copy over this base
        RNSBase newbase;
        newbase.size_ = size_ + 1;
        newbase.mod_.resize(newbase.size_);
        copy_n(mod_.data(), size_, newbase.mod_.data());

        // Extend with value
        newbase.mod_[newbase.size_ - 1] = value;

        // Initialize CRT data
        if (!newbase.initialize()) {
            throw logic_error("cannot extend by given value");
        }

        return newbase;
    }

    RNSBase RNSBase::extend(const RNSBase &other) const {
        // The bases must be coprime
        for (size_t i = 0; i < other.size_; i++) {
            for (size_t j = 0; j < size_; j++) {
                if (!are_coprime(other[i].value(), mod_[j].value())) {
                    throw invalid_argument("rnsbase is invalid");
                }
            }
        }

        // Copy over this base
        RNSBase newbase;
        newbase.size_ = size_ + other.size_;
        newbase.mod_.resize(newbase.size_);
        copy_n(mod_.data(), size_, newbase.mod_.data());

        // Extend with other base
        copy_n(other.mod_.data(), other.size_, newbase.mod_.data() + size_);

        // Initialize CRT data
        if (!newbase.initialize()) {
            throw logic_error("cannot extend by given base");
        }

        return newbase;
    }

    RNSBase RNSBase::drop() const {
        if (size_ == 1) {
            throw logic_error("cannot drop from base of size 1");
        }

        // Copy over this base
        RNSBase newbase;
        newbase.size_ = size_ - 1;
        newbase.mod_.resize(newbase.size_);
        copy_n(mod_.data(), size_ - 1, newbase.mod_.data());

        // Initialize CRT data
        newbase.initialize();

        return newbase;
    }

    RNSBase RNSBase::drop(const Modulus &value) const {
        if (size_ == 1) {
            throw logic_error("cannot drop from base of size 1");
        }
        if (!contains(value)) {
            throw logic_error("base does not contain value");
        }

        // Copy over this base
        RNSBase newbase;
        newbase.size_ = size_ - 1;
        newbase.mod_.resize(newbase.size_);
        size_t source_index = 0;
        size_t dest_index = 0;
        while (dest_index < size_ - 1) {
            if (mod_[source_index] != value) {
                newbase.mod_[dest_index] = mod_[source_index];
                dest_index++;
            }
            source_index++;
        }

        // Initialize CRT data
        newbase.initialize();
        return newbase;
    }

    RNSBase RNSBase::drop(const std::vector<Modulus> &values) const {
        size_t drop_size = values.size();

        if (size_ < drop_size + 1) {
            throw logic_error("RNSBase should contain at least one modulus after dropping");
        }

        for (auto &value: values) {
            if (!contains(value)) {
                throw logic_error("base does not contain value");
            }
        }

        // Copy over this base
        RNSBase new_base;
        new_base.size_ = size_ - drop_size;
        new_base.mod_.resize(new_base.size_);
        size_t source_index = 0;
        size_t dest_index = 0;
        while (dest_index < new_base.size_) {
            if (!std::count(values.begin(), values.end(), mod_[source_index])) {
                new_base.mod_[dest_index++] = mod_[source_index];
            }
            source_index++;
        }

        // Initialize CRT data
        new_base.initialize();
        return new_base;
    }

    // Calculate big_Q_, big_qiHat_, and qiHatInv_mod_qi_
    // Also perform the validation.
    bool RNSBase::initialize() {
        prod_mod_.resize(size_);
        prod_hat_.resize(size_ * size_);
        hat_mod_.resize(size_);
        hat_mod_shoup_.resize(size_);
        hatInv_mod_.resize(size_);
        hatInv_mod_shoup_.resize(size_);
        inv_.resize(size_);

        if (size_ > 1) {
            std::vector<uint64_t> rnsbase_values(size_);
            for (size_t i = 0; i < size_; i++)
                rnsbase_values[i] = mod_[i].value();

            // Create punctured products
            for (size_t i = 0; i < size_; i++) {
                multiply_many_uint64_except(rnsbase_values.data(), size_, i, prod_hat_.data() + i * size_);
            }

            // Compute the full product, i.e., qiHat[0] * Q_[0]
            auto temp_mpi = std::vector<uint64_t>(size_);
            multiply_uint(prod_hat_.data(), size_, mod_[0].value(), size_, temp_mpi.data());
            set_uint(temp_mpi.data(), size_, prod_mod_.data());

            // Compute inverses of punctured products mod primes
            for (size_t i = 0; i < size_; i++) {
                // punctured_prod[i] % qi
                uint64_t qiHat_mod_qi = modulo_uint(prod_hat_.data() + i * size_, size_, mod_[i]);
                // qiHat_mod_qi = qiHat_mod_qi^{-1} % qi
                uint64_t qiHatInv_mod_qi;
                if (!try_invert_uint_mod(qiHat_mod_qi, mod_[i], qiHatInv_mod_qi))
                    throw invalid_argument("invalid modulus");

                hat_mod_[i] = qiHat_mod_qi;
                hat_mod_shoup_[i] = compute_shoup(qiHat_mod_qi, mod_[i].value());
                hatInv_mod_[i] = qiHatInv_mod_qi;
                hatInv_mod_shoup_[i] = compute_shoup(qiHatInv_mod_qi, mod_[i].value());
            }

            // compute 1.0 / qi
            for (size_t i = 0; i < size_; i++) {
                uint64_t qi = mod_[i].value();
                double inv = 1.0 / static_cast<double>(qi);
                inv_[i] = inv;
            }
            return true;
        }

        // Only one single modulus
        prod_mod_[0] = mod_[0].value();
        prod_hat_[0] = 1;
        hat_mod_[0] = 1;
        hat_mod_shoup_[0] = compute_shoup(1, mod_[0].value());
        hatInv_mod_[0] = 1;
        hatInv_mod_shoup_[0] = compute_shoup(1, mod_[0].value());
        inv_[0] = 1.0 / static_cast<double>(mod_[0].value());
        return true;
    }

    void RNSBase::decompose(uint64_t *value) const {
        if (!value) {
            throw invalid_argument("value cannot be null");
        }

        if (size_ > 1) {
            // Copy the value
            auto value_copy = std::vector<uint64_t>(size_);
            std::copy_n(value, size_, value_copy.data());
            for (size_t i = 0; i < size_; i++) {
                value[i] = modulo_uint(value_copy.data(), size_, mod_[i]);
            }
        }
    }

    // input value is assumed that [1, 2, size_Q_] ... [1, 2, size_Q_], "count" elements in total
    // output value is in the form of [1, 2, count] ... [1, 2, count], "size_Q_" elements in total
    // This happens when the degree is count
    void RNSBase::decompose_array(uint64_t *value, size_t count) const {
        if (!value) {
            throw invalid_argument("value cannot be null");
        }

        if (size_ > 1) {
            // Decompose an array of multi-precision integers into an array of arrays, one per each base element
            std::vector<uint64_t> aa(count * size_);
            uint64_t *value_copy = aa.data();
            std::copy_n(value, count * size_, value_copy);
            for (size_t i = 0; i < size_; i++) {
                for (size_t j = 0; j < count; j++)
                    *(value + i * count + j) = modulo_uint(value_copy + j * size_, size_, mod_[i]);
            }
        }
    }

    // According to CRT: x = a1 * x1 * y1 + x2 * x2 * y2 + a3 * x3 * y3 % (product of all primes)
    // where x1 is the product of all prime except prime 1, i.e., big_qiHat_[0]
    // y1 is the inverse of x1 mod prime 1, i.e., qiHatInv_mod_qi_[0]
    void RNSBase::compose(uint64_t *value) const {
        if (!value) {
            throw invalid_argument("value cannot be null");
        }

        if (size_ > 1) {
            // Copy the value
            std::vector<uint64_t> aa(size_);
            uint64_t *copy_value = aa.data();
            std::copy_n(value, size_, copy_value);

            // Clear the result
            set_zero_uint(size_, value);

            auto temp_vec = std::vector<uint64_t>(size_);
            uint64_t *temp_mpi = temp_vec.data();
            uint64_t *punctured_prod = (uint64_t *)(prod_hat_.data());
            for (size_t i = 0; i < size_; i++) {
                uint64_t temp_prod = multiply_uint_mod(copy_value[i], hatInv_mod_[i], mod_[i]);
                multiply_uint(punctured_prod + i * size_, size_, temp_prod, size_, temp_mpi);
                add_uint_uint_mod(temp_mpi, value, prod_mod_.data(), size_, value);
            }
        }
    }

    void RNSBase::compose_array(uint64_t *value, size_t count) const {
        if (!value) {
            throw invalid_argument("value cannot be null");
        }
        if (size_ > 1) {
            // Merge the coefficients first
            std::vector<uint64_t> temp_array(count * size_);
            for (size_t i = 0; i < count; i++) {
                for (size_t j = 0; j < size_; j++) {
                    temp_array[j + (i * size_)] = value[(j * count) + i];
                }
            }

            // Clear the result
            set_zero_uint(count * size_, value);

            uint64_t *temp_array_iter;
            uint64_t *value_iter;
            uint64_t *punctured_prod = (uint64_t *)(prod_hat_.data());

            // Compose an array of RNS integers into a single array of multi-precision integers
            auto temp_mpi = std::vector<uint64_t>(size_);

            for (size_t i = 0; i < count; i++) {
                value_iter = value + i * size_;
                temp_array_iter = temp_array.data() + i * size_;

                for (size_t j = 0; j < size_; j++) {
                    uint64_t temp_prod = multiply_uint_mod(*(temp_array_iter + j), hatInv_mod_[j], mod_[j]);
                    multiply_uint(punctured_prod + j * size_, size_, temp_prod, size_, temp_mpi.data());
                    add_uint_uint_mod(temp_mpi.data(), value_iter, prod_mod_.data(), size_, value_iter);
                }
            }
        }
    }

    void BaseConverter::initialize() {
        // Verify that the size_QP is not too large
        size_t size_Q = ibase_.size();
        size_t size_P = obase_.size();
        auto size_QP = mul_safe(size_Q, size_P);
        if (!fits_in<std::size_t>(size_QP)) {
            throw logic_error("invalid parameters");
        }

        // Create the base-change matrix rows
        QHatModp_.resize(size_P);
        for (size_t j = 0; j < size_P; j++) {
            QHatModp_[j].resize(size_Q);
            auto ibase_big_qiHat = ibase_.big_qiHat();
            auto &pj = obase_.base()[j];
            for (size_t i = 0; i < size_Q; i++) {
                // Base-change matrix contains the punctured products of ibase elements modulo the obase
                QHatModp_[j][i] = modulo_uint(ibase_big_qiHat + i * size_Q, size_Q, pj);
            }
        }

        alphaQModp_.resize(size_Q + 1);
        for (size_t j = 0; j < size_P; j++) {
            auto big_Q = ibase_.big_modulus();
            auto &pj = obase_.base()[j];
            uint64_t big_Q_mod_pj = modulo_uint(big_Q, size_Q, pj);
            for (size_t alpha = 0; alpha < size_Q + 1; alpha++) {
                alphaQModp_[alpha].push_back(multiply_uint_mod(alpha, big_Q_mod_pj, pj));
            }
        }

        negPQHatInvModq_.resize(size_Q);
        negPQHatInvModq_shoup_.resize(size_Q);
        PModq_.resize(size_Q);
        PModq_shoup_.resize(size_Q);
        for (size_t i = 0; i < size_Q; i++) {
            auto &qi = ibase_.base()[i];
            auto QHatInvModqi = ibase_.QHatInvModq()[i];
            auto P = obase_.big_modulus();
            uint64_t PModqi = modulo_uint(P, size_P, qi);
            PModq_[i] = PModqi;
            PModq_shoup_[i] = compute_shoup(PModqi, qi.value());
            uint64_t PQHatInvModqi = multiply_uint_mod(PModqi, QHatInvModqi, qi);
            uint64_t negPQHatInvModqi = qi.value() - PQHatInvModqi;
            negPQHatInvModq_[i] = negPQHatInvModqi;
            negPQHatInvModq_shoup_[i] = compute_shoup(negPQHatInvModqi, qi.value());
        }

        QInvModp_.resize(size_P);
        for (size_t j = 0; j < size_P; j++) {
            QInvModp_[j].resize(size_Q);
            auto &pj = obase_.base()[j];
            for (size_t i = 0; i < size_Q; i++) {
                auto &qi = ibase_.base()[i];
                if (!try_invert_uint_mod(qi.value(), pj, QInvModp_[j][i])) {
                    throw logic_error("invalid rns bases in computing QInvModp");
                }
            }
        }
    }
} // namespace phantom::util

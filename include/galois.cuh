#pragma once

#include <cstdint>
#include <vector>

#include "util/modulus.h"
#include "gputype.h"
#include "util/defines.h"
#include "util/uintcore.h"

constexpr uint32_t generator_ = 5;

typedef struct PhantomGaloisTool {
    int coeff_count_power_ = 0;
    std::size_t coeff_count_ = 0;
    std::vector<size_t> galois_indexs_{};
    std::vector<uint32_t> galois_elts_{};
    std::vector<phantom::util::Pointer<uint32_t>> permutation_tables_;
    std::vector<phantom::util::Pointer<uint64_t>> index_raw_tables_; // only used by bfv
    bool is_bfv_;

    PhantomGaloisTool(const std::vector<uint32_t> &galois_elts, int coeff_count_power, bool is_bfv = false) {
        if ((coeff_count_power < phantom::util::get_power_of_two(POLY_MOD_DEGREE_MIN)) ||
            coeff_count_power > phantom::util::get_power_of_two(POLY_MOD_DEGREE_MAX)) {
            throw std::invalid_argument("coeff_count_power out of range");
        }
        coeff_count_power_ = coeff_count_power;
        coeff_count_ = size_t(1) << coeff_count_power_;

        if (galois_elts.empty()) {
            galois_elts_ = get_elts_all();
        }
        else {
            galois_elts_ = galois_elts;
        }

        is_bfv_ = is_bfv;

        galois_indexs_ = get_indexs_from_elts(galois_elts_);
        auto galois_elts_size = galois_elts_.size();
        const auto coeff_count_minus_one = static_cast<uint32_t>(coeff_count_) - 1;

        // compute permutation_tables_
        permutation_tables_.resize(galois_elts_size);
        std::vector<uint32_t> u32temp(coeff_count_);
        for (std::size_t idx{0}; idx < galois_elts_size; idx++) {
            permutation_tables_[idx].acquire(phantom::util::allocate<uint32_t>(phantom::util::global_pool(), coeff_count_));
            auto galois_elt = galois_elts_.at(idx);
            auto temp_ptr = u32temp.data();
            for (size_t i = coeff_count_; i < coeff_count_ << 1; i++) {
                uint32_t reversed = phantom::util::reverse_bits(static_cast<uint32_t>(i), coeff_count_power_ + 1);
                uint64_t index_raw = (static_cast<uint64_t>(galois_elt) * static_cast<uint64_t>(reversed)) >> 1;
                index_raw &= static_cast<uint64_t>(coeff_count_minus_one);
                *temp_ptr++ = phantom::util::reverse_bits(static_cast<uint32_t>(index_raw), coeff_count_power_);
            }
            cudaMemcpy(permutation_tables_[idx].get(), u32temp.data(), coeff_count_ * sizeof(uint32_t),
                cudaMemcpyHostToDevice);
        }

        if (is_bfv_) {
            index_raw_tables_.resize(galois_elts_size);
            std::vector<uint64_t> u64temp(coeff_count_);
            for (std::size_t idx = 0; idx < galois_elts_size; idx++) {
                index_raw_tables_[idx].
                        acquire(phantom::util::allocate<uint64_t>(phantom::util::global_pool(), coeff_count_));
                auto galois_elt = galois_elts_.at(idx);
                auto temp_ptr = u64temp.data();
                uint64_t index_raw = 0;
                for (uint64_t i = 0; i <= coeff_count_minus_one; i++) {
                    *temp_ptr++ = index_raw;
                    index_raw = (index_raw + galois_elt) & ((coeff_count_ << 1) - 1); // (mod 2n-1)
                }
                cudaMemcpy(index_raw_tables_[idx].get(), u64temp.data(), coeff_count_ * sizeof(uint64_t),
                    cudaMemcpyHostToDevice);
            }
        }
    }

    PhantomGaloisTool &operator=(PhantomGaloisTool &&move) noexcept {
        coeff_count_power_ = move.coeff_count_power_;
        coeff_count_ = move.coeff_count_;
        galois_indexs_ = move.galois_indexs_;
        galois_elts_ = move.galois_elts_;
        is_bfv_ = move.is_bfv_;
        auto permutation_tables_size = move.permutation_tables_.size();
        auto index_raw_tables_size = move.index_raw_tables_.size();

        // copy permutation_tables
        permutation_tables_.resize(permutation_tables_size);
        for (size_t idx{0}; idx < permutation_tables_size; idx++) {
            permutation_tables_[idx].acquire(permutation_tables_[idx]);
        }

        // copy index_raw_tables
        index_raw_tables_.resize(index_raw_tables_size);
        for (size_t idx{0}; idx < index_raw_tables_size; idx++) {
            index_raw_tables_[idx].acquire(move.index_raw_tables_[idx]);
        }

        return *this;
    }

    ~PhantomGaloisTool() = default;

    /**
    Compute the Galois element corresponding to a given rotation step.
    */
    [[nodiscard]] std::uint32_t get_elt_from_step(int step) const;

    [[nodiscard]] inline std::vector<std::uint32_t> get_elts_from_steps(const std::vector<int> &steps) const {
        std::vector<std::uint32_t> elts;
        for (auto step: steps)
            elts.push_back(get_elt_from_step(step));
        return elts;
    }

    /**
    Compute a vector of all necessary galois_elts.
    */
    [[nodiscard]] std::vector<uint32_t> get_elts_all() const;

    /**
    Compute the index in the range of 0 to (coeff_count_ - 1) of a given Galois element.
    */
    [[nodiscard]] static std::size_t get_index_from_elt(const uint32_t galois_elt) {
        return (galois_elt - 1) >> 1;
    }

    [[nodiscard]] static std::vector<std::size_t>
    get_indexs_from_elts(const std::vector<uint32_t> &galois_elts) {
        std::vector<std::size_t> galois_indexs{};
        for (const uint32_t galois_elt: galois_elts) {
            galois_indexs.push_back(get_index_from_elt(galois_elt));
        }
        return galois_indexs;
    }

    void apply_galois(uint64_t *operand, const DNTTTable &rns_table, size_t coeff_mod_size, size_t galois_elt_idx,
                      uint64_t *result);

    void apply_galois(uint64_t *operand, const DModulus *rns_modulus, size_t coeff_mod_size,
                      size_t galois_elt_idx, uint64_t *result);

    void apply_galois_ntt(uint64_t *operand, size_t coeff_mod_size, size_t galois_elt_idx, uint64_t *result);
} PhantomGaloisTool;

[[nodiscard]] inline auto get_elts_from_steps(const std::vector<int> &steps, size_t coeff_count) {
    std::vector<std::uint32_t> galois_elts;
    for (auto step: steps) {
        auto n = static_cast<uint32_t>(coeff_count);
        uint32_t m32 = n * 2;
        auto m = static_cast<uint64_t>(m32);

        if (step == 0) {
            galois_elts.push_back(static_cast<uint32_t>(m - 1));
        }
        else {
            // Extract sign of steps. When steps is positive, the rotation
            // is to the left; when steps is negative, it is to the right.
            bool sign = step < 0;
            auto pos_step = static_cast<uint32_t>(abs(step));

            if (pos_step >= (n >> 1)) {
                throw std::invalid_argument("step count too large");
            }

            pos_step &= m32 - 1;
            if (sign) {
                step = static_cast<int>(n >> 1) - static_cast<int>(pos_step);
            }
            else {
                step = static_cast<int>(pos_step);
            }

            // Construct Galois element for row rotation
            auto gen = static_cast<uint64_t>(generator_);
            uint64_t galois_elt = 1;
            while (step--) {
                galois_elt *= gen;
                galois_elt &= m - 1;
            }
            galois_elts.push_back(static_cast<uint32_t>(galois_elt));
        }
    }
    return galois_elts;
}

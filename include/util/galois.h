// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "modulus.h"
#include "defines.h"
#include "common.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <memory>

namespace phantom::arith {
    class GaloisTool {
    public:
        explicit GaloisTool(int coeff_count_power) {
            initialize(coeff_count_power);
        }

        GaloisTool(const GaloisTool &copy) = delete;

        GaloisTool(GaloisTool &&source) = delete;

        GaloisTool &operator=(const GaloisTool &assign) = delete;

        GaloisTool &operator=(GaloisTool &&assign) = delete;

        void apply_galois(
            std::uint64_t *operand, std::uint32_t galois_elt, const Modulus &modulus, std::uint64_t *result) const;

        /*inline void apply_galois(
            ConstRNSIter operand, std::size_t coeff_modulus_size, std::uint32_t galois_elt,
            ConstModulusIter modulus, RNSIter result) const
        {
            SEAL_ITERATE(iter(operand, modulus, result), coeff_modulus_size, [&](auto I) {
                this->apply_galois(get<0>(I), galois_elt, get<1>(I), get<2>(I));
            });
        }*/

        inline void apply_galois_rns(
            std::uint64_t *operand, std::size_t coeff_modulus_size, std::uint32_t galois_elt,
            Modulus *modulus, std::uint64_t *result) const {
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                apply_galois(operand + i * coeff_count_, galois_elt, modulus[i], result + i * coeff_count_);
            }
        }

        /*void apply_galois(
            ConstPolyIter operand, std::size_t size, std::uint32_t galois_elt, ConstModulusIter modulus,
            PolyIter result) const
        {
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(operand, result), size, [&](auto I) {
                this->apply_galois(get<0>(I), coeff_modulus_size, galois_elt, modulus, get<1>(I));
            });
        }*/

        void apply_galois_poly(
            std::uint64_t *operand, std::size_t size, std::uint32_t galois_elt, Modulus *modulus,
            std::uint64_t *result, std::size_t coeff_modulus_size) const {
            auto step_size = mul_safe(coeff_count_, coeff_modulus_size);
            for (size_t i = 0; i < size; i++)
                apply_galois_rns(operand + i * step_size, coeff_modulus_size, galois_elt, modulus,
                                 result + i * step_size);
        }

        void apply_galois_ntt(std::uint64_t *operand, std::uint32_t galois_elt, std::uint64_t *result) const;

        /*void apply_galois_ntt(
            ConstRNSIter operand, std::size_t coeff_modulus_size, std::uint32_t galois_elt, RNSIter result) const
        {
            SEAL_ITERATE(iter(operand, result), coeff_modulus_size, [&](auto I) {
                this->apply_galois_ntt(get<0>(I), galois_elt, get<1>(I));
            });
        }*/

        void apply_galois_ntt_rns(
            std::uint64_t *operand, std::size_t coeff_modulus_size, std::uint32_t galois_elt,
            std::uint64_t *result) const {
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                apply_galois_ntt(operand + i * coeff_count_, galois_elt, result + i * coeff_count_);
            }
        }

        /*void apply_galois_ntt(
            ConstPolyIter operand, std::size_t size, std::uint32_t galois_elt, PolyIter result) const
        {
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(operand, result), size, [&](auto I) {
                this->apply_galois_ntt(get<0>(I), coeff_modulus_size, galois_elt, get<1>(I));
            });
        }*/

        void apply_galois_ntt_poly(
            std::uint64_t *operand, std::size_t size, std::uint32_t galois_elt, std::uint64_t *result,
            std::size_t coeff_modulus_size) const {
            auto step_size = mul_safe(coeff_count_, coeff_modulus_size);
            for (size_t i = 0; i < size; i++)
                apply_galois_ntt_rns(operand + i * step_size, coeff_modulus_size, galois_elt, result + i * step_size);
        }

        /**
        Compute the Galois element corresponding to a given rotation step.
        */
        [[nodiscard]] std::uint32_t get_elt_from_step(int step) const;

        /**
        Compute the Galois elements corresponding to a vector of given rotation steps.
        */
        [[nodiscard]] std::vector<std::uint32_t> get_elts_from_steps(const std::vector<int> &steps) const;

        /**
        Compute a vector of all necessary galois_elts.
        */
        [[nodiscard]] std::vector<std::uint32_t> get_elts_all() const noexcept;

        /**
        Compute the index in the range of 0 to (coeff_count_ - 1) of a given Galois element.
        */
        [[nodiscard]] static inline std::size_t get_index_from_elt(std::uint32_t galois_elt) {
            return (std::size_t)((galois_elt - 1) >> 1);
        }

    private:
        void initialize(int coeff_count_power);

        void generate_table_ntt(std::uint32_t galois_elt, std::shared_ptr<std::uint32_t> &result) const;

        int coeff_count_power_ = 0;

        std::size_t coeff_count_ = 0;

        static constexpr std::uint32_t generator_ = 5;

        mutable std::shared_ptr<std::shared_ptr<std::uint32_t>> permutation_tables_;
    };
}

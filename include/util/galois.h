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

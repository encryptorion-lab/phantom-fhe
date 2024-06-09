// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "util/galois.h"
#include "util/numth.h"
#include "util/uintcore.h"

using namespace std;

namespace phantom::arith {
    void GaloisTool::generate_table_ntt(uint32_t galois_elt, std::shared_ptr<uint32_t> &result) const {
        if (result) {
            return;
        }

        auto temp = std::vector<uint32_t>(coeff_count_);
        auto temp_ptr = temp.data();

        uint32_t coeff_count_minus_one = uint32_t(coeff_count_) - 1;
        for (size_t i = coeff_count_; i < coeff_count_ << 1; i++) {
            uint32_t reversed = reverse_bits<uint32_t>((uint32_t)(i), coeff_count_power_ + 1);
            uint64_t index_raw = ((uint64_t)(galois_elt) * (uint64_t)(reversed)) >> 1;
            index_raw &= (uint64_t)(coeff_count_minus_one);
            *temp_ptr++ = reverse_bits<uint32_t>((uint32_t)(index_raw), coeff_count_power_);
        }

        if (result) {
            return;
        }
        result = std::shared_ptr<uint32_t>(temp.data());
    }

    uint32_t GaloisTool::get_elt_from_step(int step) const {
        uint32_t n = static_cast<uint32_t>(coeff_count_);
        uint32_t m32 = mul_safe(n, uint32_t(2));
        uint64_t m = static_cast<uint64_t>(m32);

        if (step == 0) {
            return static_cast<uint32_t>(m - 1);
        }
        else {
            // Extract sign of steps. When steps is positive, the rotation
            // is to the left; when steps is negative, it is to the right.
            bool sign = step < 0;
            uint32_t pos_step = static_cast<uint32_t>(abs(step));

            if (pos_step >= (n >> 1)) {
                throw invalid_argument("step count too large");
            }

            pos_step &= m32 - 1;
            if (sign) {
                step = static_cast<int>(n >> 1) - static_cast<int>(pos_step);
            }
            else {
                step = static_cast<int>(pos_step);
            }

            // Construct Galois element for row rotation
            uint64_t gen = static_cast<uint64_t>(generator_);
            uint64_t galois_elt = 1;
            while (step--) {
                galois_elt *= gen;
                galois_elt &= m - 1;
            }
            return static_cast<uint32_t>(galois_elt);
        }
    }

    vector<uint32_t> GaloisTool::get_elts_from_steps(const vector<int> &steps) const {
        vector<uint32_t> galois_elts;
        transform(steps.begin(), steps.end(), back_inserter(galois_elts),
                  [&](auto s) { return this->get_elt_from_step(s); });
        return galois_elts;
    }

    vector<uint32_t> GaloisTool::get_elts_all() const noexcept {
        uint32_t m = uint32_t((uint64_t)(coeff_count_) << 1);
        vector<uint32_t> galois_elts{};

        // Generate Galois keys for m - 1 (X -> X^{m-1})
        galois_elts.push_back(m - 1);

        // Generate Galois key for power of generator_ mod m (X -> X^{3^k}) and
        // for negative power of generator_ mod m (X -> X^{-3^k})
        uint64_t pos_power = generator_;
        uint64_t neg_power = 0;
        try_invert_uint_mod(generator_, m, neg_power);
        for (int i = 0; i < coeff_count_power_ - 1; i++) {
            galois_elts.push_back((uint32_t)(pos_power));
            pos_power *= pos_power;
            pos_power &= (m - 1);

            galois_elts.push_back((uint32_t)(neg_power));
            neg_power *= neg_power;
            neg_power &= (m - 1);
        }

        return galois_elts;
    }

    void GaloisTool::initialize(int coeff_count_power) {
        if ((coeff_count_power < get_power_of_two(POLY_MOD_DEGREE_MIN)) ||
            coeff_count_power > get_power_of_two(POLY_MOD_DEGREE_MAX)) {
            throw invalid_argument("coeff_count_power out of range");
        }

        coeff_count_power_ = coeff_count_power;
        coeff_count_ = size_t(1) << coeff_count_power_;

        // Capacity for coeff_count_ number of tables
        permutation_tables_ = std::shared_ptr<std::shared_ptr<uint32_t>>(
            std::vector<std::shared_ptr<uint32_t>>(coeff_count_).data());
    }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common.h"
#include "uintcore.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace phantom {
    namespace util {
        [[nodiscard]] inline std::string poly_to_hex_string(
            const std::uint64_t *value, std::size_t coeff_count, std::size_t coeff_uint64_count) {
            // First check if there is anything to print
            if (!coeff_count || !coeff_uint64_count) {
                return "0";
            }

            std::ostringstream result;
            bool empty = true;
            value += util::mul_safe(coeff_count - 1, coeff_uint64_count);
            while (coeff_count--) {
                if (is_zero_uint(value, coeff_uint64_count)) {
                    value -= coeff_uint64_count;
                    continue;
                }
                if (!empty) {
                    result << " + ";
                }
                result << uint_to_hex_string(value, coeff_uint64_count);
                if (coeff_count) {
                    result << "x^" << coeff_count;
                }
                empty = false;
                value -= coeff_uint64_count;
            }
            if (empty) {
                result << "0";
            }
            return result.str();
        }

        [[nodiscard]] inline std::string poly_to_dec_string(
            const std::uint64_t *value, std::size_t coeff_count, std::size_t coeff_uint64_count) {
            // First check if there is anything to print
            if (!coeff_count || !coeff_uint64_count) {
                return "0";
            }

            std::ostringstream result;
            bool empty = true;
            value += coeff_count - 1;
            while (coeff_count--) {
                if (is_zero_uint(value, coeff_uint64_count)) {
                    value -= coeff_uint64_count;
                    continue;
                }
                if (!empty) {
                    result << " + ";
                }
                result << uint_to_dec_string(value, coeff_uint64_count);
                if (coeff_count) {
                    result << "x^" << coeff_count;
                }
                empty = false;
                value -= coeff_uint64_count;
            }
            if (empty) {
                result << "0";
            }
            return result.str();
        }

        inline void set_poly(
            const std::uint64_t *poly, std::size_t coeff_count, std::size_t coeff_uint64_count, std::uint64_t *result) {
            set_uint(poly, util::mul_safe(coeff_count, coeff_uint64_count), result);
        }

        inline void set_poly_array(
            const std::uint64_t *poly, std::size_t poly_count, std::size_t coeff_count, std::size_t coeff_uint64_count,
            std::uint64_t *result) {
            set_uint(poly, util::mul_safe(poly_count, coeff_count, coeff_uint64_count), result);
        }
    }
}

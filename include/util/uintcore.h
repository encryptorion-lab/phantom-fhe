// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common.h"
#include "defines.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace phantom::arith {

    inline void set_zero_uint(std::size_t uint64_count, std::uint64_t *result) {
        std::fill_n(result, uint64_count, std::uint64_t(0));
    }

    inline void set_uint(std::uint64_t value, std::size_t uint64_count, std::uint64_t *result) {
        *result++ = value;
        for (; --uint64_count; result++) {
            *result = 0;
        }
    }

    /**
     * fill the result with the value
     * @param[in] value The value which the result will be set.
     * @param[in] uint64_cout The number of the result (in uint64) will be set
     * @param[in] result The pointer which will be set with the given value.
     */
    inline void set_uint(const std::uint64_t *value, std::size_t uint64_count, std::uint64_t *result) {
        if ((value == result) || !uint64_count) {
            return;
        }
        std::copy_n(value, uint64_count, result);
    }

    [[nodiscard]] inline bool is_zero_uint(const std::uint64_t *value, std::size_t uint64_count) {
        return std::all_of(value, value + uint64_count, [](auto coeff) -> bool { return !coeff; });
    }

    [[nodiscard]] inline bool is_equal_uint(
            const std::uint64_t *value, std::size_t uint64_count, std::uint64_t scalar) {
        if (*value++ != scalar) {
            return false;
        }
        return std::all_of(value, value + uint64_count - 1, [](auto coeff) -> bool { return !coeff; });
    }

    [[nodiscard]] inline bool is_high_bit_set_uint(const std::uint64_t *value, std::size_t uint64_count) {
        return (value[uint64_count - 1] >> (bits_per_uint64 - 1)) != 0;
    }

    [[nodiscard]] inline bool is_bit_set_uint(
            const std::uint64_t *value, std::size_t uint64_count [[maybe_unused]], int bit_index) {
        int uint64_index = bit_index / bits_per_uint64;
        int sub_bit_index = bit_index - uint64_index * bits_per_uint64;
        return ((value[static_cast<std::size_t>(uint64_index)] >> sub_bit_index) & 1) != 0;
    }

    inline void set_bit_uint(std::uint64_t *value, std::size_t uint64_count [[maybe_unused]], int bit_index) {
        int uint64_index = bit_index / bits_per_uint64;
        int sub_bit_index = bit_index % bits_per_uint64;
        value[static_cast<std::size_t>(uint64_index)] |= std::uint64_t(1) << sub_bit_index;
    }

    [[nodiscard]] inline int get_significant_bit_count_uint(const std::uint64_t *value, std::size_t uint64_count) {
        value += uint64_count - 1;
        for (; *value == 0 && uint64_count > 1; uint64_count--) {
            value--;
        }

        return static_cast<int>(uint64_count - 1) * bits_per_uint64 + get_significant_bit_count(*value);
    }

    // Return the first (right first) non-zero uint64 value.
    [[nodiscard]] inline std::size_t get_significant_uint64_count_uint(
            const std::uint64_t *value, std::size_t uint64_count) {
        value += uint64_count - 1;
        for (; uint64_count && !*value; uint64_count--) {
            value--;
        }

        return uint64_count;
    }

    [[nodiscard]] inline std::size_t get_nonzero_uint64_count_uint(
            const std::uint64_t *value, std::size_t uint64_count) {
        std::size_t nonzero_count = uint64_count;

        value += uint64_count - 1;
        for (; uint64_count; uint64_count--) {
            if (*value-- == 0) {
                nonzero_count--;
            }
        }

        return nonzero_count;
    }

    inline void set_uint(
            const std::uint64_t *value, std::size_t value_uint64_count, std::size_t result_uint64_count,
            std::uint64_t *result) {
        if (value == result || !value_uint64_count) {
            // Fast path to handle self assignment.
            std::fill(result + value_uint64_count, result + result_uint64_count, std::uint64_t(0));
        } else {
            std::size_t min_uint64_count = std::min<>(value_uint64_count, result_uint64_count);
            std::copy_n(value, min_uint64_count, result);
            std::fill(result + min_uint64_count, result + result_uint64_count, std::uint64_t(0));
        }
    }

    /**
    If the value is a power of two, return the power; otherwise, return -1.
    */
    [[nodiscard]] inline int get_power_of_two(std::uint64_t value) {
        if (value == 0 || (value & (value - 1)) != 0) {
            return -1;
        }

        unsigned long result = 0;
        get_msb_index_generic(&result, value);
        return static_cast<int>(result);
    }

    inline void filter_highbits_uint(std::uint64_t *operand, std::size_t uint64_count, int bit_count) {
        std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);
        if (unsigned_eq(bit_count, mul_safe(uint64_count, bits_per_uint64_sz))) {
            return;
        }
        int uint64_index = bit_count / bits_per_uint64;
        int subbit_index = bit_count - uint64_index * bits_per_uint64;
        operand += uint64_index;
        *operand++ &= (std::uint64_t(1) << subbit_index) - 1;
        for (int long_index = uint64_index + 1; unsigned_lt(long_index, uint64_count); long_index++) {
            *operand++ = 0;
        }
    }

    [[nodiscard]] inline int compare_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count) {
        int result = 0;
        operand1 += uint64_count - 1;
        operand2 += uint64_count - 1;

        for (; (result == 0) && uint64_count--; operand1--, operand2--) {
            result = (*operand1 > *operand2) - (*operand1 < *operand2);
        }
        return result;
    }

    [[nodiscard]] inline int compare_uint(
            const std::uint64_t *operand1, std::size_t operand1_uint64_count, const std::uint64_t *operand2,
            std::size_t operand2_uint64_count) {
        int result = 0;
        operand1 += operand1_uint64_count - 1;
        operand2 += operand2_uint64_count - 1;

        std::size_t min_uint64_count = std::min<>(operand1_uint64_count, operand2_uint64_count);

        operand1_uint64_count -= min_uint64_count;
        for (; (result == 0) && operand1_uint64_count--; operand1--) {
            result = (*operand1 > 0);
        }

        operand2_uint64_count -= min_uint64_count;
        for (; (result == 0) && operand2_uint64_count--; operand2--) {
            result = -(*operand2 > 0);
        }

        for (; (result == 0) && min_uint64_count--; operand1--, operand2--) {
            result = (*operand1 > *operand2) - (*operand1 < *operand2);
        }
        return result;
    }

    [[nodiscard]] inline bool is_greater_than_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count) {
        return compare_uint(operand1, operand2, uint64_count) > 0;
    }

    [[nodiscard]] inline bool is_greater_than_or_equal_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count) {
        return compare_uint(operand1, operand2, uint64_count) >= 0;
    }

    [[nodiscard]] inline bool is_less_than_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count) {
        return compare_uint(operand1, operand2, uint64_count) < 0;
    }

    [[nodiscard]] inline bool is_less_than_or_equal_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count) {
        return compare_uint(operand1, operand2, uint64_count) <= 0;
    }

    [[nodiscard]] inline bool is_equal_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count) {
        return compare_uint(operand1, operand2, uint64_count) == 0;
    }

    [[nodiscard]] inline bool is_greater_than_uint(
            const std::uint64_t *operand1, std::size_t operand1_uint64_count, const std::uint64_t *operand2,
            std::size_t operand2_uint64_count) {
        return compare_uint(operand1, operand1_uint64_count, operand2, operand2_uint64_count) > 0;
    }

    [[nodiscard]] inline bool is_greater_than_or_equal_uint(
            const std::uint64_t *operand1, std::size_t operand1_uint64_count, const std::uint64_t *operand2,
            std::size_t operand2_uint64_count) {
        return compare_uint(operand1, operand1_uint64_count, operand2, operand2_uint64_count) >= 0;
    }

    [[nodiscard]] inline bool is_less_than_uint(
            const std::uint64_t *operand1, std::size_t operand1_uint64_count, const std::uint64_t *operand2,
            std::size_t operand2_uint64_count) {
        return compare_uint(operand1, operand1_uint64_count, operand2, operand2_uint64_count) < 0;
    }

    [[nodiscard]] inline bool is_less_than_or_equal_uint(
            const std::uint64_t *operand1, std::size_t operand1_uint64_count, const std::uint64_t *operand2,
            std::size_t operand2_uint64_count) {
        return compare_uint(operand1, operand1_uint64_count, operand2, operand2_uint64_count) <= 0;
    }

    [[nodiscard]] inline bool is_equal_uint(
            const std::uint64_t *operand1, std::size_t operand1_uint64_count, const std::uint64_t *operand2,
            std::size_t operand2_uint64_count) {
        return compare_uint(operand1, operand1_uint64_count, operand2, operand2_uint64_count) == 0;
    }
}

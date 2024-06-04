// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "uintarith.h"
#include "uintcore.h"
#include <cstdint>

namespace phantom::arith {
    inline void increment_uint_mod(
            const std::uint64_t *operand, const std::uint64_t *modulus, std::size_t uint64_count,
            std::uint64_t *result) {
        unsigned char carry = increment_uint(operand, uint64_count, result);
        if (carry || is_greater_than_or_equal_uint(result, modulus, uint64_count)) {
            sub_uint(result, modulus, uint64_count, result);
        }
    }

    inline void decrement_uint_mod(
            const std::uint64_t *operand, const std::uint64_t *modulus, std::size_t uint64_count,
            std::uint64_t *result) {
        if (decrement_uint(operand, uint64_count, result)) {
            add_uint(result, modulus, uint64_count, result);
        }
    }

    inline void negate_uint_mod(
            const std::uint64_t *operand, const std::uint64_t *modulus, std::size_t uint64_count,
            std::uint64_t *result) {
        if (is_zero_uint(operand, uint64_count)) {
            // Negation of zero is zero.
            set_zero_uint(uint64_count, result);
        } else {
            // Otherwise, we know operand > 0 and < modulus so subtract modulus - operand.
            sub_uint(modulus, operand, uint64_count, result);
        }
    }

    inline void add_uint_uint_mod(
            const std::uint64_t *operand1, const std::uint64_t *operand2, const std::uint64_t *modulus,
            std::size_t uint64_count, std::uint64_t *result) {
        unsigned char carry = add_uint(operand1, operand2, uint64_count, result);
        if (carry || is_greater_than_or_equal_uint(result, modulus, uint64_count)) {
            sub_uint(result, modulus, uint64_count, result);
        }
    }

    inline void sub_uint_uint_mod(
            const std::uint64_t *operand1, const std::uint64_t *operand2, const std::uint64_t *modulus,
            std::size_t uint64_count, std::uint64_t *result) {
        if (sub_uint(operand1, operand2, uint64_count, result)) {
            add_uint(result, modulus, uint64_count, result);
        }
    }

    bool try_invert_uint_mod(
            const std::uint64_t *operand, const std::uint64_t *modulus, std::size_t uint64_count,
            std::uint64_t *result);
}

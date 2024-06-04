// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "modulus.h"
#include "defines.h"
#include "numth.h"
#include "uintarith.h"
#include <cstdint>
#include <type_traits>

namespace phantom::arith {
    /**
    Returns (operand++) mod modulus.
    Correctness: operand must be at most (2 * modulus -2) for correctness.
    */
    [[nodiscard]] inline std::uint64_t increment_uint_mod(std::uint64_t operand, const Modulus &modulus) {
        operand++;
        return operand - (modulus.value() &
                          static_cast<std::uint64_t>(-static_cast<std::int64_t>(operand >= modulus.value())));
    }

    /**
    Returns (operand--) mod modulus.
    @param[in] operand Must be at most (modulus - 1).
    */
    [[nodiscard]] inline std::uint64_t decrement_uint_mod(std::uint64_t operand, const Modulus &modulus) {
        std::int64_t carry = static_cast<std::int64_t>(operand == 0);
        return operand - 1 + (modulus.value() & static_cast<std::uint64_t>(-carry));
    }

    /**
    Returns (-operand) mod modulus.
    Correctness: operand must be at most modulus for correctness.
    */
    [[nodiscard]] inline std::uint64_t negate_uint_mod(std::uint64_t operand, const Modulus &modulus) {
        std::int64_t non_zero = static_cast<std::int64_t>(operand != 0);
        return (modulus.value() - operand) & static_cast<std::uint64_t>(-non_zero);
    }

    /**
    Returns (operand1 + operand2) mod modulus.
    Correctness: (operand1 + operand2) must be at most (2 * modulus - 1).
    */
    [[nodiscard]] inline std::uint64_t add_uint_mod(
            std::uint64_t operand1, std::uint64_t operand2, const Modulus &modulus) {
        // Sum of operands modulo Modulus can never wrap around 2^64
        operand1 += operand2;
        return operand1 >= modulus.value() ? operand1 - modulus.value() : operand1;
    }

    /**
    Returns (operand1 - operand2) mod modulus.
    Correctness: (operand1 - operand2) must be at most (modulus - 1) and at least (-modulus).
    @param[in] operand1 Should be at most (modulus - 1).
    @param[in] operand2 Should be at most (modulus - 1).
    */
    [[nodiscard]] inline std::uint64_t sub_uint_mod(
            std::uint64_t operand1, std::uint64_t operand2, const Modulus &modulus) {
        uint64_t temp;
        int64_t borrow = (int64_t) (sub_uint64_generic(operand1, operand2, 0, &temp));
        return (uint64_t) (temp) + (modulus.value() & (uint64_t) (-borrow));
    }

    /**
    Returns input mod modulus. This is not standard Barrett reduction.
    Correctness: modulus must be at most 63-bit.
    @param[in] input Should be at most 128-bit.
    */
    template<typename T, typename = std::enable_if_t<is_uint64_v<T>>>

    [[nodiscard]] inline std::uint64_t barrett_reduce_128(const T *input, const Modulus &modulus) {
        // Reduces input using base 2^64 Barrett reduction
        // input allocation size must be 128 bits

        //            uint64_t tmp1, tmp2[2], tmp3, carry;
        uint64_t tmp1, tmp3, carry;
        std::vector<uint64_t> tmp2(2);
        const std::uint64_t *const_ratio = modulus.const_ratio().data();

        // Multiply input and const_ratio
        // Round 1
        multiply_uint64_hw64(input[0], const_ratio[0], &carry);

        multiply_uint64(input[0], const_ratio[1], tmp2.data());

        tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, &tmp1);

        // Round 2
        multiply_uint64(input[1], const_ratio[0], tmp2.data());

        carry = tmp2[1] + add_uint64(tmp1, tmp2[0], &tmp1);

        // This is all we care about
        tmp1 = input[1] * const_ratio[1] + tmp3 + carry;

        // Barrett subtraction
        tmp3 = input[0] - tmp1 * modulus.value();

        // One more subtraction is enough
        return tmp3 >= modulus.value() ? tmp3 - modulus.value() : tmp3;
    }

    /**
    Returns input mod modulus. This is not standard Barrett reduction.
    Correctness: modulus must be at most 63-bit.
    */
    template<typename T, typename = std::enable_if_t<is_uint64_v<T>>>

    [[nodiscard]] inline std::uint64_t barrett_reduce_64(T input, const Modulus &modulus) {
        // Reduces input using base 2^64 Barrett reduction
        // floor(2^64 / mod) == floor( floor(2^128 / mod) )
        uint64_t tmp[2];
        const std::uint64_t *const_ratio = modulus.const_ratio().data();
        multiply_uint64_hw64(input, const_ratio[1], tmp + 1);

        // Barrett subtraction
        tmp[0] = input - tmp[1] * modulus.value();

        // One more subtraction is enough
        return tmp[0] >= modulus.value() ? tmp[0] - modulus.value() : tmp[0];
    }

    /**
    Returns (operand1 * operand2) mod modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    [[nodiscard]] inline std::uint64_t multiply_uint_mod(
            std::uint64_t operand1, std::uint64_t operand2, const Modulus &modulus) {
        uint64_t z[2];
        multiply_uint64(operand1, operand2, z);
        return barrett_reduce_128(z, modulus);
    }

    /**
    This struct contains a operand and a precomputed quotient: (operand << 64) / modulus, for a specific modulus.
    When passed to multiply_uint_mod, a faster variant of Barrett reduction will be performed.
    Operand must be less than modulus.
    */
    inline uint64_t compute_shoup(const uint64_t operand, const uint64_t modulus) {
        // Using __uint128_t to avoid overflow during multiplication
        __uint128_t temp = operand;
        temp <<= 64; // multiplying by 2^64
        return temp / modulus;
    }

    /**
    Returns x * y mod modulus.
    This is a highly-optimized variant of Barrett reduction.
    Correctness: modulus should be at most 63-bit, and y must be less than modulus.
    */
    [[nodiscard]] inline std::uint64_t multiply_uint_mod_shoup(
            const uint64_t x, const uint64_t y, const uint64_t y_shoup, const Modulus &modulus) {
        uint64_t tmp1;
        const std::uint64_t p = modulus.value();
        multiply_uint64_hw64(x, y_shoup, &tmp1);
        const uint64_t tmp2 = y * x - tmp1 * p;
        return tmp2 >= p ? tmp2 - p : tmp2;
    }

    /**
    Returns x * y mod modulus or x * y mod modulus + modulus.
    This is a highly-optimized variant of Barrett reduction and reduce to [0, 2 * modulus - 1].
    Correctness: modulus should be at most 63-bit, and y must be less than modulus.
    */
    [[nodiscard]] inline std::uint64_t multiply_uint_mod_lazy(
            const uint64_t x, const uint64_t y, const uint64_t y_shoup, const Modulus &modulus) {
        uint64_t tmp1;
        const std::uint64_t p = modulus.value();
        multiply_uint64_hw64(x, y_shoup, &tmp1);
        return y * x - tmp1 * p;
    }

    /**
    Returns value[0] = value mod modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    inline void modulo_uint_inplace(uint64_t *value, const size_t value_uint64_count, const Modulus &modulus) {
        if (value_uint64_count == 1) {
            if (*value < modulus.value()) {
                return;
            } else {
                *value = barrett_reduce_64(*value, modulus);
            }
        }

        // Starting from the top, reduce always 128-bit blocks
        for (std::size_t i = value_uint64_count - 1; i--;) {
            value[i] = barrett_reduce_128(value + i, modulus);
            value[i + 1] = 0;
        }
    }

    /**
    Returns value mod modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    [[nodiscard]] inline std::uint64_t modulo_uint(
            const std::uint64_t *value, const size_t value_uint64_count, const Modulus &modulus) {
        if (value_uint64_count == 1) {
            // If value < modulus no operation is needed
            if (*value < modulus.value())
                return *value;
            else
                return barrett_reduce_64(*value, modulus);
        }

        // Temporary space for 128-bit reductions
        uint64_t temp[2]{0, value[value_uint64_count - 1]};
        for (size_t k = value_uint64_count - 1; k--;) {
            temp[0] = value[k];
            temp[1] = barrett_reduce_128(temp, modulus);
        }

        // Save the result modulo i-th prime
        return temp[1];
    }

    /**
    Returns (operand1 * operand2) + operand3 mod modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    inline std::uint64_t multiply_add_uint_mod(
            std::uint64_t operand1, std::uint64_t operand2, std::uint64_t operand3, const Modulus &modulus) {
        // Lazy reduction
        uint64_t temp[2];
        multiply_uint64(operand1, operand2, temp);
        temp[1] += add_uint64(temp[0], operand3, temp);
        return barrett_reduce_128(temp, modulus);
    }

    /**
    Returns (operand1 * operand2) + operand3 mod modulus.
    Correctness: Follows the condition of multiply_uint_mod.
    */
    inline std::uint64_t multiply_add_uint_mod(
            uint64_t operand1, uint64_t operand2, uint64_t operand2_shoup, std::uint64_t operand3,
            const Modulus &modulus) {
        return add_uint_mod(
                multiply_uint_mod_shoup(operand1, operand2, operand2_shoup, modulus),
                barrett_reduce_64(operand3, modulus), modulus);
    }

    inline bool try_invert_uint_mod(std::uint64_t operand, const Modulus &modulus, std::uint64_t &result) {
        return try_invert_uint_mod(operand, modulus.value(), result);
    }

    /**
    Returns operand^exponent mod modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    [[nodiscard]] std::uint64_t exponentiate_uint_mod(
            std::uint64_t operand, std::uint64_t exponent, const Modulus &modulus);

    /**
    Computes numerator = numerator mod modulus, quotient = numerator / modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    void divide_uint_mod_inplace(
            std::uint64_t *numerator, const Modulus &modulus, std::size_t uint64_count, std::uint64_t *quotient);

    /**
    Computes <operand1, operand2> mod modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    [[nodiscard]] std::uint64_t dot_product_mod(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t count,
            const Modulus &modulus);
}

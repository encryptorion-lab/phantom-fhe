// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common.h"
#include "defines.h"
#include "uintcore.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace phantom::arith {
    /** Return the sum of two operands (with carry)
     * The sum is stored in result
     * Return value is the carry
     */
    [[nodiscard]] inline unsigned char add_uint64(
            uint64_t operand1, uint64_t operand2, unsigned char carry, uint64_t *result) {
        operand1 += operand2;
        *result = operand1 + carry;
        return (operand1 < operand2) || (~operand1 < carry);
    }

    [[nodiscard]] inline unsigned char add_uint64(uint64_t operand1, uint64_t operand2, uint64_t *result) {
        *result = operand1 + operand2;
        return static_cast<unsigned char>(*result < operand1);
    }

    template<typename T, typename S, typename = std::enable_if_t<is_uint64_v < T, S>>>

    inline unsigned char add_uint128(const T *operand1, const S *operand2, uint64_t *result) {
        unsigned char carry = add_uint64(operand1[0], operand2[0], result);
        return add_uint64(operand1[1], operand2[1], carry, result + 1);
    }

    inline unsigned char add_uint(
            const std::uint64_t *operand1, std::size_t operand1_uint64_count, const std::uint64_t *operand2,
            std::size_t operand2_uint64_count, unsigned char carry, std::size_t result_uint64_count,
            std::uint64_t *result) {
        for (std::size_t i = 0; i < result_uint64_count; i++) {
            uint64_t temp_result;
            carry = add_uint64(
                    (i < operand1_uint64_count) ? *operand1++ : 0, (i < operand2_uint64_count) ? *operand2++ : 0, carry,
                    &temp_result);
            *result++ = temp_result;
        }
        return carry;
    }

    inline unsigned char add_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count,
            std::uint64_t *result) {
        // Unroll first iteration of loop. We assume uint64_count > 0.
        unsigned char carry = add_uint64(*operand1++, *operand2++, result++);

        // Do the rest
        for (; --uint64_count; operand1++, operand2++, result++) {
            uint64_t temp_result;
            carry = add_uint64(*operand1, *operand2, carry, &temp_result);
            *result = temp_result;
        }
        return carry;
    }

    inline unsigned char add_uint(
            const std::uint64_t *operand1, std::size_t uint64_count, std::uint64_t operand2, std::uint64_t *result) {
        // Unroll first iteration of loop. We assume uint64_count > 0.
        unsigned char carry = add_uint64(*operand1++, operand2, result++);

        // Do the rest
        for (; --uint64_count; operand1++, result++) {
            uint64_t temp_result;
            carry = add_uint64(*operand1, std::uint64_t(0), carry, &temp_result);
            *result = temp_result;
        }
        return carry;
    }

    template<typename T, typename S, typename = std::enable_if_t<is_uint64_v < T, S>>>

    [[nodiscard]] inline unsigned char sub_uint64_generic(
            T operand1, S operand2, unsigned char borrow, uint64_t *result) {
        auto diff = operand1 - operand2;
        *result = diff - (borrow != 0);
        return (diff > operand1) || (diff < borrow);
    }

    template<typename T, typename S, typename = std::enable_if_t<is_uint64_v < T, S>>>

    [[nodiscard]] inline unsigned char sub_uint64(
            T operand1, S operand2, unsigned char borrow, uint64_t *result) {
        return sub_uint64_generic(operand1, operand2, borrow, result);
    }

    template<typename T, typename S, typename R, typename = std::enable_if_t<is_uint64_v < T, S, R>>>

    [[nodiscard]] inline unsigned char sub_uint64(T operand1, S operand2, R *result) {
        *result = operand1 - operand2;
        return static_cast<unsigned char>(operand2 > operand1);
    }

    inline unsigned char sub_uint(
            const std::uint64_t *operand1, std::size_t operand1_uint64_count, const std::uint64_t *operand2,
            std::size_t operand2_uint64_count, unsigned char borrow, std::size_t result_uint64_count,
            std::uint64_t *result) {
        for (std::size_t i = 0; i < result_uint64_count; i++, operand1++, operand2++, result++) {
            uint64_t temp_result;
            borrow = sub_uint64(
                    (i < operand1_uint64_count) ? *operand1 : 0, (i < operand2_uint64_count) ? *operand2 : 0, borrow,
                    &temp_result);
            *result = temp_result;
        }
        return borrow;
    }

    inline unsigned char sub_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count,
            std::uint64_t *result) {
        // Unroll first iteration of loop. We assume uint64_count > 0.
        unsigned char borrow = sub_uint64(*operand1++, *operand2++, result++);

        // Do the rest
        for (; --uint64_count; operand1++, operand2++, result++) {
            uint64_t temp_result;
            borrow = sub_uint64(*operand1, *operand2, borrow, &temp_result);
            *result = temp_result;
        }
        return borrow;
    }

    inline unsigned char sub_uint(
            const std::uint64_t *operand1, std::size_t uint64_count, std::uint64_t operand2, std::uint64_t *result) {
        // Unroll first iteration of loop. We assume uint64_count > 0.
        unsigned char borrow = sub_uint64(*operand1++, operand2, result++);

        // Do the rest
        for (; --uint64_count; operand1++, operand2++, result++) {
            uint64_t temp_result;
            borrow = sub_uint64(*operand1, std::uint64_t(0), borrow, &temp_result);
            *result = temp_result;
        }
        return borrow;
    }

    inline unsigned char increment_uint(
            const std::uint64_t *operand, std::size_t uint64_count, std::uint64_t *result) {
        return add_uint(operand, uint64_count, 1, result);
    }

    inline unsigned char decrement_uint(
            const std::uint64_t *operand, std::size_t uint64_count, std::uint64_t *result) {
        return sub_uint(operand, uint64_count, 1, result);
    }

    inline void negate_uint(const std::uint64_t *operand, std::size_t uint64_count, std::uint64_t *result) {
        // Negation is equivalent to inverting bits and adding 1.
        unsigned char carry = add_uint64(~*operand++, std::uint64_t(1), result++);
        for (; --uint64_count; operand++, result++) {
            uint64_t temp_result;
            carry = add_uint64(~*operand, std::uint64_t(0), carry, &temp_result);
            *result = temp_result;
        }
    }

    inline void left_shift_uint(
            const std::uint64_t *operand, int shift_amount, std::size_t uint64_count, std::uint64_t *result) {
        const std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);
        // How many words to shift
        std::size_t uint64_shift_amount = static_cast<std::size_t>(shift_amount) / bits_per_uint64_sz;

        // Shift words
        for (std::size_t i = 0; i < uint64_count - uint64_shift_amount; i++) {
            result[uint64_count - i - 1] = operand[uint64_count - i - 1 - uint64_shift_amount];
        }
        for (std::size_t i = uint64_count - uint64_shift_amount; i < uint64_count; i++) {
            result[uint64_count - i - 1] = 0;
        }

        // How many bits to shift in addition
        std::size_t bit_shift_amount =
                static_cast<std::size_t>(shift_amount) - (uint64_shift_amount * bits_per_uint64_sz);

        if (bit_shift_amount) {
            std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

            for (std::size_t i = uint64_count - 1; i > 0; i--) {
                result[i] = (result[i] << bit_shift_amount) | (result[i - 1] >> neg_bit_shift_amount);
            }
            result[0] = result[0] << bit_shift_amount;
        }
    }

    inline void right_shift_uint(
            const std::uint64_t *operand, int shift_amount, std::size_t uint64_count, std::uint64_t *result) {
        const std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);
        // How many words to shift
        std::size_t uint64_shift_amount = static_cast<std::size_t>(shift_amount) / bits_per_uint64_sz;

        // Shift words
        for (std::size_t i = 0; i < uint64_count - uint64_shift_amount; i++) {
            result[i] = operand[i + uint64_shift_amount];
        }
        for (std::size_t i = uint64_count - uint64_shift_amount; i < uint64_count; i++) {
            result[i] = 0;
        }

        // How many bits to shift in addition
        std::size_t bit_shift_amount =
                static_cast<std::size_t>(shift_amount) - (uint64_shift_amount * bits_per_uint64_sz);

        if (bit_shift_amount) {
            std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

            for (std::size_t i = 0; i < uint64_count - 1; i++) {
                result[i] = (result[i] >> bit_shift_amount) | (result[i + 1] << neg_bit_shift_amount);
            }
            result[uint64_count - 1] = result[uint64_count - 1] >> bit_shift_amount;
        }
    }

    inline void left_shift_uint128(const std::uint64_t *operand, int shift_amount, std::uint64_t *result) {
        const std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);
        const std::size_t shift_amount_sz = static_cast<std::size_t>(shift_amount);

        // Early return
        if (shift_amount_sz & bits_per_uint64_sz) {
            result[1] = operand[0];
            result[0] = 0;
        } else {
            result[1] = operand[1];
            result[0] = operand[0];
        }

        // How many bits to shift in addition to word shift
        std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

        // Do we have a word shift
        if (bit_shift_amount) {
            std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

            // Warning: if bit_shift_amount == 0 this is incorrect
            result[1] = (result[1] << bit_shift_amount) | (result[0] >> neg_bit_shift_amount);
            result[0] = result[0] << bit_shift_amount;
        }
    }

    inline void right_shift_uint128(const std::uint64_t *operand, int shift_amount, std::uint64_t *result) {
        const std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);

        const std::size_t shift_amount_sz = static_cast<std::size_t>(shift_amount);

        if (shift_amount_sz & bits_per_uint64_sz) {
            result[0] = operand[1];
            result[1] = 0;
        } else {
            result[1] = operand[1];
            result[0] = operand[0];
        }

        // How many bits to shift in addition to word shift
        std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

        if (bit_shift_amount) {
            std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

            // Warning: if bit_shift_amount == 0 this is incorrect
            result[0] = (result[0] >> bit_shift_amount) | (result[1] << neg_bit_shift_amount);
            result[1] = result[1] >> bit_shift_amount;
        }
    }

    inline void left_shift_uint192(const std::uint64_t *operand, int shift_amount, std::uint64_t *result) {
        const auto bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);

        const auto shift_amount_sz = static_cast<std::size_t>(shift_amount);

        if (shift_amount_sz & (bits_per_uint64_sz << 1)) {
            result[2] = operand[0];
            result[1] = 0;
            result[0] = 0;
        } else if (shift_amount_sz & bits_per_uint64_sz) {
            result[2] = operand[1];
            result[1] = operand[0];
            result[0] = 0;
        } else {
            result[2] = operand[2];
            result[1] = operand[1];
            result[0] = operand[0];
        }

        // How many bits to shift in addition to word shift
        std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

        if (bit_shift_amount) {
            std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

            // Warning: if bit_shift_amount == 0 this is incorrect
            result[2] = (result[2] << bit_shift_amount) | (result[1] >> neg_bit_shift_amount);
            result[1] = (result[1] << bit_shift_amount) | (result[0] >> neg_bit_shift_amount);
            result[0] = result[0] << bit_shift_amount;
        }
    }

    inline void left_shift_uint192_inplace(std::uint64_t *operand, size_t shift_amount) {
        // How many bits to shift in addition to word shift
        std::size_t bit_shift_amount = shift_amount & (bits_per_uint64 - 1);

        if (bit_shift_amount) {
            std::size_t neg_bit_shift_amount = bits_per_uint64 - bit_shift_amount;

            // Warning: if bit_shift_amount == 0 this is incorrect
            operand[2] = (operand[2] << bit_shift_amount) | (operand[1] >> neg_bit_shift_amount);
            operand[1] = (operand[1] << bit_shift_amount) | (operand[0] >> neg_bit_shift_amount);
            operand[0] = operand[0] << bit_shift_amount;
        }
    }

    inline void right_shift_uint192(const std::uint64_t *operand, int shift_amount, std::uint64_t *result) {
        const std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);

        const std::size_t shift_amount_sz = static_cast<std::size_t>(shift_amount);

        if (shift_amount_sz & (bits_per_uint64_sz << 1)) {
            result[0] = operand[2];
            result[1] = 0;
            result[2] = 0;
        } else if (shift_amount_sz & bits_per_uint64_sz) {
            result[0] = operand[1];
            result[1] = operand[2];
            result[2] = 0;
        } else {
            result[2] = operand[2];
            result[1] = operand[1];
            result[0] = operand[0];
        }

        // How many bits to shift in addition to word shift
        std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

        if (bit_shift_amount) {
            std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

            // Warning: if bit_shift_amount == 0 this is incorrect
            result[0] = (result[0] >> bit_shift_amount) | (result[1] << neg_bit_shift_amount);
            result[1] = (result[1] >> bit_shift_amount) | (result[2] << neg_bit_shift_amount);
            result[2] = result[2] >> bit_shift_amount;
        }
    }

    inline void half_round_up_uint(const std::uint64_t *operand, std::size_t uint64_count, std::uint64_t *result) {
        if (!uint64_count) {
            return;
        }
        // Set result to (operand + 1) / 2. To prevent overflowing operand, right shift
        // and then increment result if low-bit of operand was set.
        bool low_bit_set = operand[0] & 1;

        for (std::size_t i = 0; i < uint64_count - 1; i++) {
            result[i] = (operand[i] >> 1) | (operand[i + 1] << (bits_per_uint64 - 1));
        }
        result[uint64_count - 1] = operand[uint64_count - 1] >> 1;

        if (low_bit_set) {
            increment_uint(result, uint64_count, result);
        }
    }

    inline void not_uint(const std::uint64_t *operand, std::size_t uint64_count, std::uint64_t *result) {
        for (; uint64_count--; result++, operand++) {
            *result = ~*operand;
        }
    }

    inline void and_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count,
            std::uint64_t *result) {
        for (; uint64_count--; result++, operand1++, operand2++) {
            *result = *operand1 & *operand2;
        }
    }

    inline void or_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count,
            std::uint64_t *result) {
        for (; uint64_count--; result++, operand1++, operand2++) {
            *result = *operand1 | *operand2;
        }
    }

    inline void xor_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count,
            std::uint64_t *result) {
        for (; uint64_count--; result++, operand1++, operand2++) {
            *result = *operand1 ^ *operand2;
        }
    }

    inline void multiply_uint64(uint64_t operand1, uint64_t operand2, uint64_t *result128) {
        auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
        auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
        operand1 >>= 32;
        operand2 >>= 32;

        auto middle1 = operand1 * operand2_coeff_right;
        uint64_t middle;
        auto left = operand1 * operand2 +
                    (static_cast<uint64_t>(add_uint64(middle1, operand2 * operand1_coeff_right, &middle)) << 32);
        auto right = operand1_coeff_right * operand2_coeff_right;
        auto temp_sum = (right >> 32) + (middle & 0x00000000FFFFFFFF);

        result128[1] = static_cast<uint64_t>(left + (middle >> 32) + (temp_sum >> 32));
        result128[0] = static_cast<uint64_t>((temp_sum << 32) | (right & 0x00000000FFFFFFFF));
    }

    template<typename T, typename S, typename = std::enable_if_t<is_uint64_v < T, S>>>

    inline void multiply_uint64_hw64_generic(T operand1, S operand2, uint64_t *hw64) {
        auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
        auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
        operand1 >>= 32;
        operand2 >>= 32;

        auto middle1 = operand1 * operand2_coeff_right;
        uint64_t middle;
        auto left = operand1 * operand2 +
                    (static_cast<T>(add_uint64(middle1, operand2 * operand1_coeff_right, &middle)) << 32);
        auto right = operand1_coeff_right * operand2_coeff_right;
        auto temp_sum = (right >> 32) + (middle & 0x00000000FFFFFFFF);

        *hw64 = static_cast<uint64_t>(left + (middle >> 32) + (temp_sum >> 32));
    }

    template<typename T, typename S, typename = std::enable_if_t<is_uint64_v < T, S>>>

    inline void multiply_uint64_hw64(T operand1, S operand2, uint64_t *hw64) {
        multiply_uint64_hw64_generic(operand1, operand2, hw64);
    }

    void multiply_uint(
            const std::uint64_t *operand1, std::size_t operand1_uint64_count, const std::uint64_t *operand2,
            std::size_t operand2_uint64_count, std::size_t result_uint64_count, std::uint64_t *result);

    inline void multiply_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count,
            std::uint64_t *result) {
        multiply_uint(operand1, uint64_count, operand2, uint64_count, uint64_count * 2, result);
    }

    /** Return the product of operand1 (multiple uint64_t) with operand2 (one uint64_t)
     * @param[in] operand1 Pointer to first operand
     * @param[in] operand1_uint64_count The number of uint64_t in operand1
     * @param[in] operand2 One uint64_t to be multiplied.
     * @param[in] result_uint64_count The size of uint64_t that the result can hold
     * @param[in] result The pointer to hold the product.
     */
    void multiply_uint(
            const std::uint64_t *operand1, std::size_t operand1_uint64_count, std::uint64_t operand2,
            std::size_t result_uint64_count, std::uint64_t *result);

    inline void multiply_truncate_uint(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t uint64_count,
            std::uint64_t *result) {
        multiply_uint(operand1, uint64_count, operand2, uint64_count, uint64_count, result);
    }

    /**
     * Return the product of operands.
     * @param[in] operands The operands to multiply
     * @param[in] count The number of operands
     * @param[out] result The product result.
     */
    inline void multiply_many_uint64(uint64_t *operands, std::size_t count, uint64_t *result) {
        // Nothing to do
        if (!count)
            return;

        // Set result to operands[0]
        set_uint(operands[0], count, result);

        // Compute product
        std::vector<uint64_t> temp_mpi(count);
        for (std::size_t i = 1; i < count; i++) {
            multiply_uint(result, i, operands[i], i + 1, temp_mpi.data());
            set_uint(temp_mpi.data(), i + 1, result);
        }
    }

    template<typename T, typename = std::enable_if_t<is_uint64_v < T>>>

    inline void multiply_many_uint64_except(
            T *operands, std::size_t count, std::size_t except, T *result) {
        // An empty product; return 1
        if (count == 1 && except == 0) {
            set_uint(1, count, result);
            return;
        }

        // Set result to operands[0] unless except is 0
        set_uint(except == 0 ? std::uint64_t(1) : static_cast<std::uint64_t>(operands[0]), count, result);

        // Compute punctured product
        std::vector<uint64_t> temp_mpi(count);
        for (std::size_t i = 1; i < count; i++) {
            if (i != except) {
                multiply_uint(result, i, operands[i], i + 1, temp_mpi.data());
                set_uint(temp_mpi.data(), i + 1, result);
            }
        }
    }

    template<std::size_t Count>
    inline void multiply_accumulate_uint64(
            const std::uint64_t *operand1, const std::uint64_t *operand2, uint64_t *accumulator) {
        uint64_t qword[2];
        multiply_uint64(*operand1, *operand2, qword);
        multiply_accumulate_uint64<Count - 1>(operand1 + 1, operand2 + 1, accumulator);
        add_uint128(qword, accumulator, accumulator);
    }

    template<>
    inline void multiply_accumulate_uint64<0>(
            [[maybe_unused]] const std::uint64_t *operand1, [[maybe_unused]] const std::uint64_t *operand2,
            [[maybe_unused]] uint64_t *accumulator) {
        // Base case; nothing to do
    }

    void divide_uint_inplace(
            std::uint64_t *numerator, const std::uint64_t *denominator, std::size_t uint64_count,
            std::uint64_t *quotient);

    inline void divide_uint(
            const std::uint64_t *numerator, const std::uint64_t *denominator, std::size_t uint64_count,
            std::uint64_t *quotient, std::uint64_t *remainder) {
        set_uint(numerator, uint64_count, remainder);
        divide_uint_inplace(remainder, denominator, uint64_count, quotient);
    }

    void divide_uint128_uint64_inplace_generic(
            std::uint64_t *numerator, std::uint64_t denominator, std::uint64_t *quotient);

    inline void divide_uint128_inplace(std::uint64_t *numerator, std::uint64_t denominator,
                                       std::uint64_t *quotient) {
        divide_uint128_uint64_inplace_generic(numerator, denominator, quotient);
    }

    void divide_uint192_inplace(std::uint64_t *numerator, std::uint64_t denominator, std::uint64_t *quotient);

    [[nodiscard]] std::uint64_t exponentiate_uint_safe(std::uint64_t operand, std::uint64_t exponent);

    [[nodiscard]] std::uint64_t exponentiate_uint(std::uint64_t operand, std::uint64_t exponent);
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "util/common.h"
#include "util/uintarith.h"
#include "util/uintcore.h"
#include <algorithm>
#include <array>

using namespace std;

namespace phantom::arith {
    /** Return the product of operand1 (multiple uint64_t) with operand2 (multiple uint64_t)
     * @param[in] operand1 Pointer to first operand
     * @param[in] operand1_uint64_count The number of uint64_t in operand1
     * @param[in] operand2 Pointer to second operand
     * @param[in] operand2_uint64_count The number of uint64_t in operand2
     * @param[in] result_uint64_count The size of uint64_t that the result can hold
     * @param[in] result The pointer to hold the product.
     */
    void multiply_uint(
            const uint64_t *operand1, size_t operand1_uint64_count, const uint64_t *operand2,
            size_t operand2_uint64_count, size_t result_uint64_count, uint64_t *result) {
        if (!operand1_uint64_count || !operand2_uint64_count) {
            // If either operand is 0, then result is 0.
            set_zero_uint(result_uint64_count, result);
            return;
        }
        if (result_uint64_count == 1) {
            *result = *operand1 * *operand2;
            return;
        }

        // obtain first non-zero uint64.
        operand1_uint64_count = get_significant_uint64_count_uint(operand1, operand1_uint64_count);
        operand2_uint64_count = get_significant_uint64_count_uint(operand2, operand2_uint64_count);

        if (operand1_uint64_count == 1) {
            multiply_uint(operand2, operand2_uint64_count, *operand1, result_uint64_count, result);
            return;
        }
        if (operand2_uint64_count == 1) {
            multiply_uint(operand1, operand1_uint64_count, *operand2, result_uint64_count, result);
            return;
        }

        // Clear out result.
        set_zero_uint(result_uint64_count, result);

        // Multiply operand1 and operand2.
        size_t operand1_index_max = min(operand1_uint64_count, result_uint64_count);
        for (size_t operand1_index = 0; operand1_index < operand1_index_max; operand1_index++) {
            const uint64_t *inner_operand2 = operand2;
            uint64_t *inner_result = result++;
            uint64_t carry = 0;
            size_t operand2_index = 0;
            size_t operand2_index_max = min(operand2_uint64_count, result_uint64_count - operand1_index);
            for (; operand2_index < operand2_index_max; operand2_index++) {
                // Perform 64-bit multiplication of operand1 and operand2
                uint64_t temp_result[2];
                multiply_uint64(*operand1, *inner_operand2++, temp_result);
                carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, temp_result);
                uint64_t temp;
                carry += add_uint64(*inner_result, temp_result[0], 0, &temp);
                *inner_result++ = temp;
            }

            // Write carry if there is room in result
            if (operand1_index + operand2_index_max < result_uint64_count) {
                *inner_result = carry;
            }

            operand1++;
        }
    }

    /** Return the product of operand1 (multiple uint64_t) with operand2 (one uint64_t)
     * @param[in] operand1 Pointer to first operand
     * @param[in] operand1_uint64_count The number of uint64_t in operand1
     * @param[in] operand2 One uint64_t to be multiplied.
     * @param[in] result_uint64_count The size of uint64_t that the result can hold
     * @param[in] result The pointer to hold the product.
     */
    void multiply_uint(
            const uint64_t *operand1, size_t operand1_uint64_count, uint64_t operand2, size_t result_uint64_count,
            uint64_t *result) {
        if (!operand1_uint64_count || !operand2) {
            // If either operand is 0, then result is 0.
            set_zero_uint(result_uint64_count, result);
            return;
        }
        if (result_uint64_count == 1) {
            *result = *operand1 * operand2;
            return;
        }

        // Clear out result.
        set_zero_uint(result_uint64_count, result);

        // Multiply operand1 and operand2.
        uint64_t carry = 0;
        size_t operand1_index_max = min(operand1_uint64_count, result_uint64_count);
        for (size_t operand1_index = 0; operand1_index < operand1_index_max; operand1_index++) {
            uint64_t temp_result[2];
            multiply_uint64(*operand1++, operand2, temp_result);
            uint64_t temp;
            carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, &temp);
            *result++ = temp;
        }

        // Write carry if there is room in result
        if (operand1_index_max < result_uint64_count) {
            *result = carry;
        }
    }

    /**
     * @brief Big integer division. Remainder replaces numerator. All operands are the same size.
     * @param numerator input as numerator, output as remainder
     * @param denominator input as denominator
     * @param uint64_count bit length of all operands
     * @param quotient output as quotient
     */
    void divide_uint_inplace(
            uint64_t *numerator, const uint64_t *denominator, size_t uint64_count, uint64_t *quotient) {
        if (!uint64_count) {
            return;
        }

        // Clear quotient. Set it to zero.
        set_zero_uint(uint64_count, quotient);

        // Determine significant bits in numerator and denominator.
        int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
        int denominator_bits = get_significant_bit_count_uint(denominator, uint64_count);

        // If numerator has fewer bits than denominator, then done.
        if (numerator_bits < denominator_bits) {
            return;
        }

        // Only perform computation up to last non-zero uint64s.
        uint64_count = divide_round_up(numerator_bits, bits_per_uint64);

        // Handle fast case.
        if (uint64_count == 1) {
            *quotient = *numerator / *denominator;
            *numerator -= *quotient * *denominator;
            return;
        }
        std::vector<uint64_t> alloc_anchor(uint64_count << 1);

        // Create temporary space to store mutable copy of denominator.
        uint64_t *shifted_denominator = alloc_anchor.data();

        // Create temporary space to store difference calculation.
        uint64_t *difference = shifted_denominator + uint64_count;

        // Shift denominator to bring MSB in alignment with MSB of numerator.
        int denominator_shift = numerator_bits - denominator_bits;
        left_shift_uint(denominator, denominator_shift, uint64_count, shifted_denominator);
        denominator_bits += denominator_shift;

        // Perform bit-wise division algorithm.
        int remaining_shifts = denominator_shift;
        while (numerator_bits == denominator_bits) {
            // NOTE: MSBs of numerator and denominator are aligned.

            // Even though MSB of numerator and denominator are aligned,
            // still possible numerator < shifted_denominator.
            if (sub_uint(numerator, shifted_denominator, uint64_count, difference)) {
                // numerator < shifted_denominator and MSBs are aligned,
                // so current quotient bit is zero and next one is definitely one.
                if (remaining_shifts == 0) {
                    // No shifts remain and numerator < denominator so done.
                    break;
                }

                // Effectively shift numerator left by 1 by instead adding
                // numerator to difference (to prevent overflow in numerator).
                add_uint(difference, numerator, uint64_count, difference);

                // Adjust quotient and remaining shifts as a result of
                // shifting numerator.
                left_shift_uint(quotient, 1, uint64_count, quotient);
                remaining_shifts--;
            }
            // Difference is the new numerator with denominator subtracted.

            // Update quotient to reflect subtraction.
            quotient[0] |= 1;

            // Determine amount to shift numerator to bring MSB in alignment
            // with denominator.
            numerator_bits = get_significant_bit_count_uint(difference, uint64_count);
            int numerator_shift = denominator_bits - numerator_bits;
            if (numerator_shift > remaining_shifts) {
                // Clip the maximum shift to determine only the integer
                // (as opposed to fractional) bits.
                numerator_shift = remaining_shifts;
            }

            // Shift and update numerator.
            if (numerator_bits > 0) {
                left_shift_uint(difference, numerator_shift, uint64_count, numerator);
                numerator_bits += numerator_shift;
            } else {
                // Difference is zero so no need to shift, just set to zero.
                set_zero_uint(uint64_count, numerator);
            }

            // Adjust quotient and remaining shifts as a result of shifting numerator.
            left_shift_uint(quotient, numerator_shift, uint64_count, quotient);
            remaining_shifts -= numerator_shift;
        }

        // Correct numerator (which is also the remainder) for shifting of
        // denominator, unless it is just zero.
        if (numerator_bits > 0) {
            right_shift_uint(numerator, denominator_shift, uint64_count, numerator);
        }
    }

    void divide_uint128_uint64_inplace_generic(uint64_t *numerator, uint64_t denominator, uint64_t *quotient) {
        // We expect 128-bit input
        constexpr size_t uint64_count = 2;

        // Clear quotient. Set it to zero.
        quotient[0] = 0;
        quotient[1] = 0;

        // Determine significant bits in numerator and denominator.
        int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
        int denominator_bits = get_significant_bit_count(denominator);

        // If numerator has fewer bits than denominator, then done.
        if (numerator_bits < denominator_bits) {
            return;
        }

        // Create temporary space to store mutable copy of denominator.
        uint64_t shifted_denominator[uint64_count]{denominator, 0};

        // Create temporary space to store difference calculation.
        uint64_t difference[uint64_count]{0, 0};

        // Shift denominator to bring MSB in alignment with MSB of numerator.
        int denominator_shift = numerator_bits - denominator_bits;

        left_shift_uint128(shifted_denominator, denominator_shift, shifted_denominator);
        denominator_bits += denominator_shift;

        // Perform bit-wise division algorithm.
        int remaining_shifts = denominator_shift;
        while (numerator_bits == denominator_bits) {
            // NOTE: MSBs of numerator and denominator are aligned.

            // Even though MSB of numerator and denominator are aligned,
            // still possible numerator < shifted_denominator.
            if (sub_uint(numerator, shifted_denominator, uint64_count, difference)) {
                // numerator < shifted_denominator and MSBs are aligned,
                // so current quotient bit is zero and next one is definitely one.
                if (remaining_shifts == 0) {
                    // No shifts remain and numerator < denominator so done.
                    break;
                }

                // Effectively shift numerator left by 1 by instead adding
                // numerator to difference (to prevent overflow in numerator).
                add_uint(difference, numerator, uint64_count, difference);

                // Adjust quotient and remaining shifts as a result of shifting numerator.
                quotient[1] = (quotient[1] << 1) | (quotient[0] >> (bits_per_uint64 - 1));
                quotient[0] <<= 1;
                remaining_shifts--;
            }
            // Difference is the new numerator with denominator subtracted.

            // Determine amount to shift numerator to bring MSB in alignment
            // with denominator.
            numerator_bits = get_significant_bit_count_uint(difference, uint64_count);

            // Clip the maximum shift to determine only the integer
            // (as opposed to fractional) bits.
            int numerator_shift = min(denominator_bits - numerator_bits, remaining_shifts);

            // Shift and update numerator.
            // This may be faster; first set to zero and then update if needed

            // Difference is zero so no need to shift, just set to zero.
            numerator[0] = 0;
            numerator[1] = 0;

            if (numerator_bits > 0) {
                left_shift_uint128(difference, numerator_shift, numerator);
                numerator_bits += numerator_shift;
            }

            // Update quotient to reflect subtraction.
            quotient[0] |= 1;

            // Adjust quotient and remaining shifts as a result of shifting numerator.
            left_shift_uint128(quotient, numerator_shift, quotient);
            remaining_shifts -= numerator_shift;
        }

        // Correct numerator (which is also the remainder) for shifting of
        // denominator, unless it is just zero.
        if (numerator_bits > 0) {
            right_shift_uint128(numerator, denominator_shift, numerator);
        }
    }

    void divide_uint192_inplace(uint64_t *numerator, uint64_t denominator, uint64_t *quotient) {
        // We expect 192-bit input
        size_t uint64_count = 3;

        // Clear quotient. Set it to zero.
        quotient[0] = 0;
        quotient[1] = 0;
        quotient[2] = 0;

        // Determine significant bits in numerator and denominator.
        int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
        int denominator_bits = get_significant_bit_count(denominator);

        // If numerator has fewer bits than denominator, then done.
        if (numerator_bits < denominator_bits) {
            return;
        }

        // Only perform computation up to last non-zero uint64s.
        uint64_count = divide_round_up(numerator_bits, bits_per_uint64);

        // Handle fast case.
        if (uint64_count == 1) {
            *quotient = *numerator / denominator;
            *numerator -= *quotient * denominator;
            return;
        }

        // Create temporary space to store mutable copy of denominator.
        vector<uint64_t> shifted_denominator(uint64_count, 0);
        shifted_denominator[0] = denominator;

        // Create temporary space to store difference calculation.
        vector<uint64_t> difference(uint64_count);

        // Shift denominator to bring MSB in alignment with MSB of numerator.
        int denominator_shift = numerator_bits - denominator_bits;

        left_shift_uint192(shifted_denominator.data(), denominator_shift, shifted_denominator.data());
        denominator_bits += denominator_shift;

        // Perform bit-wise division algorithm.
        int remaining_shifts = denominator_shift;
        while (numerator_bits == denominator_bits) {
            // NOTE: MSBs of numerator and denominator are aligned.

            // Even though MSB of numerator and denominator are aligned,
            // still possible numerator < shifted_denominator.
            if (sub_uint(numerator, shifted_denominator.data(), uint64_count, difference.data())) {
                // numerator < shifted_denominator and MSBs are aligned,
                // so current quotient bit is zero and next one is definitely one.
                if (remaining_shifts == 0) {
                    // No shifts remain and numerator < denominator so done.
                    break;
                }

                // Effectively shift numerator left by 1 by instead adding
                // numerator to difference (to prevent overflow in numerator).
                add_uint(difference.data(), numerator, uint64_count, difference.data());

                // Adjust quotient and remaining shifts as a result of shifting numerator.
                left_shift_uint192(quotient, 1, quotient);
                remaining_shifts--;
            }
            // Difference is the new numerator with denominator subtracted.

            // Update quotient to reflect subtraction.
            quotient[0] |= 1;

            // Determine amount to shift numerator to bring MSB in alignment with denominator.
            numerator_bits = get_significant_bit_count_uint(difference.data(), uint64_count);
            int numerator_shift = denominator_bits - numerator_bits;
            if (numerator_shift > remaining_shifts) {
                // Clip the maximum shift to determine only the integer
                // (as opposed to fractional) bits.
                numerator_shift = remaining_shifts;
            }

            // Shift and update numerator.
            if (numerator_bits > 0) {
                left_shift_uint192(difference.data(), numerator_shift, numerator);
                numerator_bits += numerator_shift;
            } else {
                // Difference is zero so no need to shift, just set to zero.
                set_zero_uint(uint64_count, numerator);
            }

            // Adjust quotient and remaining shifts as a result of shifting numerator.
            left_shift_uint192(quotient, numerator_shift, quotient);
            remaining_shifts -= numerator_shift;
        }

        // Correct numerator (which is also the remainder) for shifting of
        // denominator, unless it is just zero.
        if (numerator_bits > 0) {
            right_shift_uint192(numerator, denominator_shift, numerator);
        }
    }

    uint64_t exponentiate_uint_safe(uint64_t operand, uint64_t exponent) {
        // Fast cases
        if (exponent == 0) {
            return 1;
        }
        if (exponent == 1) {
            return operand;
        }

        // Perform binary exponentiation.
        uint64_t power = operand;
        uint64_t product = 0;
        uint64_t intermediate = 1;

        // Initially: power = operand and intermediate = 1, product irrelevant.
        while (true) {
            if (exponent & 1) {
                product = mul_safe(power, intermediate);
                swap(product, intermediate);
            }
            exponent >>= 1;
            if (exponent == 0) {
                break;
            }
            product = mul_safe(power, power);
            swap(product, power);
        }

        return intermediate;
    }

    uint64_t exponentiate_uint(uint64_t operand, uint64_t exponent) {
        // Fast cases
        if (exponent == 0) {
            return 1;
        }
        if (exponent == 1) {
            return operand;
        }

        // Perform binary exponentiation.
        uint64_t power = operand;
        uint64_t product = 0;
        uint64_t intermediate = 1;

        // Initially: power = operand and intermediate = 1, product irrelevant.
        while (true) {
            if (exponent & 1) {
                product = power * intermediate;
                swap(product, intermediate);
            }
            exponent >>= 1;
            if (exponent == 0) {
                break;
            }
            product = power * power;
            swap(product, power);
        }

        return intermediate;
    }
}

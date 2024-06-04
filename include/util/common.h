// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "defines.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda/std/type_traits>

namespace phantom::arith {
    template<typename T, typename...>
    struct IsUInt64
            : std::conditional<
                    std::is_integral<T>::value && std::is_unsigned<T>::value && (sizeof(T) == sizeof(std::uint64_t)),
                    std::true_type, std::false_type>::type {
    };

    template<typename T, typename... Rest>
    constexpr bool is_uint64_v = IsUInt64<T, Rest...>::value;

    template<typename T, typename...>
    struct IsUInt32
            : std::conditional<
                    std::is_integral<T>::value && cuda::std::is_unsigned<T>::value && (
                            sizeof(T) == sizeof(std::uint32_t)),
                    std::true_type, std::false_type>::type {
    };

    template<typename T, typename U, typename... Rest>
    struct IsUInt32<T, U, Rest...>
            : std::conditional<IsUInt32<T>::value && IsUInt32<U, Rest...>::value, std::true_type,
                    std::false_type>::type {
    };

    template<typename T, typename... Rest>
    constexpr bool is_uint32_v = IsUInt32<T, Rest...>::value;

    template<
            typename T, typename S, typename = std::enable_if_t<std::is_integral<T>::value>,
            typename = std::enable_if_t<std::is_integral<S>::value>>
    [[nodiscard]] inline constexpr bool unsigned_lt(T in1, S in2) noexcept {
        return static_cast<std::uint64_t>(in1) < static_cast<std::uint64_t>(in2);
    }

    template<
            typename T, typename S, typename = std::enable_if_t<std::is_integral<T>::value>,
            typename = std::enable_if_t<std::is_integral<S>::value>>
    [[nodiscard]] inline constexpr bool unsigned_leq(T in1, S in2) noexcept {
        return static_cast<std::uint64_t>(in1) <= static_cast<std::uint64_t>(in2);
    }

    template<
            typename T, typename S, typename = std::enable_if_t<std::is_integral<T>::value>,
            typename = std::enable_if_t<std::is_integral<S>::value>>
    [[nodiscard]] inline constexpr bool unsigned_gt(T in1, S in2) noexcept {
        return static_cast<std::uint64_t>(in1) > static_cast<std::uint64_t>(in2);
    }

    template<
            typename T, typename S, typename = std::enable_if_t<std::is_integral<T>::value>,
            typename = std::enable_if_t<std::is_integral<S>::value>>
    [[nodiscard]] inline constexpr bool unsigned_geq(T in1, S in2) noexcept {
        return static_cast<std::uint64_t>(in1) >= static_cast<std::uint64_t>(in2);
    }

    template<
            typename T, typename S, typename = std::enable_if_t<std::is_integral<T>::value>,
            typename = std::enable_if_t<std::is_integral<S>::value>>
    [[nodiscard]] inline constexpr bool unsigned_eq(T in1, S in2) noexcept {
        return static_cast<std::uint64_t>(in1) == static_cast<std::uint64_t>(in2);
    }

    template<
            typename T, typename S, typename = std::enable_if_t<std::is_integral<T>::value>,
            typename = std::enable_if_t<std::is_integral<S>::value>>
    [[nodiscard]] inline constexpr bool unsigned_neq(T in1, S in2) noexcept {
        return static_cast<std::uint64_t>(in1) != static_cast<std::uint64_t>(in2);
    }

    template<typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
    [[nodiscard]] inline constexpr T mul_safe(T in1) noexcept {
        return in1;
    }

    template<typename T>
    [[nodiscard]] inline constexpr T mul_safe(T in1, T in2) {
        if constexpr (cuda::std::is_unsigned<T>::value) {
            if (in1 && (in2 > (std::numeric_limits<T>::max)() / in1)) {
                throw std::logic_error("unsigned overflow");
            }
        } else {
            // Positive inputs
            if ((in1 > 0) && (in2 > 0) && (in2 > (std::numeric_limits<T>::max)() / in1)) {
                throw std::logic_error("signed overflow");
            }
                // Negative inputs
            else if ((in1 < 0) && (in2 < 0) && ((-in2) > (std::numeric_limits<T>::max)() / (-in1))) {
                throw std::logic_error("signed overflow");
            }
                // Negative in1; positive in2
            else if ((in1 < 0) && (in2 > 0) && (in2 > (std::numeric_limits<T>::max)() / (-in1))) {
                throw std::logic_error("signed underflow");
            }
                // Positive in1; negative in2
            else if ((in1 > 0) && (in2 < 0) && (in2 < (std::numeric_limits<T>::min)() / in1)) {
                throw std::logic_error("signed underflow");
            }
        }
        return static_cast<T>(in1 * in2);
    }

    template<>
    [[nodiscard]] inline constexpr size_t mul_safe(size_t in1, size_t in2) {
        if (in1 && (in2 > (std::numeric_limits<size_t>::max)() / in1)) {
            throw std::logic_error("unsigned overflow");
        }
        return static_cast<size_t>(in1 * in2);
    }

    template<>
    [[nodiscard]] inline constexpr uint32_t mul_safe(uint32_t in1, uint32_t in2) {
        if (in1 && (in2 > (std::numeric_limits<uint32_t>::max)() / in1)) {
            throw std::logic_error("unsigned overflow");
        }
        return static_cast<uint32_t>(in1 * in2);
    }

    template<typename T, typename... Args, typename = std::enable_if_t<std::is_integral<T>::value>>
    [[nodiscard]] inline constexpr T mul_safe(T in1, T in2, Args &&... args) {
        return mul_safe(mul_safe(in1, in2), mul_safe(std::forward<Args>(args)...));
    }

    template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
    [[nodiscard]] inline constexpr T add_safe(T in1) noexcept {
        return in1;
    }

    template<typename T>
    [[nodiscard]] inline constexpr T add_safe(T in1, T in2) {
        if (in1 > 0 && (in2 > (std::numeric_limits<T>::max)() - in1)) {
            throw std::logic_error("signed overflow");
        } else if (in1 < 0 && (in2 < (std::numeric_limits<T>::min)() - in1)) {
            throw std::logic_error("signed underflow");
        }
        return static_cast<T>(in1 + in2);
    }

    template<>
    [[nodiscard]] inline constexpr uint32_t add_safe(uint32_t in1, uint32_t in2) {
        if (in2 > (std::numeric_limits<uint32_t>::max)() - in1) {
            throw std::logic_error("unsigned overflow");
        }
        return static_cast<uint32_t>(in1 + in2);
    }

    template<>
    [[nodiscard]] inline constexpr uint64_t add_safe(uint64_t in1, uint64_t in2) {
        if (in2 > (std::numeric_limits<uint64_t>::max)() - in1) {
            throw std::logic_error("unsigned overflow");
        }
        return static_cast<uint64_t>(in1 + in2);
    }

    template<typename T, typename... Args, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
    [[nodiscard]] inline constexpr T add_safe(T in1, T in2, Args &&... args) {
        return add_safe(add_safe(in1, in2), add_safe(std::forward<Args>(args)...));
    }

    template<typename T>
    [[nodiscard]] inline T sub_safe(T in1, T in2) {
        if (in1 < 0 && (in2 > (std::numeric_limits<T>::max)() + in1)) {
            throw std::logic_error("signed underflow");
        } else if (in1 > 0 && (in2 < (std::numeric_limits<T>::min)() + in1)) {
            throw std::logic_error("signed overflow");
        }
        return static_cast<T>(in1 - in2);
    }

    template<>
    [[nodiscard]] inline uint32_t sub_safe(uint32_t in1, uint32_t in2) {
        if (in1 < in2) {
            throw std::logic_error("unsigned underflow");
        }
        return static_cast<uint32_t>(in1 - in2);
    }

    template<>
    [[nodiscard]] inline uint64_t sub_safe(uint64_t in1, uint64_t in2) {
        if (in1 < in2) {
            throw std::logic_error("unsigned underflow");
        }
        return static_cast<uint64_t>(in1 - in2);
    }

    template<
            typename T, typename S, typename = std::enable_if_t<std::is_arithmetic<T>::value>,
            typename = std::enable_if_t<std::is_arithmetic<S>::value>>
    [[nodiscard]] inline constexpr bool fits_in(S value [[maybe_unused]]) noexcept {
        bool result = false;

        if constexpr (std::is_same<T, S>::value) {
            // Same type
            result = true;
        } else if constexpr (sizeof(S) <= sizeof(T)) {
            // Converting to bigger type
            if constexpr (std::is_integral<T>::value && std::is_integral<S>::value) {
                // Converting to at least equally big integer type
                if constexpr (
                        (cuda::std::is_unsigned<T>::value && cuda::std::is_unsigned<S>::value) ||
                        (!cuda::std::is_unsigned<T>::value && !cuda::std::is_unsigned<S>::value)) {
                    // Both either signed or unsigned
                    result = true;
                } else if constexpr (cuda::std::is_unsigned<T>::value && cuda::std::is_signed<S>::value) {
                    // Converting from signed to at least equally big unsigned type
                    result = true; // result = value >= 0;
                }
            } else if constexpr (cuda::std::is_floating_point<T>::value && cuda::std::is_floating_point<S>::value) {
                // Both floating-point
                result = true;
            }

            // Still need to consider integer-float conversions and all
            // unsigned to signed conversions
        }

        if constexpr (std::is_integral<T>::value && std::is_integral<S>::value) {
            // Both integer types
            // fixme: warning #186-D: pointless comparison of unsigned integer with zero
            if (static_cast<std::int64_t>(value) >= 0) {
                // Non-negative number; compare as std::uint64_t
                // Cannot use unsigned_leq with C++14 for lack of `if constexpr'
                result = static_cast<std::uint64_t>(value) <=
                         static_cast<std::uint64_t>((std::numeric_limits<T>::max)());
            } else {
                // Negative number; compare as std::int64_t
                result =
                        static_cast<std::int64_t>(value) >= static_cast<std::int64_t>((std::numeric_limits<
                                T>::min)());
            }
        } else if constexpr (std::is_floating_point<T>::value) {
            // Converting to floating-point
            result = (static_cast<double>(value) <= static_cast<double>((std::numeric_limits<T>::max)())) &&
                     (static_cast<double>(value) >= -static_cast<double>((std::numeric_limits<T>::max)()));
        } else {
            // Converting from floating-point
            result = (static_cast<double>(value) <= static_cast<double>((std::numeric_limits<T>::max)())) &&
                     (static_cast<double>(value) >= static_cast<double>((std::numeric_limits<T>::min)()));
        }

        return result;
    }

    constexpr int bytes_per_uint64 = sizeof(std::uint64_t);

    constexpr int bits_per_nibble = 4;

    constexpr int bits_per_byte = 8;

    constexpr int bits_per_uint64 = bytes_per_uint64 * bits_per_byte;

    constexpr int nibbles_per_byte = 2;

    constexpr int nibbles_per_uint64 = bytes_per_uint64 * nibbles_per_byte;

    [[nodiscard]] inline constexpr int hamming_weight(unsigned char value) {
        int t = static_cast<int>(value);
        t -= (t >> 1) & 0x55;
        t = (t & 0x33) + ((t >> 2) & 0x33);
        return (t + (t >> 4)) & 0x0F;
    }

    [[nodiscard]] inline constexpr uint32_t reverse_bits(uint32_t operand) noexcept {
        operand = (((operand & uint32_t(0xaaaaaaaa)) >> 1) | ((operand & uint32_t(0x55555555)) << 1));
        operand = (((operand & uint32_t(0xcccccccc)) >> 2) | ((operand & uint32_t(0x33333333)) << 2));
        operand = (((operand & uint32_t(0xf0f0f0f0)) >> 4) | ((operand & uint32_t(0x0f0f0f0f)) << 4));
        operand = (((operand & uint32_t(0xff00ff00)) >> 8) | ((operand & uint32_t(0x00ff00ff)) << 8));
        return static_cast<uint32_t>(operand >> 16) | static_cast<uint32_t>(operand << 16);
    }

    [[nodiscard]] constexpr uint64_t reverse_bits(uint64_t operand) noexcept {
        return static_cast<uint64_t>(reverse_bits(static_cast<std::uint32_t>(operand >> 32))) |
               (static_cast<uint64_t>(reverse_bits(static_cast<std::uint32_t>(operand & uint64_t(0xFFFFFFFF)))) <<
                                                                                                                32);
    }

    template<typename T, typename = std::enable_if_t<is_uint32_v<T> || is_uint64_v<T>>>
    [[nodiscard]] inline T reverse_bits(T operand, int bit_count) {
        // Just return zero if bit_count is zero
        return (bit_count == 0)
               ? T(0)
               : reverse_bits(operand) >> (sizeof(T) * static_cast<std::size_t>(bits_per_byte) -
                                           static_cast<std::size_t>(bit_count));
    }

    inline void get_msb_index_generic(unsigned long *result, std::uint64_t value) {
        static const unsigned long deBruijnTable64[64] = {
                63, 0, 58, 1, 59, 47, 53, 2, 60, 39, 48, 27, 54,
                33, 42, 3, 61, 51, 37, 40, 49, 18, 28, 20, 55, 30,
                34, 11, 43, 14, 22, 4, 62, 57, 46, 52, 38, 26, 32,
                41, 50, 36, 17, 19, 29, 10, 13, 21, 56, 45, 25, 31,
                35, 16, 9, 12, 44, 24, 15, 8, 23, 7, 6, 5
        };

        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        value |= value >> 32;

        *result = deBruijnTable64[((value - (value >> 1)) * std::uint64_t(0x07EDD5E59A4E28C2)) >> 58];
    }

    [[nodiscard]] inline int get_significant_bit_count(std::uint64_t value) {
        if (value == 0) {
            return 0;
        }

        unsigned long result = 0;
        get_msb_index_generic(&result, value);
        return static_cast<int>(result + 1);
    }

    template<typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
    [[nodiscard]] inline T divide_round_up(T value, T divisor) {
        return (add_safe(value, divisor - 1)) / divisor;
    }

    template<typename T>
    constexpr double epsilon = std::numeric_limits<T>::epsilon();

    template<typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>>
    [[nodiscard]] inline bool are_close(T value1, T value2) noexcept {
        double scale_factor = std::max<T>({std::fabs(value1), std::fabs(value2), T{1.0}});
        return std::fabs(value1 - value2) < epsilon<T> * scale_factor;
    }

    template<typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
    [[nodiscard]] inline constexpr bool is_zero(T value) noexcept {
        return value == T{0};
    }
}

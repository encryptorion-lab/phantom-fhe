// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "modulus.h"
#include "common.h"
#include "defines.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace phantom::arith {
    [[nodiscard]] inline std::vector<int> naf(int value) {
        std::vector<int> res;

        // Record the sign of the original value and compute abs
        bool sign = value < 0;
        value = std::abs(value);

        // Transform to non-adjacent form (NAF)
        for (int i = 0; value; i++) {
            int zi = (value & int(0x1)) ? 2 - (value & int(0x3)) : 0;
            value = (value - zi) >> 1;
            if (zi) {
                res.push_back((sign ? -zi : zi) * (1 << i));
            }
        }

        return res;
    }

    [[nodiscard]] inline std::uint64_t gcd(std::uint64_t x, std::uint64_t y) {
        if (x < y) {
            return gcd(y, x);
        } else if (y == 0) {
            return x;
        } else {
            std::uint64_t f = x % y;
            if (f == 0) {
                return y;
            } else {
                return gcd(y, f);
            }
        }
    }

    [[nodiscard]] inline auto xgcd(std::uint64_t x, std::uint64_t y)
    -> std::tuple<std::uint64_t, std::int64_t, std::int64_t> {
        /* Extended GCD:
        Returns (gcd, x, y) where gcd is the greatest common divisor of a and b.
        The numbers x, y are such that gcd = ax + by.
        */
        std::int64_t prev_a = 1;
        std::int64_t a = 0;
        std::int64_t prev_b = 0;
        std::int64_t b = 1;

        while (y != 0) {
            std::int64_t q = std::int64_t(x / y);
            std::int64_t temp = std::int64_t(x % y);
            x = y;
            y = std::uint64_t(temp);

            temp = a;
            a = sub_safe(prev_a, mul_safe(q, a));
            prev_a = temp;

            temp = b;
            b = sub_safe(prev_b, mul_safe(q, b));
            prev_b = temp;
        }
        return std::make_tuple(x, prev_a, prev_b);
    }

    [[nodiscard]] inline bool are_coprime(std::uint64_t x, std::uint64_t y) noexcept {
        return !(gcd(x, y) > 1);
    }

    [[nodiscard]] std::vector<std::uint64_t> multiplicative_orders(
            std::vector<std::uint64_t> conjugate_classes, std::uint64_t modulus);

    [[nodiscard]] std::vector<std::uint64_t> conjugate_classes(
            std::uint64_t modulus, std::uint64_t subgroup_generator);

    void babystep_giantstep(
            std::uint64_t modulus, std::vector<std::uint64_t> &baby_steps, std::vector<std::uint64_t> &giant_steps);

    [[nodiscard]] auto decompose_babystep_giantstep(
            std::uint64_t modulus, std::uint64_t input, const std::vector<std::uint64_t> &baby_steps,
            const std::vector<std::uint64_t> &giant_steps) -> std::pair<std::size_t, std::size_t>;

    [[nodiscard]] bool is_prime(const Modulus &modulus, std::size_t num_rounds = 40);

    [[nodiscard]] std::vector<Modulus> get_primes(std::size_t ntt_size, int bit_size, std::size_t count);

    [[nodiscard]] std::vector<Modulus> get_primes_below(size_t ntt_size, uint64_t upper_bound, size_t count);

    [[nodiscard]] inline Modulus get_prime(std::size_t ntt_size, int bit_size) {
        return get_primes(ntt_size, bit_size, 1)[0];
    }

    bool try_invert_uint_mod(std::uint64_t value, std::uint64_t modulus, std::uint64_t &result);

    bool is_primitive_root(std::uint64_t root, std::uint64_t degree, const Modulus &prime_modulus);

    // Try to find a primitive degree-th root of unity modulo small prime
    // modulus, where degree must be a power of two.
    bool try_primitive_root(std::uint64_t degree, const Modulus &prime_modulus, std::uint64_t &destination);

    // Try to find the smallest (as integer) primitive degree-th root of
    // unity modulo small prime modulus, where degree must be a power of two.
    bool try_minimal_primitive_root(std::uint64_t degree, const Modulus &prime_modulus, std::uint64_t &destination);
}

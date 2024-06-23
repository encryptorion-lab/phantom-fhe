#pragma once

#include "defines.h"
#include "hestdparms.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <limits>

namespace phantom::arith {
    /** Store pre-computation for Barrett reduction.
    Represent a non-negative integer modulus of up to 61 bits.
    */
    class Modulus {
    public:
        /**
        Creates a Modulus instance. The value of the Modulus is set to
        the given value, or to zero by default.

        @param[in] value The integer modulus
        @throws std::invalid_argument if value is 1 or more than 61 bits
        */
        Modulus(std::uint64_t value = 0) {
            set_value(value);
        }

        //Creates a new Modulus by copying a given one.
        Modulus(const Modulus &copy) = default;

        //Creates a new Modulus by moving from the provided one
        Modulus(Modulus &&source) = default;

        //Creates a new Modulus by copying a given one.
        Modulus &operator=(const Modulus &assign) = default;

        //Creates a new Modulus by moving from the provided one
        Modulus &operator=(Modulus &&assign) = default;

        /**
        Sets the value of the Modulus.

        @param[in] value The new integer modulus
        @throws std::invalid_argument if value is 1 or more than 61 bits
        */
        inline Modulus &operator=(std::uint64_t value) {
            set_value(value);
            return *this;
        }

        // Returns the significant bit count of the value of the current Modulus.
        [[nodiscard]] inline int bit_count() const noexcept {
            return bit_count_;
        }

        // Returns the size (in 64-bit words) of the value of the current Modulus.
        [[nodiscard]] inline std::size_t uint64_count() const noexcept {
            return uint64_count_;
        }

        // Returns a const pointer to the value of the current Modulus.
        [[nodiscard]] inline const uint64_t *data() const noexcept {
            return &value_;
        }

        // Returns the value of the current Modulus.
        [[nodiscard]] inline std::uint64_t value() const noexcept {
            return value_;
        }

        /**
        Returns the Barrett ratio computed for the value of the current Modulus.
        The first two components of the Barrett ratio are the floor of 2^128/value,
        and the third component is the remainder.
        */
        [[nodiscard]] inline auto &const_ratio() const noexcept {
            return const_ratio_;
        }

        // Returns whether the value of the current Modulus is zero.
        [[nodiscard]] inline bool is_zero() const noexcept {
            return value_ == 0;
        }

        // Returns whether the value of the current Modulus is a prime number.
        [[nodiscard]] inline bool is_prime() const noexcept {
            return is_prime_;
        }

        /**
        Compares two Modulus instances.

        @param[in] compare The Modulus to compare against
        */
        [[nodiscard]] inline bool operator==(const Modulus &compare) const noexcept {
            return value_ == compare.value_;
        }

        /**
        Compares a Modulus value to an unsigned integer.

        @param[in] compare The unsigned integer to compare against
        */
        [[nodiscard]] inline bool operator==(std::uint64_t compare) const noexcept {
            return value_ == compare;
        }

        /**
        Compares two Modulus instances.

        @param[in] compare The Modulus to compare against
        */
        [[nodiscard]] inline bool operator!=(const Modulus &compare) const noexcept {
            return !operator==(compare);
        }

        /**
        Compares a Modulus value to an unsigned integer.

        @param[in] compare The unsigned integer to compare against
        */
        [[nodiscard]] inline bool operator!=(std::uint64_t compare) const noexcept {
            return !operator==(compare);
        }

        /**
        Compares two Modulus instances.

        @param[in] compare The Modulus to compare against
        */
        [[nodiscard]] inline bool operator<(const Modulus &compare) const noexcept {
            return value_ < compare.value_;
        }

        /**
        Compares a Modulus value to an unsigned integer.

        @param[in] compare The unsigned integer to compare against
        */
        [[nodiscard]] inline bool operator<(std::uint64_t compare) const noexcept {
            return value_ < compare;
        }

        /**
        Compares two Modulus instances.

        @param[in] compare The Modulus to compare against
        */
        [[nodiscard]] inline bool operator<=(const Modulus &compare) const noexcept {
            return value_ <= compare.value_;
        }

        /**
        Compares a Modulus value to an unsigned integer.

        @param[in] compare The unsigned integer to compare against
        */
        [[nodiscard]] inline bool operator<=(std::uint64_t compare) const noexcept {
            return value_ <= compare;
        }

        /**
        Compares two Modulus instances.

        @param[in] compare The Modulus to compare against
        */
        [[nodiscard]] inline bool operator>(const Modulus &compare) const noexcept {
            return value_ > compare.value_;
        }

        /**
        Compares a Modulus value to an unsigned integer.

        @param[in] compare The unsigned integer to compare against
        */
        [[nodiscard]] inline bool operator>(std::uint64_t compare) const noexcept {
            return value_ > compare;
        }

        /**
        Compares two Modulus instances.

        @param[in] compare The Modulus to compare against
        */
        [[nodiscard]] inline bool operator>=(const Modulus &compare) const noexcept {
            return value_ >= compare.value_;
        }

        /**
        Compares a Modulus value to an unsigned integer.

        @param[in] compare The unsigned integer to compare against
        */
        [[nodiscard]] inline bool operator>=(std::uint64_t compare) const noexcept {
            return value_ >= compare;
        }

        /**
        Reduces a given unsigned integer modulo this modulus.

        @param[in] value The unsigned integer to reduce
        @throws std::logic_error if the Modulus is zero
        */
        [[nodiscard]] std::uint64_t reduce(std::uint64_t value) const;

        /**
        Enables access to private members of phantom::Modulus for C
        */
        struct ModulusPrivateHelper;

        void save(std::ostream &stream) const {
            save_members(stream);
        }

        void load(std::istream &stream) {
            load_members(stream);
        }

    private:
        void set_value(std::uint64_t value);

        void save_members(std::ostream &stream) const;

        void load_members(std::istream &stream);

        std::uint64_t value_ = 0;

        std::array<std::uint64_t, 3> const_ratio_{{0, 0, 0}};

        std::size_t uint64_count_ = 0;

        int bit_count_ = 0;

        bool is_prime_ = false;
    };

    /**
    Represents a standard security level according to the HomomorphicEncryption.org
    security standard.
    */
    enum class sec_level_type : int {
        // No security level specified.
        none = 0,

        // 128-bit security level
        tc128 = 128,

        // 192-bit security level
        tc192 = 192,

        // 256-bit security level
        tc256 = 256
    };

    /**
    This class contains static methods for creating a coefficient modulus easily.
    */
    class CoeffModulus {
    public:
        CoeffModulus() = delete;

        /**
        Returns the allowed largest bit-length of the coefficient modulus (product of primes)
        which is specified in HomomorphicEncryption.org security standard.
        @param[in] poly_modulus_degree The value of the poly_modulus_degree
        @param[in] sec_level The desired standard security level
        */
        [[nodiscard]] static constexpr int MaxBitCount(
                std::size_t poly_modulus_degree, sec_level_type sec_level = sec_level_type::tc128) noexcept {
            switch (sec_level) {
                case sec_level_type::tc128:
                    return util::he_std_parms_128_tc(poly_modulus_degree);

                case sec_level_type::tc192:
                    return util::he_std_parms_192_tc(poly_modulus_degree);

                case sec_level_type::tc256:
                    return util::he_std_parms_256_tc(poly_modulus_degree);

                case sec_level_type::none:
                    return (std::numeric_limits<int>::max)();

                default:
                    return 0;
            }
        }

        /**
        Returns a default coefficient modulus for the BFV scheme that guarantees
        a given security level when using a given poly_modulus_degree.

        @param[in] poly_modulus_degree The value of the poly_modulus_degree
        encryption parameter
        @param[in] sec_level The desired standard security level
        @throws std::invalid_argument if poly_modulus_degree is not a power-of-two
        or is too large
        @throws std::invalid_argument if sec_level is sec_level_type::none
        */
        [[nodiscard]] static std::vector<Modulus> BFVDefault(
                std::size_t poly_modulus_degree, sec_level_type sec_level = sec_level_type::tc128);

        /**
        Returns a custom coefficient modulus suitable for use with the specified
        poly_modulus_degree. The return value will be a vector consisting of
        Modulus elements representing distinct prime numbers of bit-lengths
        as given in the bit_sizes parameter. The bit sizes of the prime numbers
        can be at most 60 bits.

        @param[in] poly_modulus_degree The value of the poly_modulus_degree
        encryption parameter
        @param[in] bit_sizes The bit-lengths of the primes to be generated
        @throws std::invalid_argument if poly_modulus_degree is not a power-of-two
        or is too large
        @throws std::invalid_argument if bit_sizes is too large or if its elements
        are out of bounds
        @throws std::logic_error if not enough suitable primes could be found
        */
        [[nodiscard]] static std::vector<Modulus>
        Create(std::size_t poly_modulus_degree, const std::vector<int> &bit_sizes);
    };

    /**
    This class contains static methods for creating a plaintext modulus easily.
    */
    class PlainModulus {
    public:
        PlainModulus() = delete;

        /**
        Creates a prime number Modulus for use as plain_modulus encryption
        parameter that supports batching with a given poly_modulus_degree.

        @param[in] poly_modulus_degree The value of the poly_modulus_degree
        encryption parameter
        @param[in] bit_size The bit-length of the prime to be generated
        @throws std::invalid_argument if poly_modulus_degree is not a power-of-two
        or is too large
        @throws std::invalid_argument if bit_size is out of bounds
        @throws std::logic_error if a suitable prime could not be found
        */
        [[nodiscard]] static inline Modulus Batching(std::size_t poly_modulus_degree, int bit_size) {
            return CoeffModulus::Create(poly_modulus_degree, {bit_size})[0];
        }
    };
}

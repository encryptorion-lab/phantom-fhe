#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "host/modulus.h"
#include "host/common.h"
#include "host/uintarith.h"
#include "host/numth.h"
#include "host/uintarithsmallmod.h"
#include "host/globals.h"

using namespace std;

namespace phantom::arith {
    void Modulus::set_value(uint64_t value) {
        if (value == 0) {
            // Zero settings
            bit_count_ = 0;
            uint64_count_ = 1;
            value_ = 0;
            const_ratio_ = {{0, 0, 0}};
            is_prime_ = false;
        } else if ((value >> MOD_BIT_COUNT_MAX != 0) || (value == 1)) {
            throw invalid_argument("value can be at most 61-bit and cannot be 1");
        } else {
            value_ = value;
            bit_count_ = get_significant_bit_count(value_);

            // Compute Barrett ratios for 64-bit words (barrett_reduce_128)
            uint64_t numerator[3]{0, 0, 1};
            uint64_t quotient[3]{0, 0, 0};

            // quotient = numerator（1<<128）/ value_,
            // numerator = numerator - quotient * value
            divide_uint192_inplace(numerator, value_, quotient);

            const_ratio_[0] = quotient[0];
            const_ratio_[1] = quotient[1];

            // We store also the remainder
            const_ratio_[2] = numerator[0];

            uint64_count_ = 1;

            // Set the primality flag
            is_prime_ = ::phantom::arith::is_prime(*this);
        }
    }

    uint64_t Modulus::reduce(uint64_t value) const {
        if (value_ == 0) {
            throw logic_error("cannot reduce modulo a zero modulus");
        }
        return barrett_reduce_64(value, *this);
    }

    vector <Modulus> CoeffModulus::BFVDefault(size_t poly_modulus_degree, sec_level_type sec_level) {
        if (!MaxBitCount(poly_modulus_degree, sec_level)) {
            throw invalid_argument("non-standard poly_modulus_degree");
        }
        if (sec_level == sec_level_type::none) {
            throw invalid_argument("invalid security level");
        }

        switch (sec_level) {
            case sec_level_type::tc128:
                return util::global_variables::GetDefaultCoeffModulus128().at(poly_modulus_degree);

            case sec_level_type::tc192:
                return util::global_variables::GetDefaultCoeffModulus192().at(poly_modulus_degree);

            case sec_level_type::tc256:
                return util::global_variables::GetDefaultCoeffModulus256().at(poly_modulus_degree);

            default:
                throw runtime_error("invalid security level");
        }
    }

    vector <Modulus> CoeffModulus::Create(size_t poly_modulus_degree, const vector<int> &bit_sizes) {
        if (poly_modulus_degree > POLY_MOD_DEGREE_MAX || poly_modulus_degree < POLY_MOD_DEGREE_MIN ||
            get_power_of_two(static_cast<uint64_t>(poly_modulus_degree)) < 0) {
            throw invalid_argument("poly_modulus_degree is invalid");
        }
        if (bit_sizes.size() > COEFF_MOD_COUNT_MAX) {
            throw invalid_argument("bit_sizes is invalid");
        }
        if (accumulate(
                bit_sizes.cbegin(), bit_sizes.cend(), USER_MOD_BIT_COUNT_MIN,
                [](int a, int b) { return max(a, b); }) > USER_MOD_BIT_COUNT_MAX ||
            accumulate(bit_sizes.cbegin(), bit_sizes.cend(), USER_MOD_BIT_COUNT_MAX,
                       [](int a, int b) { return min(a, b); }) < USER_MOD_BIT_COUNT_MIN) {
            throw invalid_argument("bit_sizes is invalid");
        }

        unordered_map<int, size_t> count_table;
        unordered_map<int, vector<Modulus>> prime_table;
        for (int size: bit_sizes) {
            ++count_table[size];
        }
        for (const auto &table_elt: count_table) {
            prime_table[table_elt.first] = get_primes(poly_modulus_degree, table_elt.first, table_elt.second);
        }

        vector<Modulus> result;
        for (int size: bit_sizes) {
            result.emplace_back(prime_table[size].back());
            prime_table[size].pop_back();
        }
        return result;
    }
}

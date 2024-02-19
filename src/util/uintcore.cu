// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "util/common.h"
#include "util/uintarith.h"
#include "util/uintcore.h"
#include <algorithm>
#include <string>

using namespace std;

namespace phantom {
    namespace util {
        string uint_to_hex_string(const uint64_t *value, size_t uint64_count) {
            // Start with a string with a zero for each nibble in the array.
            size_t num_nibbles = mul_safe(uint64_count, static_cast<size_t>(nibbles_per_uint64));
            string output(num_nibbles, '0');

            // Iterate through each uint64 in array and set string with correct nibbles in hex.
            size_t nibble_index = num_nibbles;
            size_t leftmost_non_zero_pos = num_nibbles;
            for (size_t i = 0; i < uint64_count; i++) {
                uint64_t part = *value++;

                // Iterate through each nibble in the current uint64.
                for (size_t j = 0; j < nibbles_per_uint64; j++) {
                    size_t nibble = (size_t)(part & uint64_t(0x0F));
                    size_t pos = --nibble_index;
                    if (nibble != 0) {
                        // If nibble is not zero, then update string and save this pos to determine
                        // number of leading zeros.
                        output[pos] = nibble_to_upper_hex(static_cast<int>(nibble));
                        leftmost_non_zero_pos = pos;
                    }
                    part >>= 4;
                }
            }

            // Trim string to remove leading zeros.
            output.erase(0, leftmost_non_zero_pos);

            // Return 0 if nothing remains.
            if (output.empty()) {
                return string("0");
            }

            return output;
        }

        string uint_to_dec_string(const uint64_t *value, size_t uint64_count) {
            if (!uint64_count) {
                return string("0");
            }
            auto remainder = std::vector<uint64_t>(uint64_count);
            auto quotient = std::vector<uint64_t>(uint64_count);
            auto base = std::vector<uint64_t>(uint64_count);

            uint64_t *remainderptr = remainder.data();
            uint64_t *quotientptr = quotient.data();
            uint64_t *baseptr = base.data();
            set_uint(10, uint64_count, baseptr);
            set_uint(value, uint64_count, remainderptr);
            string output;
            while (!is_zero_uint(remainderptr, uint64_count)) {
                divide_uint_inplace(remainderptr, baseptr, uint64_count, quotientptr);
                char digit = (char)(remainderptr[0] + uint64_t('0'));
                output += digit;
                swap(remainderptr, quotientptr);
            }
            reverse(output.begin(), output.end());

            // Return 0 if nothing remains.
            if (output.empty()) {
                return string("0");
            }

            return output;
        }
    }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "blake2.h"
#include "common.h"
#include "defines.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace phantom {
    namespace util {
        class HashFunction {
        public:
            HashFunction() = delete;

            static constexpr std::size_t hash_block_uint64_count = 4;

            static constexpr std::size_t hash_block_byte_count = hash_block_uint64_count * arith::bytes_per_uint64;

            using hash_block_type = std::array<std::uint64_t, hash_block_uint64_count>;

            static constexpr hash_block_type hash_zero_block{{0, 0, 0, 0}};

            inline static void hash(const std::uint64_t *input, std::size_t uint64_count,
                                    hash_block_type &destination) {
                if (blake2b(&destination, hash_block_byte_count, input, uint64_count * arith::bytes_per_uint64, nullptr, 0) !=
                    0) {
                    throw std::runtime_error("blake2b failed");
                }
            }
        };
    }
}

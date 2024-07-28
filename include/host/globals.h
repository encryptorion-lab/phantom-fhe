// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <vector>

#include "hestdparms.h"
#include "common.h"
#include "modulus.h"
#include "cuda_wrapper.cuh"

namespace phantom::util::global_variables {
    /**
    Default value for the standard deviation of the noise (error) distribution.
    */
    constexpr std::size_t prng_seed_uint64_count = 8;
    constexpr std::size_t prng_seed_byte_count = prng_seed_uint64_count * arith::bytes_per_uint64;

    constexpr double noise_standard_deviation = distributionParameter;

    constexpr double noise_distribution_width_multiplier = 6;

    constexpr double noise_max_deviation = noise_standard_deviation * noise_distribution_width_multiplier;

    /**
    This data structure is a key-value storage that maps degrees of the polynomial modulus
    to vectors of Modulus elements so that when used with the default value for the
    standard deviation of the noise distribution (noise_standard_deviation), the security
    level is at least 128 bits according to https://HomomorphicEncryption.org. This makes
    it easy for non-expert users to select secure parameters.
    */
    const std::map<std::size_t, std::vector<arith::Modulus>> &GetDefaultCoeffModulus128();

    /**
    This data structure is a key-value storage that maps degrees of the polynomial modulus
    to vectors of Modulus elements so that when used with the default value for the
    standard deviation of the noise distribution (noise_standard_deviation), the security
    level is at least 192 bits according to https://HomomorphicEncryption.org. This makes
    it easy for non-expert users to select secure parameters.
    */
    const std::map<std::size_t, std::vector<arith::Modulus>> &GetDefaultCoeffModulus192();

    /**
    This data structure is a key-value storage that maps degrees of the polynomial modulus
    to vectors of Modulus elements so that when used with the default value for the
    standard deviation of the noise distribution (noise_standard_deviation), the security
    level is at least 256 bits according to https://HomomorphicEncryption.org. This makes
    it easy for non-expert users to select secure parameters.
    */
    const std::map<std::size_t, std::vector<arith::Modulus>> &GetDefaultCoeffModulus256();

    // Global default CUDA stream, implement and init at context.cu
    extern std::unique_ptr<phantom::util::cuda_stream_wrapper> default_stream;
}



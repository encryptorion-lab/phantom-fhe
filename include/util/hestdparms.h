#pragma once

#include "defines.h"
#include <cstddef>

namespace phantom {
    namespace util {
        /**
        Largest allowed bit counts for coeff_modulus based on the security estimates from
        HomomorphicEncryption.org security standard. Microsoft seal samples the secret key
        from a ternary {-1, 0, 1} distribution.
        */
        // Ternary secret; 128 bits classical security
        [[nodiscard]] constexpr int he_std_parms_128_tc(std::size_t poly_modulus_degree) noexcept {
            switch (poly_modulus_degree) {
                case std::size_t(1024):
                    return 27;
                case std::size_t(2048):
                    return 54;
                case std::size_t(4096):
                    return 109;
                case std::size_t(8192):
                    return 218;
                case std::size_t(16384):
                    return 438;
                case std::size_t(32768):
                    return 881;
                case std::size_t(65536):
                    return 1777;
                case std::size_t(131072):
                    return 3576;
            }
            return 0;
        }

        // Ternary secret; 192 bits classical security
        [[nodiscard]] constexpr int he_std_parms_192_tc(std::size_t poly_modulus_degree) noexcept {
            switch (poly_modulus_degree) {
                case std::size_t(1024):
                    return 19;
                case std::size_t(2048):
                    return 37;
                case std::size_t(4096):
                    return 75;
                case std::size_t(8192):
                    return 151;
                case std::size_t(16384):
                    return 304;
                case std::size_t(32768):
                    return 611;
                case std::size_t(65536):
                    return 1229;
                case std::size_t(131072):
                    return 2469;
            }
            return 0;
        }

        // Ternary secret; 256 bits classical security
        [[nodiscard]] constexpr int he_std_parms_256_tc(std::size_t poly_modulus_degree) noexcept {
            switch (poly_modulus_degree) {
                case std::size_t(1024):
                    return 14;
                case std::size_t(2048):
                    return 29;
                case std::size_t(4096):
                    return 58;
                case std::size_t(8192):
                    return 118;
                case std::size_t(16384):
                    return 237;
                case std::size_t(32768):
                    return 476;
                case std::size_t(65536):
                    return 955;
                case std::size_t(131072):
                    return 1918;
            }
            return 0;
        }

        // Ternary secret; 128 bits quantum security
        [[nodiscard]] constexpr int he_std_parms_128_tq(std::size_t poly_modulus_degree) noexcept {
            switch (poly_modulus_degree) {
                case std::size_t(1024):
                    return 25;
                case std::size_t(2048):
                    return 51;
                case std::size_t(4096):
                    return 101;
                case std::size_t(8192):
                    return 204;
                case std::size_t(16384):
                    return 410;
                case std::size_t(32768):
                    return 826;
                case std::size_t(65536):
                    return 1664;
                case std::size_t(131072):
                    return 3349;
            }
            return 0;
        }

        // Ternary secret; 192 bits quantum security
        [[nodiscard]] constexpr int he_std_parms_192_tq(std::size_t poly_modulus_degree) noexcept {
            switch (poly_modulus_degree) {
                case std::size_t(1024):
                    return 17;
                case std::size_t(2048):
                    return 35;
                case std::size_t(4096):
                    return 70;
                case std::size_t(8192):
                    return 141;
                case std::size_t(16384):
                    return 284;
                case std::size_t(32768):
                    return 570;
                case std::size_t(65536):
                    return 1145;
                case std::size_t(131072):
                    return 2302;
            }
            return 0;
        }

        // Ternary secret; 256 bits quantum security
        [[nodiscard]] constexpr int he_std_parms_256_tq(std::size_t poly_modulus_degree) noexcept {
            switch (poly_modulus_degree) {
                case std::size_t(1024):
                    return 13;
                case std::size_t(2048):
                    return 27;
                case std::size_t(4096):
                    return 54;
                case std::size_t(8192):
                    return 109;
                case std::size_t(16384):
                    return 220;
                case std::size_t(32768):
                    return 443;
                case std::size_t(65536):
                    return 889;
                case std::size_t(131072):
                    return 1784;
            }
            return 0;
        }

        // Standard deviation for error distribution
        constexpr float distributionParameter = 3.2f;
        constexpr float assuranceMeasure = 36;
    }
}

#pragma once

#include <cuComplex.h>
#include <cstddef>
#include <stdexcept>

namespace phantom {
    namespace util {
        class ComplexRoots {
        public:
            ComplexRoots() = delete;

            ComplexRoots(std::size_t degree_of_roots);

            cuDoubleComplex get_root(std::size_t index) const;

        private:
            static constexpr double PI_ = 3.1415926535897932384626433832795028842;

            // Contains 0~(n/8-1)-th powers of the n-th primitive root.
            cuDoubleComplex *roots_;

            std::size_t degree_of_roots_;
        };

        __host__ __device__ __forceinline__ cuDoubleComplex polar(const double &__rho, const double &__theta) {
            return make_cuDoubleComplex(__rho * cos(__theta), __rho * sin(__theta));
        }
    }
}

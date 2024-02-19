#include "croots.h"
#include <cuComplex.h>

using namespace std;

namespace phantom {
    namespace util {
        // Required for C++14 compliance: static constexpr member variables are not necessarily inlined so need to
        // ensure symbol is created.
        constexpr double ComplexRoots::PI_;

        ComplexRoots::ComplexRoots(size_t degree_of_roots) : degree_of_roots_(degree_of_roots) {
            roots_ = (cuDoubleComplex *)malloc((degree_of_roots_ / 8 + 1) * sizeof(cuDoubleComplex));

            // Generate 1/8 of all roots.
            // Alternatively, choose from precomputed high-precision roots in files.
            for (size_t i = 0; i <= degree_of_roots_ / 8; i++) {
                roots_[i] = polar(1.0, 2 * PI_ * static_cast<double>(i) / static_cast<double>(degree_of_roots_));
            }
        }

        cuDoubleComplex ComplexRoots::get_root(size_t index) const {
            index &= degree_of_roots_ - 1;
            auto mirror = [](cuDoubleComplex a) {
                return make_cuDoubleComplex(a.y, a.x);
            };

            // This express the 8-fold symmetry of all n-th roots.
            if (index <= degree_of_roots_ / 8) {
                return roots_[index];
            }
            else if (index <= degree_of_roots_ / 4) {
                return mirror(roots_[degree_of_roots_ / 4 - index]);
            }
            else if (index <= degree_of_roots_ / 2) {
                return cuCsub({0, 0}, cuConj(get_root(degree_of_roots_ / 2 - index)));
            }
            else if (index <= 3 * degree_of_roots_ / 4) {
                return cuCsub({0, 0}, get_root(index - degree_of_roots_ / 2));
            }
            else {
                return cuConj(get_root(degree_of_roots_ - index));
            }
        }
    }
}

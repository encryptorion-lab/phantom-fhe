#include "util/ntt.h"
#include "util/uintarith.h"
#include "util/uintarithsmallmod.h"

using namespace std;
using namespace phantom;
using namespace phantom::util;

namespace phantom {
    namespace arith {
        NTT::NTT(const int coeff_count_power, const Modulus &modulus) {
            coeff_count_power_ = coeff_count_power;
            coeff_count_ = 1 << coeff_count_power_;
            modulus_ = modulus;
            // We defer parameter checking to try_minimal_primitive_root(...)
            if (!try_minimal_primitive_root(2 * coeff_count_, modulus_, root_)) {
                throw invalid_argument("invalid modulus in try_minimal_primitive_root");
            }
            if (!try_invert_uint_mod(root_, modulus_, inv_root_)) {
                throw invalid_argument("invalid modulus in try_invert_uint_mod");
            }
            // Populate tables with powers of root in specific orders.
            root_powers_.resize(coeff_count_);
            root_powers_shoup_.resize(coeff_count_);
            const uint64_t root_shoup = compute_shoup(root_, modulus_.value());
            uint64_t power = root_;
            for (size_t i = 1; i < coeff_count_; i++) {
                root_powers_[reverse_bits(i, coeff_count_power_)] = power;
                root_powers_shoup_[reverse_bits(i, coeff_count_power_)] = compute_shoup(power, modulus_.value());
                power = multiply_uint_mod_shoup(power, root_, root_shoup, modulus_);
            }
            root_powers_[0] = 1;
            root_powers_shoup_[0] = compute_shoup(1, modulus_.value());

            inv_root_powers_.resize(coeff_count_);
            inv_root_powers_shoup_.resize(coeff_count_);
            const uint64_t inv_root_shoup = compute_shoup(inv_root_, modulus_.value());
            power = inv_root_;
            for (size_t i = 1; i < coeff_count_; i++) {
                inv_root_powers_[reverse_bits(i, coeff_count_power_)] = power;
                inv_root_powers_shoup_[reverse_bits(i, coeff_count_power_)] = compute_shoup(power, modulus_.value());
                power = multiply_uint_mod_shoup(power, inv_root_, inv_root_shoup, modulus_);
            }
            inv_root_powers_[0] = 1;
            inv_root_powers_shoup_[0] = compute_shoup(1, modulus_.value());

            // Compute n^(-1) modulo q.
            if (!try_invert_uint_mod(coeff_count_, modulus_, inv_degree_modulo_)) {
                throw invalid_argument("invalid modulus in computing n^(-1) modulo q");
            }
            inv_degree_modulo_shoup_ = compute_shoup(inv_degree_modulo_, modulus_.value());

            inv_root_powers_[1] = multiply_uint_mod_shoup(inv_root_powers_[1], inv_degree_modulo_,
                                                          inv_degree_modulo_shoup_, modulus_);
            inv_root_powers_shoup_[1] = compute_shoup(inv_root_powers_[1], modulus_.value());
        }

        RNSNTT::RNSNTT(const size_t log_N, const vector<Modulus> &modulus) {
            if (log_N == 0) {
                throw invalid_argument("log_N must be positive");
            }

            if (modulus.empty()) {
                throw invalid_argument("RNS modulus is empty");
            }

            log_N_ = log_N;
            N_ = 1 << log_N;
            modulus_ = modulus;

            for (size_t i = 0; i < modulus.size(); i++) {
                rns_ntt_.push_back(NTT(log_N, modulus[i]));
            }
        }
    } // namespace arith
} // namespace phantom

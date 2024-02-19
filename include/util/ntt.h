#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include "defines.h"
#include "modulus.h"
#include "uintarithsmallmod.h"
#include "uintcore.h"

namespace phantom {
    namespace arith {
        class NTT {
        public:
            explicit NTT(int coeff_count_power, const Modulus &modulus);

            uint64_t get_root() const { return root_; }

            auto &get_from_root_powers() const { return root_powers_; }

            auto &get_from_root_powers_shoup() const { return root_powers_shoup_; }

            auto &get_from_inv_root_powers() const { return inv_root_powers_; }

            auto &get_from_inv_root_powers_shoup() const { return inv_root_powers_shoup_; }

            const uint64_t &inv_degree_modulo() const { return inv_degree_modulo_; }

            const uint64_t &inv_degree_modulo_shoup() const { return inv_degree_modulo_shoup_; }

            const Modulus &modulus() const { return modulus_; }

            int coeff_count_power() const { return coeff_count_power_; }

            size_t coeff_count() const { return coeff_count_; }

        private:
            std::uint64_t root_ = 0;
            std::uint64_t inv_root_ = 0;
            int coeff_count_power_ = 0;
            std::size_t coeff_count_ = 0;
            Modulus modulus_;

            // Inverse of coeff_count_ modulo modulus_.
            uint64_t inv_degree_modulo_;
            uint64_t inv_degree_modulo_shoup_;

            // Holds 1~(n-1)-th powers of root_ in bit-reversed order, the 0-th power is left unset.
            std::vector<uint64_t> root_powers_;
            std::vector<uint64_t> root_powers_shoup_;

            // Holds 1~(n-1)-th powers of inv_root_ in scrambled order, the 0-th power is left unset.
            std::vector<uint64_t> inv_root_powers_;
            std::vector<uint64_t> inv_root_powers_shoup_;
        };

        class RNSNTT {
        public:
            explicit RNSNTT(size_t log_N, const std::vector<Modulus> &rns_modulus);
            auto &get_modulus_at(const size_t index) const { return modulus_.at(index); }
            auto &get_ntt_at(const size_t index) const { return rns_ntt_.at(index); }
            auto size() const { return modulus_.size(); }

        private:
            size_t log_N_;
            size_t N_;
            std::vector<Modulus> modulus_;
            std::vector<NTT> rns_ntt_;
        };
    } // namespace arith
} // namespace phantom

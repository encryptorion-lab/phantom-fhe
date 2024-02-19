#pragma once

#include "context.cuh"
#include "ntt.cuh"
#include "plaintext.h"

typedef struct PhantomBatchEncoder {
    std::size_t slots_;

    phantom::util::Pointer<uint64_t> matrix_reps_index_map_;

    phantom::util::Pointer<uint64_t> data_;

    /**
    Creates a BatchEncoder. It is necessary that the encryption parameters
    given through the SEALContext object support batching.

    @param[in] parms The PhantomContext
    @throws std::invalid_argument if the encryption parameters are not valid for batching
    @throws std::invalid_argument if scheme is not scheme_type::bfv
    */
    explicit PhantomBatchEncoder(const PhantomContext &context);

    /**
    Creates a plaintext from a given matrix. This function "batches" a given matrix
    of integers modulo the plaintext modulus into a plaintext element, and stores
    the result in the destination parameter. The input vector must have size at most equal
    to the degree of the polynomial modulus. The first half of the elements represent the
    first row of the matrix, and the second half represent the second row. The numbers
    in the matrix can be at most equal to the plaintext modulus for it to represent
    a valid plaintext.

    If the destination plaintext overlaps the input values in memory, the behavior of
    this function is undefined.

    @param[in] values The matrix of integers modulo plaintext modulus to batch
    @param[out] destination The plaintext polynomial to overwrite with the result
    @throws std::invalid_argument if values is too large
    */
    void
    encode(const PhantomContext &context, const std::vector<int64_t> &values_matrix, PhantomPlaintext &destination);

    /**
    Inverse of encode. This function "unbatches" a given plaintext into a matrix
    of integers modulo the plaintext modulus, and stores the result in the destination
    parameter. The input plaintext must have degrees less than the polynomial modulus,
    and coefficients less than the plaintext modulus, i.e. it must be a valid plaintext
    for the encryption parameters. Dynamic memory allocations in the process are
    allocated from the global memory pool.

    @param[in] plain The plaintext polynomial to unbatch
    @param[out] destination The matrix to be overwritten with the values in the slots
    @throws std::invalid_argument if plain is not valid for the encryption parameters
    @throws std::invalid_argument if plain is in NTT form
    */
    void decode(const PhantomContext &context, const PhantomPlaintext &plain,
                std::vector<std::int64_t> &destination) const;

    /**
     Returns the number of complex numbers encoded.
     */
    [[nodiscard]] inline std::size_t slot_count() const noexcept {
        return slots_;
    }

    [[nodiscard]] inline uint64_t *matrix_reps_index_map() const noexcept {
        return matrix_reps_index_map_.get();
    }

    void populate_matrix_reps_index_map() const;
} PhantomBatchEncoder;

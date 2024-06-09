#ifndef _GPU_SCALINGVARIANT_H
#define _GPU_SCALINGVARIANT_H

#include "ciphertext.h"
#include "context.cuh"
#include "plaintext.h"

/** For BFV cipher + ceil(m*q/t);
 */
void multiply_add_plain_with_scaling_variant(const PhantomContext &context,
                                             const PhantomPlaintext &plain,
                                             size_t chain_index,
                                             PhantomCiphertext &cipher,
                                             const cudaStream_t &stream);

void multiply_sub_plain_with_scaling_variant(const PhantomContext &context,
                                             const PhantomPlaintext &plain,
                                             size_t chain_index,
                                             PhantomCiphertext &cipher,
                                             const cudaStream_t &stream);
#endif //_GPU_SCALINGVARIANT_H

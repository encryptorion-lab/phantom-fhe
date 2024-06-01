#pragma once

#include "gputype.h"
#include "context.cuh"

void special_fft_forward(const PhantomContext &context, DCKKSEncoderInfo *gpu_ckks_msg_vec_, uint32_t msg_vec_size);

void special_fft_backward(const PhantomContext &context, DCKKSEncoderInfo *gpu_ckks_msg_vec_, uint32_t msg_vec_size,
                          double scalar = 0.0);

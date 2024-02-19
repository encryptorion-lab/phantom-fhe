#pragma once
#include <gputype.h>

void special_fft_forward(DCKKSEncoderInfo *gpu_ckks_msg_vec_,
                         uint32_t msg_vec_size);

void special_fft_backward(DCKKSEncoderInfo *gpu_ckks_msg_vec_,
                          uint32_t msg_vec_size,
                          double scalar = 0.0);

#pragma once

#include "gputype.h"
#include "context.cuh"

void special_fft_forward(DCKKSEncoderInfo &gp);

void special_fft_backward(DCKKSEncoderInfo &gp, double scalar = 0.0);

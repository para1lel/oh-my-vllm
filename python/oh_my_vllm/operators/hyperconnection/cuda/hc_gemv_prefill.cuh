#pragma once

namespace hc_projection {

__global__ void gemv_prefill_kernel(const float *__restrict__ input,
                                    const float *__restrict__ weight, float *__restrict__ output,
                                    int rows, int width, int outputs) {
  int out = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
  int row = blockIdx.y * 2, lane = threadIdx.x;
  if (out >= outputs)
    return;
  const float *first_input = input + int64_t(row) * width;
  const float *second_input = first_input + width;
  float first_left = 0.f, first_right = 0.f;
  float second_left = 0.f, second_right = 0.f;
  for (int k = lane; k < width; k += 32) {
    float left = weight[int64_t(out) * width + k];
    float right = weight[int64_t(out + 1) * width + k];
    float first = first_input[k];
    float second = row + 1 < rows ? second_input[k] : 0.f;
    first_left = __fmaf_rn(first, left, first_left);
    first_right = __fmaf_rn(first, right, first_right);
    second_left = __fmaf_rn(second, left, second_left);
    second_right = __fmaf_rn(second, right, second_right);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    first_left = __fadd_rn(first_left, __shfl_down_sync(0xffffffff, first_left, offset));
    first_right = __fadd_rn(first_right, __shfl_down_sync(0xffffffff, first_right, offset));
    second_left = __fadd_rn(second_left, __shfl_down_sync(0xffffffff, second_left, offset));
    second_right = __fadd_rn(second_right, __shfl_down_sync(0xffffffff, second_right, offset));
  }
  if (lane == 0) {
    output[int64_t(row) * outputs + out] = first_left;
    output[int64_t(row) * outputs + out + 1] = first_right;
    if (row + 1 < rows) {
      output[int64_t(row + 1) * outputs + out] = second_left;
      output[int64_t(row + 1) * outputs + out + 1] = second_right;
    }
  }
}

} // namespace hc_projection

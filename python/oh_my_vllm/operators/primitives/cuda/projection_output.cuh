#pragma once

#include "groupwise.cuh"
#include "projection_decode_16dp.cuh"

__global__ void output_projection_merge(const float *partials, __nv_bfloat16 *output, int count,
                                        int splits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    float value = 0;
    for (int split = 0; split < splits; ++split)
      value += partials[split * count + index];
    output[index] = __float2bfloat16_rn(value);
  }
}

template <bool Dependent = false>
inline void output_projection(TensorView input, TensorView weight, TensorView input_scale,
                              TensorView weight_scale, TensorView partials, TensorView output,
                              int rows, cudaStream_t stream) {
  int splits = rows <= 8 ? 8 : 3;
  if (rows <= 8)
    down16dp::launch<Dependent>(input, weight, input_scale, weight_scale, partials, rows, splits,
                                stream);
  else
    groupwise<128, 32, 256, 128, 1, true, 3, float>(
        static_cast<const __nv_fp8_e4m3 *>(weight.data_ptr()),
        static_cast<const __nv_fp8_e4m3 *>(input.data_ptr()),
        static_cast<const float *>(weight_scale.data_ptr()),
        static_cast<const float *>(input_scale.data_ptr()), partials.data_ptr(), 5120, rows,
        6144 / 3, stream, Dependent);
  output_projection_merge<<<(rows * 5120 + 255) / 256, 256, 0, stream>>>(
      static_cast<const float *>(partials.data_ptr()),
      static_cast<__nv_bfloat16 *>(output.data_ptr()), rows * 5120, splits);
  check_cuda(cudaGetLastError());
}

#pragma once

#include "common.cuh"

__global__ void attention_gate_kernel(const __nv_bfloat16 *input, const __nv_bfloat16 *gate,
                                      __nv_bfloat16 *output, int rows, int heads, int width,
                                      int64_t input_row, int64_t input_head, int64_t gate_row,
                                      int64_t gate_head) {
  pdl_dependency_wait();
  pdl_launch_next();
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * heads * width)
    return;
  const int row = index / (heads * width), head = index / width % heads, column = index % width;
  const float value = __bfloat162float(gate[row * gate_row + head * gate_head + column]);
  const auto factor = __float2bfloat16_rn(1.f / (1.f + expf(-value)));
  output[index] = __hmul(input[row * input_row + head * input_head + column], factor);
}

void attention_gate(TensorView input, TensorView gate, TensorView output) {
  TVM_FFI_ICHECK(input.ndim() == 3 && input.size(0) >= 0 && input.size(1) > 0 &&
                 input.size(2) > 0 && gate.ndim() == 3 && output.ndim() == 3);
  TVM_FFI_ICHECK(input.size(1) <= INT32_MAX / input.size(2) &&
                 input.size(0) <= INT32_MAX / (input.size(1) * input.size(2)));
  for (auto tensor : {input, gate, output}) {
    TVM_FFI_ICHECK(same_cuda_device(tensor, input) && has_dtype(tensor, kDLBfloat, 16) &&
                   tensor.size(0) == input.size(0) && tensor.size(1) == input.size(1) &&
                   tensor.size(2) == input.size(2) && tensor.stride(2) == 1 &&
                   tensor.stride(1) >= tensor.size(2) &&
                   tensor.stride(1) <= INT32_MAX / tensor.size(1) &&
                   tensor.stride(0) >= tensor.size(1) * tensor.stride(1) &&
                   tensor.stride(0) <= INT32_MAX && tensor.size(0) <= INT32_MAX / tensor.stride(0));
  }
  TVM_FFI_ICHECK(output.stride(1) == output.size(2) &&
                 output.stride(0) == output.size(1) * output.size(2) &&
                 disjoint_storage(output, input) && disjoint_storage(output, gate));
  if (input.size(0) == 0)
    return;
  auto stream = stream_for(input, "attention_gate");
  const int count = input.size(0) * input.size(1) * input.size(2);
  launch_kernel(attention_gate_kernel, (count - 1) / 256 + 1, 256, 0, stream,
                static_cast<const __nv_bfloat16 *>(input.data_ptr()),
                static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
                static_cast<__nv_bfloat16 *>(output.data_ptr()), input.size(0), input.size(1),
                input.size(2), input.stride(0), input.stride(1), gate.stride(0), gate.stride(1));
  finish_cuda_launch(stream, "attention_gate");
}

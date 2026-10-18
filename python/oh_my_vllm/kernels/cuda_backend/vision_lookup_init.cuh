#pragma once

__global__ void vision_lookup_initialize_kernel(cutlass::bfloat16_t *table) {
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  float input = float(cutlass::bfloat16_t::bitcast(uint16_t(index)));
  cutlass::NumericConverter<cutlass::bfloat16_t, float> convert;
  table[index] = convert(VisionRoundedGelu<float>{}(input));
}

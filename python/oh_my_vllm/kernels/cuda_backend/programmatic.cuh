#pragma once

#include "common.cuh"

template <typename Kernel, typename... Args>
void launch_programmatic(Kernel kernel, dim3 grid, dim3 block, size_t shared_bytes,
                         cudaStream_t stream, Args... args) {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shared_bytes;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  check_cuda(cudaLaunchKernelEx(&config, kernel, args...));
}

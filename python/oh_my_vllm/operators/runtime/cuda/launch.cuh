#pragma once

#include "contracts.cuh"

template <typename Kernel, typename... Args>
void launch_kernel(Kernel kernel, dim3 grid, dim3 block, size_t shared_bytes, cudaStream_t stream,
                   Args... args) {
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shared_bytes;
  config.stream = stream;
  check_cuda(cudaLaunchKernelEx(&config, kernel, args...));
}

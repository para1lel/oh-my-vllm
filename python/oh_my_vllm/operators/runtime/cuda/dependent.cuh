#pragma once

#include "launch.cuh"

__device__ inline void wait_for_producer() {
  cudaGridDependencySynchronize();
  asm volatile("" ::: "memory");
}

__device__ inline void launch_consumer() { cudaTriggerProgrammaticLaunchCompletion(); }

template <bool Dependent = true, typename Kernel, typename... Args>
inline void launch_dependent_kernel(Kernel kernel, dim3 grid, dim3 block, size_t shared_bytes,
                                    cudaStream_t stream, Args... args) {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shared_bytes;
  config.stream = stream;
  if constexpr (Dependent) {
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
  check_cuda(cudaLaunchKernelEx(&config, kernel, args...));
}

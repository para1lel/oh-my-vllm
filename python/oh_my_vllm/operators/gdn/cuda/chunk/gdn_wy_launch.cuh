#pragma once

#include "gdn_wy_native_output.cuh"
#include "gdn_wy_native_prepare.cuh"
#include "gdn_wy_register.cuh"
#include "gdn_wy_metadata.cuh"

namespace gdn_wy {

inline bool selected(int rows, int batch, int max_queries, bool verify) {
  return !verify && max_queries > 128;
}

inline void workspace_contract(TensorView reference, TensorView metadata, TensorView offsets,
                               TensorView transforms, TensorView prefixes, TensorView snapshots,
                               TensorView coefficients, int chunks, int batch) {
  tensor_contract(metadata, reference, kDLInt, 32, {chunks, 4});
  tensor_contract(offsets, reference, kDLInt, 32, {chunks ? batch + 1 : 0});
  tensor_contract(transforms, reference, kDLBfloat, 16, {chunks, 48, 1, 2, 64, 128});
  tensor_contract(prefixes, reference, kDLFloat, 32, {chunks, 48, 128});
  tensor_contract(snapshots, reference, kDLBfloat, 16, {chunks, 48, 128, 128});
  tensor_contract(coefficients, reference, kDLBfloat, 16, {chunks, 48, 64, 64});
}

inline void native_launch(const __nv_bfloat16 *qkv, const float *gates, const int *starts,
                          const int *reads, const int *writes, float *state, __nv_bfloat16 *output,
                          int *metadata, int *offsets, __nv_bfloat16 *transforms, float *prefixes,
                          __nv_bfloat16 *snapshots, __nv_bfloat16 *coefficients, int rows,
                          int slots, int batch, int capacity, int chunks, cudaStream_t stream) {
  check_cuda(cudaFuncSetAttribute(native_prepare_bf16_inverse_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  sizeof(NativePrepareStorage)));
  auto carry = batch == 4 ? gdn_wy_register::register_carry_kernel<true>
                          : gdn_wy_register::register_carry_kernel<false>;
  check_cuda(cudaFuncSetAttribute(carry, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  sizeof(gdn_wy_register::RegisterCarryStorage)));
  check_cuda(cudaFuncSetAttribute(native_output_qk_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  sizeof(NativeOutputStorage)));
  if (capacity >= 32) {
    gdn_metadata_parallel::launch(starts, reads, writes, metadata, offsets, snapshots, capacity,
                                  chunks, rows, slots, batch, stream);
  } else {
    metadata_kernel<<<batch, 32, 0, stream>>>(starts, reads, writes, metadata, offsets, chunks,
                                              capacity, rows, slots);
    check_cuda(cudaGetLastError());
  }
  native_prepare_bf16_inverse_kernel<<<dim3(chunks, 48), 256, sizeof(NativePrepareStorage),
                                       stream>>>(qkv, gates, metadata, transforms, prefixes,
                                                 coefficients);
  check_cuda(cudaGetLastError());
  carry<<<dim3(batch, 48, 2), 128, sizeof(gdn_wy_register::RegisterCarryStorage), stream>>>(
      qkv, metadata, reads, transforms, prefixes, state, snapshots, offsets);
  check_cuda(cudaGetLastError());
  native_output_qk_kernel<<<dim3(chunks, 48), 256, sizeof(NativeOutputStorage), stream>>>(
      qkv, gates, metadata, transforms, prefixes, snapshots, output, coefficients);
  check_cuda(cudaGetLastError());
}

} // namespace gdn_wy

#pragma once

#include "select/kernel.cuh"

namespace {

template <typename Config> void launch_selection(const TopkSelectArgs &args) {
  using Kernel = topk_select_bf16_normal::TopkSelectKernelBF16Normal<Config>;
  auto kernel = topk_select_bf16_normal::topk_kernel<Kernel>;
  constexpr size_t shared_bytes = sizeof(typename Kernel::SharedMemoryPlan);
  TVM_FFI_ICHECK(shared_bytes * Kernel::TARGET_OCCUPANCY <= args.shared_memory_size_per_sm);
  check_cuda(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  typename Kernel::TmaParams maps = {Kernel::make_topk_tensor_map(args)};
  kernel<<<args.batch_size, Kernel::NUM_THREADS, shared_bytes, args.stream>>>(args, maps);
  check_cuda(cudaGetLastError());
}

} // namespace

void topk_bf16(TensorView scores, TensorView output, int64_t logical_width) {
  TVM_FFI_ICHECK(scores.ndim() == 2);
  int64_t rows = scores.size(0), physical_width = scores.size(1);
  TVM_FFI_ICHECK(rows >= 0 && rows <= 32768 && logical_width >= 512 && logical_width < (1 << 23));
  TVM_FFI_ICHECK(physical_width == (logical_width + 511) / 512 * 512);
  tensor_contract(scores, scores, kDLBfloat, 16, {rows, physical_width});
  auto source = reinterpret_cast<uintptr_t>(scores.data_ptr());
  tensor_contract(output, scores, kDLInt, 64, {rows, 512});
  auto destination = reinterpret_cast<uintptr_t>(output.data_ptr());
  int64_t score_bytes = rows * physical_width * 2;
  int64_t output_bytes = rows * 512 * 8;
  TVM_FFI_ICHECK(source + score_bytes <= destination || destination + output_bytes <= source)
      << "operator tensors overlap";
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(scores.device().device_id);
  cudaDeviceProp properties;
  check_cuda(cudaGetDeviceProperties(&properties, scores.device().device_id));
  TopkSelectArgs args{};
  args.batch_size = rows;
  args.vocab_size = logical_width;
  args.topk = 512;
  args.input = scores.data_ptr();
  args.output_index = output.data_ptr();
  args.stride_input_batch = physical_width;
  args.stride_output_index_batch = 512;
  args.sorted_index = true;
  args.return_value = false;
  args.idx_oob_fill_value = -1;
  args.value_oob_fill_value = -INFINITY;
  args.abort_when_nan_found = false;
  args.shared_memory_size_per_sm = properties.sharedMemPerBlockOptin;
  args.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, scores.device().device_id));
  if (rows <= properties.multiProcessorCount)
    launch_selection<
        TopkSelectConfig<nv_bfloat16, int64_t, false, true, false, 512, 512, 1, 8192, 4096, 5>>(
        args);
  else
    launch_selection<
        TopkSelectConfig<nv_bfloat16, int64_t, false, true, false, 512, 256, 2, 4096, 4096, 4>>(
        args);
}

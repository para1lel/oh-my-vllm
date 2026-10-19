#pragma once
#include "mx_fp8_norm.cuh"
#include "mx_fp8_plan.cuh"
#include "mx_fp8_quant.cuh"
#include "normalization.cuh"

namespace ffn_mx_native {
inline void lt_alignment(TensorView tensor) {
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 256 == 0)
      << "MXFP8 GEMM storage requires 256-byte alignment";
}

inline void add_tensor(std::vector<Tensor> &owned, const Tensor &tensor) {
  for (const auto &current : owned)
    if (current.data_ptr() == tensor.data_ptr() && current.numel() == tensor.numel())
      return;
  owned.push_back(tensor);
}

inline void add_weights(std::vector<MXWeights> &owned, const MXWeights &weights) {
  for (const auto &current : owned)
    if (current.get() == weights.get())
      return;
  owned.push_back(weights);
}

inline void convert(const Tensor &original, const Tensor &scales, const Tensor &prepared,
                    const Tensor &prepared_scales, bool weights, cudaStream_t stream) {
  int rows = original.size(0), width = original.size(1), padded = prepared.size(0);
  convert_kernel<<<(padded * (width / 4) + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_fp8_e4m3 *>(original.data_ptr()),
      static_cast<const float *>(scales.data_ptr()),
      static_cast<__nv_fp8_e4m3 *>(prepared.data_ptr()),
      static_cast<unsigned char *>(prepared_scales.data_ptr()), rows, padded, width, weights);
  check_cuda(cudaGetLastError());
}

inline void owner_check(const MXContext &owner) {
  owner->binding.check(owner->binding.device);
  TVM_FFI_ICHECK(owner->handle && !owner->poisoned) << "MX context is closed or failed";
}

inline void retain(const MXContext &owner, const MXWeights &weights,
                   const std::array<Tensor, 6> &tensors, cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  unsigned long long id = 0;
  check_cuda(cudaStreamGetCaptureInfo(stream, &status, &id));
  if (status == cudaStreamCaptureStatusNone) {
    TVM_FFI_ICHECK(!owner->lease) << "MX context has an exclusive live graph lease";
    if (!owner->eager_stream_bound) {
      owner->eager_stream = stream;
      owner->eager_stream_bound = true;
    }
    TVM_FFI_ICHECK(owner->eager_stream == stream) << "MX eager calls require the same CUDA stream";
    for (const auto &tensor : tensors)
      add_tensor(owner->eager_tensors, tensor);
  } else {
    TVM_FFI_ICHECK(status == cudaStreamCaptureStatusActive && owner->lease &&
                   owner->lease_thread == std::this_thread::get_id())
        << "MX capture requires its thread's graph lease";
    if (!owner->capture_bound) {
      owner->capture_id = id;
      owner->capture_stream = stream;
      owner->capture_bound = true;
    }
    TVM_FFI_ICHECK(owner->capture_id == id && owner->capture_stream == stream)
        << "MX lease belongs to another capture or stream";
    add_weights(owner->lease->weights, weights);
    for (const auto &tensor : tensors)
      add_tensor(owner->lease->tensors, tensor);
  }
  add_weights(owner->seen_weights, weights);
}
} // namespace ffn_mx_native

ffn_mx_native::MXWeights
ffn_mx_prepare_weights(tvm::ffi::Tensor norm, tvm::ffi::Tensor gate_up,
                       tvm::ffi::Tensor gate_up_scale, tvm::ffi::Tensor down,
                       tvm::ffi::Tensor down_scale, tvm::ffi::Tensor prepared_gu,
                       tvm::ffi::Tensor prepared_gu_sf, tvm::ffi::Tensor prepared_down,
                       tvm::ffi::Tensor prepared_down_sf, tvm::ffi::Tensor silu_table) {
  using namespace ffn_mx_native;
  tensor_contract(norm, norm, kDLFloat, 32, {5120});
  tensor_contract(gate_up, norm, kDLFloat8_e4m3fn, 8, {34816, 5120});
  tensor_contract(gate_up_scale, norm, kDLFloat, 32, {272, 40});
  tensor_contract(down, norm, kDLFloat8_e4m3fn, 8, {5120, 17408});
  tensor_contract(down_scale, norm, kDLFloat, 32, {40, 136});
  tensor_contract(prepared_gu, norm, kDLFloat8_e4m3fn, 8, {34816, 5120});
  tensor_contract(prepared_gu_sf, norm, kDLUInt, 8, {34816 * 160});
  tensor_contract(prepared_down, norm, kDLFloat8_e4m3fn, 8, {5120, 17408});
  tensor_contract(prepared_down_sf, norm, kDLUInt, 8, {5120 * 544});
  tensor_contract(silu_table, norm, kDLFloat, 32, {65536});
  all_disjoint({norm, gate_up, gate_up_scale, down, down_scale, prepared_gu, prepared_gu_sf,
                prepared_down, prepared_down_sf, silu_table});
  for (const auto &tensor : {prepared_gu, prepared_gu_sf, prepared_down, prepared_down_sf})
    lt_alignment(tensor);
  tvm::ffi::CUDADeviceGuard guard(norm.device().device_id);
  auto stream = stream_for(norm.device().device_id);
  outside_capture(stream);
  auto owner = tvm::ffi::make_object<MXWeightsObj>();
  owner->binding.initialize(norm.device().device_id);
  owner->tensors = {
      std::move(norm),           std::move(gate_up),       std::move(gate_up_scale),
      std::move(down),           std::move(down_scale),    std::move(prepared_gu),
      std::move(prepared_gu_sf), std::move(prepared_down), std::move(prepared_down_sf),
      std::move(silu_table)};
  const auto &w = owner->tensors;
  silu_table_initialize_kernel<<<256, 256, 0, stream>>>(static_cast<float *>(w[9].data_ptr()));
  check_cuda(cudaGetLastError());
  convert(w[1], w[2], w[5], w[6], true, stream);
  convert(w[3], w[4], w[7], w[8], true, stream);
  check_cuda(cudaStreamSynchronize(stream));
  return MXWeights(std::move(owner));
}

ffn_mx_native::MXContext ffn_mx_prepare(ffn_mx_native::MXWeights weights,
                                        tvm::ffi::Tensor workspace, tvm::ffi::Tensor input,
                                        tvm::ffi::Tensor input_sf, tvm::ffi::Tensor gate_up,
                                        tvm::ffi::Tensor product, tvm::ffi::Tensor product_sf,
                                        tvm::ffi::Tensor down, int rows) {
  using namespace ffn_mx_native;
  TVM_FFI_ICHECK(rows > 128 && rows <= 32768);
  int padded = (rows + 127) / 128 * 128;
  const auto &w = weights->tensors;
  tensor_contract(workspace, w[0], kDLUInt, 8, {WorkspaceBytes});
  tensor_contract(input, w[0], kDLFloat8_e4m3fn, 8, {padded, 5120});
  tensor_contract(input_sf, w[0], kDLUInt, 8, {padded * 160});
  tensor_contract(gate_up, w[0], kDLBfloat, 16, {padded, 34816});
  tensor_contract(product, w[0], kDLFloat8_e4m3fn, 8, {padded, 17408});
  tensor_contract(product_sf, w[0], kDLUInt, 8, {padded * 544});
  tensor_contract(down, w[0], kDLBfloat, 16, {padded, 5120});
  for (const auto &tensor : {workspace, input, input_sf, gate_up, product, product_sf, down})
    lt_alignment(tensor);
  std::vector<TensorView> tensors = {workspace, input,      input_sf, gate_up,
                                     product,   product_sf, down};
  for (const auto &tensor : w)
    tensors.push_back(tensor);
  all_disjoint(tensors);
  tvm::ffi::CUDADeviceGuard guard(w[0].device().device_id);
  auto stream = stream_for(w[0].device().device_id);
  outside_capture(stream);
  weights->binding.check(w[0].device().device_id);
  auto owner = tvm::ffi::make_object<MXContextObj>();
  owner->binding.initialize(w[0].device().device_id);
  owner->rows = rows;
  owner->padded_rows = padded;
  owner->buffers = {std::move(workspace), std::move(input),   std::move(input_sf),
                    std::move(gate_up),   std::move(product), std::move(product_sf),
                    std::move(down)};
  owner->seen_weights.push_back(weights);
  owner->upper.binding = owner->lower.binding = &owner->binding;
  check_lt(cublasLtCreate(&owner->handle));
  int matrix_rows = rows % 16 == 0 ? rows : padded;
  owner->upper.initialize(owner->handle, matrix_rows, 34816, 5120, w[6], owner->buffers[2]);
  owner->lower.initialize(owner->handle, matrix_rows, 5120, 17408, w[8], owner->buffers[5]);
  return MXContext(std::move(owner));
}

void ffn_mx(ffn_mx_native::MXContext owner, ffn_mx_native::MXWeights weights, tvm::ffi::Tensor x,
            tvm::ffi::Tensor residual, tvm::ffi::Tensor output, tvm::ffi::Tensor summed,
            tvm::ffi::Tensor quantized, tvm::ffi::Tensor input_scale) {
  using namespace ffn_mx_native;
  std::lock_guard<std::mutex> lock(owner->mutex);
  tvm::ffi::CUDADeviceGuard guard(owner->binding.device);
  owner_check(owner);
  weights->binding.check(owner->binding.device);
  int rows = owner->rows;
  for (const auto &tensor : {x, residual, output, summed})
    tensor_contract(tensor, owner->buffers[0], kDLBfloat, 16, {rows, 5120});
  tensor_contract(quantized, x, kDLFloat8_e4m3fn, 8, {rows, 5120});
  tensor_contract(input_scale, x, kDLFloat, 32, {rows, 40});
  std::vector<TensorView> tensors = {x, residual, output, summed, quantized, input_scale};
  for (const auto &tensor : owner->buffers)
    tensors.push_back(tensor);
  for (const auto &tensor : weights->tensors)
    tensors.push_back(tensor);
  all_disjoint(tensors);
  // Other retained layers must not alias this context's mutable buffers.
  for (const auto &previous : owner->seen_weights)
    for (const auto &weight : previous->tensors)
      for (size_t index = 0; index < 13; ++index)
        disjoint(weight, tensors[index]);
  auto stream = stream_for(owner->binding.device);
  retain(owner, weights, {x, residual, output, summed, quantized, input_scale}, stream);
  const auto &b = owner->buffers;
  const auto &w = weights->tensors;
  try {
    owner->upper.weight_scale(w[6]);
    owner->lower.weight_scale(w[8]);
    norm_mx_kernel<320><<<owner->padded_rows, 320, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(x.data_ptr()),
        static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
        static_cast<const float *>(w[0].data_ptr()),
        static_cast<__nv_bfloat16 *>(summed.data_ptr()), nullptr,
        static_cast<__nv_fp8_e4m3 *>(b[1].data_ptr()),
        static_cast<unsigned char *>(b[2].data_ptr()), rows);
    check_cuda(cudaGetLastError());
    owner->upper.run(owner->handle, w[5], b[1], b[3], b[0], stream);
    activate_mx_kernel<<<(owner->padded_rows * 136 + 31) / 32, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(b[3].data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(b[4].data_ptr()),
        static_cast<unsigned char *>(b[5].data_ptr()), rows, owner->padded_rows,
        static_cast<const float *>(w[9].data_ptr()));
    check_cuda(cudaGetLastError());
    bool direct = rows % 16 == 0 && reinterpret_cast<uintptr_t>(output.data_ptr()) % 256 == 0;
    owner->lower.run(owner->handle, w[7], b[4], direct ? output : b[6], b[0], stream);
    if (!direct)
      check_cuda(cudaMemcpyAsync(output.data_ptr(), b[6].data_ptr(), int64_t(rows) * 5120 * 2,
                                 cudaMemcpyDeviceToDevice, stream));
    cudaStreamCaptureStatus status;
    check_cuda(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusNone)
      owner->warmed = true;
  } catch (...) {
    owner->poisoned = true;
    throw;
  }
}

ffn_mx_native::MXGraphLease ffn_mx_graph_lease(ffn_mx_native::MXContext owner) {
  using namespace ffn_mx_native;
  std::lock_guard<std::mutex> lock(owner->mutex);
  tvm::ffi::CUDADeviceGuard guard(owner->binding.device);
  owner_check(owner);
  outside_capture(stream_for(owner->binding.device));
  TVM_FFI_ICHECK(!owner->lease && owner->warmed)
      << "MX graph lease requires warmup and no live graph";
  check_cuda(cudaDeviceSynchronize());
  owner->eager_tensors.clear();
  auto lease = tvm::ffi::make_object<MXGraphLeaseObj>(owner);
  owner->lease = lease.get();
  owner->capture_bound = false;
  owner->capture_stream = nullptr;
  owner->lease_thread = std::this_thread::get_id();
  return MXGraphLease(std::move(lease));
}

void ffn_mx_close(ffn_mx_native::MXContext owner) {
  using namespace ffn_mx_native;
  std::lock_guard<std::mutex> lock(owner->mutex);
  tvm::ffi::CUDADeviceGuard guard(owner->binding.device);
  owner->binding.check(owner->binding.device);
  outside_capture(stream_for(owner->binding.device));
  TVM_FFI_ICHECK(!owner->lease) << "MX context still has live graphs";
  if (!owner->handle)
    return;
  check_cuda(cudaDeviceSynchronize());
  owner->release();
}

tvm::ffi::Array<tvm::ffi::String> ffn_mx_images() {
  tvm::ffi::Array<tvm::ffi::String> paths;
  for (const void *symbol : {reinterpret_cast<const void *>(&cublasLtCreate),
                             reinterpret_cast<const void *>(&cublasLtMatmul),
                             reinterpret_cast<const void *>(&cublasLtMatmulAlgoGetHeuristic),
                             reinterpret_cast<const void *>(&cudaRuntimeGetVersion)}) {
    Dl_info image{};
    TVM_FFI_ICHECK(dladdr(symbol, &image) != 0 && image.dli_fname);
    paths.push_back(tvm::ffi::String(image.dli_fname));
  }
  return paths;
}

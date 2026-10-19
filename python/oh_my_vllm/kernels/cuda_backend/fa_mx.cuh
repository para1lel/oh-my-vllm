#pragma once
#include "fa_mx_plan.cuh"
#include "fa_output_mx.cuh"
#include "mx_fp8_convert.cuh"
#include "mx_fp8_norm.cuh"

namespace fa_mxio_native {
using mx_fp8::convert_kernel;

inline void lt_alignment(TensorView tensor) {
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 256 == 0)
      << "FA MXFP8 GEMM storage requires 256-byte alignment";
}

inline void owner_check(const MXContext &owner) {
  owner->binding.check(owner->binding.device);
  TVM_FFI_ICHECK(owner->handle && !owner->poisoned) << "FA MX context is closed or failed";
}

inline void add_tensor(std::vector<Tensor> &owned, const Tensor &tensor) {
  for (const auto &previous : owned)
    if (previous.data_ptr() == tensor.data_ptr() && previous.numel() == tensor.numel())
      return;
  owned.push_back(tensor);
}

inline void add_weights(std::vector<MXWeights> &owned, const MXWeights &weights) {
  for (const auto &previous : owned)
    if (previous.get() == weights.get())
      return;
  owned.push_back(weights);
}

inline bool same_storage(const Tensor &a, const Tensor &b) {
  return a.data_ptr() == b.data_ptr() && a.numel() == b.numel() && a.dtype() == b.dtype();
}

class Call {
  tvm::ffi::Optional<MXContext> context;
  tvm::ffi::Optional<MXWeights> weights;
  std::unique_lock<std::mutex> lock;
  int exceptions = std::uncaught_exceptions();
  bool capturing = false;

public:
  Call(tvm::ffi::Optional<MXContext> owner, tvm::ffi::Optional<MXWeights> layer, const Tensor &norm,
       const Tensor &weight, const Tensor &scale, const Tensor &output_weight,
       const Tensor &output_scale, std::initializer_list<Tensor> tensors, int rows,
       cudaStream_t stream)
      : context(std::move(owner)), weights(std::move(layer)) {
    TVM_FFI_ICHECK(context.has_value() == weights.has_value());
    if (!context.has_value())
      return;
    const auto &c = context.value();
    const auto &w = weights.value();
    lock = std::unique_lock<std::mutex>(c->mutex);
    owner_check(c);
    w->binding.check(c->binding.device);
    TVM_FFI_ICHECK(rows > 128 && rows <= 32768 && c->rows == rows && c->columns == 14336 &&
                   w->columns == 14336 && tensors.size() == 27);
    TVM_FFI_ICHECK(same_storage(norm, w->tensors[0]) && same_storage(weight, w->tensors[1]) &&
                   same_storage(scale, w->tensors[2]) &&
                   same_storage(output_weight, w->tensors[5]) &&
                   same_storage(output_scale, w->tensors[6]))
        << "FA MX weights do not match the original layer";
    for (const auto &buffer : c->buffers)
      for (const auto &tensor : tensors)
        disjoint(buffer, tensor);
    auto immutable = [&](const MXWeights &previous) {
      for (const auto &tensor : previous->tensors) {
        for (const auto &buffer : c->buffers)
          disjoint(tensor, buffer);
        size_t index = 0;
        for (const auto &argument : tensors) {
          bool read_only_weight = index <= 7 || (index >= 9 && index <= 14);
          if (!(read_only_weight && same_storage(tensor, argument)))
            disjoint(tensor, argument);
          ++index;
        }
      }
    };
    immutable(w);
    for (const auto &previous : c->seen_weights)
      if (previous.get() != w.get())
        immutable(previous);
    cudaStreamCaptureStatus status;
    unsigned long long id = 0;
    check_cuda(cudaStreamGetCaptureInfo(stream, &status, &id));
    if (status == cudaStreamCaptureStatusNone) {
      TVM_FFI_ICHECK(!c->lease) << "FA MX context has an exclusive graph lease";
      if (!c->eager_stream_bound) {
        c->eager_stream = stream;
        c->eager_stream_bound = true;
      }
      TVM_FFI_ICHECK(c->eager_stream == stream) << "FA MX eager calls require the same stream";
      for (const auto &tensor : tensors)
        add_tensor(c->eager_tensors, tensor);
    } else {
      TVM_FFI_ICHECK(status == cudaStreamCaptureStatusActive && c->lease &&
                     c->lease_thread == std::this_thread::get_id())
          << "FA MX capture requires its thread's graph lease";
      if (!c->capture_bound) {
        c->capture_id = id;
        c->capture_stream = stream;
        c->capture_bound = true;
      }
      TVM_FFI_ICHECK(c->capture_id == id && c->capture_stream == stream)
          << "FA MX lease belongs to another capture or stream";
      add_weights(c->lease->weights, w);
      for (const auto &tensor : tensors)
        add_tensor(c->lease->tensors, tensor);
      capturing = true;
    }
    add_weights(c->seen_weights, w);
  }

  ~Call() {
    if (active() && std::uncaught_exceptions() > exceptions)
      context.value()->poisoned = true;
  }

  bool active() const { return context.has_value(); }

  void normalize(const __nv_bfloat16 *input, const Tensor &residual, const Tensor &summed,
                 const Tensor &codes, const Tensor &scales, cudaStream_t stream) const {
    const auto &c = context.value();
    ffn_mx_native::norm_mx_kernel<320><<<c->padded_rows, 320, 0, stream>>>(
        input, static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
        static_cast<const float *>(weights.value()->tensors[0].data_ptr()),
        static_cast<__nv_bfloat16 *>(summed.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(codes.data_ptr()), static_cast<float *>(scales.data_ptr()),
        nullptr, static_cast<__nv_fp8_e4m3 *>(c->buffers[1].data_ptr()),
        static_cast<unsigned char *>(c->buffers[2].data_ptr()), c->rows);
    check_cuda(cudaGetLastError());
  }

  void project(const Tensor &output, cudaStream_t stream) const {
    const auto &c = context.value();
    const auto &w = weights.value()->tensors;
    c->projection.weight_scale(w[4]);
    bool direct =
        c->matrix_rows == c->rows && reinterpret_cast<uintptr_t>(output.data_ptr()) % 256 == 0;
    c->projection.run(c->handle, w[3], c->buffers[1], direct ? output : c->buffers[3],
                      c->buffers[0], stream);
    if (!direct)
      check_cuda(cudaMemcpyAsync(output.data_ptr(), c->buffers[3].data_ptr(),
                                 int64_t(c->rows) * c->columns * 2, cudaMemcpyDeviceToDevice,
                                 stream));
  }

  void quantize_output(const __nv_bfloat16 *attended, const Tensor &gate, const Tensor &codes,
                       const Tensor &scales, cudaStream_t stream) const {
    const auto &c = context.value();
    fa_output_mx_kernel<<<c->padded_rows * 6, 256, 0, stream>>>(
        attended, static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(codes.data_ptr()), static_cast<float *>(scales.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(c->buffers[4].data_ptr()),
        static_cast<unsigned char *>(c->buffers[5].data_ptr()), c->rows);
    check_cuda(cudaGetLastError());
  }

  void project_output(const Tensor &output, cudaStream_t stream) const {
    const auto &c = context.value();
    const auto &w = weights.value()->tensors;
    c->output_projection.weight_scale(w[8]);
    bool direct =
        c->matrix_rows == c->rows && reinterpret_cast<uintptr_t>(output.data_ptr()) % 256 == 0;
    c->output_projection.run(c->handle, w[7], c->buffers[4], direct ? output : c->buffers[6],
                             c->buffers[0], stream);
    if (!direct)
      check_cuda(cudaMemcpyAsync(output.data_ptr(), c->buffers[6].data_ptr(),
                                 int64_t(c->rows) * 5120 * 2, cudaMemcpyDeviceToDevice, stream));
  }

  void finish() const {
    if (active() && !capturing)
      context.value()->warmed = true;
  }
};
} // namespace fa_mxio_native

fa_mxio_native::MXWeights
fa_mx_prepare_weights(tvm::ffi::Tensor norm, tvm::ffi::Tensor qkvz, tvm::ffi::Tensor scale,
                      tvm::ffi::Tensor prepared, tvm::ffi::Tensor prepared_sf,
                      tvm::ffi::Tensor output, tvm::ffi::Tensor output_scale,
                      tvm::ffi::Tensor prepared_output, tvm::ffi::Tensor prepared_output_sf) {
  using namespace fa_mxio_native;
  tensor_contract(norm, norm, kDLFloat, 32, {5120});
  tensor_contract(qkvz, norm, kDLFloat8_e4m3fn, 8, {14336, 5120});
  tensor_contract(scale, norm, kDLFloat, 32, {112, 40});
  tensor_contract(prepared, norm, kDLFloat8_e4m3fn, 8, {14336, 5120});
  tensor_contract(prepared_sf, norm, kDLUInt, 8, {14336 * 160});
  tensor_contract(output, norm, kDLFloat8_e4m3fn, 8, {5120, 6144});
  tensor_contract(output_scale, norm, kDLFloat, 32, {40, 48});
  tensor_contract(prepared_output, norm, kDLFloat8_e4m3fn, 8, {5120, 6144});
  tensor_contract(prepared_output_sf, norm, kDLUInt, 8, {5120 * 192});
  all_disjoint({norm, qkvz, scale, prepared, prepared_sf, output, output_scale, prepared_output,
                prepared_output_sf});
  for (const auto &tensor : {prepared, prepared_sf, prepared_output, prepared_output_sf})
    lt_alignment(tensor);
  tvm::ffi::CUDADeviceGuard guard(norm.device().device_id);
  auto stream = stream_for(norm.device().device_id);
  outside_capture(stream);
  auto owner = tvm::ffi::make_object<MXWeightsObj>();
  owner->binding.initialize(norm.device().device_id);
  owner->tensors = {std::move(norm),
                    std::move(qkvz),
                    std::move(scale),
                    std::move(prepared),
                    std::move(prepared_sf),
                    std::move(output),
                    std::move(output_scale),
                    std::move(prepared_output),
                    std::move(prepared_output_sf)};
  const auto &w = owner->tensors;
  convert_kernel<<<(14336 * (5120 / 4) + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_fp8_e4m3 *>(w[1].data_ptr()),
      static_cast<const float *>(w[2].data_ptr()), static_cast<__nv_fp8_e4m3 *>(w[3].data_ptr()),
      static_cast<unsigned char *>(w[4].data_ptr()), 14336, 14336, 5120, true);
  check_cuda(cudaGetLastError());
  convert_kernel<<<(5120 * (6144 / 4) + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_fp8_e4m3 *>(w[5].data_ptr()),
      static_cast<const float *>(w[6].data_ptr()), static_cast<__nv_fp8_e4m3 *>(w[7].data_ptr()),
      static_cast<unsigned char *>(w[8].data_ptr()), 5120, 5120, 6144, true);
  check_cuda(cudaGetLastError());
  check_cuda(cudaStreamSynchronize(stream));
  return MXWeights(std::move(owner));
}

fa_mxio_native::MXContext fa_mx_prepare(fa_mxio_native::MXWeights weights,
                                        tvm::ffi::Tensor workspace, tvm::ffi::Tensor input,
                                        tvm::ffi::Tensor input_sf, tvm::ffi::Tensor projected,
                                        tvm::ffi::Tensor attended_input,
                                        tvm::ffi::Tensor attended_sf, tvm::ffi::Tensor output,
                                        int rows) {
  using namespace fa_mxio_native;
  TVM_FFI_ICHECK(rows > 128 && rows <= 32768 && weights->columns == 14336);
  int padded = (rows + 127) / 128 * 128;
  const auto &w = weights->tensors;
  tensor_contract(workspace, w[0], kDLUInt, 8, {WorkspaceBytes});
  tensor_contract(input, w[0], kDLFloat8_e4m3fn, 8, {padded, 5120});
  tensor_contract(input_sf, w[0], kDLUInt, 8, {padded * 160});
  tensor_contract(projected, w[0], kDLBfloat, 16, {padded, 14336});
  tensor_contract(attended_input, w[0], kDLFloat8_e4m3fn, 8, {padded, 6144});
  tensor_contract(attended_sf, w[0], kDLUInt, 8, {padded * 192});
  tensor_contract(output, w[0], kDLBfloat, 16, {padded, 5120});
  for (const auto &tensor :
       {workspace, input, input_sf, projected, attended_input, attended_sf, output})
    lt_alignment(tensor);
  std::vector<TensorView> tensors = {workspace,      input,       input_sf, projected,
                                     attended_input, attended_sf, output};
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
  owner->matrix_rows = rows % 16 == 0 ? rows : padded;
  owner->columns = weights->columns;
  owner->buffers = {std::move(workspace), std::move(input),          std::move(input_sf),
                    std::move(projected), std::move(attended_input), std::move(attended_sf),
                    std::move(output)};
  owner->seen_weights.push_back(weights);
  owner->projection.binding = &owner->binding;
  owner->output_projection.binding = &owner->binding;
  check_lt(cublasLtCreate(&owner->handle));
  owner->projection.initialize(owner->handle, owner->matrix_rows, owner->columns, 5120, w[4],
                               owner->buffers[2]);
  owner->output_projection.initialize(owner->handle, owner->matrix_rows, 5120, 6144, w[8],
                                      owner->buffers[5]);
  return MXContext(std::move(owner));
}

fa_mxio_native::MXGraphLease fa_mx_graph_lease(fa_mxio_native::MXContext owner) {
  using namespace fa_mxio_native;
  std::lock_guard<std::mutex> lock(owner->mutex);
  tvm::ffi::CUDADeviceGuard guard(owner->binding.device);
  owner_check(owner);
  outside_capture(stream_for(owner->binding.device));
  TVM_FFI_ICHECK(!owner->lease && owner->warmed)
      << "FA MX graph lease requires warmup and no live graph";
  check_cuda(cudaDeviceSynchronize());
  owner->eager_tensors.clear();
  auto lease = tvm::ffi::make_object<MXGraphLeaseObj>(owner);
  owner->lease = lease.get();
  owner->capture_bound = false;
  owner->capture_stream = nullptr;
  owner->lease_thread = std::this_thread::get_id();
  return MXGraphLease(std::move(lease));
}

void fa_mx_close(fa_mxio_native::MXContext owner) {
  using namespace fa_mxio_native;
  std::lock_guard<std::mutex> lock(owner->mutex);
  tvm::ffi::CUDADeviceGuard guard(owner->binding.device);
  owner->binding.check(owner->binding.device);
  outside_capture(stream_for(owner->binding.device));
  TVM_FFI_ICHECK(!owner->lease) << "FA MX context still has live graphs";
  if (!owner->handle)
    return;
  check_cuda(cudaDeviceSynchronize());
  owner->release();
}

tvm::ffi::Array<tvm::ffi::String> fa_mx_images() {
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

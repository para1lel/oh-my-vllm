#pragma once

#include "common.cuh"
#include "vision_mlp.cuh"
#include "vision_lookup_init.cuh"
#include <mutex>
#include <thread>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/optional.h>

constexpr int64_t VisionClassicWorkspaceBytes = 32 * 1024 * 1024;
using VisionContextGet = CUresult(CUDAAPI *)(CUcontext *);
using VisionContextPush = CUresult(CUDAAPI *)(CUcontext);

inline void vision_classic_check(cublasStatus_t status) {
  TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS) << "cuBLAS status " << int(status);
}

inline void *vision_classic_driver_entry(const char *name) {
  void *entry = nullptr;
  cudaDriverEntryPointQueryResult status;
  check_cuda(cudaGetDriverEntryPointByVersion(name, &entry, 4000, cudaEnableDefault, &status));
  TVM_FFI_ICHECK(status == cudaDriverEntryPointSuccess && entry);
  return entry;
}

inline void vision_classic_driver_check(CUresult status) {
  TVM_FFI_ICHECK(status == CUDA_SUCCESS) << "CUDA driver status " << int(status);
}

class VisionClassicOwnerObj : public tvm::ffi::Object {
public:
  mutable cublasHandle_t handle = nullptr;
  mutable tvm::ffi::Tensor workspace;
  int device = -1;
  CUcontext context = nullptr;
  VisionContextGet get_current = nullptr;
  VisionContextPush push_current = nullptr;
  VisionContextGet pop_current = nullptr;
  mutable const cutlass::bfloat16_t *gelu_table = nullptr;
  mutable std::mutex lifetime_mutex;
  mutable bool graph_leased = false;
  mutable bool capture_bound = false;
  mutable unsigned long long capture_id = 0;
  mutable std::thread::id lease_thread;

  ~VisionClassicOwnerObj() {
    if (handle || gelu_table) {
      if (push_current && pop_current && push_current(context) == CUDA_SUCCESS) {
        cudaDeviceSynchronize();
        if (handle)
          cublasDestroy(handle);
        if (gelu_table)
          cudaFree(const_cast<cutlass::bfloat16_t *>(gelu_table));
        workspace = tvm::ffi::Tensor();
        CUcontext popped = nullptr;
        pop_current(&popped);
      }
    }
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.VisionClassicOwner", VisionClassicOwnerObj,
                                    tvm::ffi::Object);
};

class VisionClassicOwner : public tvm::ffi::ObjectRef {
public:
  explicit VisionClassicOwner(tvm::ffi::ObjectPtr<VisionClassicOwnerObj> object)
      : ObjectRef(std::move(object)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VisionClassicOwner, tvm::ffi::ObjectRef,
                                                VisionClassicOwnerObj);
};

inline void vision_classic_context(const VisionClassicOwner &owner, int device) {
  TVM_FFI_ICHECK(owner->handle && owner->workspace.defined() && owner->gelu_table &&
                 owner->device == device);
  CUcontext current = nullptr;
  vision_classic_driver_check(owner->get_current(&current));
  TVM_FFI_ICHECK(current && current == owner->context)
      << "Vision classic owner is bound to a different CUDA context";
}

inline void vision_classic_outside_capture(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  check_cuda(cudaStreamIsCapturing(stream, &status));
  TVM_FFI_ICHECK(status == cudaStreamCaptureStatusNone)
      << "Vision classic preparation and cleanup must precede or follow capture";
}

class VisionClassicGraphLeaseObj : public tvm::ffi::Object {
public:
  VisionClassicOwner owner;

  explicit VisionClassicGraphLeaseObj(VisionClassicOwner context) : owner(std::move(context)) {
    std::lock_guard<std::mutex> lock(owner->lifetime_mutex);
    vision_classic_context(owner, owner->device);
    TVM_FFI_ICHECK(!owner->graph_leased) << "Vision context already has an active graph lease";
    owner->graph_leased = true;
    owner->capture_bound = false;
    owner->lease_thread = std::this_thread::get_id();
  }

  ~VisionClassicGraphLeaseObj() {
    std::lock_guard<std::mutex> lock(owner->lifetime_mutex);
    owner->graph_leased = false;
    owner->capture_bound = false;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.VisionClassicGraphLease",
                                    VisionClassicGraphLeaseObj, tvm::ffi::Object);
};

class VisionClassicGraphLease : public tvm::ffi::ObjectRef {
public:
  explicit VisionClassicGraphLease(tvm::ffi::ObjectPtr<VisionClassicGraphLeaseObj> object)
      : ObjectRef(std::move(object)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VisionClassicGraphLease, tvm::ffi::ObjectRef,
                                                VisionClassicGraphLeaseObj);
};

VisionClassicGraphLease vision_classic_graph_lease(VisionClassicOwner owner) {
  tvm::ffi::CUDADeviceGuard guard(owner->device);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, owner->device));
  vision_classic_outside_capture(stream);
  return VisionClassicGraphLease(
      tvm::ffi::make_object<VisionClassicGraphLeaseObj>(std::move(owner)));
}

inline void vision_classic_capture(const VisionClassicOwner &owner, cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  unsigned long long id = 0;
  check_cuda(cudaStreamGetCaptureInfo(stream, &status, &id));
  if (status == cudaStreamCaptureStatusNone)
    return;
  TVM_FFI_ICHECK(status == cudaStreamCaptureStatusActive);
  TVM_FFI_ICHECK(owner->graph_leased) << "Vision capture requires a graph owner lease";
  if (!owner->capture_bound) {
    TVM_FFI_ICHECK(owner->lease_thread == std::this_thread::get_id());
    owner->capture_id = id;
    owner->capture_bound = true;
  }
  TVM_FFI_ICHECK(owner->capture_id == id) << "Vision owner lease is bound to another graph";
}

inline void vision_classic_storage(const VisionClassicOwner &owner,
                                   std::initializer_list<TensorView> tensors) {
  auto table = reinterpret_cast<uintptr_t>(owner->gelu_table);
  for (auto tensor : tensors) {
    disjoint(tensor, owner->workspace);
    auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    TVM_FFI_ICHECK(begin + tvm::ffi::GetDataSize(tensor) <= table || table + 131072 <= begin)
        << "Vision tensor overlaps the private GELU table";
  }
}

VisionClassicOwner vision_classic_prepare(tvm::ffi::Tensor workspace) {
  tensor_contract(workspace, workspace, kDLUInt, 8, {VisionClassicWorkspaceBytes});
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 256 == 0);
  tvm::ffi::CUDADeviceGuard guard(workspace.device().device_id);
  auto stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, workspace.device().device_id));
  vision_classic_outside_capture(stream);
  auto owner = tvm::ffi::make_object<VisionClassicOwnerObj>();
  owner->workspace = std::move(workspace);
  owner->device = owner->workspace.device().device_id;
  owner->get_current =
      reinterpret_cast<VisionContextGet>(vision_classic_driver_entry("cuCtxGetCurrent"));
  owner->push_current =
      reinterpret_cast<VisionContextPush>(vision_classic_driver_entry("cuCtxPushCurrent"));
  owner->pop_current =
      reinterpret_cast<VisionContextGet>(vision_classic_driver_entry("cuCtxPopCurrent"));
  vision_classic_driver_check(owner->get_current(&owner->context));
  TVM_FFI_ICHECK(owner->context);
  vision_classic_check(cublasCreate(&owner->handle));
  vision_classic_check(cublasSetPointerMode(owner->handle, CUBLAS_POINTER_MODE_HOST));
  vision_classic_check(cublasSetMathMode(
      owner->handle,
      cublasMath_t(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  vision_classic_check(cublasSetStream(owner->handle, stream));
  vision_classic_check(
      cublasSetWorkspace(owner->handle, owner->workspace.data_ptr(), VisionClassicWorkspaceBytes));
  void *table = nullptr;
  check_cuda(cudaMalloc(&table, 131072));
  owner->gelu_table = static_cast<cutlass::bfloat16_t *>(table);
  vision_lookup_initialize_kernel<<<256, 256, 0, stream>>>(
      static_cast<cutlass::bfloat16_t *>(table));
  check_cuda(cudaGetLastError());
  check_cuda(cudaStreamSynchronize(stream));
  VisionClassicOwner result(std::move(owner));
  vision_classic_context(result, result->device);
  return result;
}

void vision_classic_close(VisionClassicOwner owner) {
  tvm::ffi::CUDADeviceGuard guard(owner->device);
  std::lock_guard<std::mutex> lock(owner->lifetime_mutex);
  TVM_FFI_ICHECK(!owner->graph_leased) << "Vision context still has live graphs";
  if (!owner->handle)
    return;
  vision_classic_context(owner, owner->device);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, owner->device));
  vision_classic_outside_capture(stream);
  check_cuda(cudaDeviceSynchronize());
  vision_classic_check(cublasSetStream(owner->handle, stream));
  auto status = cublasDestroy(owner->handle);
  if (status == CUBLAS_STATUS_SUCCESS) {
    owner->handle = nullptr;
    check_cuda(cudaFree(const_cast<cutlass::bfloat16_t *>(owner->gelu_table)));
    owner->gelu_table = nullptr;
    owner->workspace = tvm::ffi::Tensor();
  }
  vision_classic_check(status);
}

tvm::ffi::Array<int64_t> vision_classic_state(VisionClassicOwner owner) {
  std::lock_guard<std::mutex> lock(owner->lifetime_mutex);
  TVM_FFI_ICHECK(owner->handle && owner->workspace.defined());
  tvm::ffi::CUDADeviceGuard guard(owner->device);
  vision_classic_context(owner, owner->device);
  cublasMath_t math;
  cublasPointerMode_t pointer;
  cudaStream_t stream;
  vision_classic_check(cublasGetMathMode(owner->handle, &math));
  vision_classic_check(cublasGetPointerMode(owner->handle, &pointer));
  vision_classic_check(cublasGetStream(owner->handle, &stream));
  return {owner->device,
          int64_t(math),
          int64_t(pointer),
          VisionClassicWorkspaceBytes,
          int64_t(reinterpret_cast<uintptr_t>(owner->workspace.data_ptr())),
          int64_t(reinterpret_cast<uintptr_t>(stream)),
          int64_t(reinterpret_cast<uintptr_t>(owner->context))};
}

inline void vision_classic_linear(const VisionClassicOwner &owner, const __nv_bfloat16 *input,
                                  const __nv_bfloat16 *weight, float *output, int rows, int columns,
                                  int inner, int device, cudaStream_t stream) {
  vision_classic_context(owner, device);
  vision_classic_check(cublasSetStream(owner->handle, stream));
  vision_classic_check(
      cublasSetWorkspace(owner->handle, owner->workspace.data_ptr(), VisionClassicWorkspaceBytes));
  float alpha = 1.f, beta = 0.f;
  vision_classic_check(cublasGemmEx(owner->handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, inner,
                                    &alpha, weight, CUDA_R_16BF, inner, input, CUDA_R_16BF, inner,
                                    &beta, output, CUDA_R_32F, columns, CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT));
}

inline void vision_classic_down(const VisionClassicOwner &owner, const __nv_bfloat16 *input,
                                const __nv_bfloat16 *weight, float *output, int rows, int device,
                                cudaStream_t stream) {
  vision_classic_linear(owner, input, weight, output, rows, 1152, 4304, device, stream);
}

tvm::ffi::Array<tvm::ffi::String> vision_classic_images() {
  tvm::ffi::Array<tvm::ffi::String> paths;
  for (const void *symbol : {reinterpret_cast<const void *>(&cublasCreate_v2),
                             reinterpret_cast<const void *>(&cublasLtCreate),
                             reinterpret_cast<const void *>(&cudaRuntimeGetVersion)}) {
    Dl_info image{};
    TVM_FFI_ICHECK(dladdr(symbol, &image) != 0 && image.dli_fname);
    paths.push_back(tvm::ffi::String(image.dli_fname));
  }
  return paths;
}

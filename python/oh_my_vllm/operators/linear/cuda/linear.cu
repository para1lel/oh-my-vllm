#include "../../runtime/cuda/contracts.cuh"
#include <cuda.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <mutex>
#include <thread>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>

constexpr int64_t LinearWorkspaceBytes = 32 * 1024 * 1024;
using ContextGet = CUresult(CUDAAPI *)(CUcontext *);
using ContextPush = CUresult(CUDAAPI *)(CUcontext);

inline void linear_check(cublasStatus_t status) {
  TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS) << "cuBLAS status " << int(status);
}

inline void *linear_entry(const char *name) {
  void *entry = nullptr;
  cudaDriverEntryPointQueryResult status;
  check_cuda(cudaGetDriverEntryPointByVersion(name, &entry, 4000, cudaEnableDefault, &status));
  TVM_FFI_ICHECK(status == cudaDriverEntryPointSuccess && entry);
  return entry;
}

inline void linear_outside_capture(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  check_cuda(cudaStreamIsCapturing(stream, &status));
  TVM_FFI_ICHECK(status == cudaStreamCaptureStatusNone)
      << "prepare and lease transitions must be outside capture";
}

class LinearOwnerObj : public tvm::ffi::Object {
public:
  mutable cublasHandle_t handle = nullptr;
  tvm::ffi::Tensor scratch;
  int device = -1;
  CUcontext context = nullptr;
  ContextGet get_current = nullptr, pop_current = nullptr;
  ContextPush push_current = nullptr;
  cudaStream_t stream = nullptr;
  std::thread::id thread;
  mutable std::mutex mutex;
  mutable bool leased = false, capture_bound = false;
  mutable unsigned long long capture_id = 0;

  ~LinearOwnerObj() {
    if (!handle)
      return;
    bool pushed = push_current && push_current(context) == CUDA_SUCCESS;
    bool finished = pushed && cudaDeviceSynchronize() == cudaSuccess &&
                    cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS;
    if (!finished) {
      // Keep scratch mapped on failed completion; worker teardown owns recovery.
      auto *retained = new tvm::ffi::Tensor(std::move(scratch));
      (void)retained;
    }
    if (pushed) {
      CUcontext popped = nullptr;
      pop_current(&popped);
    }
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.LinearOwner", LinearOwnerObj, tvm::ffi::Object);
};

class LinearOwner : public tvm::ffi::ObjectRef {
public:
  explicit LinearOwner(tvm::ffi::ObjectPtr<LinearOwnerObj> object) : ObjectRef(std::move(object)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LinearOwner, tvm::ffi::ObjectRef, LinearOwnerObj);
};

inline void linear_context(const LinearOwner &owner, cudaStream_t stream) {
  CUcontext current = nullptr;
  TVM_FFI_ICHECK(owner->handle && owner->get_current(&current) == CUDA_SUCCESS &&
                 current == owner->context && stream == owner->stream &&
                 owner->thread == std::this_thread::get_id())
      << "linear workspace requires its preparation context, caller thread, and stream";
}

LinearOwner linear_prepare(tvm::ffi::Tensor scratch) {
  tensor_contract(scratch, scratch, kDLUInt, 8, {LinearWorkspaceBytes});
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(scratch.data_ptr()) % 256 == 0);
  tvm::ffi::CUDADeviceGuard guard(scratch.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, scratch.device().device_id));
  linear_outside_capture(stream);
  auto owner = tvm::ffi::make_object<LinearOwnerObj>();
  owner->scratch = std::move(scratch);
  owner->device = owner->scratch.device().device_id;
  owner->stream = stream;
  owner->thread = std::this_thread::get_id();
  owner->get_current = reinterpret_cast<ContextGet>(linear_entry("cuCtxGetCurrent"));
  owner->push_current = reinterpret_cast<ContextPush>(linear_entry("cuCtxPushCurrent"));
  owner->pop_current = reinterpret_cast<ContextGet>(linear_entry("cuCtxPopCurrent"));
  TVM_FFI_ICHECK(owner->get_current(&owner->context) == CUDA_SUCCESS && owner->context);
  linear_check(cublasCreate(&owner->handle));
  linear_check(cublasSetPointerMode(owner->handle, CUBLAS_POINTER_MODE_HOST));
  linear_check(cublasSetMathMode(owner->handle, CUBLAS_PEDANTIC_MATH));
  linear_check(cublasSetStream(owner->handle, stream));
  linear_check(cublasSetWorkspace(owner->handle, owner->scratch.data_ptr(), LinearWorkspaceBytes));
  return LinearOwner(std::move(owner));
}

class LinearLeaseObj : public tvm::ffi::Object {
public:
  LinearOwner owner;

  explicit LinearLeaseObj(LinearOwner value) : owner(std::move(value)) {
    std::lock_guard<std::mutex> lock(owner->mutex);
    auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, owner->device));
    linear_context(owner, stream);
    linear_outside_capture(stream);
    TVM_FFI_ICHECK(!owner->leased) << "linear workspace already has a graph lease";
    owner->leased = true;
    owner->capture_bound = false;
  }

  ~LinearLeaseObj() {
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->leased = false;
    owner->capture_bound = false;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.LinearLease", LinearLeaseObj, tvm::ffi::Object);
};

class LinearLease : public tvm::ffi::ObjectRef {
public:
  explicit LinearLease(tvm::ffi::ObjectPtr<LinearLeaseObj> object) : ObjectRef(std::move(object)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LinearLease, tvm::ffi::ObjectRef, LinearLeaseObj);
};

LinearLease linear_graph_lease(LinearOwner owner) {
  tvm::ffi::CUDADeviceGuard guard(owner->device);
  return LinearLease(tvm::ffi::make_object<LinearLeaseObj>(std::move(owner)));
}

void linear_project(LinearOwner owner, TensorView input, TensorView weight, TensorView output,
                    int64_t groups) {
  TVM_FFI_ICHECK(groups == 1 || groups == 2 || groups == 8);
  TVM_FFI_ICHECK(input.ndim() == (groups == 1 ? 2 : 3) && weight.ndim() == input.ndim());
  int64_t rows = input.size(0), width = input.size(input.ndim() - 1);
  int64_t columns = weight.size(weight.ndim() - 2);
  bool f32 = input.dtype().code == kDLFloat;
  bool output_f32 = output.dtype().code == kDLFloat;
  TVM_FFI_ICHECK(rows >= 0 && rows <= 384 && width >= 1 && width <= 8192 && columns >= 1 &&
                 columns <= 129280 && (!f32 || output_f32));
  if (groups != 1)
    TVM_FFI_ICHECK(!f32 && !output_f32 && width == 4096 && columns == 1024);
  int code = f32 ? kDLFloat : kDLBfloat, bits = f32 ? 32 : 16;
  int out_code = output_f32 ? kDLFloat : kDLBfloat, out_bits = output_f32 ? 32 : 16;
  if (groups == 1) {
    tensor_contract(input, owner->scratch, code, bits, {rows, width});
    tensor_contract(weight, input, code, bits, {columns, width});
    tensor_contract(output, input, out_code, out_bits, {rows, columns});
  } else {
    tensor_contract(input, owner->scratch, code, bits, {rows, groups, width});
    tensor_contract(weight, input, code, bits, {groups, columns, width});
    tensor_contract(output, input, out_code, out_bits, {rows, groups, columns});
  }
  disjoint(input, output);
  disjoint(weight, output);
  disjoint(input, weight);
  for (auto tensor : {input, weight, output})
    disjoint(tensor, owner->scratch);
  tvm::ffi::CUDADeviceGuard guard(owner->device);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, owner->device));
  std::lock_guard<std::mutex> lock(owner->mutex);
  linear_context(owner, owner->stream);
  cudaStreamCaptureStatus status;
  unsigned long long id = 0;
  check_cuda(cudaStreamGetCaptureInfo(stream, &status, &id));
  if (status != cudaStreamCaptureStatusNone) {
    TVM_FFI_ICHECK(status == cudaStreamCaptureStatusActive && owner->leased)
        << "linear capture requires a graph lease";
    if (!owner->capture_bound) {
      owner->capture_id = id;
      owner->capture_bound = true;
    }
    TVM_FFI_ICHECK(id == owner->capture_id) << "linear lease belongs to another graph";
  } else {
    TVM_FFI_ICHECK(stream == owner->stream) << "linear execution requires the preparation stream";
  }
  if (!rows)
    return;
  linear_check(cublasSetStream(owner->handle, stream));
  linear_check(cublasSetWorkspace(owner->handle, owner->scratch.data_ptr(), LinearWorkspaceBytes));
  float alpha = 1.f, beta = 0.f;
  cudaDataType_t kind = f32 ? CUDA_R_32F : CUDA_R_16BF;
  cudaDataType_t out_kind = output_f32 ? CUDA_R_32F : CUDA_R_16BF;
  for (int64_t group = 0; group < groups; ++group) {
    auto a = static_cast<const char *>(weight.data_ptr()) + group * columns * width * (bits / 8);
    auto b = static_cast<const char *>(input.data_ptr()) + group * width * (bits / 8);
    auto c = static_cast<char *>(output.data_ptr()) + group * columns * (out_bits / 8);
    linear_check(cublasGemmEx(owner->handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, width, &alpha,
                              a, kind, width, b, kind, groups * width, &beta, c, out_kind,
                              groups * columns, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
  }
}

tvm::ffi::Array<tvm::ffi::String> linear_images() {
  tvm::ffi::Array<tvm::ffi::String> paths;
  for (const void *symbol : {reinterpret_cast<const void *>(&cublasCreate_v2),
                             reinterpret_cast<const void *>(&cublasLtCreate),
                             reinterpret_cast<const void *>(&cudaRuntimeGetVersion)}) {
    Dl_info image{};
    TVM_FFI_ICHECK(dladdr(symbol, &image) && image.dli_fname);
    paths.push_back(tvm::ffi::String(image.dli_fname));
  }
  return paths;
}

#pragma once
#include "common.cuh"
#include <cuda.h>
#include <cublasLt.h>
#include <array>
#include <mutex>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>

namespace ffn_mx_native {
using Tensor = tvm::ffi::Tensor;
constexpr int64_t WorkspaceBytes = 128 * 1024 * 1024;

inline void check_lt(cublasStatus_t status) {
  TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS) << "cuBLASLt status " << int(status);
}

inline void outside_capture(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  check_cuda(cudaStreamIsCapturing(stream, &status));
  TVM_FFI_ICHECK(status == cudaStreamCaptureStatusNone)
      << "MX preparation and cleanup require no capture";
}

struct Binding {
  using Get = CUresult(CUDAAPI *)(CUcontext *);
  using Push = CUresult(CUDAAPI *)(CUcontext);
  int device = -1;
  CUcontext context = nullptr;
  Get get = nullptr, pop = nullptr;
  Push push = nullptr;

  static void *entry(const char *name) {
    void *result = nullptr;
    cudaDriverEntryPointQueryResult status;
    check_cuda(cudaGetDriverEntryPointByVersion(name, &result, 4000, cudaEnableDefault, &status));
    TVM_FFI_ICHECK(status == cudaDriverEntryPointSuccess && result);
    return result;
  }

  void initialize(int ordinal) {
    device = ordinal;
    get = reinterpret_cast<Get>(entry("cuCtxGetCurrent"));
    push = reinterpret_cast<Push>(entry("cuCtxPushCurrent"));
    pop = reinterpret_cast<Get>(entry("cuCtxPopCurrent"));
    TVM_FFI_ICHECK(get(&context) == CUDA_SUCCESS && context);
  }

  void check(int ordinal) const {
    CUcontext current = nullptr;
    TVM_FFI_ICHECK(get && ordinal == device && get(&current) == CUDA_SUCCESS && current == context)
        << "MX owner is bound to another CUDA context";
  }
};

struct Scope {
  const Binding &binding;
  bool active;

  explicit Scope(const Binding &value)
      : binding(value),
        active(value.push && value.context && value.push(value.context) == CUDA_SUCCESS) {}

  ~Scope() {
    if (active) {
      CUcontext previous = nullptr;
      binding.pop(&previous);
    }
  }
};

struct Plan {
  const Binding *binding = nullptr;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algo{};
  Plan() = default;
  Plan(const Plan &) = delete;
  Plan &operator=(const Plan &) = delete;

  ~Plan() {
    if (binding && (desc || a || b || c)) {
      Scope scope(*binding);
      if (scope.active)
        clear();
    }
  }

  void clear() noexcept {
    if (a) {
      cublasLtMatrixLayoutDestroy(a);
      a = nullptr;
    }
    if (b) {
      cublasLtMatrixLayoutDestroy(b);
      b = nullptr;
    }
    if (c) {
      cublasLtMatrixLayoutDestroy(c);
      c = nullptr;
    }
    if (desc) {
      cublasLtMatmulDescDestroy(desc);
      desc = nullptr;
    }
  }

  void weight_scale(const Tensor &scales) {
    void *pointer = scales.data_ptr();
    check_lt(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pointer,
                                            sizeof(pointer)));
  }

  void initialize(cublasLtHandle_t handle, int rows, int columns, int inner,
                  const Tensor &weight_scales, const Tensor &input_scales) {
    check_lt(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    check_lt(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    check_lt(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    auto mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    check_lt(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode,
                                            sizeof(mode)));
    check_lt(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode,
                                            sizeof(mode)));
    weight_scale(weight_scales);
    void *sb = input_scales.data_ptr();
    check_lt(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb,
                                            sizeof(sb)));
    check_lt(cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, inner, columns, inner));
    check_lt(cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, inner, rows, inner));
    check_lt(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, columns, rows, columns));

    struct Preference {
      cublasLtMatmulPreference_t value = nullptr;

      ~Preference() {
        if (value)
          cublasLtMatmulPreferenceDestroy(value);
      }
    } preference;

    check_lt(cublasLtMatmulPreferenceCreate(&preference.value));
    size_t bytes = WorkspaceBytes;
    check_lt(cublasLtMatmulPreferenceSetAttribute(
        preference.value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    check_lt(cublasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, c, preference.value, 1, &result,
                                            &found));
    TVM_FFI_ICHECK(found == 1) << "no MXFP8 GEMM heuristic";
    check_lt(result.state);
    TVM_FFI_ICHECK(result.workspaceSize <= WorkspaceBytes) << "MXFP8 GEMM exceeds owned workspace";
    algo = result.algo;
  }

  void run(cublasLtHandle_t handle, const Tensor &weight, const Tensor &input, const Tensor &output,
           const Tensor &workspace, cudaStream_t stream) {
    float alpha = 1.f, beta = 0.f;
    check_lt(cublasLtMatmul(handle, desc, &alpha, weight.data_ptr(), a, input.data_ptr(), b, &beta,
                            output.data_ptr(), c, output.data_ptr(), c, &algo, workspace.data_ptr(),
                            WorkspaceBytes, stream));
  }
};

inline void all_disjoint(const std::vector<TensorView> &tensors) {
  for (size_t i = 0; i < tensors.size(); ++i)
    for (size_t j = i + 1; j < tensors.size(); ++j)
      disjoint(tensors[i], tensors[j]);
}

inline cudaStream_t stream_for(int device) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
}

class MXWeightsObj : public tvm::ffi::Object {
public:
  Binding binding;
  std::array<Tensor, 10> tensors;

  ~MXWeightsObj() {
    Scope scope(binding);
    if (scope.active) {
      cudaDeviceSynchronize();
      for (auto &tensor : tensors)
        tensor = Tensor();
    }
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.MXLutWeights", MXWeightsObj, tvm::ffi::Object);
};

class MXWeights : public tvm::ffi::ObjectRef {
public:
  explicit MXWeights(tvm::ffi::ObjectPtr<MXWeightsObj> object) : ObjectRef(std::move(object)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MXWeights, tvm::ffi::ObjectRef, MXWeightsObj);
};
class MXGraphLeaseObj;

class MXContextObj : public tvm::ffi::Object {
public:
  Binding binding;
  int rows = 0, padded_rows = 0;
  mutable std::array<Tensor, 7> buffers;
  mutable std::vector<MXWeights> seen_weights;
  mutable std::vector<Tensor> eager_tensors;
  mutable std::mutex mutex;
  mutable cublasLtHandle_t handle = nullptr;
  mutable Plan upper, lower;
  mutable bool poisoned = false, warmed = false, eager_stream_bound = false;
  mutable cudaStream_t eager_stream = nullptr, capture_stream = nullptr;
  mutable MXGraphLeaseObj *lease = nullptr;
  mutable bool capture_bound = false;
  mutable unsigned long long capture_id = 0;
  mutable std::thread::id lease_thread;

  void release() const noexcept {
    upper.clear();
    lower.clear();
    if (handle) {
      cublasLtDestroy(handle);
      handle = nullptr;
    }
    eager_tensors.clear();
    seen_weights.clear();
    for (auto &buffer : buffers)
      buffer = Tensor();
  }

  ~MXContextObj() {
    Scope scope(binding);
    if (scope.active) {
      cudaDeviceSynchronize();
      release();
    }
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.MXLutContext", MXContextObj, tvm::ffi::Object);
};

class MXContext : public tvm::ffi::ObjectRef {
public:
  explicit MXContext(tvm::ffi::ObjectPtr<MXContextObj> object) : ObjectRef(std::move(object)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MXContext, tvm::ffi::ObjectRef, MXContextObj);
};

class MXGraphLeaseObj : public tvm::ffi::Object {
public:
  MXContext owner;
  std::vector<MXWeights> weights;
  std::vector<Tensor> tensors;

  explicit MXGraphLeaseObj(MXContext context) : owner(std::move(context)) {}

  ~MXGraphLeaseObj() {
    std::lock_guard<std::mutex> lock(owner->mutex);
    Scope scope(owner->binding);
    if (scope.active) {
      cudaDeviceSynchronize();
      weights.clear();
      tensors.clear();
    }
    if (owner->lease == this) {
      owner->lease = nullptr;
      owner->capture_bound = false;
      owner->capture_stream = nullptr;
    }
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.MXLutGraphLease", MXGraphLeaseObj,
                                    tvm::ffi::Object);
};

class MXGraphLease : public tvm::ffi::ObjectRef {
public:
  explicit MXGraphLease(tvm::ffi::ObjectPtr<MXGraphLeaseObj> object)
      : ObjectRef(std::move(object)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MXGraphLease, tvm::ffi::ObjectRef, MXGraphLeaseObj);
};
} // namespace ffn_mx_native

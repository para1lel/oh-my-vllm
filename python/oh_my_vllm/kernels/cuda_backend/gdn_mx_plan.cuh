#pragma once
#include "mx_fp8_plan.cuh"
#include <exception>
#include <tvm/ffi/optional.h>

namespace gdn_mx_native {
using ffn_mx_native::all_disjoint;
using ffn_mx_native::Binding;
using ffn_mx_native::check_lt;
using ffn_mx_native::outside_capture;
using ffn_mx_native::Plan;
using ffn_mx_native::Scope;
using ffn_mx_native::stream_for;
using ffn_mx_native::Tensor;
using ffn_mx_native::WorkspaceBytes;

class MXWeightsObj : public tvm::ffi::Object {
public:
  Binding binding;
  std::array<Tensor, 9> tensors;
  int columns = 16384;

  ~MXWeightsObj() {
    Scope scope(binding);
    if (scope.active) {
      cudaDeviceSynchronize();
      for (auto &tensor : tensors)
        tensor = Tensor();
    }
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.GDNMXIOWeights", MXWeightsObj, tvm::ffi::Object);
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
  int rows = 0, padded_rows = 0, matrix_rows = 0, columns = 16384;
  mutable std::array<Tensor, 7> buffers;
  mutable std::vector<MXWeights> seen_weights;
  mutable std::vector<Tensor> eager_tensors;
  mutable std::mutex mutex;
  mutable cublasLtHandle_t handle = nullptr;
  mutable Plan projection, output_projection;
  mutable bool poisoned = false, warmed = false, eager_stream_bound = false;
  mutable cudaStream_t eager_stream = nullptr, capture_stream = nullptr;
  mutable MXGraphLeaseObj *lease = nullptr;
  mutable bool capture_bound = false;
  mutable unsigned long long capture_id = 0;
  mutable std::thread::id lease_thread;

  void release() const noexcept {
    projection.clear();
    output_projection.clear();
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

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.GDNMXIOContext", MXContextObj, tvm::ffi::Object);
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

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.GDNMXIOGraphLease", MXGraphLeaseObj,
                                    tvm::ffi::Object);
};

class MXGraphLease : public tvm::ffi::ObjectRef {
public:
  explicit MXGraphLease(tvm::ffi::ObjectPtr<MXGraphLeaseObj> object)
      : ObjectRef(std::move(object)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MXGraphLease, tvm::ffi::ObjectRef, MXGraphLeaseObj);
};
} // namespace gdn_mx_native

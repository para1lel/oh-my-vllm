#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>

using tvm::ffi::TensorView;

#ifndef OH_MY_VLLM_ENABLE_PDL
#define OH_MY_VLLM_ENABLE_PDL 1
#endif

// Every participating consumer waits before activation or mutable-state reads.
// Triggering permits another grid to start; the wait establishes visibility.
__device__ __forceinline__ void pdl_dependency_wait() {
#if OH_MY_VLLM_ENABLE_PDL && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ void pdl_launch_next() {
#if OH_MY_VLLM_ENABLE_PDL && __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename Kernel, typename... Args>
void launch_kernel(Kernel kernel, dim3 grid, dim3 block, size_t shared, cudaStream_t stream,
                   Args... args) {
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shared;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = OH_MY_VLLM_ENABLE_PDL;
  config.attrs = &attribute;
  config.numAttrs = 1;
  auto status = cudaLaunchKernelEx(&config, kernel, args...);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "CUDA extended launch failed: " << cudaGetErrorString(status);
}

// Host dispatches include direct calls and CUDA Graph capture, but not replay.
// A relaxed atomic keeps variants observable across Python/worker threads.
// Keep this order in sync with _VARIANT_OPERATIONS in cuda_backend/__init__.py.
enum VariantOperation {
  kNorm,
  kAddNorm,
  kGatedNorm,
  kQk,
  kRecurrent,
  kAppend,
  kConvolution,
  kGatedNormFp8Linear,
  kVariantCount
};
std::atomic<int64_t> variant_launches[kVariantCount][2]{};
void record_variant(int operation, bool fast) {
  variant_launches[operation][fast].fetch_add(1, std::memory_order_relaxed);
}
int64_t variant_launch_count(int64_t operation, bool fast) {
  TVM_FFI_ICHECK(operation >= 0 && operation < kVariantCount) << "unknown CUDA variant operation";
  return variant_launches[operation][fast].load(std::memory_order_relaxed);
}

bool cuda_debug_sync_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("OH_MY_VLLM_CUDA_DEBUG_SYNC");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}
void begin_cuda_launch(cudaStream_t stream, const char *operation) {
  if (!cuda_debug_sync_enabled())
    return;
  cudaError_t status = cudaPeekAtLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << operation << " prior CUDA error before launch: " << cudaGetErrorString(status);
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  status = cudaStreamIsCapturing(stream, &capture);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << operation
      << " prior CUDA work or capture query failed before launch: " << cudaGetErrorString(status);
  TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone)
      << operation << " CUDA debug sync requires eager execution; set "
      << "OH_MY_VLLM_ENFORCE_EAGER=1";
  status = cudaStreamSynchronize(stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << operation << " prior CUDA work failed before launch: " << cudaGetErrorString(status);
}
void finish_cuda_launch(cudaStream_t stream, const char *operation) {
  cudaError_t status = cudaPeekAtLastError();
  if (cuda_debug_sync_enabled()) {
    TVM_FFI_ICHECK(status == cudaSuccess)
        << operation << " CUDA error observed after launch: " << cudaGetErrorString(status);
    status = cudaStreamSynchronize(stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << operation
        << " CUDA execution error observed after launch: " << cudaGetErrorString(status);
  } else {
    TVM_FFI_ICHECK(status == cudaSuccess)
        << operation << " CUDA launch or prior asynchronous error: " << cudaGetErrorString(status);
  }
}
cudaStream_t stream_for(TensorView x, const char *operation) {
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  begin_cuda_launch(stream, operation);
  return stream;
}

bool has_dtype(TensorView tensor, int code, int bits) {
  auto dtype = tensor.dtype();
  return dtype.code == code && dtype.bits == bits && dtype.lanes == 1;
}
bool same_cuda_device(TensorView tensor, TensorView source) {
  auto device = tensor.device();
  auto expected = source.device();
  return device.device_type == kDLCUDA && expected.device_type == kDLCUDA &&
         device.device_id == expected.device_id;
}
bool is_index_dtype(TensorView tensor) {
  auto dtype = tensor.dtype();
  return dtype.code == kDLInt && (dtype.bits == 32 || dtype.bits == 64) && dtype.lanes == 1;
}
bool fits_int32_flat_offsets(TensorView tensor, bool kernel_width) {
  if (tensor.ndim() != 2)
    return false;
  int64_t rows = tensor.size(0), width = tensor.size(1);
  constexpr int64_t kMaxDimension = (int64_t(1) << 31) - 1;
  constexpr int64_t kMaxElements = int64_t(1) << 31;
  return rows > 0 && width > 0 && rows <= kMaxDimension &&
         width <= (kernel_width ? kMaxDimension : kMaxElements) && rows <= kMaxElements / width;
}

template <typename Input> struct alignas(sizeof(Input) * 4) Four {
  Input values[4];
};
template <typename Input, bool Aligned> __device__ Four<Input> load_four(const Input *x) {
  if constexpr (Aligned)
    return *reinterpret_cast<const Four<Input> *>(x);
  Four<Input> values;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    values.values[j] = x[j];
  return values;
}
__device__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return __shfl_sync(0xffffffff, value, 0);
}
__device__ int64_t index_at(const void *data, bool wide, int index) {
  return wide ? static_cast<const int64_t *>(data)[index]
              : static_cast<const int32_t *>(data)[index];
}
template <typename T, int Vector> struct alignas(sizeof(T) * Vector) AlignedVector {
  T value[Vector];
};
template <typename T, int Vector>
__device__ AlignedVector<T, Vector> load_aligned_vector(const T *p) {
  return *reinterpret_cast<const AlignedVector<T, Vector> *>(p);
}
template <int Vector, bool Streaming>
__device__ void store_rms_vector(__nv_bfloat16 *destination,
                                 AlignedVector<__nv_bfloat16, Vector> value) {
  if constexpr (Streaming) {
    static_assert(Vector == 4);
    // Medium residual batches otherwise evict reused inputs with both outputs.
    // The cache hint changes eviction priority, not visibility or stored bits.
    unsigned lo = static_cast<unsigned>(__bfloat16_as_ushort(value.value[0])) |
                  (static_cast<unsigned>(__bfloat16_as_ushort(value.value[1])) << 16);
    unsigned hi = static_cast<unsigned>(__bfloat16_as_ushort(value.value[2])) |
                  (static_cast<unsigned>(__bfloat16_as_ushort(value.value[3])) << 16);
    asm volatile("st.global.cs.v2.u32 [%0], {%1,%2};" ::"l"(destination), "r"(lo), "r"(hi)
                 : "memory");
  } else {
    *reinterpret_cast<AlignedVector<__nv_bfloat16, Vector> *>(destination) = value;
  }
}

bool aligned(TensorView tensor, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(tensor.data_ptr()) & (bytes - 1)) == 0;
}

struct Strides {
  int64_t token, head, dim;
};
// Validated callers use positive dense/row-strided views. Conservative byte
// spans include padding so a fast restricted path never assumes false independence.
uint64_t tensor_span_bytes(TensorView tensor) {
  if (!tensor.numel())
    return 0;
  uint64_t elements = 1;
  for (int i = 0; i < tensor.ndim(); ++i) {
    TVM_FFI_ICHECK(tensor.stride(i) >= 0) << "CUDA storage spans require nonnegative strides";
    elements += (tensor.size(i) - 1) * tensor.stride(i);
  }
  return elements * (tensor.dtype().bits / 8);
}
bool disjoint_storage(TensorView output, TensorView input) {
  auto first = reinterpret_cast<uintptr_t>(output.data_ptr());
  auto second = reinterpret_cast<uintptr_t>(input.data_ptr());
  return first + tensor_span_bytes(output) <= second || second + tensor_span_bytes(input) <= first;
}

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

// Host dispatches include direct calls and CUDA Graph capture, but not replay.
// A relaxed atomic keeps variants observable across Python/worker threads.
// Keep this order in sync with _VARIANT_OPERATIONS in cuda_backend/__init__.py.
enum VariantOperation {
  kNorm, kAddNorm, kGatedNorm, kQk, kRecurrent, kAppend, kConvolution, kVariantCount
};
std::atomic<int64_t> variant_launches[kVariantCount][2]{};
void record_variant(int operation, bool fast) {
  variant_launches[operation][fast].fetch_add(1, std::memory_order_relaxed);
}
int64_t variant_launch_count(int64_t operation, bool fast) {
  TVM_FFI_ICHECK(operation >= 0 && operation < kVariantCount)
      << "unknown CUDA variant operation";
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
      << operation << " prior CUDA work or capture query failed before launch: "
      << cudaGetErrorString(status);
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
        << operation << " CUDA execution error observed after launch: "
        << cudaGetErrorString(status);
  } else {
    TVM_FFI_ICHECK(status == cudaSuccess)
        << operation << " CUDA launch or prior asynchronous error: "
        << cudaGetErrorString(status);
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
  return dtype.code == kDLInt && (dtype.bits == 32 || dtype.bits == 64) &&
         dtype.lanes == 1;
}
bool fits_int32_flat_offsets(TensorView tensor, bool kernel_width) {
  if (tensor.ndim() != 2)
    return false;
  int64_t rows = tensor.size(0), width = tensor.size(1);
  constexpr int64_t kMaxDimension = (int64_t(1) << 31) - 1;
  constexpr int64_t kMaxElements = int64_t(1) << 31;
  return rows > 0 && width > 0 && rows <= kMaxDimension &&
         width <= (kernel_width ? kMaxDimension : kMaxElements) &&
         rows <= kMaxElements / width;
}

// BF16 SiLU keeps both rounding boundaries. The fast exponential produces
// identical rounded SiLU for every finite BF16 input on the required SM100.
// One warp owns one complete 128-value scaling group. Independent warps share
// a CTA; reductions do not require shared memory or block-wide synchronization.
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
template <typename Input, bool Silu, bool Column, int RowsPerWarp, bool Flat = false, int Width = 0,
          bool Aligned = false>
__global__ void quantize_kernel(const Input *__restrict__ x, __nv_fp8_e4m3 *__restrict__ out,
                                float *__restrict__ scales, int rows, int runtime_width) {
  const int width = Width ? Width : runtime_width;
  const int lane = threadIdx.x & 31;
  const int groups = width / 128;
  const int linear_group = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int group = Flat ? linear_group % groups : blockIdx.y;
  const int col = group * 128;
#pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
    const int row =
        Flat ? linear_group / groups
             : blockIdx.x * (blockDim.x / 32) * RowsPerWarp + (threadIdx.x / 32) * RowsPerWarp + r;
    if (row >= rows)
      continue;
    float value[4], maximum = 0;
    int offset = row * width * (Silu ? 2 : 1) + col + lane * 4;
    Four<Input> packed;
    packed = load_four<Input, Aligned>(x + offset);
    Four<Input> up;
    if constexpr (Silu)
      up = load_four<Input, Aligned>(x + offset + width);
    if constexpr (Silu) {
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        float x0 = static_cast<float>(packed.values[j]);
        float x1 = static_cast<float>(packed.values[j + 1]);
        auto activated = __floats2bfloat162_rn(x0 / (1.f + __expf(-x0)), x1 / (1.f + __expf(-x1)));
        auto product =
            __hmul2(activated,
                    __halves2bfloat162(__float2bfloat16_rn(static_cast<float>(up.values[j])),
                                       __float2bfloat16_rn(static_cast<float>(up.values[j + 1]))));
        auto pair = __bfloat1622float2(product);
        value[j] = pair.x;
        value[j + 1] = pair.y;
        maximum = fmaxf(maximum, fmaxf(fabsf(pair.x), fabsf(pair.y)));
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float v = static_cast<float>(packed.values[j]);
        value[j] = v;
        maximum = fmaxf(maximum, fabsf(v));
      }
    }
    // Absolute nonnegative FP32 bit patterns preserve unsigned ordering.
    maximum = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(maximum)));
    float scale = __fdiv_rn(fmaxf(maximum, 1e-10f), 448.f);
    if (lane == 0)
      scales[Column ? group * rows + row : row * groups + group] = scale;
    // BF16 scale and reciprocal stay normal throughout the finite domain.
    // One residual-refinement step uses two FMA operations and preserves
    // FP8 rounding; FP16/FP32 and nonfinite maxima retain exact division.
    // This is not FP32 equivalence.
    float inverse = 0.f;
    if constexpr (std::is_same_v<Input, __nv_bfloat16>)
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(scale));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float divided;
      if constexpr (std::is_same_v<Input, __nv_bfloat16>) {
        if (isfinite(maximum)) {
          float magnitude = fabsf(value[j]);
          float initial = magnitude * inverse;
          float corrected = __fmaf_rn(__fmaf_rn(-initial, scale, magnitude), inverse, initial);
          divided = copysignf(corrected, value[j]);
        } else {
          divided = __fdiv_rn(value[j], scale);
        }
      } else {
        divided = __fdiv_rn(value[j], scale);
      }
      value[j] = fminf(448.f, fmaxf(-448.f, divided));
    }
    reinterpret_cast<__nv_fp8x4_e4m3 *>(out)[(row * width + col) / 4 + lane] =
        __nv_fp8x4_e4m3(make_float4(value[0], value[1], value[2], value[3]));
  }
}

template <typename Input, int Width, bool Aligned>
void launch_quantize(TensorView x, TensorView out, TensorView scales, bool column, bool silu,
                     cudaStream_t stream) {
  int rows = out.size(0), width = out.size(1);
#define CALL(S, C, R, T)                                                                           \
  quantize_kernel<Input, S, C, R, false, Width, Aligned>                                           \
      <<<dim3((rows + (T / 32) * R - 1) / ((T / 32) * R), width / 128), T, 0, stream>>>(           \
          static_cast<const Input *>(x.data_ptr()), static_cast<__nv_fp8_e4m3 *>(out.data_ptr()),  \
          static_cast<float *>(scales.data_ptr()), rows, width)
#define LAUNCH(S, C)                                                                               \
  if (width / 128 > 65535 || (rows >= 4 && rows < 128) ||                                         \
      (rows == 1 && width == 6144 && C && !S)) {                                                  \
    quantize_kernel<Input, S, C, 1, true, Width, Aligned>                                          \
        <<<(rows * (width / 128) + 3) / 4, 128, 0, stream>>>(                                      \
            static_cast<const Input *>(x.data_ptr()),                                              \
            static_cast<__nv_fp8_e4m3 *>(out.data_ptr()), static_cast<float *>(scales.data_ptr()), \
            rows, width);                                                                          \
  } else if (S && rows >= 128) {                                                                   \
    CALL(S, C, 1, 128);                                                                            \
  } else if (rows >= 128) {                                                                        \
    CALL(S, C, 4, 128);                                                                            \
  } else {                                                                                         \
    CALL(S, C, 1, 32);                                                                             \
  }
  if (silu) {
    if (column) {
      LAUNCH(true, true);
    } else {
      LAUNCH(true, false);
    }
  } else {
    if (column) {
      LAUNCH(false, true);
    } else {
      LAUNCH(false, false);
    }
  }
#undef LAUNCH
#undef CALL
}

template <typename Input, bool Aligned>
void dispatch(TensorView x, TensorView out, TensorView scales, bool column, bool silu,
              cudaStream_t stream) {
  switch (out.size(1)) {
  case 5120:
    launch_quantize<Input, 5120, Aligned>(x, out, scales, column, silu, stream);
    break;
  case 6144:
    launch_quantize<Input, 6144, Aligned>(x, out, scales, column, silu, stream);
    break;
  case 17408:
    launch_quantize<Input, 17408, Aligned>(x, out, scales, column, silu, stream);
    break;
  default:
    launch_quantize<Input, 0, Aligned>(x, out, scales, column, silu, stream);
  }
}
template <typename Input>
void dispatch_alignment(TensorView x, TensorView out, TensorView scales, bool column, bool silu,
                        cudaStream_t stream) {
  // Contiguous rows and128-value groups keep every four-element load aligned
  // whenever the input base is aligned. Storage-offset views retain scalar loads.
  if ((reinterpret_cast<uintptr_t>(x.data_ptr()) & (sizeof(Input) * 4 - 1)) == 0)
    dispatch<Input, true>(x, out, scales, column, silu, stream);
  else
    dispatch<Input, false>(x, out, scales, column, silu, stream);
}
void quantize(TensorView x, TensorView out, TensorView scales, bool column, bool silu) {
  TVM_FFI_ICHECK(same_cuda_device(x, x) && same_cuda_device(out, x) &&
                 same_cuda_device(scales, x))
      << "CUDA quantize tensors must share one CUDA device";
  const auto dtype = x.dtype();
  TVM_FFI_ICHECK(dtype.lanes == 1 &&
                 ((dtype.code == kDLBfloat && dtype.bits == 16) ||
                  (dtype.code == kDLFloat && (dtype.bits == 16 || dtype.bits == 32))))
      << "CUDA quantize requires BF16, FP16, or FP32 input";
  TVM_FFI_ICHECK(!silu || has_dtype(x, kDLBfloat, 16))
      << "CUDA fused SiLU quantize requires BF16 input";
  TVM_FFI_ICHECK(has_dtype(out, kDLFloat8_e4m3fn, 8) && has_dtype(scales, kDLFloat, 32))
      << "CUDA quantize requires FP8 output and FP32 scales";
  TVM_FFI_ICHECK(x.ndim() == 2 && out.ndim() == 2 && scales.ndim() == 2 &&
                 out.size(0) == x.size(0) && out.size(0) > 0 &&
                 x.size(1) > 0 && out.size(1) > 0 && out.size(1) % 128 == 0 &&
                 x.size(1) % (silu ? 2 : 1) == 0 &&
                 x.size(1) / (silu ? 2 : 1) == out.size(1) &&
                 scales.size(0) == out.size(0) && scales.size(1) == out.size(1) / 128 &&
                 x.stride(1) == 1 && x.stride(0) == x.size(1) &&
                 out.stride(1) == 1 && out.stride(0) == out.size(1) &&
                 scales.stride(0) == (column ? 1 : scales.size(1)) &&
                 scales.stride(1) == (column ? out.size(0) : 1))
      << "CUDA quantize output shape or layout is invalid";
  TVM_FFI_ICHECK(fits_int32_flat_offsets(x, false) && fits_int32_flat_offsets(out, true))
      << "CUDA quantize input/output flat offsets exceed signed int32";
  auto stream = stream_for(x, "quantize");
  if (dtype.code == kDLBfloat)
    dispatch_alignment<__nv_bfloat16>(x, out, scales, column, silu, stream);
  else if (dtype.bits == 16)
    dispatch_alignment<__half>(x, out, scales, column, silu, stream);
  else
    dispatch_alignment<float>(x, out, scales, column, silu, stream);
  finish_cuda_launch(stream, "quantize");
}
__global__ void silu_kernel(const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ out,
                            int rows, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width)
    return;
  int offset = (i / width) * width * 2 + i % width;
  float g = __bfloat162float(x[offset]);
  float a = __bfloat162float(__float2bfloat16_rn(g / (1.f + __expf(-g))));
  out[i] = __float2bfloat16_rn(a * __bfloat162float(x[offset + width]));
}
void silu_mul(TensorView x, TensorView out) {
  TVM_FFI_ICHECK(same_cuda_device(x, x) && same_cuda_device(out, x))
      << "CUDA SiLU tensors must share one CUDA device";
  TVM_FFI_ICHECK(x.ndim() == 2 && out.ndim() == 2 && x.size(0) == out.size(0) &&
                 x.size(0) >= 0 && x.size(1) >= 0 && x.size(1) % 2 == 0 &&
                 out.size(1) == x.size(1) / 2 &&
                 has_dtype(x, kDLBfloat, 16) && has_dtype(out, kDLBfloat, 16))
      << "CUDA SiLU requires matching BF16 packed input and output";
  if (out.size(0) == 0 || out.size(1) == 0)
    return;
  TVM_FFI_ICHECK(x.stride(1) == 1 && x.stride(0) == x.size(1) &&
                 out.stride(1) == 1 && out.stride(0) == out.size(1))
      << "CUDA SiLU requires contiguous input and output";
  TVM_FFI_ICHECK(fits_int32_flat_offsets(x, false) && fits_int32_flat_offsets(out, true))
      << "CUDA SiLU input/output flat offsets exceed signed int32";
  auto stream = stream_for(x, "silu_mul");
  int n = out.size(0) * out.size(1);
  if (!n)
    return;
  silu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const __nv_bfloat16 *>(x.data_ptr()),
                                                   static_cast<__nv_bfloat16 *>(out.data_ptr()),
                                                   out.size(0), out.size(1));
  finish_cuda_launch(stream, "silu_mul");
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
__global__ void gates_kernel(const __nv_bfloat16 *__restrict__ ba, const float *__restrict__ log,
                             const float *__restrict__ bias, float *__restrict__ decay,
                             float *__restrict__ beta, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * 48)
    return;
  int head = i % 48, offset = i / 48 * 96 + head;
  float b = __bfloat162float(ba[offset]);
  float a = __bfloat162float(ba[offset + 48]) + bias[head];
  float softplus = a > 20.f ? a : log1pf(expf(a));
  decay[i] = -expf(log[head]) * softplus;
  float e = __expf(-fabsf(b));
  beta[i] = __fdividef(b >= 0.f ? 1.f : e, 1.f + e);
}
void gates(TensorView ba, TensorView log, TensorView bias, TensorView decay, TensorView beta) {
  TVM_FFI_ICHECK(same_cuda_device(ba, ba) && same_cuda_device(log, ba) &&
                 same_cuda_device(bias, ba) && same_cuda_device(decay, ba) &&
                 same_cuda_device(beta, ba))
      << "CUDA gates tensors must share one CUDA device";
  TVM_FFI_ICHECK(ba.ndim() == 2 && ba.size(1) == 96 &&
                 log.ndim() == 1 && log.size(0) == 48 &&
                 bias.ndim() == 1 && bias.size(0) == 48 &&
                 decay.ndim() == 2 && decay.size(0) == ba.size(0) && decay.size(1) == 48 &&
                 beta.ndim() == 2 && beta.size(0) == ba.size(0) && beta.size(1) == 48 &&
                 has_dtype(ba, kDLBfloat, 16) && has_dtype(log, kDLFloat, 32) &&
                 has_dtype(bias, kDLFloat, 32) && has_dtype(decay, kDLFloat, 32) &&
                 has_dtype(beta, kDLFloat, 32) &&
                 ba.stride(1) == 1 && ba.stride(0) == 96 &&
                 log.stride(0) == 1 && bias.stride(0) == 1 &&
                 decay.stride(1) == 1 && decay.stride(0) == 48 &&
                 beta.stride(1) == 1 && beta.stride(0) == 48)
      << "CUDA gates requires BF16 projection and FP32 [rows,48] outputs";
  if (ba.size(0) == 0)
    return;
  auto stream = stream_for(ba, "gates");
  gates_kernel<<<(ba.size(0) * 48 + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(ba.data_ptr()), static_cast<const float *>(log.data_ptr()),
      static_cast<const float *>(bias.data_ptr()), static_cast<float *>(decay.data_ptr()),
      static_cast<float *>(beta.data_ptr()), ba.size(0));
  finish_cuda_launch(stream, "gates");
}

template <typename T, int Vector>
struct alignas(sizeof(T) * Vector) AlignedVector {
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

template <bool Residual, int Threads, int Vector, bool Streaming>
__global__ void rms5120_kernel(const __nv_bfloat16 *__restrict__ x,
                               const __nv_bfloat16 *__restrict__ residual,
                               const float *__restrict__ w,
                               __nv_bfloat16 *__restrict__ out,
                               __nv_bfloat16 *__restrict__ summed, float eps) {
  static_assert(Threads % 32 == 0 && 5120 % (Threads * Vector) == 0);
  constexpr int Chunks = 5120 / (Threads * Vector);
  int row = blockIdx.x, lane = threadIdx.x & 31;
  float values[Chunks][Vector], total = 0;
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int col = chunk * Threads * Vector + threadIdx.x * Vector;
    auto q = load_aligned_vector<__nv_bfloat16, Vector>(
        x + static_cast<int64_t>(row) * 5120 + col);
    AlignedVector<__nv_bfloat16, Vector> r;
    if constexpr (Residual)
      r = load_aligned_vector<__nv_bfloat16, Vector>(
          residual + static_cast<int64_t>(row) * 5120 + col);
    if constexpr (Residual) {
#pragma unroll
      for (int j = 0; j < Vector; j += 2) {
        auto pair = __hadd2(__halves2bfloat162(q.value[j], q.value[j + 1]),
                            __halves2bfloat162(r.value[j], r.value[j + 1]));
        q.value[j] = __low2bfloat16(pair);
        q.value[j + 1] = __high2bfloat16(pair);
      }
    }
#pragma unroll
    for (int j = 0; j < Vector; ++j) {
      float v = __bfloat162float(q.value[j]);
      values[chunk][j] = v;
      total += v * v;
    }
    if constexpr (Residual)
      store_rms_vector<Vector, Streaming>(summed + static_cast<int64_t>(row) * 5120 + col, q);
  }
  total = warp_sum(total);
  __shared__ float partial[Threads / 32];
  if (lane == 0)
    partial[threadIdx.x / 32] = total;
  __syncthreads();
  total = warp_sum(lane < Threads / 32 ? partial[lane] : 0.f);
  float inv = rsqrtf(total * (1.f / 5120.f) + eps);
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int col = chunk * Threads * Vector + threadIdx.x * Vector;
    auto weight = load_aligned_vector<float, Vector>(w + col);
    AlignedVector<__nv_bfloat16, Vector> value;
#pragma unroll
    for (int j = 0; j < Vector; ++j)
      value.value[j] =
          __float2bfloat16_rn(values[chunk][j] * inv * weight.value[j]);
    store_rms_vector<Vector, Streaming>(out + static_cast<int64_t>(row) * 5120 + col, value);
  }
}

bool aligned(TensorView tensor, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(tensor.data_ptr()) & (bytes - 1)) == 0;
}

// Public wrappers allocate both destinations independently. The fast path is
// entered only for aligned contiguous model-width rows; other layouts use RMS.
void launch_rms5120(TensorView x, TensorView residual, TensorView weight,
                    TensorView out, TensorView summed, bool add,
                    float epsilon) {
  auto stream = stream_for(x, add ? "add_norm" : "norm");
#define RMS5120(R, Threads, Vector, Streaming)                                            \
  rms5120_kernel<R, Threads, Vector, Streaming><<<x.size(0), Threads, 0, stream>>>(       \
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),                        \
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),                 \
      static_cast<const float *>(weight.data_ptr()),                           \
      static_cast<__nv_bfloat16 *>(out.data_ptr()),                            \
      static_cast<__nv_bfloat16 *>(summed.data_ptr()), epsilon)
  // Larger rows are bandwidth-bound; residual traffic benefits from more warps.
  // Medium batches need fewer threads per row to avoid excess scheduling waves.
  if (x.size(0) >= 4096) {
    if (add) {
      if (aligned(x, 16) && aligned(residual, 16) && aligned(weight, 32)) {
        RMS5120(true, 320, 8, false);
      } else {
        RMS5120(true, 320, 4, false);
      }
    } else {
      RMS5120(false, 160, 4, false);
    }
  } else if (x.size(0) >= 2048) {
    if (add) {
      RMS5120(true, 128, 4, true);
    } else {
      RMS5120(false, 128, 4, false);
    }
  } else if (add) {
    RMS5120(true, 256, 4, false);
  } else {
    RMS5120(false, 256, 4, false);
  }
#undef RMS5120
  finish_cuda_launch(stream, add ? "add_norm" : "norm");
}

struct Strides {
  int64_t token, head, dim;
};
// Fixed model head width retains the input values across the reduction.
// The stable sigmoid keeps the fast division denominator in [1, 2], including
// extreme finite BF16 gates; it does not round the gate activation to BF16.
__global__ void gated_rms128_kernel(const __nv_bfloat16 *__restrict__ x,
                                    const float *__restrict__ weight,
                                    const __nv_bfloat16 *__restrict__ gate,
                                    __nv_bfloat16 *__restrict__ out, int rows,
                                    Strides xs, Strides gs, float epsilon) {
  int lane = threadIdx.x & 31, row = blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows)
    return;
  int64_t offset = (row / 48) * xs.token + (row % 48) * xs.head;
  int64_t goffset = (row / 48) * gs.token + (row % 48) * gs.head;
  float values[4], total = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float v = __bfloat162float(x[offset + (lane + j * 32) * xs.dim]);
    values[j] = v;
    total += v * v;
  }
  float inv = rsqrtf(warp_sum(total) * (1.f / 128.f) + epsilon);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int col = lane + j * 32;
    float g = __bfloat162float(gate[goffset + col * gs.dim]);
    float e = __expf(-fabsf(g));
    float sigmoid = __fdividef(g >= 0.f ? 1.f : e, 1.f + e);
    float v = values[j] * inv * weight[col];
    out[static_cast<int64_t>(row) * 128 + col] = __float2bfloat16_rn(v * (g * sigmoid));
  }
}

template <bool Large, bool Gated, bool Residual>
__global__ void rms_kernel(const __nv_bfloat16 *__restrict__ x,
                           const float *__restrict__ weight,
                           const __nv_bfloat16 *__restrict__ gate,
                           __nv_bfloat16 *__restrict__ out,
                           __nv_bfloat16 *__restrict__ summed, int rows,
                           int heads, int width, Strides xs, Strides gs,
                           float epsilon) {
  int lane = threadIdx.x & 31;
  int row = Large ? blockIdx.x : blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows)
    return;
  int step = Large ? 256 : 32;
  int first = Large ? threadIdx.x : lane;
  int64_t offset = (row / heads) * xs.token + (row % heads) * xs.head;
  int64_t goffset = (row / heads) * gs.token + (row % heads) * gs.head;
  float total = 0;
  for (int i = first; i < width; i += step) {
    float v = __bfloat162float(x[offset + i * xs.dim]);
    if constexpr (Residual) {
      v = __bfloat162float(__float2bfloat16_rn(v + __bfloat162float(gate[goffset + i * gs.dim])));
      summed[static_cast<int64_t>(row) * width + i] = __float2bfloat16_rn(v);
    }
    total += v * v;
  }
  total = warp_sum(total);
  if constexpr (Large) {
    __shared__ float partial[8];
    if (lane == 0)
      partial[threadIdx.x / 32] = total;
    __syncthreads();
    total = warp_sum(lane < 8 ? partial[lane] : 0.f);
  }
  float inv = rsqrtf(total / width + epsilon);
  for (int i = first; i < width; i += step) {
    float v = __bfloat162float(x[offset + i * xs.dim]);
    if constexpr (Residual)
      v = __bfloat162float(summed[static_cast<int64_t>(row) * width + i]);
    v = v * inv * weight[i];
    if constexpr (Gated) {
      float g = __bfloat162float(gate[goffset + i * gs.dim]);
      v = v * g / (1.f + expf(-g));
    }
    out[static_cast<int64_t>(row) * width + i] = __float2bfloat16_rn(v);
  }
}
void rms(TensorView x, TensorView weight, TensorView gate, TensorView out, double epsilon,
         bool gated) {
  int h = x.size(1), d = x.size(2), rows = x.size(0) * h;
  if (!gated && h == 1 && d == 5120 && x.stride(0) == 5120 && x.stride(1) == 5120 &&
      x.stride(2) == 1 && aligned(x, 8) && aligned(weight, 16)) {
    launch_rms5120(x, gate, weight, out, out, false, epsilon);
    record_variant(kNorm, true);
    return;
  }
  Strides xs{x.stride(0), x.stride(1), x.stride(2)};
  Strides gs{gate.stride(0), gate.stride(1), gate.stride(2)};
  if (gated && h == 48 && d == 128) {
    auto stream = stream_for(x, "gated_norm");
    gated_rms128_kernel<<<(rows + 3) / 4, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(x.data_ptr()),
        static_cast<const float *>(weight.data_ptr()),
        static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
        static_cast<__nv_bfloat16 *>(out.data_ptr()), rows, xs, gs, epsilon);
    finish_cuda_launch(stream, "gated_norm");
    record_variant(kGatedNorm, true);
    return;
  }
#define RMS(L, G)                                                                                  \
  rms_kernel<L, G, false><<<L ? rows : (rows + 3) / 4, L ? 256 : 128, 0, stream>>>(                 \
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                            \
      static_cast<const float *>(weight.data_ptr()),                                               \
      static_cast<const __nv_bfloat16 *>(gate.data_ptr()),                                         \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), nullptr, rows, h, d, xs, gs, epsilon)
  auto stream = stream_for(x, gated ? "gated_norm" : "norm");
  if (d > 256) {
    if (gated) {
      RMS(true, true);
    } else {
      RMS(true, false);
    }
  } else {
    if (gated) {
      RMS(false, true);
    } else {
      RMS(false, false);
    }
  }
#undef RMS
  finish_cuda_launch(stream, gated ? "gated_norm" : "norm");
  record_variant(gated ? kGatedNorm : kNorm, false);
}
void add_rms(TensorView x, TensorView residual, TensorView weight, TensorView summed,
             TensorView out) {
  int d = x.size(1), rows = x.size(0);
  if (d == 5120 && aligned(x, 8) && aligned(residual, 8) && aligned(weight, 16)) {
    launch_rms5120(x, residual, weight, out, summed, true, 1e-6f);
    record_variant(kAddNorm, true);
    return;
  }
  auto stream = stream_for(x, "add_norm");
  rms_kernel<true, false, true><<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      static_cast<const float *>(weight.data_ptr()),
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
      static_cast<__nv_bfloat16 *>(out.data_ptr()), static_cast<__nv_bfloat16 *>(summed.data_ptr()),
      rows, 1, d, {d, d, 1}, {d, d, 1}, 1e-6f);
  finish_cuda_launch(stream, "add_norm");
  record_variant(kAddNorm, false);
}

__device__ float phase(int64_t position, int channel, int rotary, double theta) {
  double a = static_cast<double>(position) * exp(-log(theta) * channel * 2. / rotary);
  constexpr double tau = 6.283185307179586476925286766559;
  return static_cast<float>(a - nearbyint(a / tau) * tau);
}
// Fixed model frequencies, computed in FP64 as exp(-log(1e7) * channel /32).
// Global read-only storage avoids divergent constant-memory serialization.
__device__ const double rotary_frequency[32] = {
    0x1.0000000000000p+0,  0x1.3566562253c7dp-1,  0x1.75f034d79e964p-2,  0x1.c3f06b4e265cep-3,
    0x1.111aedafb9a9dp-3,  0x1.4a12ad8379eb6p-4,  0x1.8eec7def5d56dp-5,  0x1.e222ec75094e2p-6,
    0x1.235a71c5ee5cbp-6,  0x1.6020a364b1285p-7,  0x1.a99428b3d26b0p-8,  0x1.012cfaad0da12p-8,
    0x1.36d219065ac0dp-9,  0x1.77a7d87ee61aep-10, 0x1.c603c39630445p-11, 0x1.125c04ab2d71cp-11,
    0x1.4b96be9c2da2ap-12, 0x1.90c181b38fe22p-13, 0x1.e459c57e28a49p-14, 0x1.24b0fd0e890cep-14,
    0x1.61bea2721385ap-15, 0x1.ab88830de4ad1p-16, 0x1.025b57369650ap-16, 0x1.383f8796f72a9p-17,
    0x1.7961810874aa1p-18, 0x1.c8198c91f9fc4p-19, 0x1.139e9527f964ap-19, 0x1.4d1c97f4e952dp-20,
    0x1.9298ace36f12dp-21, 0x1.e69338f8adcd5p-22, 0x1.26091b11c865ep-22, 0x1.635e883bbc810p-23};

__global__ void prepare_attention_kernel(const __nv_bfloat16 *packed, const float *qw,
                                         const float *kw, const void *positions,
                                         __nv_bfloat16 *cache, const void *slots,
                                         __nv_bfloat16 *out, int n, int64_t capacity, bool pw,
                                         bool sw) {
  int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= n * 28)
    return;
  int token = row / 28, head = row % 28;
  bool key = head >= 24;
  int64_t offset = (int64_t)token * 14336 + (key ? 12288 + (head - 24) * 256 : head * 512);
  const float *weight = key ? kw : qw;
  float values[8], total = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    values[j] = __bfloat162float(packed[offset + lane + j * 32]);
    total += values[j] * values[j];
  }
  float inv = rsqrtf(warp_sum(total) / 256.f + 1e-6f);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    values[j] = __bfloat162float(__float2bfloat16_rn(values[j] * inv * weight[lane + j * 32]));
  double angle = static_cast<double>(index_at(positions, pw, token)) * rotary_frequency[lane];
  constexpr double tau = 6.283185307179586476925286766559;
  float a = (float)(angle - nearbyint(angle / tau) * tau);
  float c = cosf(a), s = sinf(a);
  float left = values[0], right = values[1];
  values[0] = left * c - right * s;
  values[1] = right * c + left * s;
  if (key) {
    int64_t slot = index_at(slots, sw, token);
    if (slot < 0)
      return;
    if (slot >= capacity)
      asm volatile("trap;");
    int64_t dst = ((slot / 784) * 1568 + slot % 784) * 1024 + (head - 24) * 256;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      cache[dst + lane + j * 32] = __float2bfloat16_rn(values[j]);
      cache[dst + 802816 + lane + j * 32] =
          packed[(int64_t)token * 14336 + 13312 + (head - 24) * 256 + lane + j * 32];
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      out[((int64_t)token * 24 + head) * 256 + lane + j * 32] = __float2bfloat16_rn(values[j]);
  }
}
void prepare_attention(TensorView p, TensorView qw, TensorView kw, TensorView pos, TensorView cache,
                       TensorView slots, TensorView out) {
  auto stream = stream_for(p, "prepare_attention");
  prepare_attention_kernel<<<(p.size(0) * 28 + 3) / 4, 128, 0, stream>>>(
      (const __nv_bfloat16 *)p.data_ptr(), (const float *)qw.data_ptr(),
      (const float *)kw.data_ptr(), pos.data_ptr(), (__nv_bfloat16 *)cache.data_ptr(),
      slots.data_ptr(), (__nv_bfloat16 *)out.data_ptr(), p.size(0), cache.size(0) * 784,
      pos.dtype().bits == 64, slots.dtype().bits == 64);
  finish_cuda_launch(stream, "prepare_attention");
}

__global__ void rms_rope_kernel(const __nv_bfloat16 *__restrict__ x,
                                const float *__restrict__ weight,
                                const void *__restrict__ positions, __nv_bfloat16 *__restrict__ out,
                                int rows, int heads, Strides xs, bool wide) {
  int row = blockIdx.x * 4 + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows)
    return;
  int64_t offset = row / heads * xs.token + row % heads * xs.head;
  float values[8], total = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    values[j] = __bfloat162float(x[offset + (lane + j * 32) * xs.dim]);
    total += values[j] * values[j];
  }
  float inv = rsqrtf(warp_sum(total) / 256.f + 1e-6f);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    values[j] = __bfloat162float(__float2bfloat16_rn(values[j] * inv * weight[lane + j * 32]));
  double angle =
      static_cast<double>(index_at(positions, wide, row / heads)) * rotary_frequency[lane];
  constexpr double tau = 6.283185307179586476925286766559;
  float a = static_cast<float>(angle - nearbyint(angle / tau) * tau);
  float c = cosf(a), s = sinf(a);
  float left = values[0], right = values[1];
  values[0] = left * c - right * s;
  values[1] = right * c + left * s;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    out[static_cast<int64_t>(row) * 256 + lane + j * 32] = __float2bfloat16_rn(values[j]);
}
void rms_rope(TensorView x, TensorView weight, TensorView positions, TensorView out) {
  int rows = x.size(0) * x.size(1);
  auto stream = stream_for(x, "rms_rope");
  rms_rope_kernel<<<(rows + 3) / 4, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      static_cast<const float *>(weight.data_ptr()), positions.data_ptr(),
      static_cast<__nv_bfloat16 *>(out.data_ptr()), rows, x.size(1),
      {x.stride(0), x.stride(1), x.stride(2)}, positions.dtype().bits == 64);
  finish_cuda_launch(stream, "rms_rope");
}
__global__ void rope_kernel(const __nv_bfloat16 *__restrict__ x, const void *__restrict__ positions,
                            __nv_bfloat16 *__restrict__ out, int n, int h, int d, int rotary,
                            double theta, bool wide) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * h * d)
    return;
  int col = i % d;
  float v = __bfloat162float(x[i]);
  if (col < rotary) {
    int half = rotary / 2;
    float other = __bfloat162float(x[i + (col < half ? half : -half)]);
    float a = phase(index_at(positions, wide, i / (h * d)), col % half, rotary, theta);
    v = v * cosf(a) + (col < half ? -other : other) * sinf(a);
  }
  out[i] = __float2bfloat16_rn(v);
}
void rope(TensorView x, TensorView positions, TensorView out, int64_t rotary, double theta) {
  int n = x.size(0), h = x.size(1), d = x.size(2);
  auto stream = stream_for(x, "rope");
  rope_kernel<<<(n * h * d + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(x.data_ptr()), positions.data_ptr(),
      static_cast<__nv_bfloat16 *>(out.data_ptr()), n, h, d, rotary, theta,
      positions.dtype().bits == 64);
  finish_cuda_launch(stream, "rope");
}

__device__ float half_warp_sum(float value) {
#pragma unroll
  for (int offset = 8; offset; offset >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, offset, 16);
  return value;
}

template <int Threads, bool Joint>
__global__ void __launch_bounds__(Threads, 1)
    model_qk_kernel(const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                    __nv_bfloat16 *__restrict__ oq, __nv_bfloat16 *__restrict__ ok) {
  static_assert(Threads == 128 || Threads == 256);
  // Fixed16-head rows fill every CTA, so no partial-head mask is needed.

  int64_t input_offset = (int64_t(blockIdx.x) * (Threads) / 256) * 10240 +
                         ((int64_t(blockIdx.x) * (Threads)) % 256) * 8 + int64_t(threadIdx.x) * 8;
  auto a = *reinterpret_cast<const AlignedVector<__nv_bfloat16, 8> *>(q + input_offset);
  auto b = *reinterpret_cast<const AlignedVector<__nv_bfloat16, 8> *>(k + input_offset);
  float2 av[4], bv[4], as = make_float2(0, 0), bs = make_float2(0, 0);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    av[j] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(a.value)[j]);
    bv[j] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(b.value)[j]);
    as = __fadd2_rn(as, __fmul2_rn(av[j], av[j]));
    bs = __fadd2_rn(bs, __fmul2_rn(bv[j], bv[j]));
  }
  float2 sums = make_float2(as.x + as.y, bs.x + bs.y);
  if constexpr (Joint) {
#pragma unroll
    for (int i = 8; i; i >>= 1)
      sums = __fadd2_rn(sums, make_float2(__shfl_xor_sync(0xffffffff, sums.x, i, 16),
                                          __shfl_xor_sync(0xffffffff, sums.y, i, 16)));
  } else {
    sums.x = half_warp_sum(sums.x);
    sums.y = half_warp_sum(sums.y);
  }
  float ai = rsqrtf(sums.x + 1e-6f), bi = rsqrtf(sums.y + 1e-6f);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    reinterpret_cast<__nv_bfloat162 *>(a.value)[j] =
        __float22bfloat162_rn(__fmul2_rn(av[j], make_float2(ai, ai)));
    reinterpret_cast<__nv_bfloat162 *>(b.value)[j] =
        __float22bfloat162_rn(__fmul2_rn(bv[j], make_float2(bi, bi)));
  }
  *reinterpret_cast<AlignedVector<__nv_bfloat16, 8> *>(oq + int64_t(blockIdx.x) * Threads * 8 +
                                                       threadIdx.x * 8) = a;
  *reinterpret_cast<AlignedVector<__nv_bfloat16, 8> *>(ok + int64_t(blockIdx.x) * Threads * 8 +
                                                       threadIdx.x * 8) = b;
}
__global__ void qk_kernel(const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                          __nv_bfloat16 *__restrict__ oq, __nv_bfloat16 *__restrict__ ok, int rows,
                          int heads, int64_t qs, int64_t ks) {
  int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows)
    return;
  float qv[4], kv[4], qsum = 0, ksum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    qv[j] = __bfloat162float(q[row / heads * qs + row % heads * 128 + lane + j * 32]);
    kv[j] = __bfloat162float(k[row / heads * ks + row % heads * 128 + lane + j * 32]);
    qsum += qv[j] * qv[j];
    ksum += kv[j] * kv[j];
  }
  float qi = rsqrtf(warp_sum(qsum) + 1e-6f), ki = rsqrtf(warp_sum(ksum) + 1e-6f);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    oq[static_cast<int64_t>(row) * 128 + lane + j * 32] = __float2bfloat16_rn(qv[j] * qi);
    ok[static_cast<int64_t>(row) * 128 + lane + j * 32] = __float2bfloat16_rn(kv[j] * ki);
  }
}
void normalize_qk(TensorView q, TensorView k, TensorView oq, TensorView ok) {
  bool packed = q.size(1) == 16 && q.stride(0) == 10240 && k.stride(0) == 10240 &&
                reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0;
  auto stream = stream_for(q, "normalize_qk");
  if (packed) {
#define MODEL_QK(T, J)                                                                             \
  model_qk_kernel<T, J><<<q.size(0) * (256 / T), T, 0, stream>>>(                                   \
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),                                            \
      static_cast<__nv_bfloat16 *>(oq.data_ptr()), static_cast<__nv_bfloat16 *>(ok.data_ptr()))
    if (q.size(0) < 1024) {
      MODEL_QK(128, true);
    } else if (q.size(0) < 4096) {
      MODEL_QK(256, false);
    } else {
      MODEL_QK(256, true);
    }
#undef MODEL_QK
    finish_cuda_launch(stream, "normalize_qk");
    record_variant(kQk, true);
    return;
  }
  int rows = q.size(0) * q.size(1);
  qk_kernel<<<(rows + 3) / 4, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()), static_cast<__nv_bfloat16 *>(oq.data_ptr()),
      static_cast<__nv_bfloat16 *>(ok.data_ptr()), rows, q.size(1), q.stride(0), k.stride(0));
  finish_cuda_launch(stream, "normalize_qk");
  record_variant(kQk, false);
}
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
template <typename State, int Rows, int Warps>
__global__ void
recurrent_vector_kernel(const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                        const __nv_bfloat16 *__restrict__ v, const float *__restrict__ decay,
                        const float *__restrict__ beta, State *__restrict__ pool,
                        const void *__restrict__ starts, const void *__restrict__ reads,
                        const void *__restrict__ writes, bool sw, bool rw, bool ww,
                        __nv_bfloat16 *__restrict__ out, int64_t qs, int64_t ks, int64_t vs) {
  static_assert(128 % (Rows * Warps * 2) == 0);
  constexpr int hq = 16, hv = 48;
  int seq = blockIdx.x, head = blockIdx.y, lane = threadIdx.x & 15;
  int first_v = blockIdx.z * (Rows * Warps * 2) + threadIdx.x / 16 * Rows;
  int qhead = head / (hv / hq);
  int64_t source = index_at(reads, rw, seq);
  float state[Rows][8];
#pragma unroll
  for (int i = 0; i < Rows; ++i) {
    auto v = *reinterpret_cast<const AlignedVector<State, 8> *>(
        pool + ((source * hv + head) * 128 + first_v + i) * 128 + lane * 8);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      state[i][j] = static_cast<float>(v.value[j]);
  }
  // Persistent FP32 values survive all candidates; only destination snapshots round.
  for (int token = index_at(starts, sw, seq); token < index_at(starts, sw, seq + 1); ++token) {
    float qv[8], kv[8], qq = 0, kk = 0;
    auto qpack = *reinterpret_cast<const AlignedVector<__nv_bfloat16, 8> *>(q + token * qs +
                                                                            qhead * 128 + lane * 8);
    auto kpack = *reinterpret_cast<const AlignedVector<__nv_bfloat16, 8> *>(k + token * ks +
                                                                            qhead * 128 + lane * 8);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      qv[j] = __bfloat162float(qpack.value[j]);
      kv[j] = __bfloat162float(kpack.value[j]);
      qq += qv[j] * qv[j];
      kk += kv[j] * kv[j];
    }
    float qi = rsqrtf(half_warp_sum(qq) + 1e-6f) * 0.08838834764831844f;
    float ki = rsqrtf(half_warp_sum(kk) + 1e-6f);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      qv[j] *= qi;
      kv[j] *= ki;
    }
    float g = expf(decay[token * hv + head]), b = beta[token * hv + head];
    int64_t target = index_at(writes, ww, token);
#pragma unroll
    for (int i = 0; i < Rows; ++i) {
      float recall = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        state[i][j] *= g;
        recall += state[i][j] * kv[j];
      }
      recall = half_warp_sum(recall);
      float residual = (__bfloat162float(v[token * vs + head * 128 + first_v + i]) - recall) * b;
      float result = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        state[i][j] += residual * kv[j];
        result += state[i][j] * qv[j];
      }
      if (target >= 0) {
        AlignedVector<State, 8> snapshot;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if constexpr (std::is_same_v<State, __nv_bfloat16>)
            reinterpret_cast<__nv_bfloat162 *>(snapshot.value)[j] =
                __floats2bfloat162_rn(state[i][j * 2], state[i][j * 2 + 1]);
          else {
            snapshot.value[j * 2] = state[i][j * 2];
            snapshot.value[j * 2 + 1] = state[i][j * 2 + 1];
          }
        }
        *reinterpret_cast<AlignedVector<State, 8> *>(
            pool + ((target * hv + head) * 128 + first_v + i) * 128 + lane * 8) = snapshot;
      }
      result = half_warp_sum(result);
      if (lane == 0)
        out[(static_cast<int64_t>(token) * hv + head) * 128 + first_v + i] =
            __float2bfloat16_rn(result);
    }
  }
}
template <typename State>
__global__ void recurrent_kernel(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                 const __nv_bfloat16 *v, const float *decay, const float *beta,
                                 State *pool, const void *starts, const void *reads,
                                 const void *writes, bool sw, bool rw, bool ww, __nv_bfloat16 *out,
                                 int hq, int hv, int64_t qs, int64_t ks, int64_t vs) {
  int seq = blockIdx.x, head = blockIdx.y, lane = threadIdx.x & 31;
  int first_v = blockIdx.z * 16 + threadIdx.x / 32 * 4;
  int qhead = head / (hv / hq);
  int64_t source = index_at(reads, rw, seq);
  float state[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      state[i][j] = static_cast<float>(
          pool[((source * hv + head) * 128 + first_v + i) * 128 + lane + j * 32]);
  // Persistent FP32 values survive all candidates; only destination snapshots round.
  for (int token = index_at(starts, sw, seq); token < index_at(starts, sw, seq + 1); ++token) {
    float qv[4], kv[4], qq = 0, kk = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qv[j] = __bfloat162float(q[token * qs + qhead * 128 + lane + j * 32]);
      kv[j] = __bfloat162float(k[token * ks + qhead * 128 + lane + j * 32]);
      qq += qv[j] * qv[j];
      kk += kv[j] * kv[j];
    }
    float qi = rsqrtf(warp_sum(qq) + 1e-6f) * 0.08838834764831844f;
    float ki = rsqrtf(warp_sum(kk) + 1e-6f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qv[j] *= qi;
      kv[j] *= ki;
    }
    float g = expf(decay[token * hv + head]), b = beta[token * hv + head];
    int64_t target = index_at(writes, ww, token);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float recall = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        state[i][j] *= g;
        recall += state[i][j] * kv[j];
      }
      recall = warp_sum(recall);
      float residual = (__bfloat162float(v[token * vs + head * 128 + first_v + i]) - recall) * b;
      float result = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        state[i][j] += residual * kv[j];
        result += state[i][j] * qv[j];
        if (target >= 0)
          pool[((target * hv + head) * 128 + first_v + i) * 128 + lane + j * 32] =
              static_cast<State>(state[i][j]);
      }
      result = warp_sum(result);
      if (lane == 0)
        out[(static_cast<int64_t>(token) * hv + head) * 128 + first_v + i] =
            __float2bfloat16_rn(result);
    }
  }
}
void recurrent(TensorView q, TensorView k, TensorView v, TensorView decay, TensorView beta,
               TensorView pool, TensorView starts, TensorView reads, TensorView writes,
               TensorView out) {
  // Caller guarantees starts=[0,...,q.size(0)] is monotone, reads are in
  // [0,pool.size(0)), and writes are -1 (skip) or in that range. Their values
  // stay on GPU to avoid a hot-path sync.
  TVM_FFI_ICHECK(same_cuda_device(q, q) && same_cuda_device(k, q) &&
                 same_cuda_device(v, q) && same_cuda_device(decay, q) &&
                 same_cuda_device(beta, q) && same_cuda_device(pool, q) &&
                 same_cuda_device(starts, q) && same_cuda_device(reads, q) &&
                 same_cuda_device(writes, q) && same_cuda_device(out, q))
      << "CUDA recurrence tensors must share one CUDA device";
  TVM_FFI_ICHECK(q.ndim() == 3 && k.ndim() == 3 && v.ndim() == 3 &&
                 q.size(0) > 0 && q.size(1) > 0 && q.size(2) == 128 &&
                 k.size(0) == q.size(0) && k.size(1) == q.size(1) && k.size(2) == 128 &&
                 v.size(0) == q.size(0) && v.size(1) > 0 && v.size(2) == 128 &&
                 v.size(1) % q.size(1) == 0 &&
                 has_dtype(q, kDLBfloat, 16) && has_dtype(k, kDLBfloat, 16) &&
                 has_dtype(v, kDLBfloat, 16) &&
                 q.stride(2) == 1 && q.stride(1) == 128 &&
                 k.stride(2) == 1 && k.stride(1) == 128 &&
                 v.stride(2) == 1 && v.stride(1) == 128 &&
                 q.stride(0) >= q.size(1) * 128 && k.stride(0) >= k.size(1) * 128 &&
                 v.stride(0) >= v.size(1) * 128)
      << "CUDA recurrence requires nonempty BF16 q/k/v rows";
  TVM_FFI_ICHECK(decay.ndim() == 2 && beta.ndim() == 2 &&
                 decay.size(0) == q.size(0) && decay.size(1) == v.size(1) &&
                 beta.size(0) == q.size(0) && beta.size(1) == v.size(1) &&
                 has_dtype(decay, kDLFloat, 32) && has_dtype(beta, kDLFloat, 32) &&
                 decay.stride(1) == 1 && decay.stride(0) == v.size(1) &&
                 beta.stride(1) == 1 && beta.stride(0) == v.size(1))
      << "CUDA recurrence requires contiguous FP32 gates";
  TVM_FFI_ICHECK(starts.ndim() == 1 && reads.ndim() == 1 && writes.ndim() == 1 &&
                 reads.size(0) > 0 && starts.size(0) == reads.size(0) + 1 &&
                 writes.size(0) == q.size(0) &&
                 is_index_dtype(starts) && is_index_dtype(reads) &&
                 is_index_dtype(writes) && starts.stride(0) == 1 &&
                 reads.stride(0) == 1 && writes.stride(0) == 1)
      << "CUDA recurrence requires nonempty int32/int64 metadata";
  const auto state_dtype = pool.dtype();
  const bool bf16_state = state_dtype.code == kDLBfloat && state_dtype.bits == 16 &&
                          state_dtype.lanes == 1;
  const bool fp32_state = state_dtype.code == kDLFloat && state_dtype.bits == 32 &&
                          state_dtype.lanes == 1;
  TVM_FFI_ICHECK(bf16_state || fp32_state) << "CUDA recurrence state requires BF16 or FP32";
  TVM_FFI_ICHECK(pool.ndim() == 4 && pool.size(0) > 0 &&
                 pool.size(1) == v.size(1) && pool.size(2) == 128 && pool.size(3) == 128 &&
                 pool.stride(3) == 1 && pool.stride(2) == 128 &&
                 pool.stride(1) == 128 * 128 &&
                 pool.stride(0) == v.size(1) * 128 * 128)
      << "CUDA recurrence state shape or layout is invalid";
  TVM_FFI_ICHECK(out.ndim() == 3 && out.size(0) == v.size(0) && out.size(1) == v.size(1) &&
                 out.size(2) == 128 && has_dtype(out, kDLBfloat, 16) &&
                 out.stride(2) == 1 && out.stride(1) == 128 &&
                 out.stride(0) == v.size(1) * 128)
      << "CUDA recurrence output requires contiguous BF16 value shape";
  const auto state_alignment = bf16_state ? alignof(AlignedVector<__nv_bfloat16, 8>)
                                          : alignof(AlignedVector<float, 8>);
  bool vector = q.size(1) == 16 && v.size(1) == 48 && q.stride(0) % 8 == 0 &&
                k.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(pool.data_ptr()) % state_alignment == 0;
  for (auto input : {q, k, v, decay, beta, starts, reads, writes})
    vector = vector && disjoint_storage(pool, input);
  auto stream = stream_for(q, "recurrent");
  if (vector) {
#define VECTOR_REC(S, R, W)                                                                        \
  recurrent_vector_kernel<S, R, W>                                                                 \
      <<<dim3(reads.size(0), 48, 128 / (R * W * 2)), W * 32, 0, stream>>>(                          \
          static_cast<const __nv_bfloat16 *>(q.data_ptr()),                                        \
          static_cast<const __nv_bfloat16 *>(k.data_ptr()),                                        \
          static_cast<const __nv_bfloat16 *>(v.data_ptr()),                                        \
          static_cast<const float *>(decay.data_ptr()),                                            \
          static_cast<const float *>(beta.data_ptr()), static_cast<S *>(pool.data_ptr()),          \
          starts.data_ptr(), reads.data_ptr(), writes.data_ptr(), starts.dtype().bits == 64,       \
          reads.dtype().bits == 64, writes.dtype().bits == 64,                                     \
          static_cast<__nv_bfloat16 *>(out.data_ptr()), q.stride(0), k.stride(0), v.stride(0))
    if (bf16_state) {
      if (reads.size(0) == 1) {
        VECTOR_REC(__nv_bfloat16, 2, 2);
      } else {
        VECTOR_REC(__nv_bfloat16, 4, 2);
      }
    } else {
      if (reads.size(0) <= 2) {
        VECTOR_REC(float, 2, 2);
      } else if (reads.size(0) == 3) {
        VECTOR_REC(float, 2, 4);
      } else {
        VECTOR_REC(float, 4, 2);
      }
    }
#undef VECTOR_REC
    finish_cuda_launch(stream, "recurrent");
    record_variant(kRecurrent, true);
    return;
  }
  dim3 grid(reads.size(0), v.size(1), 8);
#define REC(S)                                                                                     \
  recurrent_kernel<S><<<grid, 128, 0, stream>>>(                                                    \
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),                                            \
      static_cast<const float *>(decay.data_ptr()), static_cast<const float *>(beta.data_ptr()),   \
      static_cast<S *>(pool.data_ptr()), starts.data_ptr(), reads.data_ptr(), writes.data_ptr(),   \
      starts.dtype().bits == 64, reads.dtype().bits == 64, writes.dtype().bits == 64,              \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), q.size(1), v.size(1), q.stride(0),             \
      k.stride(0), v.stride(0))
  if (bf16_state) {
    REC(__nv_bfloat16);
  } else {
    REC(float);
  }
#undef REC
  finish_cuda_launch(stream, "recurrent");
  record_variant(kRecurrent, false);
}
template <bool Vector>
__global__ void append_kernel(const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *cache,
                              const void *slots, int width, int64_t capacity, bool wide) {
  int token = blockIdx.x;
  int64_t slot = index_at(slots, wide, token);
  if (slot < 0)
    return;
  if (slot >= capacity)
    asm volatile("trap;");
  int64_t dst = ((slot / 784) * 1568 + slot % 784) * width;
  if constexpr (Vector) {
    for (int j = threadIdx.x; j < width / 8; j += blockDim.x) {
      reinterpret_cast<uint4 *>(cache + dst)[j] =
          reinterpret_cast<const uint4 *>(k + static_cast<int64_t>(token) * width)[j];
      reinterpret_cast<uint4 *>(cache + dst + 784 * width)[j] =
          reinterpret_cast<const uint4 *>(v + static_cast<int64_t>(token) * width)[j];
    }
  } else {
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
      cache[dst + j] = k[static_cast<int64_t>(token) * width + j];
      cache[dst + 784 * width + j] = v[static_cast<int64_t>(token) * width + j];
    }
  }
}
void append(TensorView k, TensorView v, TensorView cache, TensorView slots) {
  int width = k.size(1) * k.size(2);
  bool vector = width % 8 == 0 && reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0;
  auto stream = stream_for(k, "append");
#define APPEND(V)                                                                                  \
  append_kernel<V>                                                                                 \
      <<<k.size(0), 128, 0, stream>>>(static_cast<const __nv_bfloat16 *>(k.data_ptr()),             \
                                             static_cast<const __nv_bfloat16 *>(v.data_ptr()),     \
                                             static_cast<__nv_bfloat16 *>(cache.data_ptr()),       \
                                             slots.data_ptr(), width, cache.size(0) * 784,         \
                                             slots.dtype().bits == 64)
  if (vector) {
    APPEND(true);
  } else {
    APPEND(false);
  }
#undef APPEND
  finish_cuda_launch(stream, "append");
  record_variant(kAppend, vector);
}

template <int Rows>
__global__ void
model_convolution_kernel(const __nv_bfloat16 *__restrict__ x,
                         const __nv_bfloat16 *__restrict__ weight, __nv_bfloat16 *__restrict__ pool,
                         const void *__restrict__ ids, const void *__restrict__ starts,
                         const __nv_bfloat16 *__restrict__ sources, const void *__restrict__ writes,
                         __nv_bfloat16 *__restrict__ out, int n, bool iw, bool sw, bool ww) {
  constexpr int c = 10240, stride = 16384;
  int channel = blockIdx.y * 128 + threadIdx.x;
  if (channel >= c)
    return;
  auto packed = *reinterpret_cast<const Four<__nv_bfloat16> *>(weight + channel * 4);
  float w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    w[j] = __bfloat162float(packed.values[j]);
#pragma unroll
  for (int i = 0; i < Rows; ++i) {
    int token = blockIdx.x * Rows + i;
    if (token >= n)
      continue;
    int64_t seq = index_at(ids, iw, token), first = index_at(starts, sw, seq);
    int64_t target = index_at(writes, ww, token);
    float total = 0;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
      int64_t pos = token + tap - 3;
      __nv_bfloat16 value = pos >= first ? x[pos * stride + channel]
                                         : sources[(seq * c + channel) * 3 + pos - first + 3];
      total += __bfloat162float(value) * w[tap];
      if (tap > 0 && target >= 0)
        pool[(target * c + channel) * 3 + tap - 1] = value;
    }
    float e = __expf(-fabsf(total));
    float activated = total * __fdividef(total >= 0.f ? 1.f : e, 1.f + e);
    out[static_cast<int64_t>(token) * c + channel] = __float2bfloat16_rn(activated);
  }
}
template <int Rows>
__global__ void convolution_kernel(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                                   __nv_bfloat16 *pool, const void *ids, const void *starts,
                                   const __nv_bfloat16 *sources, const void *writes,
                                   __nv_bfloat16 *out, int n, int c, int64_t stride, bool iw,
                                   bool sw, bool ww) {
  int channel = blockIdx.y * 128 + threadIdx.x;
  if (channel >= c)
    return;
  float w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    w[j] = __bfloat162float(weight[channel * 4 + j]);
#pragma unroll
  for (int i = 0; i < Rows; ++i) {
    int token = blockIdx.x * Rows + i;
    if (token >= n)
      continue;
    int64_t seq = index_at(ids, iw, token), first = index_at(starts, sw, seq);
    int64_t target = index_at(writes, ww, token);
    float total = 0;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
      int64_t pos = token + tap - 3;
      __nv_bfloat16 value = pos >= first ? x[pos * stride + channel]
                                         : sources[(seq * c + channel) * 3 + pos - first + 3];
      total += __bfloat162float(value) * w[tap];
      if (tap > 0 && target >= 0)
        pool[(target * c + channel) * 3 + tap - 1] = value;
    }
    out[static_cast<int64_t>(token) * c + channel] =
        __float2bfloat16_rn(total / (1.f + expf(-total)));
  }
}
void convolution(TensorView x, TensorView weight, TensorView pool, TensorView ids,
                 TensorView starts, TensorView sources, TensorView writes, TensorView out) {
  int n = x.size(0), c = x.size(1);
  bool model =
      c == 10240 && x.stride(0) == 16384 && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 8 == 0;
  for (auto input : {x, weight, ids, starts, sources, writes})
    model = model && disjoint_storage(pool, input);
  auto stream = stream_for(x, "convolution");
  if (model) {
#define MODEL_CONV(R)                                                                              \
  model_convolution_kernel<R><<<dim3((n + R - 1) / R, 80), 128, 0, stream>>>(                       \
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(weight.data_ptr()),                                       \
      static_cast<__nv_bfloat16 *>(pool.data_ptr()), ids.data_ptr(), starts.data_ptr(),            \
      static_cast<const __nv_bfloat16 *>(sources.data_ptr()), writes.data_ptr(),                   \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), n, ids.dtype().bits == 64,                     \
      starts.dtype().bits == 64, writes.dtype().bits == 64)
    if (n >= 4096) {
      MODEL_CONV(8);
    } else if (n >= 128) {
      MODEL_CONV(4);
    } else {
      MODEL_CONV(1);
    }
#undef MODEL_CONV
    finish_cuda_launch(stream, "convolution");
    record_variant(kConvolution, true);
    return;
  }
#define CONV(R)                                                                                    \
  convolution_kernel<R><<<dim3((n + R - 1) / R, (c + 127) / 128), 128, 0, stream>>>(                \
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(weight.data_ptr()),                                       \
      static_cast<__nv_bfloat16 *>(pool.data_ptr()), ids.data_ptr(), starts.data_ptr(),            \
      static_cast<const __nv_bfloat16 *>(sources.data_ptr()), writes.data_ptr(),                   \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), n, c, x.stride(0), ids.dtype().bits == 64,     \
      starts.dtype().bits == 64, writes.dtype().bits == 64)
  if (n >= 128) {
    CONV(8);
  } else {
    CONV(1);
  }
#undef CONV
  finish_cuda_launch(stream, "convolution");
  record_variant(kConvolution, false);
}
__device__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  return __shfl_sync(0xffffffff, value, 0);
}

// Softmax uses the same hardware base-two exponential family as the frozen
// TileLang path. Keep approximation local: RoPE and FP8 retain their math rules.
__device__ __forceinline__ float softmax_exp2(float value) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

// PTX mma.m16n8k16 BF16 fragment mapping follows the NVIDIA ISA guide.
// XOR eight-element sectors across rows to avoid shared-memory bank conflicts.
__device__ __forceinline__ int shared_index(int row, int col, int stride = 256) {
  return row * stride + (col ^ ((row & 7) * 8));
}
__device__ __forceinline__ void load_a(unsigned (&a)[4], const __nv_bfloat16 *data, int inner,
                                       int lane, int stride = 256) {
  unsigned address =
      __cvta_generic_to_shared(data + shared_index(lane % 16, inner + (lane / 16) * 8, stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(address));
}
__device__ __forceinline__ void load_k(unsigned (&b)[2], const __nv_bfloat16 *data, int inner,
                                       int warp, int lane, int stride = 256) {
  unsigned address = __cvta_generic_to_shared(
      data + shared_index(warp * 8 + lane % 8, inner + ((lane / 8) % 2) * 8, stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(address));
}
__device__ __forceinline__ void load_v(unsigned (&b)[2], const __nv_bfloat16 *data, int inner,
                                       int col, int lane, int stride = 256) {
  unsigned address = __cvta_generic_to_shared(data + shared_index(inner + lane % 16, col, stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(address));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Keep the complete PV accumulator in registers across all KV tiles. Only
// scores/probabilities and online-softmax statistics cross warp boundaries.
template <int BK, typename Position>
__device__ __forceinline__ void stage_kv(const __nv_bfloat16 *cache, const void *tables, int start,
                                         int table_width, Position base, Position last, int hk,
                                         int kh, bool tw, __nv_bfloat16 *k, __nv_bfloat16 *v) {
  // A BK-wide tile can touch at most two 784-token pages. Each thread loads
  // those table entries once per tile, rather than once per K/V chunk.
  Position first_page = min(base, last - 1) / 784;
  Position final_page = min(base + BK - 1, last - 1) / 784;
  int64_t table_row = static_cast<int64_t>(start) * table_width;
  int64_t first_block = index_at(tables, tw, table_row + first_page);
  int64_t final_block = first_page == final_page
                            ? first_block
                            : index_at(tables, tw, table_row + final_page);
#pragma unroll
  for (int i = threadIdx.x; i < BK * 32; i += 128) {
    Position pos = base + i / 32;
    int64_t page = pos / 784 == first_page ? first_block : final_block;
    int64_t offset = ((page * 1568 + pos % 784) * hk + kh) * 256 + (i % 32) * 8;
    unsigned kd = __cvta_generic_to_shared(k + shared_index(i / 32, (i % 32) * 8));
    unsigned vd = __cvta_generic_to_shared(v + shared_index(i / 32, (i % 32) * 8));
    int bytes = pos < last ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(kd),
                 "l"(cache + offset), "r"(bytes)
                 : "memory");
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(vd),
                 "l"(cache + offset + 784 * hk * 256), "r"(bytes)
                 : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <bool ModelShape, bool Grouped, typename Position, int Buffers>
__global__ void
attention_partial_kernel(const __nv_bfloat16 *__restrict__ query,
                         const __nv_bfloat16 *__restrict__ cache, const void *__restrict__ tables,
                         const void *__restrict__ lengths, const void *__restrict__ starts,
                         float *__restrict__ partial, float *__restrict__ lse, int runtime_h,
                         int runtime_hk, int table_width, int splits, int first, bool tw, bool lw,
                         bool sw, int query_tiles, int max_query_len) {
  constexpr int BQ = Grouped ? 32 : 16, BK = Grouped ? 64 : 32;
  constexpr int RowGroups = BQ / 8, ScoreTiles = BQ * BK / 512, AccTiles = BQ / 2;
  const int h = ModelShape ? 24 : runtime_h, hk = ModelShape ? 4 : runtime_hk;
  extern __shared__ __align__(32) unsigned char storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *key_buffers = q + BQ * 256;
  auto *value_buffers = key_buffers + Buffers * BK * 256;
  auto *p = value_buffers + Buffers * BK * 256;
  auto *maxima = reinterpret_cast<float *>(p + BQ * 64);
  auto *sums = maxima + 4 * BQ;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int fr = lane / 4, fc = (lane % 4) * 2;
  const int seq = blockIdx.x / query_tiles, qbase = (blockIdx.x % query_tiles) * BQ;
  const int kh = blockIdx.y, split = blockIdx.z, ratio = h / hk;
  const int start = Grouped ? index_at(starts, sw, seq) : seq;
  const int end = Grouped ? index_at(starts, sw, seq + 1) : seq + 1;
  if constexpr (Grouped) {
    if (end <= start || end - start > max_query_len)
      asm volatile("trap;");
  }
  const Position last = index_at(lengths, lw, end - 1);
  const Position chunk = ((max(last - first, Position(0)) + splits * BK - 1) / (splits * BK)) * BK;
  const Position begin = first + split * chunk;
  float acc[AccTiles][4] = {}, denominator[RowGroups] = {}, maximum[RowGroups];
#pragma unroll
  for (int rg = 0; rg < RowGroups; ++rg)
    maximum[rg] = -INFINITY;
#pragma unroll
  for (int i = threadIdx.x; i < BQ * 32; i += 128) {
    int r = i / 32, col = (i % 32) * 8;
    int row = start + (qbase + r) / ratio;
    auto *destination = q + shared_index(r, col);
    int64_t offset = (static_cast<int64_t>(row) * h + kh * ratio + (qbase + r) % ratio) * 256 + col;
    if ((reinterpret_cast<uintptr_t>(query) & 15) == 0) {
      uint4 values =
          row < end ? *reinterpret_cast<const uint4 *>(query + offset) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4 *>(destination) = values;
    } else {
      // A contiguous tensor may begin at a BF16 storage offset, not a16B boundary.
#pragma unroll
      for (int j = 0; j < 8; ++j)
        destination[j] = row < end ? query[offset + j] : __float2bfloat16(0);
    }
  }
  __syncthreads();
  const Position limit = min(begin + chunk, last);
  if (begin < limit)
    stage_kv<BK>(cache, tables, start, table_width, begin, last, hk, kh, tw, key_buffers,
                 value_buffers);
  int step = 0;
  for (Position base = begin; base < limit; base += BK, ++step) {
    auto *k = key_buffers + (step % Buffers) * BK * 256;
    auto *v = value_buffers + (step % Buffers) * BK * 256;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if constexpr (Buffers == 2) {
      if (base + BK < limit)
        stage_kv<BK>(cache, tables, start, table_width, base + BK, last, hk, kh, tw,
                     key_buffers + ((step + 1) % Buffers) * BK * 256,
                     value_buffers + ((step + 1) % Buffers) * BK * 256);
    }
    float score[ScoreTiles][4] = {};
#pragma unroll 8
    for (int inner = 0; inner < 256; inner += 16) {
      unsigned qa[BQ / 16][4], kb[BK / 32][2];
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg)
        load_a(qa[rg], q + rg * 16 * 256, inner, lane);
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt)
        load_k(kb[kt], k, inner, kt * 4 + warp, lane);
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg) {
#pragma unroll
        for (int kt = 0; kt < BK / 32; ++kt) {
          mma_bf16(score[rg * (BK / 32) + kt], qa[rg][0], qa[rg][1], qa[rg][2], qa[rg][3],
                   kb[kt][0], kb[kt][1]);
        }
      }
    }
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int row = start + (qbase + r) / ratio;
      Position length = row < end ? index_at(lengths, lw, row) : 0;
      float m = -INFINITY;
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt) {
        int tile = (rg / 2) * (BK / 32) + kt;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          int col = (kt * 4 + warp) * 8 + fc + j;
          float value = row < end && base + col < length
                            ? score[tile][(rg % 2) * 2 + j] * 0.09016844005556021f
                            : -INFINITY;
          score[tile][(rg % 2) * 2 + j] = value;
          m = fmaxf(m, value);
        }
      }
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
      if (lane % 4 == 0)
        maxima[warp * BQ + r] = m;
    }
    __syncthreads();
    float alpha[RowGroups];
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      float m = maximum[rg];
#pragma unroll
      for (int w = 0; w < 4; ++w)
        m = fmaxf(m, maxima[w * BQ + r]);
      float safe = m == -INFINITY ? 0.f : m;
      alpha[rg] = softmax_exp2(maximum[rg] - safe);
      maximum[rg] = m;
      float sum = 0;
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt) {
        int tile = (rg / 2) * (BK / 32) + kt;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          int col = (kt * 4 + warp) * 8 + fc + j;
          float probability = softmax_exp2(score[tile][(rg % 2) * 2 + j] - safe);
          sum += probability;
          p[shared_index(r, col, 64)] = __float2bfloat16_rn(probability);
        }
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      sum += __shfl_xor_sync(0xffffffff, sum, 2);
      if (lane % 4 == 0)
        sums[warp * BQ + r] = sum;
    }
    __syncthreads();
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      float sum = 0;
#pragma unroll
      for (int w = 0; w < 4; ++w)
        sum += sums[w * BQ + r];
      denominator[rg] = denominator[rg] * alpha[rg] + sum;
    }
#pragma unroll
    for (int tile = 0; tile < AccTiles; ++tile) {
      // Four column warps make the row group independent of warp. Keep this
      // index static so alpha and denominator remain register arrays.
      int row_base = (tile / 8) * 16;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[tile][j] *= alpha[row_base / 8 + j / 2];
    }
#pragma unroll
    for (int inner = 0; inner < BK; inner += 16) {
      unsigned pa[BQ / 16][4];
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg)
        load_a(pa[rg], p + rg * 16 * 64, inner, lane, 64);
#pragma unroll
      for (int ct = 0; ct < 8; ++ct) {
        unsigned vb[2];
        load_v(vb, v, inner, (ct * 4 + warp) * 8, lane);
#pragma unroll
        for (int rg = 0; rg < BQ / 16; ++rg)
          mma_bf16(acc[rg * 8 + ct], pa[rg][0], pa[rg][1], pa[rg][2], pa[rg][3], vb[0], vb[1]);
      }
    }
    __syncthreads();
    if constexpr (Buffers == 1) {
      if (base + BK < limit)
        stage_kv<BK>(cache, tables, start, table_width, base + BK, last, hk, kh, tw, key_buffers,
                     value_buffers);
    }
  }
#pragma unroll
  for (int rg = 0; rg < RowGroups; ++rg)
    denominator[rg] = 1.f / (denominator[rg] > 0 ? denominator[rg] : 1.f);
#pragma unroll
  for (int tile = 0; tile < AccTiles; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int rg = (tile / 8) * 2 + j / 2;
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int qr = qbase + r, row = start + qr / ratio;
      if (row < end) {
        int col = ((tile % 8) * 4 + warp) * 8 + fc + j % 2;
        int64_t dst =
            ((static_cast<int64_t>(row) * h + kh * ratio + qr % ratio) * splits + split) * 256 +
            col;
        partial[dst] = acc[tile][j] * denominator[rg];
      }
    }
  }
  if (warp == 0 && lane % 4 == 0) {
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int qr = qbase + r, row = start + qr / ratio;
      if (row < end)
        lse[(static_cast<int64_t>(row) * h + kh * ratio + qr % ratio) * splits + split] =
            maximum[rg] - log2f(denominator[rg]);
    }
  }
}
template <bool ModelShape, bool Grouped, typename Position, int Buffers>
void configure_attention_shared_memory(int shared_bytes) {
  // ModelShape, Grouped and Buffers determine this kernel's shared footprint.
  // Once-only setup also keeps subsequent CUDA Graph captures free of setters.
  static std::once_flag configured;
  std::call_once(configured, [shared_bytes] {
    TVM_FFI_ICHECK(cudaFuncSetAttribute(
                      attention_partial_kernel<ModelShape, Grouped, Position, Buffers>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes) == cudaSuccess);
  });
}
template <typename Position>
void launch_attention(TensorView q, TensorView cache, TensorView tables, TensorView lengths,
                      TensorView starts, TensorView partial, TensorView lse, int64_t first,
                      bool grouped, int max_query_len, cudaStream_t stream) {
  int h = q.size(1), hk = cache.size(3), splits = lse.size(2);
  int bq = grouped ? 32 : 16, bk = grouped ? 64 : 32;
  int tiles = ((grouped ? max_query_len : 1) * (h / hk) + bq - 1) / bq;
  // Only four ungrouped queries benefit from double buffering on B200.
  // Larger eager batches need the occupancy afforded by one shared KV tile.
  int shared_bytes =
      (bq * 256 + 2 * ((!grouped && q.size(0) == 4) ? 2 : 1) * bk * 256 + bq * 64) * 2 +
      (8 * bq) * 4;
  dim3 grid((grouped ? starts.size(0) - 1 : q.size(0)) * tiles, hk, splits);
#define ATTENTION(M, G, B)                                                                         \
  configure_attention_shared_memory<M, G, Position, B>(shared_bytes);                             \
  attention_partial_kernel<M, G, Position, B><<<grid, 128, shared_bytes, stream>>>(                 \
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(cache.data_ptr()), tables.data_ptr(), lengths.data_ptr(), \
      starts.data_ptr(), static_cast<float *>(partial.data_ptr()),                                 \
      static_cast<float *>(lse.data_ptr()), h, hk, tables.size(1), splits, first,                  \
      tables.dtype().bits == 64, lengths.dtype().bits == 64, starts.dtype().bits == 64, tiles,       \
      max_query_len)
  if (h == 24 && hk == 4) {
    if (grouped) {
      ATTENTION(true, true, 1);
    } else {
      if (q.size(0) == 4) {
        ATTENTION(true, false, 2);
      } else {
        ATTENTION(true, false, 1);
      }
    }
  } else {
    if (grouped) {
      ATTENTION(false, true, 1);
    } else {
      if (q.size(0) == 4) {
        ATTENTION(false, false, 2);
      } else {
        ATTENTION(false, false, 1);
      }
    }
  }
#undef ATTENTION
}
void attention_partial(TensorView q, TensorView cache, TensorView tables, TensorView lengths,
                       TensorView starts, TensorView partial, TensorView lse, int64_t first,
                       bool grouped, bool position64, int64_t max_query_len) {
  TVM_FFI_ICHECK(max_query_len >= 1 && max_query_len <= 8);
  auto stream = stream_for(q, "attention_partial");
  if (position64)
    launch_attention<int64_t>(q, cache, tables, lengths, starts, partial, lse, first, grouped,
                              max_query_len, stream);
  else
    launch_attention<int32_t>(q, cache, tables, lengths, starts, partial, lse, first, grouped,
                              max_query_len, stream);
  finish_cuda_launch(stream, "attention_partial");
}
template <int Splits>
__global__ void attention_merge_kernel(const float *__restrict__ partial,
                                       const float *__restrict__ lse,
                                       __nv_bfloat16 *__restrict__ out) {
  int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  __shared__ float weights[Splits], scratch[8];
  float log = tid < Splits ? lse[row * Splits + tid] : -INFINITY;
  float maximum = warp_max(log);
  if (lane == 0)
    scratch[warp] = maximum;
  __syncthreads();
  maximum = warp_max(lane < 4 ? scratch[lane] : -INFINITY);
  float safe_maximum = maximum == -INFINITY ? 0.f : maximum;
  float weight = tid < Splits ? softmax_exp2(log - safe_maximum) : 0.f;
  if (tid < Splits)
    weights[tid] = weight;
  float total = warp_sum(weight);
  if (lane == 0)
    scratch[warp + 4] = total;
  __syncthreads();
  total = warp_sum(lane < 4 ? scratch[lane + 4] : 0.f);
  float2 accum = make_float2(0.f, 0.f);
#pragma unroll 128
  for (int i = 0; i < Splits; ++i) {
    float2 value = reinterpret_cast<const float2 *>(
        partial)[(static_cast<int64_t>(row) * Splits + i) * 128 + tid];
    float w = weights[i];
    accum.x += value.x * w;
    accum.y += value.y * w;
  }
  reinterpret_cast<__nv_bfloat162 *>(out)[row * 128 + tid] =
      __floats2bfloat162_rn(total > 0.f ? accum.x / total : 0.f,
                           total > 0.f ? accum.y / total : 0.f);
}

void attention_merge(TensorView partial, TensorView lse, TensorView out) {
  auto stream = stream_for(out, "attention_merge");
#define MERGE(S)                                                                                   \
  attention_merge_kernel<S><<<lse.size(0) * lse.size(1), 128, 0, stream>>>(                         \
      static_cast<const float *>(partial.data_ptr()), static_cast<const float *>(lse.data_ptr()),  \
      static_cast<__nv_bfloat16 *>(out.data_ptr()))
  if (lse.size(2) == 16) {
    MERGE(16);
  } else if (lse.size(2) == 64) {
    MERGE(64);
  } else {
    TVM_FFI_ICHECK(lse.size(2) == 128);
    MERGE(128);
  }
#undef MERGE
  finish_cuda_launch(stream, "attention_merge");
}

// DSpark keeps the retained prefix in pages and the bidirectional seven-row
// noise block in separate tensors. No noise KV is written to the shared pages.
template <bool Wide, bool Narrow>
__global__ void __launch_bounds__(128, 1) dspark_append_kernel(
    const __nv_bfloat16 *__restrict__ key, const __nv_bfloat16 *__restrict__ value,
    __nv_bfloat16 *__restrict__ cache, const void *__restrict__ slots, int rows, int64_t capacity) {
  constexpr int Width = 1024;
  int token = blockIdx.x, column = threadIdx.x & 127;
  uint4 k, v;
  const auto *source_k = reinterpret_cast<const uint4 *>(key + int64_t(token) * Width);
  const auto *source_v = reinterpret_cast<const uint4 *>(value + int64_t(token) * Width);
  // Independent source reads overlap the slot load and address arithmetic.
  // All source rows exist even when a negative slot suppresses its cache write.
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(k.x), "=r"(k.y),
               "=r"(k.z), "=r"(k.w) : "l"(source_k + column) : "memory");
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y),
               "=r"(v.z), "=r"(v.w) : "l"(source_v + column) : "memory");
  int64_t slot = index_at(slots, Wide, token);
  if (static_cast<uint64_t>(slot) >= static_cast<uint64_t>(capacity)) {
    if (slot >= 0)
      asm volatile("trap;");
    return;
  }
  // Host dispatch proves a nonnegative slot fits int32 before narrowing.
  int64_t page, within;
  if constexpr (Narrow) {
    uint32_t index = static_cast<uint32_t>(slot);
    page = index / 784;
    within = index % 784;
  } else {
    page = slot / 784;
    within = slot % 784;
  }
  int64_t offset = (page * 1568 + within) * Width;
  auto *destination_k = reinterpret_cast<uint4 *>(cache + offset);
  auto *destination_v = reinterpret_cast<uint4 *>(cache + offset + 784 * Width);
  destination_k[column] = k;
  destination_v[column] = v;
}

template <bool Wide>
void launch_dspark_append(TensorView key, TensorView value, TensorView cache,
                          TensorView slots, cudaStream_t stream) {
  int rows = key.size(0);
#define DS_APPEND(Narrow)                                                         \
  dspark_append_kernel<Wide, Narrow><<<rows, 128, 0, stream>>>( \
      static_cast<const __nv_bfloat16 *>(key.data_ptr()),                               \
      static_cast<const __nv_bfloat16 *>(value.data_ptr()),                             \
      static_cast<__nv_bfloat16 *>(cache.data_ptr()), slots.data_ptr(), rows,            \
      cache.size(0) * 784)
  if (cache.size(0) * 784 <= 2147483647) {
    DS_APPEND(true);
  } else {
    DS_APPEND(false);
  }
#undef DS_APPEND
}

void dspark_append(TensorView key, TensorView value, TensorView cache, TensorView slots) {
  bool fast = key.ndim() == 3 && key.size(1) == 8 && key.size(2) == 128 &&
              aligned(key, 16) && aligned(value, 16) && aligned(cache, 16) &&
              disjoint_storage(cache, key) && disjoint_storage(cache, value) &&
              disjoint_storage(cache, slots);
  if (!fast) {
    // Preserve the generic operation for unaligned or overlapping views.
    append(key, value, cache, slots);
    return;
  }
  auto stream = stream_for(key, "dspark_append");
  if (slots.dtype().bits == 64) {
    launch_dspark_append<true>(key, value, cache, slots, stream);
  } else {
    launch_dspark_append<false>(key, value, cache, slots, stream);
  }
  finish_cuda_launch(stream, "dspark_append");
  record_variant(kAppend, true);
}

template <int Heads, int Warps, bool Wide, bool Small = false>
__global__ void dspark_norm_rope_kernel(
    const __nv_bfloat16 *__restrict__ x, const float *__restrict__ weight,
    const void *__restrict__ positions, const float *__restrict__ inv_freq,
    __nv_bfloat16 *__restrict__ output, int rows, int heads, int64_t row_stride,
    float attention_factor, float epsilon) {
  constexpr int Lanes = Small ? 64 : 32, Columns = 128 / Lanes;
  int head_row = Small ? blockIdx.x : blockIdx.x * Warps + threadIdx.x / 32;
  if (head_row >= rows)
    return;
  int count = Heads == 0 ? heads : Heads;
  int token = head_row / count, head = head_row % count;
  int lane = Small ? threadIdx.x : threadIdx.x & 31;
  const auto *row = x + static_cast<int64_t>(token) * row_stride + head * 128;
  float values[Columns], sum = 0.f;
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    values[j] = __bfloat162float(row[lane + j * Lanes]);
    sum += values[j] * values[j];
  }
  sum = warp_sum(sum);
  if constexpr (Small) {
    __shared__ float partial[2];
    if ((lane & 31) == 0)
      partial[lane / 32] = sum;
    __syncthreads();
    sum = partial[0] + partial[1];
  }
  float inverse = rsqrtf(sum / 128.f + epsilon);
  __nv_bfloat16 weighted[Columns];
#pragma unroll
  for (int j = 0; j < Columns / 2; ++j) {
    auto normalized = __floats2bfloat162_rn(values[j] * inverse,
                                            values[j + Columns / 2] * inverse);
    auto multiplier = __floats2bfloat162_rn(weight[lane + j * Lanes],
                                            weight[lane + (j + Columns / 2) * Lanes]);
    auto result = __hmul2_rn(normalized, multiplier);
    weighted[j] = __low2bfloat16(result);
    weighted[j + Columns / 2] = __high2bfloat16(result);
  }
  // A lane owns both NeoX halves. Reuse the trigonometry without another load.
#pragma unroll
  for (int j = 0; j < Columns / 2; ++j) {
    int column = lane + j * Lanes;
    double phase = static_cast<double>(index_at(positions, Wide, token)) *
                   static_cast<double>(inv_freq[column]);
    constexpr double tau = 6.283185307179586476925286766559;
    phase -= nearbyint(phase / tau) * tau;
    float sine, cosine;
    sincosf(static_cast<float>(phase), &sine, &cosine);
    auto pair = __halves2bfloat162(weighted[j], weighted[j + Columns / 2]);
    auto c = __float2bfloat16_rn(cosine * attention_factor);
    auto v = __float2bfloat16_rn(sine * attention_factor);
    auto own_cos = __hmul2_rn(pair, __halves2bfloat162(c, c));
    auto own_sin = __hmul2_rn(pair, __halves2bfloat162(v, v));
    auto *destination = output + static_cast<int64_t>(head_row) * 128;
    destination[column] = __hsub_rn(__low2bfloat16(own_cos), __high2bfloat16(own_sin));
    destination[column + 64] = __hadd_rn(__high2bfloat16(own_cos), __low2bfloat16(own_sin));
  }
}

template <int Heads, bool Wide>
void launch_dspark_norm(TensorView x, TensorView weight, TensorView positions,
                        TensorView inv_freq, TensorView output, float factor,
                        float epsilon, cudaStream_t stream) {
  int rows = x.size(0) * x.size(1);
#define DS_NORM(W)                                                                      \
  dspark_norm_rope_kernel<Heads, W, Wide><<<(rows + W - 1) / W, W * 32, 0, stream>>>(      \
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                  \
      static_cast<const float *>(weight.data_ptr()), positions.data_ptr(),              \
      static_cast<const float *>(inv_freq.data_ptr()),                                  \
      static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, x.size(1), x.stride(0),       \
      factor, epsilon)
  if (rows >= 4096) {
    DS_NORM(4);
  } else {
    dspark_norm_rope_kernel<Heads, 1, Wide, true><<<rows, 64, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(x.data_ptr()),
        static_cast<const float *>(weight.data_ptr()), positions.data_ptr(),
        static_cast<const float *>(inv_freq.data_ptr()),
        static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, x.size(1), x.stride(0),
        factor, epsilon);
  }
#undef DS_NORM
}

void dspark_norm_rope(TensorView x, TensorView weight, TensorView positions,
                      TensorView inv_freq, TensorView output, double attention_factor,
                      double epsilon) {
  TVM_FFI_ICHECK(x.ndim() == 3 && x.size(2) == 128 && x.size(0) > 0);
  TVM_FFI_ICHECK(weight.numel() == 128 && inv_freq.numel() == 64);
  TVM_FFI_ICHECK(is_index_dtype(positions) && positions.numel() == x.size(0));
  auto stream = stream_for(x, "dspark_norm_rope");
#define DS_NORM_HEADS(H)                                                                \
  if (positions.dtype().bits == 64) {                                                   \
    launch_dspark_norm<H, true>(x, weight, positions, inv_freq, output,                  \
                                attention_factor, epsilon, stream);                    \
  } else {                                                                             \
    launch_dspark_norm<H, false>(x, weight, positions, inv_freq, output,                 \
                                 attention_factor, epsilon, stream);                   \
  }
  if (x.size(1) == 8) {
    DS_NORM_HEADS(8)
  } else if (x.size(1) == 32) {
    DS_NORM_HEADS(32)
  } else {
    DS_NORM_HEADS(0)
  }
#undef DS_NORM_HEADS
  finish_cuda_launch(stream, "dspark_norm_rope");
}

__device__ __forceinline__ void stage_dspark_copy(__nv_bfloat16 *destination,
                                                const __nv_bfloat16 *source, bool valid) {
  if ((reinterpret_cast<uintptr_t>(source) & 15) == 0) {
    unsigned target = __cvta_generic_to_shared(destination);
    int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(target),
                 "l"(source), "r"(bytes) : "memory");
  } else {
    // A contiguous BF16 view can start at an odd storage offset. Scalar
    // staging preserves that legal layout without changing the CUDA provider.
#pragma unroll
    for (int j = 0; j < 8; ++j)
      destination[j] = valid ? source[j] : __float2bfloat16_rn(0.f);
  }
}

__device__ __forceinline__ void stage_dspark_kv(
    const __nv_bfloat16 *cache, const int *tables, const __nv_bfloat16 *block_key,
    const __nv_bfloat16 *block_value, int seq, int table_width, int cache_pages,
    int base, int context, int kh, __nv_bfloat16 *key, __nv_bfloat16 *value) {
#pragma unroll
  for (int i = threadIdx.x; i < 64 * 16; i += 128) {
    int pos = base + i / 16, col = (i % 16) * 8;
    const __nv_bfloat16 *source_k = block_key, *source_v = block_value;
    bool valid = pos < context + 7;
    if (pos < context) {
      int page = tables[seq * table_width + pos / 784];
      if (page < 0 || page >= cache_pages)
        asm volatile("trap;");
      int64_t offset = ((static_cast<int64_t>(page) * 1568 + pos % 784) * 8 + kh) * 128 + col;
      source_k = cache + offset;
      source_v = source_k + 784 * 8 * 128;
    } else if (valid) {
      int64_t offset = ((static_cast<int64_t>(seq) * 7 + pos - context) * 8 + kh) * 128 + col;
      source_k = block_key + offset;
      source_v = block_value + offset;
    }
    stage_dspark_copy(key + shared_index(i / 16, col, 128), source_k, valid);
    stage_dspark_copy(value + shared_index(i / 16, col, 128), source_v, valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__global__ void dspark_attention_partial_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ cache,
    const int *__restrict__ tables, const int *__restrict__ contexts,
    const __nv_bfloat16 *__restrict__ block_key, const __nv_bfloat16 *__restrict__ block_value,
    float *__restrict__ partial, float *__restrict__ lse, int table_width, int cache_pages,
    int splits) {
  constexpr int BQ = 32, BK = 64, Buffers = 1;
  constexpr int RowGroups = BQ / 8, ScoreTiles = BQ * BK / 512, AccTiles = BQ / 4;
  constexpr int h = 32, hk = 8;
  extern __shared__ __align__(32) unsigned char storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *key_buffers = q + BQ * 128;
  auto *value_buffers = key_buffers + Buffers * BK * 128;
  auto *p = value_buffers + Buffers * BK * 128;
  auto *maxima = reinterpret_cast<float *>(p + BQ * 64);
  auto *sums = maxima + 4 * BQ;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int fr = lane / 4, fc = (lane % 4) * 2;
  const int seq = blockIdx.x, qbase = 0;
  const int kh = blockIdx.y, split = blockIdx.z, ratio = h / hk;
  const int start = seq * 7, end = start + 7;
  const int context = contexts[seq], last = context + 7;
  if (context < 0 || context > table_width * 784)
    asm volatile("trap;");
  const int chunk = ((last + splits * BK - 1) / (splits * BK)) * BK;
  const int begin = split * chunk;
  float acc[AccTiles][4] = {}, denominator[RowGroups] = {}, maximum[RowGroups];
#pragma unroll
  for (int rg = 0; rg < RowGroups; ++rg)
    maximum[rg] = -INFINITY;
#pragma unroll
  for (int i = threadIdx.x; i < BQ * 16; i += 128) {
    int r = i / 16, col = (i % 16) * 8;
    int row = start + (qbase + r) / ratio;
    auto *destination = q + shared_index(r, col, 128);
    int64_t offset = (static_cast<int64_t>(row) * h + kh * ratio + (qbase + r) % ratio) * 128 + col;
    if ((reinterpret_cast<uintptr_t>(query) & 15) == 0) {
      uint4 values =
          row < end ? *reinterpret_cast<const uint4 *>(query + offset) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4 *>(destination) = values;
    } else {
      // A contiguous tensor may begin at a BF16 storage offset, not a16B boundary.
#pragma unroll
      for (int j = 0; j < 8; ++j)
        destination[j] = row < end ? query[offset + j] : __float2bfloat16(0);
    }
  }
  __syncthreads();
  const int limit = min(begin + chunk, last);
  if (begin < limit)
    stage_dspark_kv(cache, tables, block_key, block_value, seq, table_width, cache_pages,
                    begin, context, kh, key_buffers, value_buffers);
  int step = 0;
  for (int base = begin; base < limit; base += BK, ++step) {
    auto *k = key_buffers + (step % Buffers) * BK * 128;
    auto *v = value_buffers + (step % Buffers) * BK * 128;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if constexpr (Buffers == 2) {
      if (base + BK < limit)
        stage_dspark_kv(cache, tables, block_key, block_value, seq, table_width, cache_pages,
                        base + BK, context, kh, key_buffers, value_buffers);
    }
    float score[ScoreTiles][4] = {};
#pragma unroll 8
    for (int inner = 0; inner < 128; inner += 16) {
      unsigned qa[BQ / 16][4], kb[BK / 32][2];
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg)
        load_a(qa[rg], q + rg * 16 * 128, inner, lane, 128);
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt)
        load_k(kb[kt], k, inner, kt * 4 + warp, lane, 128);
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg) {
#pragma unroll
        for (int kt = 0; kt < BK / 32; ++kt) {
          mma_bf16(score[rg * (BK / 32) + kt], qa[rg][0], qa[rg][1], qa[rg][2], qa[rg][3],
                   kb[kt][0], kb[kt][1]);
        }
      }
    }
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int row = start + (qbase + r) / ratio;
      int length = row < end ? last : 0;
      float m = -INFINITY;
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt) {
        int tile = (rg / 2) * (BK / 32) + kt;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          int col = (kt * 4 + warp) * 8 + fc + j;
          float value = row < end && base + col < length
                            ? score[tile][(rg % 2) * 2 + j] * 0.12751743082459868f
                            : -INFINITY;
          score[tile][(rg % 2) * 2 + j] = value;
          m = fmaxf(m, value);
        }
      }
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
      if (lane % 4 == 0)
        maxima[warp * BQ + r] = m;
    }
    __syncthreads();
    float alpha[RowGroups];
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      float m = maximum[rg];
#pragma unroll
      for (int w = 0; w < 4; ++w)
        m = fmaxf(m, maxima[w * BQ + r]);
      float safe = m == -INFINITY ? 0.f : m;
      alpha[rg] = softmax_exp2(maximum[rg] - safe);
      maximum[rg] = m;
      float sum = 0;
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt) {
        int tile = (rg / 2) * (BK / 32) + kt;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          int col = (kt * 4 + warp) * 8 + fc + j;
          float probability = softmax_exp2(score[tile][(rg % 2) * 2 + j] - safe);
          sum += probability;
          p[shared_index(r, col, 64)] = __float2bfloat16_rn(probability);
        }
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      sum += __shfl_xor_sync(0xffffffff, sum, 2);
      if (lane % 4 == 0)
        sums[warp * BQ + r] = sum;
    }
    __syncthreads();
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      float sum = 0;
#pragma unroll
      for (int w = 0; w < 4; ++w)
        sum += sums[w * BQ + r];
      denominator[rg] = denominator[rg] * alpha[rg] + sum;
    }
#pragma unroll
    for (int tile = 0; tile < AccTiles; ++tile) {
      // Four column warps make the row group independent of warp. Keep this
      // index static so alpha and denominator remain register arrays.
      int row_base = (tile / 4) * 16;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[tile][j] *= alpha[row_base / 8 + j / 2];
    }
#pragma unroll
    for (int inner = 0; inner < BK; inner += 16) {
      unsigned pa[BQ / 16][4];
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg)
        load_a(pa[rg], p + rg * 16 * 64, inner, lane, 64);
#pragma unroll
      for (int ct = 0; ct < 4; ++ct) {
        unsigned vb[2];
        load_v(vb, v, inner, (ct * 4 + warp) * 8, lane, 128);
#pragma unroll
        for (int rg = 0; rg < BQ / 16; ++rg)
          mma_bf16(acc[rg * 4 + ct], pa[rg][0], pa[rg][1], pa[rg][2], pa[rg][3], vb[0], vb[1]);
      }
    }
    __syncthreads();
    if constexpr (Buffers == 1) {
      if (base + BK < limit)
        stage_dspark_kv(cache, tables, block_key, block_value, seq, table_width, cache_pages,
                        base + BK, context, kh, key_buffers, value_buffers);
    }
  }
#pragma unroll
  for (int rg = 0; rg < RowGroups; ++rg)
    denominator[rg] = 1.f / (denominator[rg] > 0 ? denominator[rg] : 1.f);
#pragma unroll
  for (int tile = 0; tile < AccTiles; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int rg = (tile / 4) * 2 + j / 2;
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int qr = qbase + r, row = start + qr / ratio;
      if (row < end) {
        int col = ((tile % 4) * 4 + warp) * 8 + fc + j % 2;
        int64_t dst =
            ((static_cast<int64_t>(row) * h + kh * ratio + qr % ratio) * splits + split) * 128 +
            col;
        partial[dst] = acc[tile][j] * denominator[rg];
      }
    }
  }
  if (warp == 0 && lane % 4 == 0) {
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int qr = qbase + r, row = start + qr / ratio;
      if (row < end)
        lse[(static_cast<int64_t>(row) * h + kh * ratio + qr % ratio) * splits + split] =
            maximum[rg] - log2f(denominator[rg]);
    }
  }
}

template <typename Weight, int Threads, int Vector, bool Streaming>
__global__ void dspark_rms_norm_kernel(const __nv_bfloat16 *__restrict__ input,
                                      const Weight *__restrict__ weight,
                                      __nv_bfloat16 *__restrict__ output,
                                      float epsilon) {
  constexpr int Width = 5120, Chunks = Width / (Threads * Vector);
  static_assert(Width % (Threads * Vector) == 0);
  int row = blockIdx.x, lane = threadIdx.x & 31;
  float values[Chunks][Vector], partial_sums[Vector] = {};
#pragma unroll
  for (int i = 0; i < Chunks; ++i) {
    int column = threadIdx.x * Vector + i * Threads * Vector;
    auto loaded = load_aligned_vector<__nv_bfloat16, Vector>(
        input + static_cast<int64_t>(row) * Width + column);
#pragma unroll
    for (int j = 0; j < Vector; ++j) {
      values[i][j] = __bfloat162float(loaded.value[j]);
      partial_sums[j] += values[i][j] * values[i][j];
    }
  }
  float total = 0.f;
#pragma unroll
  for (int j = 0; j < Vector; ++j)
    total += partial_sums[j];
  total = warp_sum(total);
  __shared__ float sums[Threads / 32];
  if (lane == 0)
    sums[threadIdx.x / 32] = total;
  __syncthreads();
  total = warp_sum(lane < Threads / 32 ? sums[lane] : 0.f);
  float inverse = rsqrtf(total * (1.f / Width) + epsilon);
#pragma unroll
  for (int i = 0; i < Chunks; ++i) {
    int column = threadIdx.x * Vector + i * Threads * Vector;
    auto multipliers = load_aligned_vector<Weight, Vector>(weight + column);
    AlignedVector<__nv_bfloat16, Vector> result;
    // Preserve the two checkpoint BF16 roundings with packed multiplication.
#pragma unroll
    for (int j = 0; j < Vector; j += 2) {
      if constexpr (Vector == 1) {
        auto normalized = __float2bfloat16_rn(values[i][j] * inverse);
        auto multiplier = __float2bfloat16_rn(float(multipliers.value[j]));
        result.value[j] = __hmul_rn(normalized, multiplier);
      } else {
        auto normalized = __floats2bfloat162_rn(values[i][j] * inverse,
                                                values[i][j + 1] * inverse);
        __nv_bfloat162 multiplier;
        if constexpr (std::is_same_v<Weight, float>) {
          multiplier = __floats2bfloat162_rn(multipliers.value[j], multipliers.value[j + 1]);
        } else {
          multiplier = __halves2bfloat162(multipliers.value[j], multipliers.value[j + 1]);
        }
        auto product = __hmul2_rn(normalized, multiplier);
        result.value[j] = __low2bfloat16(product);
        result.value[j + 1] = __high2bfloat16(product);
      }
    }
    auto *destination = output + static_cast<int64_t>(row) * Width + column;
    store_rms_vector<Vector, Streaming>(destination, result);
  }
}

template <typename Weight>
void launch_dspark_rms(TensorView input, TensorView weight, TensorView output,
                       float epsilon, cudaStream_t stream) {
#define DS_RMS(T, V, S)                                                                     \
  dspark_rms_norm_kernel<Weight, T, V, S><<<input.size(0), T, 0, stream>>>(                   \
      static_cast<const __nv_bfloat16 *>(input.data_ptr()),                              \
      static_cast<const Weight *>(weight.data_ptr()),                                    \
      static_cast<__nv_bfloat16 *>(output.data_ptr()), epsilon)
  if (aligned(input, 16) && aligned(output, 16) && aligned(weight, sizeof(Weight) * 8)) {
    if (input.size(0) >= 4096) {
      DS_RMS(256, 4, true);
    } else {
      DS_RMS(256, 4, false);
    }
  } else {
    // Contiguous views can have an odd BF16 storage offset.
    DS_RMS(256, 1, false);
  }
#undef DS_RMS
}

void dspark_rms_norm(TensorView input, TensorView weight, TensorView output, double epsilon) {
  TVM_FFI_ICHECK(input.ndim() == 2 && input.size(1) == 5120 && weight.ndim() == 1 &&
                weight.size(0) == 5120 && output.ndim() == 2 &&
                output.size(0) == input.size(0) && output.size(1) == 5120);
  TVM_FFI_ICHECK(has_dtype(input, kDLBfloat, 16) && has_dtype(output, kDLBfloat, 16) &&
                (has_dtype(weight, kDLBfloat, 16) || has_dtype(weight, kDLFloat, 32)));
  auto stream = stream_for(input, "dspark_rms_norm");
  if (has_dtype(weight, kDLFloat, 32)) {
    launch_dspark_rms<float>(input, weight, output, epsilon, stream);
  } else {
    launch_dspark_rms<__nv_bfloat16>(input, weight, output, epsilon, stream);
  }
  finish_cuda_launch(stream, "dspark_rms_norm");
}

void dspark_attention_partial(TensorView query, TensorView cache, TensorView tables,
                              TensorView contexts, TensorView block_key, TensorView block_value,
                              TensorView partial, TensorView lse) {
  TVM_FFI_ICHECK(query.ndim() == 4 && query.size(1) == 7 && query.size(2) == 32 &&
                query.size(3) == 128);
  TVM_FFI_ICHECK(cache.ndim() == 5 && cache.size(1) == 2 && cache.size(2) == 784 &&
                cache.size(3) == 8 && cache.size(4) == 128);
  TVM_FFI_ICHECK(has_dtype(tables, kDLInt, 32) && has_dtype(contexts, kDLInt, 32));
  TVM_FFI_ICHECK(lse.size(2) == 16 || lse.size(2) == 64);
  auto stream = stream_for(query, "dspark_attention_partial");
  constexpr int shared_bytes = (32 * 128 + 2 * 64 * 128 + 32 * 64) * 2 + 8 * 32 * 4;
  dim3 grid(query.size(0), 8, lse.size(2));
  dspark_attention_partial_kernel<<<grid, 128, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16 *>(query.data_ptr()),
      static_cast<const __nv_bfloat16 *>(cache.data_ptr()),
      static_cast<const int *>(tables.data_ptr()), static_cast<const int *>(contexts.data_ptr()),
      static_cast<const __nv_bfloat16 *>(block_key.data_ptr()),
      static_cast<const __nv_bfloat16 *>(block_value.data_ptr()),
      static_cast<float *>(partial.data_ptr()), static_cast<float *>(lse.data_ptr()),
      tables.size(1), cache.size(0), lse.size(2));
  finish_cuda_launch(stream, "dspark_attention_partial");
}
template <int Splits>
__global__ void dspark_attention_merge_kernel(const float *__restrict__ partial,
                                       const float *__restrict__ lse,
                                       __nv_bfloat16 *__restrict__ out) {
  int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  __shared__ float weights[Splits], scratch[8];
  float log = tid < Splits ? lse[row * Splits + tid] : -INFINITY;
  float maximum = warp_max(log);
  if (lane == 0)
    scratch[warp] = maximum;
  __syncthreads();
  maximum = warp_max(lane < 4 ? scratch[lane] : -INFINITY);
  float safe_maximum = maximum == -INFINITY ? 0.f : maximum;
  float weight = tid < Splits ? softmax_exp2(log - safe_maximum) : 0.f;
  if (tid < Splits)
    weights[tid] = weight;
  float total = warp_sum(weight);
  if (lane == 0)
    scratch[warp + 4] = total;
  __syncthreads();
  total = warp_sum(lane < 4 ? scratch[lane + 4] : 0.f);
  float2 accum = make_float2(0.f, 0.f);
#pragma unroll 128
  for (int i = 0; i < Splits; ++i) {
    float2 value = make_float2(0.f, 0.f);
    if (tid < 64)
      value = reinterpret_cast<const float2 *>(
        partial)[(static_cast<int64_t>(row) * Splits + i) * 64 + tid];
    float w = weights[i];
    accum.x += value.x * w;
    accum.y += value.y * w;
  }
  if (tid < 64)
    reinterpret_cast<__nv_bfloat162 *>(out)[row * 64 + tid] =
      __floats2bfloat162_rn(total > 0.f ? accum.x / total : 0.f,
                           total > 0.f ? accum.y / total : 0.f);
}

void dspark_attention_merge(TensorView partial, TensorView lse, TensorView output) {
  auto stream = stream_for(output, "dspark_attention_merge");
#define DSPARK_MERGE(S) \
  dspark_attention_merge_kernel<S><<<lse.size(0) * lse.size(1), 128, 0, stream>>>( \
      static_cast<const float *>(partial.data_ptr()), static_cast<const float *>(lse.data_ptr()), \
      static_cast<__nv_bfloat16 *>(output.data_ptr()))
  if (lse.size(2) == 16) {
    DSPARK_MERGE(16);
  } else {
    TVM_FFI_ICHECK(lse.size(2) == 64);
    DSPARK_MERGE(64);
  }
#undef DSPARK_MERGE
  finish_cuda_launch(stream, "dspark_attention_merge");
}

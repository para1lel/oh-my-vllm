#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

using tvm::ffi::TensorView;

// BF16 SiLU keeps both rounding boundaries. The fast exponential produces
// identical rounded SiLU for every finite BF16 input on the required SM100.
// One warp owns one complete 128-value scaling group. Independent warps share
// a CTA; reductions do not require shared memory or block-wide synchronization.
template <typename Input> struct alignas(sizeof(Input) * 4) Four {
  Input values[4];
};
template <typename Input> __device__ Four<Input> load_four(const Input *x) {
  if ((reinterpret_cast<uintptr_t>(x) & (sizeof(Input) * 4 - 1)) == 0)
    return *reinterpret_cast<const Four<Input> *>(x);
  Four<Input> values;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    values.values[j] = x[j];
  return values;
}
template <typename Input, bool Silu, bool Column, int RowsPerWarp, bool Flat = false, int Width = 0>
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
    if constexpr (Silu)
      packed = load_four(x + offset);
    Four<Input> up;
    if constexpr (Silu)
      up = load_four(x + offset + width);
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
        float v = static_cast<float>(x[row * width + col + lane + j * 32]);
        value[j] = v;
        maximum = fmaxf(maximum, fabsf(v));
      }
    }
    // Absolute nonnegative FP32 bit patterns preserve unsigned ordering.
    maximum = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(maximum)));
    float scale = __fdiv_rn(fmaxf(maximum, 1e-10f), 448.f);
    if (lane == 0)
      scales[Column ? group * rows + row : row * groups + group] = scale;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      value[j] = fminf(448.f, fmaxf(-448.f, __fdiv_rn(value[j], scale)));
    if constexpr (Silu)
      reinterpret_cast<__nv_fp8x4_e4m3 *>(out)[(row * width + col) / 4 + lane] =
          __nv_fp8x4_e4m3(make_float4(value[0], value[1], value[2], value[3]));
    else {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        out[row * width + col + lane + j * 32] = __nv_fp8_e4m3(value[j]);
    }
  }
}

template <typename Input, int Width>
void launch_quantize(TensorView x, TensorView out, TensorView scales, bool column, bool silu,
                     cudaStream_t stream) {
  int rows = out.size(0), width = out.size(1);
#define CALL(S, C, R, T)                                                                           \
  quantize_kernel<Input, S, C, R, false, Width>                                                    \
      <<<dim3((rows + (T / 32) * R - 1) / ((T / 32) * R), width / 128), T, 0, stream>>>(           \
          static_cast<const Input *>(x.data_ptr()), static_cast<__nv_fp8_e4m3 *>(out.data_ptr()),  \
          static_cast<float *>(scales.data_ptr()), rows, width)
#define LAUNCH(S, C)                                                                               \
  if (width / 128 > 65535 || (S && rows >= 4 && rows < 128)) {                                     \
    quantize_kernel<Input, S, C, 1, true, Width>                                                   \
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

template <typename Input>
void dispatch(TensorView x, TensorView out, TensorView scales, bool column, bool silu,
              cudaStream_t stream) {
  switch (out.size(1)) {
  case 5120:
    launch_quantize<Input, 5120>(x, out, scales, column, silu, stream);
    break;
  case 6144:
    launch_quantize<Input, 6144>(x, out, scales, column, silu, stream);
    break;
  case 17408:
    launch_quantize<Input, 17408>(x, out, scales, column, silu, stream);
    break;
  default:
    launch_quantize<Input, 0>(x, out, scales, column, silu, stream);
  }
}
void quantize(TensorView x, TensorView out, TensorView scales, bool column, bool silu) {
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  if (x.dtype().code == kDLBfloat)
    dispatch<__nv_bfloat16>(x, out, scales, column, silu, stream);
  else if (x.dtype().bits == 16)
    dispatch<__half>(x, out, scales, column, silu, stream);
  else
    dispatch<float>(x, out, scales, column, silu, stream);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA quantize launch failed";
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
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  int n = out.size(0) * out.size(1);
  if (!n)
    return;
  silu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const __nv_bfloat16 *>(x.data_ptr()),
                                                   static_cast<__nv_bfloat16 *>(out.data_ptr()),
                                                   out.size(0), out.size(1));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA SiLU launch failed";
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
cudaStream_t stream_for(TensorView x) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
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
  beta[i] = 1.f / (1.f + expf(-b));
}
void gates(TensorView ba, TensorView log, TensorView bias, TensorView decay, TensorView beta) {
  gates_kernel<<<(ba.size(0) * 48 + 255) / 256, 256, 0, stream_for(ba)>>>(
      static_cast<const __nv_bfloat16 *>(ba.data_ptr()), static_cast<const float *>(log.data_ptr()),
      static_cast<const float *>(bias.data_ptr()), static_cast<float *>(decay.data_ptr()),
      static_cast<float *>(beta.data_ptr()), ba.size(0));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA gate launch failed";
}

template <typename T> struct alignas(sizeof(T) * 4) AlignedFour {
  T value[4];
};
template <typename T> __device__ AlignedFour<T> load_aligned_four(const T *p) {
  return *reinterpret_cast<const AlignedFour<T> *>(p);
}
template <bool Residual, int Threads>
__global__ void rms5120_kernel(const __nv_bfloat16 *__restrict__ x,
                               const __nv_bfloat16 *__restrict__ residual,
                               const float *__restrict__ w, __nv_bfloat16 *__restrict__ out,
                               __nv_bfloat16 *__restrict__ summed, float eps) {
  constexpr int Chunks = 5120 / (Threads * 4);
  int row = blockIdx.x, lane = threadIdx.x & 31;
  float values[Chunks][4], total = 0;
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int col = chunk * Threads * 4 + threadIdx.x * 4;
    auto q = load_aligned_four(x + static_cast<int64_t>(row) * 5120 + col);
    AlignedFour<__nv_bfloat16> r;
    if constexpr (Residual)
      r = load_aligned_four(residual + static_cast<int64_t>(row) * 5120 + col);
    if constexpr (Residual) {
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        auto pair = __hadd2(__halves2bfloat162(q.value[j], q.value[j + 1]),
                            __halves2bfloat162(r.value[j], r.value[j + 1]));
        q.value[j] = __low2bfloat16(pair);
        q.value[j + 1] = __high2bfloat16(pair);
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float v = __bfloat162float(q.value[j]);
      values[chunk][j] = v;
      total += v * v;
    }
    if constexpr (Residual)
      *reinterpret_cast<AlignedFour<__nv_bfloat16> *>(summed + static_cast<int64_t>(row) * 5120 +
                                                      col) = q;
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
    int col = chunk * Threads * 4 + threadIdx.x * 4;
    auto weight = load_aligned_four(w + col);
    AlignedFour<__nv_bfloat16> value;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      value.value[j] = __float2bfloat16_rn(values[chunk][j] * inv * weight.value[j]);
    *reinterpret_cast<AlignedFour<__nv_bfloat16> *>(out + static_cast<int64_t>(row) * 5120 + col) =
        value;
  }
}

// Public wrappers allocate both destinations independently. The fast path is
// entered only for aligned contiguous model-width rows; other layouts use RMS.
void launch_rms5120(TensorView x, TensorView residual, TensorView weight, TensorView out,
                    TensorView summed, bool add, float epsilon) {
  auto stream = stream_for(x);
#define RMS5120(R)                                                                                 \
  rms5120_kernel<R, 256><<<x.size(0), 256, 0, stream>>>(                                           \
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),                                     \
      static_cast<const float *>(weight.data_ptr()), static_cast<__nv_bfloat16 *>(out.data_ptr()), \
      static_cast<__nv_bfloat16 *>(summed.data_ptr()), epsilon)
  if (add) {
    RMS5120(true);
  } else {
    RMS5120(false);
  }
#undef RMS5120
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA model-width RMS launch failed";
}
bool aligned(TensorView tensor, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(tensor.data_ptr()) & (bytes - 1)) == 0;
}

struct Strides {
  int64_t token, head, dim;
};
template <bool Large, bool Gated, bool Residual>
__global__ void rms_kernel(const __nv_bfloat16 *__restrict__ x, const float *__restrict__ weight,
                           const __nv_bfloat16 *__restrict__ gate, __nv_bfloat16 *__restrict__ out,
                           __nv_bfloat16 *__restrict__ summed, int rows, int heads, int width,
                           Strides xs, Strides gs, float epsilon) {
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
    return;
  }
  Strides xs{x.stride(0), x.stride(1), x.stride(2)};
  Strides gs{gate.stride(0), gate.stride(1), gate.stride(2)};
#define RMS(L, G)                                                                                  \
  rms_kernel<L, G, false><<<L ? rows : (rows + 3) / 4, L ? 256 : 128, 0, stream_for(x)>>>(         \
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                            \
      static_cast<const float *>(weight.data_ptr()),                                               \
      static_cast<const __nv_bfloat16 *>(gate.data_ptr()),                                         \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), nullptr, rows, h, d, xs, gs, epsilon)
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
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA RMS launch failed";
}
void add_rms(TensorView x, TensorView residual, TensorView weight, TensorView summed,
             TensorView out) {
  int d = x.size(1), rows = x.size(0);
  if (d == 5120 && aligned(x, 8) && aligned(residual, 8) && aligned(weight, 16)) {
    launch_rms5120(x, residual, weight, out, summed, true, 1e-6f);
    return;
  }
  rms_kernel<true, false, true><<<rows, 256, 0, stream_for(x)>>>(
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      static_cast<const float *>(weight.data_ptr()),
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
      static_cast<__nv_bfloat16 *>(out.data_ptr()), static_cast<__nv_bfloat16 *>(summed.data_ptr()),
      rows, 1, d, {d, d, 1}, {d, d, 1}, 1e-6f);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA residual RMS launch failed";
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
  rms_rope_kernel<<<(rows + 3) / 4, 128, 0, stream_for(x)>>>(
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      static_cast<const float *>(weight.data_ptr()), positions.data_ptr(),
      static_cast<__nv_bfloat16 *>(out.data_ptr()), rows, x.size(1),
      {x.stride(0), x.stride(1), x.stride(2)}, positions.dtype().bits == 64);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA RMS/RoPE launch failed";
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
  rope_kernel<<<(n * h * d + 255) / 256, 256, 0, stream_for(x)>>>(
      static_cast<const __nv_bfloat16 *>(x.data_ptr()), positions.data_ptr(),
      static_cast<__nv_bfloat16 *>(out.data_ptr()), n, h, d, rotary, theta,
      positions.dtype().bits == 64);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA RoPE launch failed";
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
  int rows = q.size(0) * q.size(1);
  qk_kernel<<<(rows + 3) / 4, 128, 0, stream_for(q)>>>(
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()), static_cast<__nv_bfloat16 *>(oq.data_ptr()),
      static_cast<__nv_bfloat16 *>(ok.data_ptr()), rows, q.size(1), q.stride(0), k.stride(0));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA Q/K normalization launch failed";
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
  dim3 grid(reads.size(0), v.size(1), 8);
#define REC(S)                                                                                     \
  recurrent_kernel<S><<<grid, 128, 0, stream_for(q)>>>(                                            \
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),                                            \
      static_cast<const float *>(decay.data_ptr()), static_cast<const float *>(beta.data_ptr()),   \
      static_cast<S *>(pool.data_ptr()), starts.data_ptr(), reads.data_ptr(), writes.data_ptr(),   \
      starts.dtype().bits == 64, reads.dtype().bits == 64, writes.dtype().bits == 64,              \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), q.size(1), v.size(1), q.stride(0),             \
      k.stride(0), v.stride(0))
  if (pool.dtype().code == kDLBfloat) {
    REC(__nv_bfloat16);
  } else {
    REC(float);
  }
#undef REC
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA recurrence launch failed";
}

template <bool Vector>
__global__ void append_kernel(const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *cache,
                              const void *slots, int width, bool wide) {
  int token = blockIdx.x;
  int64_t slot = index_at(slots, wide, token);
  if (slot < 0)
    return;
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
#define APPEND(V)                                                                                  \
  append_kernel<V>                                                                                 \
      <<<k.size(0), 128, 0, stream_for(k)>>>(static_cast<const __nv_bfloat16 *>(k.data_ptr()),     \
                                             static_cast<const __nv_bfloat16 *>(v.data_ptr()),     \
                                             static_cast<__nv_bfloat16 *>(cache.data_ptr()),       \
                                             slots.data_ptr(), width, slots.dtype().bits == 64)
  if (vector) {
    APPEND(true);
  } else {
    APPEND(false);
  }
#undef APPEND
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA KV append launch failed";
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
#define CONV(R)                                                                                    \
  convolution_kernel<R><<<dim3((n + R - 1) / R, (c + 127) / 128), 128, 0, stream_for(x)>>>(        \
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
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA convolution launch failed";
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
                                       int warp, int lane) {
  unsigned address = __cvta_generic_to_shared(
      data + shared_index(warp * 8 + lane % 8, inner + ((lane / 8) % 2) * 8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(address));
}
__device__ __forceinline__ void load_v(unsigned (&b)[2], const __nv_bfloat16 *data, int inner,
                                       int col, int lane) {
  unsigned address = __cvta_generic_to_shared(data + shared_index(inner + lane % 16, col));
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
#pragma unroll
  for (int i = threadIdx.x; i < BK * 32; i += 128) {
    Position pos = base + i / 32;
    int64_t page =
        index_at(tables, tw, static_cast<int64_t>(start) * table_width + min(pos, last - 1) / 784);
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
                         bool sw, int query_tiles) {
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
#pragma unroll
    for (int tile = 0; tile < ScoreTiles; ++tile) {
      int column_tile = (tile * 4 + warp) % (BK / 8);
      int row_base = (tile * 4 + warp) / (BK / 8) * 16;
#pragma unroll 1
      for (int inner = 0; inner < 256; inner += 16) {
        unsigned qa[4], kb[2];
        load_a(qa, q + row_base * 256, inner, lane);
        load_k(kb, k, inner, column_tile, lane);
        mma_bf16(score[tile], qa[0], qa[1], qa[2], qa[3], kb[0], kb[1]);
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
      int row_base = (tile * 4 + warp) / 32 * 16;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[tile][j] *= alpha[row_base / 8 + j / 2];
#pragma unroll
      for (int inner = 0; inner < BK; inner += 16) {
        unsigned pa[4], vb[2];
        load_a(pa, p + row_base * 64, inner, lane, 64);
        load_v(vb, v, inner, ((tile * 4 + warp) % 32) * 8, lane);
        mma_bf16(acc[tile], pa[0], pa[1], pa[2], pa[3], vb[0], vb[1]);
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
      int rg = (tile * 4 + warp) / 32 * 2 + j / 2;
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int qr = qbase + r, row = start + qr / ratio;
      if (row < end) {
        int col = ((tile * 4 + warp) % 32) * 8 + fc + j % 2;
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
template <typename Position>
void launch_attention(TensorView q, TensorView cache, TensorView tables, TensorView lengths,
                      TensorView starts, TensorView partial, TensorView lse, int64_t first,
                      bool grouped) {
  int h = q.size(1), hk = cache.size(3), splits = lse.size(2);
  int bq = grouped ? 32 : 16, bk = grouped ? 64 : 32;
  int tiles = ((grouped ? 5 : 1) * (h / hk) + bq - 1) / bq;
  int shared_bytes =
      (bq * 256 + 2 * ((!grouped && q.size(0) > 2) ? 2 : 1) * bk * 256 + bq * 64) * 2 +
      (8 * bq) * 4;
  dim3 grid((grouped ? starts.size(0) - 1 : q.size(0)) * tiles, hk, splits);
#define ATTENTION(M, G, B)                                                                         \
  TVM_FFI_ICHECK(cudaFuncSetAttribute(attention_partial_kernel<M, G, Position, B>,                 \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,                 \
                                      shared_bytes) == cudaSuccess);                               \
  attention_partial_kernel<M, G, Position, B><<<grid, 128, shared_bytes, stream_for(q)>>>(         \
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(cache.data_ptr()), tables.data_ptr(), lengths.data_ptr(), \
      starts.data_ptr(), static_cast<float *>(partial.data_ptr()),                                 \
      static_cast<float *>(lse.data_ptr()), h, hk, tables.size(1), splits, first,                  \
      tables.dtype().bits == 64, lengths.dtype().bits == 64, starts.dtype().bits == 64, tiles)
  if (h == 24 && hk == 4) {
    if (grouped) {
      ATTENTION(true, true, 1);
    } else {
      if (q.size(0) > 2) {
        ATTENTION(true, false, 2);
      } else {
        ATTENTION(true, false, 1);
      }
    }
  } else {
    if (grouped) {
      ATTENTION(false, true, 1);
    } else {
      if (q.size(0) > 2) {
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
                       bool grouped, bool position64) {
  if (position64)
    launch_attention<int64_t>(q, cache, tables, lengths, starts, partial, lse, first, grouped);
  else
    launch_attention<int32_t>(q, cache, tables, lengths, starts, partial, lse, first, grouped);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA attention partial launch failed";
}
template <int Splits>
__global__ void attention_merge_kernel(const float *__restrict__ partial,
                                       const float *__restrict__ lse,
                                       __nv_bfloat16 *__restrict__ out) {
  constexpr int splits = Splits;
  int row = blockIdx.x, lane = threadIdx.x & 31;
  extern __shared__ float weights[];
  // Compute each split weight once, shared by all256 output columns.
  if (threadIdx.x < 32) {
    float maximum = -INFINITY;
    for (int i = lane; i < splits; i += 32)
      maximum = fmaxf(maximum, lse[row * splits + i]);
    maximum = warp_max(maximum);
    float total = 0;
    for (int i = lane; i < splits; i += 32) {
      float weight = softmax_exp2(lse[row * splits + i] - maximum);
      weights[i] = weight;
      total += weight;
    }
    total = warp_sum(total);
    if (lane == 0)
      weights[splits] = total;
  }
  __syncthreads();
  float accumulators[2][8] = {};
#pragma unroll 1
  for (int base = 0; base < splits; base += 8) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float weight = weights[base + j];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int col = threadIdx.x + half * 128;
        accumulators[half][j] +=
            partial[(static_cast<int64_t>(row) * splits + base + j) * 256 + col] * weight;
      }
    }
  }
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float value = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      value += accumulators[half][j];
    out[static_cast<int64_t>(row) * 256 + threadIdx.x + half * 128] =
        __float2bfloat16_rn(value / weights[splits]);
  }
}
void attention_merge(TensorView partial, TensorView lse, TensorView out) {
#define MERGE(S)                                                                                   \
  attention_merge_kernel<S>                                                                        \
      <<<lse.size(0) * lse.size(1), 128, (S + 1) * sizeof(float), stream_for(out)>>>(              \
          static_cast<const float *>(partial.data_ptr()),                                          \
          static_cast<const float *>(lse.data_ptr()),                                              \
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
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA attention merge launch failed";
}

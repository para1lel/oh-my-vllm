#pragma once

#include "common.cuh"

template <bool Residual, int Threads, int Vector, bool Streaming>
__global__ void rms5120_kernel(const __nv_bfloat16 *__restrict__ x,
                               const __nv_bfloat16 *__restrict__ residual,
                               const float *__restrict__ w, __nv_bfloat16 *__restrict__ out,
                               __nv_bfloat16 *__restrict__ summed, float eps) {
  pdl_dependency_wait();
  pdl_launch_next();
  static_assert(Threads % 32 == 0 && 5120 % (Threads * Vector) == 0);
  constexpr int Chunks = 5120 / (Threads * Vector);
  int row = blockIdx.x, lane = threadIdx.x & 31;
  float values[Chunks][Vector], total = 0;
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int col = chunk * Threads * Vector + threadIdx.x * Vector;
    auto q = load_aligned_vector<__nv_bfloat16, Vector>(x + static_cast<int64_t>(row) * 5120 + col);
    AlignedVector<__nv_bfloat16, Vector> r;
    if constexpr (Residual)
      r = load_aligned_vector<__nv_bfloat16, Vector>(residual + static_cast<int64_t>(row) * 5120 +
                                                     col);
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
      value.value[j] = __float2bfloat16_rn(values[chunk][j] * inv * weight.value[j]);
    store_rms_vector<Vector, Streaming>(out + static_cast<int64_t>(row) * 5120 + col, value);
  }
}

// Public wrappers allocate both destinations independently. The fast path is
// entered only for aligned contiguous model-width rows; other layouts use RMS.
void launch_rms5120(TensorView x, TensorView residual, TensorView weight, TensorView out,
                    TensorView summed, bool add, float epsilon) {
  auto stream = stream_for(x, add ? "add_norm" : "norm");
#define RMS5120(R, Threads, Vector, Streaming)                                                     \
  launch_kernel(rms5120_kernel<R, Threads, Vector, Streaming>, x.size(0), Threads, 0, stream,      \
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                  \
                static_cast<const __nv_bfloat16 *>(residual.data_ptr()),                           \
                static_cast<const float *>(weight.data_ptr()),                                     \
                static_cast<__nv_bfloat16 *>(out.data_ptr()),                                      \
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
    if (x.size(0) >= 128) {
      RMS5120(true, 256, 4, true);
    } else {
      RMS5120(true, 256, 4, false);
    }
  } else {
    RMS5120(false, 256, 4, false);
  }
#undef RMS5120
  finish_cuda_launch(stream, add ? "add_norm" : "norm");
}

// Fixed model head width retains the input values across the reduction.
// The stable sigmoid keeps the fast division denominator in [1, 2], including
// extreme finite BF16 gates; it does not round the gate activation to BF16.
template <int RowsPerWarp>
__global__ void
gated_rms128_kernel(const __nv_bfloat16 *__restrict__ x, const float *__restrict__ weight,
                    const __nv_bfloat16 *__restrict__ gate, __nv_bfloat16 *__restrict__ out,
                    int rows, Strides xs, Strides gs, float epsilon) {
  pdl_dependency_wait();
  pdl_launch_next();
  int lane = threadIdx.x & 31;
  float w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    w[j] = weight[lane + j * 32];
#pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
    int row = blockIdx.x * 4 * RowsPerWarp + (threadIdx.x / 32) * RowsPerWarp + r;
    if (row >= rows)
      continue;
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
      float v = values[j] * inv * w[j];
      out[static_cast<int64_t>(row) * 128 + col] = __float2bfloat16_rn(v * (g * sigmoid));
    }
  }
}

template <bool Large, bool Gated, bool Residual>
__global__ void rms_kernel(const __nv_bfloat16 *__restrict__ x, const float *__restrict__ weight,
                           const __nv_bfloat16 *__restrict__ gate, __nv_bfloat16 *__restrict__ out,
                           __nv_bfloat16 *__restrict__ summed, int rows, int heads, int width,
                           Strides xs, Strides gs, float epsilon) {
  pdl_dependency_wait();
  pdl_launch_next();
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
#define GATED_RMS(R)                                                                               \
  launch_kernel(gated_rms128_kernel<R>, (rows + 4 * R - 1) / (4 * R), 128, 0, stream,              \
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                  \
                static_cast<const float *>(weight.data_ptr()),                                     \
                static_cast<const __nv_bfloat16 *>(gate.data_ptr()),                               \
                static_cast<__nv_bfloat16 *>(out.data_ptr()), rows, xs, gs,                        \
                static_cast<float>(epsilon))
    if (rows >= 4096 * 48) {
      GATED_RMS(4);
    } else {
      GATED_RMS(1);
    }
#undef GATED_RMS
    finish_cuda_launch(stream, "gated_norm");
    record_variant(kGatedNorm, true);
    return;
  }
#define RMS(L, G)                                                                                  \
  launch_kernel(rms_kernel<L, G, false>, L ? rows : (rows + 3) / 4, L ? 256 : 128, 0, stream,      \
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                  \
                static_cast<const float *>(weight.data_ptr()),                                     \
                static_cast<const __nv_bfloat16 *>(gate.data_ptr()),                               \
                static_cast<__nv_bfloat16 *>(out.data_ptr()), nullptr, rows, h, d, xs, gs,         \
                epsilon)
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
  launch_kernel(rms_kernel<true, false, true>, rows, 256, 0, stream,
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),
                static_cast<const float *>(weight.data_ptr()),
                static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
                static_cast<__nv_bfloat16 *>(out.data_ptr()),
                static_cast<__nv_bfloat16 *>(summed.data_ptr()), rows, 1, d, Strides{d, d, 1},
                Strides{d, d, 1}, 1e-6f);
  finish_cuda_launch(stream, "add_norm");
  record_variant(kAddNorm, false);
}

template <typename Weight, int Threads, int Vector, bool Streaming>
__global__ void dspark_rms_norm_kernel(const __nv_bfloat16 *__restrict__ input,
                                       const Weight *__restrict__ weight,
                                       __nv_bfloat16 *__restrict__ output, float epsilon) {
  pdl_dependency_wait();
  pdl_launch_next();
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
        auto normalized = __floats2bfloat162_rn(values[i][j] * inverse, values[i][j + 1] * inverse);
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
void launch_dspark_rms(TensorView input, TensorView weight, TensorView output, float epsilon,
                       cudaStream_t stream) {
#define DS_RMS(T, V, S)                                                                            \
  launch_kernel(dspark_rms_norm_kernel<Weight, T, V, S>, input.size(0), T, 0, stream,              \
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
                 weight.size(0) == 5120 && output.ndim() == 2 && output.size(0) == input.size(0) &&
                 output.size(1) == 5120);
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

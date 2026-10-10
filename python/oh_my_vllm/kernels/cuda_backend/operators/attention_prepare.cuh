#pragma once

#include "common.cuh"

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

template <int Mode = 0>
__global__ void
prepare_attention_kernel(const __nv_bfloat16 *packed, const float *qw, const float *kw,
                         const void *positions, __nv_bfloat16 *cache, const void *slots,
                         __nv_bfloat16 *out, int n, int64_t capacity, bool pw, bool sw) {
  pdl_dependency_wait();
  pdl_launch_next();
  int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
  constexpr int heads = Mode == 1 ? 4 : Mode == 2 ? 24 : 28;
  constexpr int width = Mode == 1 ? 2048 : Mode == 2 ? 12288 : 14336;
  if (row >= n * heads)
    return;
  int token = row / heads, head = row % heads;
  bool key = Mode == 1 || (Mode == 0 && head >= 24);
  int key_head = Mode == 1 ? head : head - 24;
  int64_t offset =
      (int64_t)token * width + (key ? (Mode == 1 ? 0 : 12288) + key_head * 256 : head * 512);
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
    int64_t dst = ((slot / 784) * 1568 + slot % 784) * 1024 + key_head * 256;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      cache[dst + lane + j * 32] = __float2bfloat16_rn(values[j]);
      cache[dst + 802816 + lane + j * 32] =
          packed[(int64_t)token * width + (Mode == 1 ? 1024 : 13312) + key_head * 256 + lane +
                 j * 32];
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
  launch_kernel(prepare_attention_kernel<0>, (p.size(0) * 28 + 3) / 4, 128, 0, stream,
                (const __nv_bfloat16 *)p.data_ptr(), (const float *)qw.data_ptr(),
                (const float *)kw.data_ptr(), pos.data_ptr(), (__nv_bfloat16 *)cache.data_ptr(),
                slots.data_ptr(), (__nv_bfloat16 *)out.data_ptr(), p.size(0), cache.size(0) * 784,
                pos.dtype().bits == 64, slots.dtype().bits == 64);
  finish_cuda_launch(stream, "prepare_attention");
}

bool disjoint_storage(TensorView output, TensorView input);
void check_partial_prepare(TensorView p, TensorView weight, TensorView pos, int width) {
  TVM_FFI_ICHECK(same_cuda_device(p, p) && same_cuda_device(weight, p) &&
                 same_cuda_device(pos, p) && has_dtype(p, kDLBfloat, 16) &&
                 has_dtype(weight, kDLFloat, 32) && is_index_dtype(pos))
      << "partial attention preparation requires BF16 data and FP32 norm on one GPU";
  TVM_FFI_ICHECK(p.ndim() == 2 && p.size(1) == width && p.size(0) > 0 &&
                 p.size(0) <= ((int64_t(1) << 31) - 1) / 28 && p.stride(0) == width &&
                 p.stride(1) == 1 && weight.ndim() == 1 && weight.size(0) == 256 &&
                 weight.stride(0) == 1 && pos.ndim() == 1 && pos.size(0) == p.size(0) &&
                 pos.stride(0) == 1)
      << "partial attention preparation requires contiguous model shapes";
}
void prepare_context(TensorView p, TensorView kw, TensorView pos, TensorView cache,
                     TensorView slots) {
  check_partial_prepare(p, kw, pos, 2048);
  TVM_FFI_ICHECK(same_cuda_device(cache, p) && has_dtype(cache, kDLBfloat, 16) &&
                 cache.ndim() == 5 && cache.size(0) > 0 && cache.size(1) == 2 &&
                 cache.size(2) == 784 && cache.size(3) == 4 && cache.size(4) == 256)
      << "context preparation requires model KV cache";
  int64_t stride = 1;
  for (int axis = 4; axis >= 0; --axis) {
    TVM_FFI_ICHECK(cache.stride(axis) == stride) << "context cache must be contiguous";
    stride *= cache.size(axis);
  }
  TVM_FFI_ICHECK(same_cuda_device(slots, p) && is_index_dtype(slots) && slots.ndim() == 1 &&
                 slots.size(0) == p.size(0) && slots.stride(0) == 1)
      << "context preparation needs one integer slot per row";
  for (auto input : {p, kw, pos, slots})
    TVM_FFI_ICHECK(disjoint_storage(cache, input)) << "context cache overlaps input";
  auto stream = stream_for(p, "prepare_context");
  launch_kernel(prepare_attention_kernel<1>, p.size(0), 128, 0, stream,
                (const __nv_bfloat16 *)p.data_ptr(), (const float *)kw.data_ptr(),
                (const float *)kw.data_ptr(), pos.data_ptr(), (__nv_bfloat16 *)cache.data_ptr(),
                slots.data_ptr(), (__nv_bfloat16 *)nullptr, p.size(0), cache.size(0) * 784,
                pos.dtype().bits == 64, slots.dtype().bits == 64);
  finish_cuda_launch(stream, "prepare_context");
}

void prepare_query(TensorView p, TensorView qw, TensorView pos, TensorView out) {
  check_partial_prepare(p, qw, pos, 12288);
  TVM_FFI_ICHECK(same_cuda_device(out, p) && has_dtype(out, kDLBfloat, 16) && out.ndim() == 3 &&
                 out.size(0) == p.size(0) && out.size(1) == 24 && out.size(2) == 256 &&
                 out.stride(0) == 6144 && out.stride(1) == 256 && out.stride(2) == 1)
      << "query preparation requires contiguous model output";
  for (auto input : {p, qw, pos})
    TVM_FFI_ICHECK(disjoint_storage(out, input)) << "query output overlaps input";
  auto stream = stream_for(p, "prepare_query");
  launch_kernel(prepare_attention_kernel<2>, (p.size(0) * 24 + 3) / 4, 128, 0, stream,
                (const __nv_bfloat16 *)p.data_ptr(), (const float *)qw.data_ptr(),
                (const float *)qw.data_ptr(), pos.data_ptr(), (__nv_bfloat16 *)nullptr,
                (const void *)nullptr, (__nv_bfloat16 *)out.data_ptr(), p.size(0), 0,
                pos.dtype().bits == 64, false);
  finish_cuda_launch(stream, "prepare_query");
}

__global__ void rms_rope_kernel(const __nv_bfloat16 *__restrict__ x,
                                const float *__restrict__ weight,
                                const void *__restrict__ positions, __nv_bfloat16 *__restrict__ out,
                                int rows, int heads, Strides xs, bool wide) {
  pdl_dependency_wait();
  pdl_launch_next();
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
  launch_kernel(rms_rope_kernel, (rows + 3) / 4, 128, 0, stream,
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),
                static_cast<const float *>(weight.data_ptr()), positions.data_ptr(),
                static_cast<__nv_bfloat16 *>(out.data_ptr()), rows, x.size(1),
                Strides{x.stride(0), x.stride(1), x.stride(2)}, positions.dtype().bits == 64);
  finish_cuda_launch(stream, "rms_rope");
}
__global__ void rope_kernel(const __nv_bfloat16 *__restrict__ x, const void *__restrict__ positions,
                            __nv_bfloat16 *__restrict__ out, int n, int h, int d, int rotary,
                            double theta, bool wide) {
  pdl_dependency_wait();
  pdl_launch_next();
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
  launch_kernel(rope_kernel, (n * h * d + 255) / 256, 256, 0, stream,
                static_cast<const __nv_bfloat16 *>(x.data_ptr()), positions.data_ptr(),
                static_cast<__nv_bfloat16 *>(out.data_ptr()), n, h, d, rotary, theta,
                positions.dtype().bits == 64);
  finish_cuda_launch(stream, "rope");
}

__global__ void contiguous_bf16_rows_kernel(const uint4 *__restrict__ input,
                                            uint4 *__restrict__ output, int64_t rows, int width,
                                            int64_t stride) {
  pdl_dependency_wait();
  int64_t index = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
  if (index < rows * width) {
    int64_t row = index / width, column = index % width;
    output[index] = input[row * stride + column];
  }
  pdl_launch_next();
}

void contiguous_bf16_rows(TensorView input, TensorView output) {
  TVM_FFI_ICHECK(same_cuda_device(input, input) && same_cuda_device(output, input) &&
                 has_dtype(input, kDLBfloat, 16) && has_dtype(output, kDLBfloat, 16))
      << "CUDA row copy requires BF16 tensors on one CUDA device";
  TVM_FFI_ICHECK(input.ndim() == 2 && output.ndim() == 2 && fits_int32_flat_offsets(input, false) &&
                 output.size(0) == input.size(0) && output.size(1) == input.size(1) &&
                 input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                 output.stride(1) == 1 && output.stride(0) == output.size(1))
      << "CUDA row copy requires nonempty dense BF16 rows and a contiguous output";
  TVM_FFI_ICHECK(input.size(1) % 8 == 0 && input.stride(0) % 8 == 0 &&
                 reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0)
      << "CUDA row copy requires aligned eight-value vectors";
  TVM_FFI_ICHECK(disjoint_storage(output, input))
      << "CUDA row copy requires disjoint input and output storage";
  auto stream = stream_for(input, "contiguous_bf16_rows");
  int64_t vectors = input.numel() / 8;
  launch_kernel(contiguous_bf16_rows_kernel, (vectors + 255) / 256, 256, 0, stream,
                static_cast<const uint4 *>(input.data_ptr()),
                static_cast<uint4 *>(output.data_ptr()), input.size(0),
                static_cast<int>(input.size(1) / 8), input.stride(0) / 8);
  finish_cuda_launch(stream, "contiguous_bf16_rows");
}

template <bool Vector>
__global__ void append_kernel(const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *cache,
                              const void *slots, int width, int64_t capacity, bool wide) {
  pdl_dependency_wait();
  pdl_launch_next();
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
  launch_kernel(append_kernel<V>, k.size(0), 128, 0, stream,                                       \
                static_cast<const __nv_bfloat16 *>(k.data_ptr()),                                  \
                static_cast<const __nv_bfloat16 *>(v.data_ptr()),                                  \
                static_cast<__nv_bfloat16 *>(cache.data_ptr()), slots.data_ptr(), width,           \
                cache.size(0) * 784, slots.dtype().bits == 64)
  if (vector) {
    APPEND(true);
  } else {
    APPEND(false);
  }
#undef APPEND
  finish_cuda_launch(stream, "append");
  record_variant(kAppend, vector);
}

template <bool Wide, bool Narrow>
__global__ void __launch_bounds__(128, 1)
    dspark_append_kernel(const __nv_bfloat16 *__restrict__ key,
                         const __nv_bfloat16 *__restrict__ value, __nv_bfloat16 *__restrict__ cache,
                         const void *__restrict__ slots, int rows, int64_t capacity) {
  pdl_dependency_wait();
  pdl_launch_next();
  constexpr int Width = 1024;
  int token = blockIdx.x, column = threadIdx.x & 127;
  uint4 k, v;
  const auto *source_k = reinterpret_cast<const uint4 *>(key + int64_t(token) * Width);
  const auto *source_v = reinterpret_cast<const uint4 *>(value + int64_t(token) * Width);
  // Independent source reads overlap the slot load and address arithmetic.
  // All source rows exist even when a negative slot suppresses its cache write.
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(k.x), "=r"(k.y), "=r"(k.z), "=r"(k.w)
               : "l"(source_k + column)
               : "memory");
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(source_v + column)
               : "memory");
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
void launch_dspark_append(TensorView key, TensorView value, TensorView cache, TensorView slots,
                          cudaStream_t stream) {
  int rows = key.size(0);
#define DS_APPEND(Narrow)                                                                          \
  launch_kernel(dspark_append_kernel<Wide, Narrow>, rows, 128, 0, stream,                          \
                static_cast<const __nv_bfloat16 *>(key.data_ptr()),                                \
                static_cast<const __nv_bfloat16 *>(value.data_ptr()),                              \
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
  bool fast = key.ndim() == 3 && key.size(1) == 8 && key.size(2) == 128 && aligned(key, 16) &&
              aligned(value, 16) && aligned(cache, 16) && disjoint_storage(cache, key) &&
              disjoint_storage(cache, value) && disjoint_storage(cache, slots);
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
__global__ void
dspark_norm_rope_kernel(const __nv_bfloat16 *__restrict__ x, const float *__restrict__ weight,
                        const void *__restrict__ positions, const float *__restrict__ inv_freq,
                        __nv_bfloat16 *__restrict__ output, int rows, int heads, int64_t row_stride,
                        float attention_factor, float epsilon) {
  pdl_dependency_wait();
  pdl_launch_next();
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
    auto normalized = __floats2bfloat162_rn(values[j] * inverse, values[j + Columns / 2] * inverse);
    auto multiplier =
        __floats2bfloat162_rn(weight[lane + j * Lanes], weight[lane + (j + Columns / 2) * Lanes]);
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
void launch_dspark_norm(TensorView x, TensorView weight, TensorView positions, TensorView inv_freq,
                        TensorView output, float factor, float epsilon, cudaStream_t stream) {
  int rows = x.size(0) * x.size(1);
#define DS_NORM(W)                                                                                 \
  launch_kernel(dspark_norm_rope_kernel<Heads, W, Wide>, (rows + W - 1) / W, W * 32, 0, stream,    \
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                  \
                static_cast<const float *>(weight.data_ptr()), positions.data_ptr(),               \
                static_cast<const float *>(inv_freq.data_ptr()),                                   \
                static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, x.size(1), x.stride(0),     \
                factor, epsilon)
  if (rows >= 4096) {
    DS_NORM(4);
  } else {
    launch_kernel(dspark_norm_rope_kernel<Heads, 1, Wide, true>, rows, 64, 0, stream,
                  static_cast<const __nv_bfloat16 *>(x.data_ptr()),
                  static_cast<const float *>(weight.data_ptr()), positions.data_ptr(),
                  static_cast<const float *>(inv_freq.data_ptr()),
                  static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, x.size(1), x.stride(0),
                  factor, epsilon);
  }
#undef DS_NORM
}

void dspark_norm_rope(TensorView x, TensorView weight, TensorView positions, TensorView inv_freq,
                      TensorView output, double attention_factor, double epsilon) {
  TVM_FFI_ICHECK(x.ndim() == 3 && x.size(2) == 128 && x.size(0) > 0);
  TVM_FFI_ICHECK(weight.numel() == 128 && inv_freq.numel() == 64);
  TVM_FFI_ICHECK(is_index_dtype(positions) && positions.numel() == x.size(0));
  auto stream = stream_for(x, "dspark_norm_rope");
#define DS_NORM_HEADS(H)                                                                           \
  if (positions.dtype().bits == 64) {                                                              \
    launch_dspark_norm<H, true>(x, weight, positions, inv_freq, output, attention_factor, epsilon, \
                                stream);                                                           \
  } else {                                                                                         \
    launch_dspark_norm<H, false>(x, weight, positions, inv_freq, output, attention_factor,         \
                                 epsilon, stream);                                                 \
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

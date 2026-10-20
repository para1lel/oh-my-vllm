#pragma once
#include "../../../runtime/cuda/dependent.cuh"

#include "../../../primitives/cuda/common.cuh"
#include "../../../runtime/cuda/launch.cuh"

__device__ inline void gdn_require(bool condition) {
  if (!condition)
    asm volatile("trap;");
}

__device__ inline float gdn_sigmoid(float value) {
  float exponent = expf(-fabsf(value));
  return (value >= 0.f ? 1.f : exponent) / (1.f + exponent);
}

__global__ void gdn_gather_kernel(const __nv_bfloat16 *embedding, const int64_t *tokens,
                                  const __nv_bfloat16 *media, const int *media_rows,
                                  __nv_bfloat16 *output, int rows, int vocabulary,
                                  int media_count) {
  int row = blockIdx.x, column = threadIdx.x;
  int64_t token = tokens[row];
  gdn_require(token >= 0 && token < vocabulary);
  __shared__ int media_index;
  if (threadIdx.x == 0) {
    media_index = -1;
    for (int index = 0; index < media_count; ++index) {
      int target = media_rows[index];
      gdn_require(target >= 0 && target < rows);
      if (target == row) {
        gdn_require(media_index == -1);
        media_index = index;
      }
    }
  }
  __syncthreads();
  const __nv_bfloat16 *source =
      media_index < 0 ? embedding + token * 5120 : media + int64_t(media_index) * 5120;
  for (; column < 5120; column += blockDim.x)
    output[int64_t(row) * 5120 + column] = source[column];
}

__device__ inline float2 gdn_ba_projection(const __nv_bfloat16 *normalized,
                                           const __nv_bfloat16 *ba_weight, __nv_bfloat16 *ba,
                                           int row, int head) {
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  __shared__ float projection_partial[2][8];
  float projected_b, projected_a;
  float b = 0, a = 0;
#pragma unroll
  for (int column = threadIdx.x; threadIdx.x < 256 && column < 5120; column += 256) {
    float value = __bfloat162float(normalized[int64_t(row) * 5120 + column]);
    b += value * __bfloat162float(ba_weight[head * 5120 + column]);
    a += value * __bfloat162float(ba_weight[(head + 48) * 5120 + column]);
  }
  b = warp_sum(b);
  a = warp_sum(a);
  if (lane == 0 && warp < 8) {
    projection_partial[0][warp] = b;
    projection_partial[1][warp] = a;
  }
  __syncthreads();
  projected_b =
      __bfloat162float(__float2bfloat16_rn(warp_sum(lane < 8 ? projection_partial[0][lane] : 0.f)));
  projected_a =
      __bfloat162float(__float2bfloat16_rn(warp_sum(lane < 8 ? projection_partial[1][lane] : 0.f)));
  if (threadIdx.x == 0) {
    ba[int64_t(row) * 96 + head] = __float2bfloat16_rn(projected_b);
    ba[int64_t(row) * 96 + 48 + head] = __float2bfloat16_rn(projected_a);
  }
  return make_float2(projected_b, projected_a);
}

template <bool ProjectBA>
__global__ void gdn_convolution_kernel(const __nv_bfloat16 *projection, const __nv_bfloat16 *weight,
                                       __nv_bfloat16 *pool, const int *starts, const int *reads,
                                       const int *writes, __nv_bfloat16 *output, int rows,
                                       int batch, int slots, bool verify,
                                       const __nv_bfloat16 *normalized,
                                       const __nv_bfloat16 *ba_weight, __nv_bfloat16 *ba) {
  if constexpr (ProjectBA) {
    if (blockIdx.y >= 20) {
      gdn_ba_projection(normalized, ba_weight, ba, blockIdx.x, blockIdx.y - 20);
      return;
    }
    if (threadIdx.x >= 128)
      return;
  }
  int group = blockIdx.y * 4 + threadIdx.x / 32;
  int row = blockIdx.x, lane = threadIdx.x & 31;
  int sequence = 0;
  gdn_require(starts[0] == 0 && starts[batch] == rows);
  while (sequence + 1 < batch && row >= starts[sequence + 1])
    ++sequence;
  int first = starts[sequence], last = starts[sequence + 1];
  int source = reads[sequence], target = writes[row];
  gdn_require(first >= 0 && first <= row && row < last && last <= rows);
  gdn_require(!verify || last - first <= 5);
  gdn_require(source >= 0 && source < slots && target >= -1 && target < slots);
  if (target >= 0)
    for (int request = 0; request < batch; ++request)
      gdn_require(target != reads[request]);
  float values[4], total_square = 0;
#pragma unroll
  for (int element = 0; element < 4; ++element) {
    int channel = group * 128 + lane * 4 + element;
    float sum = 0;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
      int position = row + tap - 3;
      auto value = position >= first
                       ? projection[int64_t(position) * 16384 + channel]
                       : pool[(int64_t(source) * 10240 + channel) * 3 + position - first + 3];
      sum += __bfloat162float(value) * __bfloat162float(weight[channel * 4 + tap]);
      if (target >= 0 && tap > 0)
        pool[(int64_t(target) * 10240 + channel) * 3 + tap - 1] = value;
    }
    values[element] = __bfloat162float(__float2bfloat16_rn(sum * gdn_sigmoid(sum)));
    total_square += values[element] * values[element];
  }
  float inverse = !verify && group < 32 ? rsqrtf(warp_sum(total_square) + 1e-6f) : 1.f;
#pragma unroll
  for (int element = 0; element < 4; ++element)
    output[int64_t(row) * 10240 + group * 128 + lane * 4 + element] =
        __float2bfloat16_rn(values[element] * inverse);
}

__global__ void gdn_gate_kernel(const __nv_bfloat16 *ba, const float *a_log, const float *dt_bias,
                                float *gates, int rows) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * 48)
    return;
  int row = index / 48, head = index % 48;
  float projected_b = __bfloat162float(ba[int64_t(row) * 96 + head]);
  float projected_a = __bfloat162float(ba[int64_t(row) * 96 + 48 + head]);
  float a = projected_a + dt_bias[head];
  float softplus = a > 20.f ? a : log1pf(expf(a));
  float decay = expf(-expf(a_log[head]) * softplus);
  float beta = gdn_sigmoid(projected_b);
  gates[int64_t(row) * 96 + head] = beta;
  gates[int64_t(row) * 96 + 48 + head] = decay;
}

constexpr int GDNPrefetchRows = 16;
constexpr bool GDNStageQ = true;
constexpr bool GDNStageK = false;
constexpr int GDNStageQueryBytes = 128 * (GDNStageQ ? 4 : 2);
constexpr int GDNStageKeyBytes = 128 * (GDNStageK ? 4 : 2);
constexpr int GDNStageQKBytes = GDNStageQueryBytes + GDNStageKeyBytes;
template <int ValueTile>
constexpr int GDNPrefetchSlotBytes = GDNPrefetchRows * (GDNStageQKBytes + 12 + ValueTile * 2) + 16;

template <int Bytes>
__device__ inline void gdn_prefetch_copy(void *destination, const void *source, bool valid) {
  unsigned address = __cvta_generic_to_shared(destination);
  int source_bytes = valid ? Bytes : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(address), "l"(source),
               "n"(Bytes), "r"(source_bytes)
               : "memory");
}

template <bool Widen, bool Scale>
__device__ __forceinline__ void gdn_stage_operand(unsigned char *destination,
                                                  const __nv_bfloat16 *source, bool valid) {
  if constexpr (Widen) {
    uint4 words = valid ? *reinterpret_cast<const uint4 *>(source) : make_uint4(0, 0, 0, 0);
    float values[8] = {__uint_as_float(words.x << 16), __uint_as_float(words.x & 0xffff0000u),
                       __uint_as_float(words.y << 16), __uint_as_float(words.y & 0xffff0000u),
                       __uint_as_float(words.z << 16), __uint_as_float(words.z & 0xffff0000u),
                       __uint_as_float(words.w << 16), __uint_as_float(words.w & 0xffff0000u)};
    if constexpr (Scale)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        values[i] = __fmul_rn(values[i], 0.08838834764831844f);
    auto *out = reinterpret_cast<float4 *>(destination);
    out[0] = make_float4(values[0], values[1], values[2], values[3]);
    out[1] = make_float4(values[4], values[5], values[6], values[7]);
  } else {
    gdn_prefetch_copy<16>(destination, source, valid);
  }
}

template <bool Wide>
__device__ __forceinline__ void gdn_stage_load(const unsigned char *source, int vector,
                                               float (&values)[4]) {
  if constexpr (Wide) {
    float4 v = reinterpret_cast<const float4 *>(source)[vector];
    values[0] = v.x;
    values[1] = v.y;
    values[2] = v.z;
    values[3] = v.w;
  } else {
    uint2 v = reinterpret_cast<const uint2 *>(source)[vector];
    values[0] = __uint_as_float(v.x << 16);
    values[1] = __uint_as_float(v.x & 0xffff0000u);
    values[2] = __uint_as_float(v.y << 16);
    values[3] = __uint_as_float(v.y & 0xffff0000u);
  }
}

template <int ValueTile>
__device__ inline void gdn_prefetch_rows(unsigned char *storage, const __nv_bfloat16 *qkv,
                                         const float *gates, const int *writes, int base, int last,
                                         int head, int value_base, int source_slot, int slots) {
  auto *query = storage;
  auto *key = query + GDNPrefetchRows * GDNStageQueryBytes;
  auto *value = reinterpret_cast<__nv_bfloat16 *>(key + GDNPrefetchRows * GDNStageKeyBytes);
  auto *coefficients = reinterpret_cast<float *>(value + GDNPrefetchRows * ValueTile);
  auto *targets = reinterpret_cast<int *>(coefficients + GDNPrefetchRows * 2);
  for (int index = threadIdx.x; index < GDNPrefetchRows * 16; index += blockDim.x) {
    int token = index / 16, column = (index % 16) * 8;
    int row = base + token, source = min(row, last - 1);
    int64_t offset = int64_t(source) * 10240 + head / 3 * 128 + column;
    gdn_stage_operand<GDNStageQ, true>(query + (token * 128 + column) * (GDNStageQ ? 4 : 2),
                                       qkv + offset, row < last);
    gdn_stage_operand<GDNStageK, false>(key + (token * 128 + column) * (GDNStageK ? 4 : 2),
                                        qkv + offset + 2048, row < last);
  }
  for (int index = threadIdx.x; index < GDNPrefetchRows * (ValueTile / 8); index += blockDim.x) {
    int token = index / (ValueTile / 8), column = (index % (ValueTile / 8)) * 8;
    int row = base + token, source = min(row, last - 1);
    int64_t offset = int64_t(source) * 10240 + 4096 + head * 128 + value_base + column;
    gdn_prefetch_copy<16>(value + token * ValueTile + column, qkv + offset, row < last);
  }
  for (int token = threadIdx.x; token < GDNPrefetchRows; token += blockDim.x) {
    int row = base + token, source = min(row, last - 1);
    gdn_prefetch_copy<4>(coefficients + token * 2, gates + int64_t(source) * 96 + head, row < last);
    gdn_prefetch_copy<4>(coefficients + token * 2 + 1, gates + int64_t(source) * 96 + 48 + head,
                         row < last);
    gdn_prefetch_copy<4>(targets + token, writes + source, row < last);
  }
  if (threadIdx.x < 32) {
    int row = base + threadIdx.x;
    int target = threadIdx.x < GDNPrefetchRows && row < last ? writes[row] : -1;
    gdn_require(target >= -1 && target < slots && target != source_slot);
    unsigned mask = __ballot_sync(0xffffffffu, target >= 0);
    if (threadIdx.x == 0)
      *reinterpret_cast<unsigned *>(storage + GDNPrefetchSlotBytes<ValueTile> - 16) = mask;
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <bool FusedBA>
__global__ void __launch_bounds__(256, 1)
    gdn_scan_quant_kernel(const __nv_bfloat16 *qkv, const __nv_bfloat16 *qkvz, __nv_bfloat16 *ba,
                          const __nv_bfloat16 *normalized, const __nv_bfloat16 *ba_weight,
                          const float *a_log, const float *dt_bias, const float *norm, float *pool,
                          const int *starts, const int *reads, const int *writes,
                          __nv_fp8_e4m3 *output, float *scales, int rows, int slots) {
  launch_consumer();
  int sequence = blockIdx.x, head = blockIdx.y, lane = threadIdx.x & 31;
  int warp = threadIdx.x / 32;
  constexpr int Warps = 8, Values = 16;
  int value_base = warp * Values;
  int first = starts[sequence], last = starts[sequence + 1], source = reads[sequence];
  gdn_require(first >= 0 && first <= last && last <= rows && source >= 0 && source < slots);
  float state[Values][4];
#pragma unroll
  for (int value = 0; value < Values; ++value)
#pragma unroll
    for (int key = 0; key < 4; ++key)
      state[value][key] =
          pool[((int64_t(source) * 48 + head) * 128 + value_base + value) * 128 + lane * 4 + key];
  __shared__ float result[128], partial[Warps];
  for (int row = first; row < last; ++row) {
    int target = writes[row];
    gdn_require(target >= -1 && target < slots && target != source);
    float q[4], k[4];
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      int column = head / 3 * 128 + lane * 4 + element;
      q[element] = __bfloat162float(qkv[int64_t(row) * 10240 + column]);
      k[element] = __bfloat162float(qkv[int64_t(row) * 10240 + 2048 + column]);
    }
#pragma unroll
    for (int element = 0; element < 4; ++element)
      q[element] *= 0.08838834764831844f;
    float projected_b, projected_a;
    if constexpr (FusedBA) {
      float2 projected = gdn_ba_projection(normalized, ba_weight, ba, row, head);
      projected_b = projected.x;
      projected_a = projected.y;
    } else {
      projected_b = __bfloat162float(ba[int64_t(row) * 96 + head]);
      projected_a = __bfloat162float(ba[int64_t(row) * 96 + 48 + head]);
    }
    float a = projected_a + dt_bias[head];
    float softplus = a > 20.f ? a : log1pf(expf(a));
    float decay = expf(-expf(a_log[head]) * softplus);
    float beta = gdn_sigmoid(projected_b);
#pragma unroll
    for (int value = 0; value < Values; ++value) {
      float recalled = 0;
#pragma unroll
      for (int key = 0; key < 4; ++key) {
        state[value][key] *= decay;
        recalled += state[value][key] * k[key];
      }
      recalled = warp_sum(recalled);
      float input_value =
          __bfloat162float(qkv[int64_t(row) * 10240 + 4096 + head * 128 + value_base + value]);
      float delta = beta * (input_value - recalled);
      float sum = 0;
#pragma unroll
      for (int key = 0; key < 4; ++key) {
        state[value][key] += delta * k[key];
        sum += state[value][key] * q[key];
        if (target >= 0)
          pool[((int64_t(target) * 48 + head) * 128 + value_base + value) * 128 + lane * 4 + key] =
              state[value][key];
      }
      sum = warp_sum(sum);
      if (lane == 0)
        result[value_base + value] = __bfloat162float(__float2bfloat16_rn(sum));
    }
    __syncthreads();
    float square = threadIdx.x < 128 ? result[threadIdx.x] * result[threadIdx.x] : 0;
    square = warp_sum(square);
    if (lane == 0)
      partial[warp] = square;
    __syncthreads();
    float total = warp_sum(lane < Warps ? partial[lane] : 0);
    float inverse = rsqrtf(total / 128.f + 1e-6f);
    if (warp == 0) {
      float values[4];
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        int column = lane * 4 + element;
        float normalized = result[column] * inverse * norm[column];
        float gate = __bfloat162float(qkvz[int64_t(row) * 16384 + 10240 + head * 128 + column]);
        float activated_gate = gate * gdn_sigmoid(gate);
        values[element] = __bfloat162float(__float2bfloat16_rn(normalized * activated_gate));
      }
      quantize_four(make_float4(values[0], values[1], values[2], values[3]),
                    output + int64_t(row) * 6144 + head * 128 + lane * 4,
                    scales + int64_t(row) * 48 + head);
    }
    __syncthreads();
  }
}

template <bool Programmatic>
__global__ void gdn_output_quant_kernel(const __nv_bfloat16 *recurrent, const __nv_bfloat16 *qkvz,
                                        const float *norm, __nv_fp8_e4m3 *output, float *scales) {
  if constexpr (Programmatic)
    launch_consumer();
  int row = blockIdx.x / 6, head = (blockIdx.x % 6) * 8 + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  const auto *source = recurrent + (int64_t(row) * 48 + head) * 128;
  float partial[4];
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    float value = __bfloat162float(source[group * 32 + lane]);
    partial[group] = value;
  }
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    float2 x = make_float2(partial[pair * 2], partial[pair * 2 + 1]);
    float2 squared = __fmul2_rn(x, x);
    partial[pair * 2] = squared.x;
    partial[pair * 2 + 1] = squared.y;
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      float2 x = make_float2(partial[pair * 2], partial[pair * 2 + 1]);
      float2 other = make_float2(__shfl_xor_sync(0xffffffffu, x.x, offset),
                                 __shfl_xor_sync(0xffffffffu, x.y, offset));
      float2 sum = __fadd2_rn(x, other);
      partial[pair * 2] = sum.x;
      partial[pair * 2 + 1] = sum.y;
    }
  }
  float total = __shfl_sync(0xffffffffu, (partial[0] + partial[2]) + (partial[1] + partial[3]), 0);
  float inverse = rsqrtf(total / 128.f + 1e-6f);
  int column = lane * 4;
  uint2 input = *reinterpret_cast<const uint2 *>(source + column);
  uint2 gate =
      *reinterpret_cast<const uint2 *>(qkvz + int64_t(row) * 16384 + 10240 + head * 128 + column);
  float values[4] = {__uint_as_float(input.x << 16), __uint_as_float(input.x & 0xffff0000u),
                     __uint_as_float(input.y << 16), __uint_as_float(input.y & 0xffff0000u)};
  float gates[4] = {__uint_as_float(gate.x << 16), __uint_as_float(gate.x & 0xffff0000u),
                    __uint_as_float(gate.y << 16), __uint_as_float(gate.y & 0xffff0000u)};
  float4 raw_weight = *reinterpret_cast<const float4 *>(norm + column);
  float weights[4] = {raw_weight.x, raw_weight.y, raw_weight.z, raw_weight.w};
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    int element = pair * 2;
    float2 normalized = __fmul2_rn(make_float2(values[element], values[element + 1]),
                                   make_float2(inverse, inverse));
    normalized = __fmul2_rn(normalized, make_float2(weights[element], weights[element + 1]));
    float2 gate = make_float2(gates[element], gates[element + 1]);
    float2 sigmoid = make_float2(gdn_sigmoid(gate.x), gdn_sigmoid(gate.y));
    float2 activated = __fmul2_rn(gate, sigmoid);
    float2 product = __fmul2_rn(normalized, activated);
    __nv_bfloat162 rounded = __floats2bfloat162_rn(product.x, product.y);
    unsigned bits = *reinterpret_cast<const unsigned *>(&rounded);
    values[element] = __uint_as_float(bits << 16);
    values[element + 1] = __uint_as_float(bits & 0xffff0000u);
  }
  quantize_four(make_float4(values[0], values[1], values[2], values[3]),
                output + int64_t(row) * 6144 + head * 128 + column,
                scales + int64_t(row) * 48 + head);
}

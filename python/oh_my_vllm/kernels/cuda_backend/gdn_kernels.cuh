#pragma once

#include "common.cuh"
#include "launch.cuh"

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
  for (int column = threadIdx.x; column < 5120; column += 256) {
    float value = __bfloat162float(normalized[int64_t(row) * 5120 + column]);
    b += value * __bfloat162float(ba_weight[head * 5120 + column]);
    a += value * __bfloat162float(ba_weight[(head + 48) * 5120 + column]);
  }
  b = warp_sum(b);
  a = warp_sum(a);
  if (lane == 0) {
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

template <bool FusedBA>
__global__ void __launch_bounds__(256, 1)
    gdn_scan_quant_kernel(const __nv_bfloat16 *qkv, const __nv_bfloat16 *qkvz, __nv_bfloat16 *ba,
                          const __nv_bfloat16 *normalized, const __nv_bfloat16 *ba_weight,
                          const float *a_log, const float *dt_bias, const float *norm,
                          __nv_bfloat16 *pool, const int *starts, const int *reads,
                          const int *writes, __nv_fp8_e4m3 *output, float *scales, int rows,
                          int slots, bool verify) {
  int sequence = blockIdx.x, head = blockIdx.y, lane = threadIdx.x & 31;
  int warp = threadIdx.x / 32;
  int first = starts[sequence], last = starts[sequence + 1], source = reads[sequence];
  gdn_require(first >= 0 && first <= last && last <= rows && source >= 0 && source < slots);
  gdn_require(!verify || last - first <= 5);
  float state[16][4];
#pragma unroll
  for (int value = 0; value < 16; ++value)
#pragma unroll
    for (int key = 0; key < 4; ++key)
      state[value][key] = __bfloat162float(
          pool[((int64_t(source) * 48 + head) * 128 + warp * 16 + value) * 128 + lane * 4 + key]);
  __shared__ float result[128], partial[8];
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
    float q_scale = 0.08838834764831844f, k_scale = 1.f;
    if (verify) {
      float q_square = 0, k_square = 0;
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        q_square += q[element] * q[element];
        k_square += k[element] * k[element];
      }
      q_scale *= rsqrtf(warp_sum(q_square) + 1e-6f);
      k_scale = rsqrtf(warp_sum(k_square) + 1e-6f);
    }
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      q[element] *= q_scale;
      k[element] *= k_scale;
    }
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
    for (int value = 0; value < 16; ++value) {
      float recalled = 0;
#pragma unroll
      for (int key = 0; key < 4; ++key) {
        state[value][key] *= decay;
        recalled += state[value][key] * k[key];
      }
      recalled = warp_sum(recalled);
      float delta =
          beta *
          (__bfloat162float(qkv[int64_t(row) * 10240 + 4096 + head * 128 + warp * 16 + value]) -
           recalled);
      float sum = 0;
#pragma unroll
      for (int key = 0; key < 4; ++key) {
        state[value][key] += delta * k[key];
        sum += state[value][key] * q[key];
        if (target >= 0)
          pool[((int64_t(target) * 48 + head) * 128 + warp * 16 + value) * 128 + lane * 4 + key] =
              __float2bfloat16_rn(state[value][key]);
      }
      sum = warp_sum(sum);
      if (lane == 0)
        result[warp * 16 + value] = __bfloat162float(__float2bfloat16_rn(sum));
    }
    __syncthreads();
    float square = threadIdx.x < 128 ? result[threadIdx.x] * result[threadIdx.x] : 0;
    square = warp_sum(square);
    if (lane == 0)
      partial[warp] = square;
    __syncthreads();
    float total = warp_sum(lane < 8 ? partial[lane] : 0);
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

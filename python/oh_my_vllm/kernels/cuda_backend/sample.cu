#include "linear.cuh"
#include <cub/cub.cuh>
#include <curand_kernel.h>

constexpr int Vocabulary = 248320;
constexpr int TokenBits = 18;
constexpr uint64_t TokenMask = (1ull << TokenBits) - 1;
constexpr int SampleNormThreads = 640;

#include "sample_accept.cuh"

inline cudaError_t sort_scores(void *workspace, size_t &bytes, const uint64_t *keys,
                               uint64_t *ordered, int rows, cudaStream_t stream) {
  int row_bits = 0;
  while ((1 << row_bits) < rows)
    ++row_bits;
  return cub::DeviceRadixSort::SortKeysDescending(workspace, bytes, keys, ordered,
                                                  rows * Vocabulary, TokenBits,
                                                  TokenBits + 32 + row_bits, stream);
}

int64_t sample_workspace(int64_t rows) {
  TVM_FFI_ICHECK(rows > 0 && rows <= 20);
  size_t bytes = 0;
  check_cuda(sort_scores(nullptr, bytes, nullptr, nullptr, rows, nullptr));
  return bytes;
}

__global__ void sample_norm_kernel(const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
                                   const float *weight, __nv_bfloat16 *normalized) {
  int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  constexpr int Values = 5120 / SampleNormThreads, Warps = SampleNormThreads / 32;
  float values[Values], total = 0;
#pragma unroll
  for (int j = 0; j < Values; ++j) {
    int column = threadIdx.x + SampleNormThreads * j;
    values[j] =
        __bfloat162float(x[row * 5120 + column]) + __bfloat162float(residual[row * 5120 + column]);
    total += values[j] * values[j];
  }
  __shared__ float partial[Warps];
  total = warp_sum(total);
  if (!lane)
    partial[warp] = total;
  __syncthreads();
  total = warp_sum(lane < Warps ? partial[lane] : 0);
  float inverse = rsqrtf(total / 5120.f + 1e-6f);
#pragma unroll
  for (int j = 0; j < Values; ++j) {
    int column = threadIdx.x + SampleNormThreads * j;
    normalized[row * 5120 + column] =
        __float2bfloat16_rn(values[j] * inverse * (weight[column] + 1.f));
  }
}

__device__ uint64_t sample_key(const __nv_bfloat16 *logits, const int *history,
                               const uint8_t *allowed, const float *parameters, int row,
                               int token) {
  int index = row * Vocabulary + token;
  const float *p = parameters + row * 7;
  int prompt = history[row * 2 * Vocabulary + token];
  int generated = history[(row * 2 + 1) * Vocabulary + token];
  if (!(isfinite(p[0]) && p[0] >= 0 && p[1] >= -1 && p[1] <= Vocabulary && p[1] == truncf(p[1]) &&
        p[2] > 0 && p[2] <= 1 && p[3] >= 0 && p[3] <= 1 && isfinite(p[4]) && isfinite(p[5]) &&
        p[6] > 0 && isfinite(p[6]) && prompt >= 0 && generated >= 0)) {
    asm volatile("trap;");
    return 0;
  }
  float value = __bfloat162float(logits[index]);
  if (!isfinite(value)) {
    asm volatile("trap;");
    return 0;
  }
  if (prompt || generated)
    value = value > 0 ? value / p[6] : value * p[6];
  value -= p[4] * generated + p[5] * (generated > 0);
  float score = allowed[index] ? value / (p[0] > 0 ? p[0] : 1.f) : -INFINITY;
  unsigned bits = __float_as_uint(score == 0 ? 0.f : score);
  unsigned ordered = bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
  return (uint64_t(ordered) << TokenBits) | (TokenMask - token);
}

__global__ void scores_kernel(const __nv_bfloat16 *logits, const int *history,
                              const uint8_t *allowed, const float *parameters, uint64_t *keys,
                              int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  int row = index / Vocabulary, token = index % Vocabulary;
  keys[index] = (uint64_t(count / Vocabulary - 1 - row) << (TokenBits + 32)) |
                sample_key(logits, history, allowed, parameters, row, token);
}

#include "sample_greedy.cuh"

__global__ void unpack_scores_kernel(const uint64_t *ordered, float *scores, int *order,
                                     float *probability, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  uint64_t key = ordered[index];
  unsigned encoded = unsigned(key >> TokenBits);
  unsigned bits = encoded ^ ((encoded >> 31) ? 0x80000000u : 0xffffffffu);
  scores[index] = __uint_as_float(bits);
  order[index] = int(TokenMask - (key & TokenMask));
  probability[index] = 0;
}

__global__ void distribution_kernel(const float *sorted, const int *order, const float *parameters,
                                    uint64_t *rng, float *probability, int64_t *tokens,
                                    float *selected_probability, bool allow_invalid) {
  using Reduce = cub::BlockReduce<float, 256>;
  using Scan = cub::BlockScan<float, 256>;

  __shared__ union {
    typename Reduce::TempStorage reduce;
    typename Scan::TempStorage scan;
  } temporary;

  __shared__ float total, prefix, retained, uniform;
  __shared__ int selected, active;
  int row = blockIdx.x, lane = threadIdx.x;
  const float *scores = sorted + row * Vocabulary;
  const int *ids = order + row * Vocabulary;
  float *output = probability + row * Vocabulary;
  const float *p = parameters + row * 7;
  if (!isfinite(scores[0])) {
    if (allow_invalid) {
      if (!lane) {
        tokens[row] = -1;
        selected_probability[row] = 0;
      }
      return;
    }
    asm volatile("trap;");
    return;
  }
  int topk = p[1] > 0 ? int(p[1]) : Vocabulary;
  float threshold = scores[topk - 1];
  float maximum = scores[0], local = 0;
  if (!lane) {
    int first = 0, last = Vocabulary;
    while (first < last) {
      int middle = first + (last - first) / 2;
      float value = expf(scores[middle] - maximum);
      if (scores[middle] >= threshold && value > 0 && value >= p[3])
        first = middle + 1;
      else
        last = middle;
    }
    active = p[0] == 0 ? 1 : first;
  }
  __syncthreads();
  for (int i = lane; i < active; i += 256)
    local += expf(scores[i] - maximum);
  float sum = Reduce(temporary.reduce).Sum(local);
  if (!lane) {
    total = sum;
    prefix = 0;
    retained = 0;
    selected = Vocabulary;
    curandStatePhilox4_32_10_t state;
    curand_init(rng[row * 2], 0, rng[row * 2 + 1], &state);
    uniform = curand_uniform(&state);
    if (p[0] > 0)
      ++rng[row * 2 + 1];
  }
  __syncthreads();
  if (p[0] == 0) {
    if (!lane) {
      output[ids[0]] = 1;
      tokens[row] = ids[0];
      selected_probability[row] = 1;
    }
    return;
  }
  for (int base = 0; base < active; base += 256) {
    int i = base + lane;
    float value = i < active ? expf(scores[i] - maximum) : 0;
    float inclusive, aggregate;
    Scan(temporary.scan).InclusiveSum(value, inclusive, aggregate);
    float before = prefix + (inclusive - value);
    bool keep = value > 0 && before < p[2] * total;
    if (i < active)
      output[ids[i]] = keep ? value : 0;
    __syncthreads();
    float kept = Reduce(temporary.reduce).Sum(keep ? value : 0);
    if (!lane) {
      retained += kept;
      prefix += aggregate;
    }
    __syncthreads();
  }
  if (!lane)
    prefix = 0;
  __syncthreads();
  for (int base = 0; base < active; base += 256) {
    int i = base + lane;
    float value = i < active ? output[ids[i]] : 0;
    float inclusive, aggregate;
    Scan(temporary.scan).InclusiveSum(value, inclusive, aggregate);
    if (i < active && value > 0 && prefix + inclusive >= uniform * retained)
      atomicMin(&selected, i);
    if (i < active)
      output[ids[i]] = value / retained;
    __syncthreads();
    if (!lane)
      prefix += aggregate;
    __syncthreads();
  }
  if (!lane) {
    if (selected == Vocabulary) {
      asm volatile("trap;");
      return;
    }
    tokens[row] = ids[selected];
    selected_probability[row] = output[ids[selected]];
  }
}

void sample(TensorView x, TensorView residual, TensorView norm, TensorView weight,
            TensorView history, TensorView allowed, TensorView parameters, TensorView rng,
            TensorView normalized, TensorView logits, TensorView keys, TensorView ordered,
            TensorView sorted, TensorView order, TensorView probability, TensorView tokens,
            TensorView selected_probability, TensorView workspace, bool allow_invalid,
            bool greedy) {
  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(0) > 0 && x.size(0) <= 20 && x.size(1) == 5120);
  int rows = x.size(0);
  for (auto tensor : {x, residual, normalized})
    tensor_contract(tensor, x, kDLBfloat, 16, {rows, 5120});
  tensor_contract(norm, x, kDLFloat, 32, {5120});
  tensor_contract(weight, x, kDLBfloat, 16, {Vocabulary, 5120});
  tensor_contract(logits, x, kDLBfloat, 16, {rows, Vocabulary});
  tensor_contract(history, x, kDLInt, 32, {rows, 2, Vocabulary});
  tensor_contract(allowed, x, kDLUInt, 8, {rows, Vocabulary});
  tensor_contract(parameters, x, kDLFloat, 32, {rows, 7});
  tensor_contract(rng, x, kDLUInt, 64, {rows, 2});
  for (auto tensor : {sorted, probability})
    tensor_contract(tensor, x, kDLFloat, 32, {rows, Vocabulary});
  for (auto tensor : {keys, ordered})
    tensor_contract(tensor, x, kDLUInt, 64, {rows, Vocabulary});
  tensor_contract(order, x, kDLInt, 32, {rows, Vocabulary});
  tensor_contract(tokens, x, kDLInt, 64, {rows});
  tensor_contract(selected_probability, x, kDLFloat, 32, {rows});
  TVM_FFI_ICHECK(workspace.ndim() == 1);
  tensor_contract(workspace, x, kDLUInt, 8, {workspace.size(0)});
  auto tensors = {x,
                  residual,
                  norm,
                  weight,
                  history,
                  allowed,
                  parameters,
                  rng,
                  normalized,
                  logits,
                  keys,
                  ordered,
                  sorted,
                  order,
                  probability,
                  tokens,
                  selected_probability,
                  workspace};
  int left = 0;
  for (auto a : tensors) {
    int right = 0;
    for (auto b : tensors)
      if (right++ > left)
        disjoint(a, b);
    ++left;
  }
  tvm::ffi::CUDADeviceGuard guard(x.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  size_t bytes = 0;
  if (!greedy) {
    check_cuda(sort_scores(nullptr, bytes, nullptr, nullptr, rows, stream));
    TVM_FFI_ICHECK(workspace.numel() >= bytes);
  }
  sample_norm_kernel<<<rows, SampleNormThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
      static_cast<const float *>(norm.data_ptr()),
      static_cast<__nv_bfloat16 *>(normalized.data_ptr()));
  check_cuda(cudaGetLastError());
  bf16_linear(static_cast<const __nv_bfloat16 *>(normalized.data_ptr()),
              static_cast<const __nv_bfloat16 *>(weight.data_ptr()), logits.data_ptr(), rows,
              Vocabulary, 5120, stream);
  if (greedy) {
    greedy_scores_kernel<<<dim3(GreedyBlocks, rows), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(logits.data_ptr()),
        static_cast<const int *>(history.data_ptr()),
        static_cast<const uint8_t *>(allowed.data_ptr()),
        static_cast<const float *>(parameters.data_ptr()), static_cast<uint64_t *>(keys.data_ptr()),
        static_cast<float *>(probability.data_ptr()));
    check_cuda(cudaGetLastError());
    greedy_select_kernel<<<rows, 256, 0, stream>>>(
        static_cast<const uint64_t *>(keys.data_ptr()),
        static_cast<float *>(probability.data_ptr()), static_cast<int64_t *>(tokens.data_ptr()),
        static_cast<float *>(selected_probability.data_ptr()), allow_invalid);
    check_cuda(cudaGetLastError());
    return;
  }
  scores_kernel<<<(rows * Vocabulary + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(logits.data_ptr()),
      static_cast<const int *>(history.data_ptr()),
      static_cast<const uint8_t *>(allowed.data_ptr()),
      static_cast<const float *>(parameters.data_ptr()), static_cast<uint64_t *>(keys.data_ptr()),
      rows * Vocabulary);
  check_cuda(cudaGetLastError());
  check_cuda(sort_scores(workspace.data_ptr(), bytes,
                         static_cast<const uint64_t *>(keys.data_ptr()),
                         static_cast<uint64_t *>(ordered.data_ptr()), rows, stream));
  unpack_scores_kernel<<<(rows * Vocabulary + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint64_t *>(ordered.data_ptr()), static_cast<float *>(sorted.data_ptr()),
      static_cast<int *>(order.data_ptr()), static_cast<float *>(probability.data_ptr()),
      rows * Vocabulary);
  check_cuda(cudaGetLastError());
  distribution_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float *>(sorted.data_ptr()), static_cast<const int *>(order.data_ptr()),
      static_cast<const float *>(parameters.data_ptr()), static_cast<uint64_t *>(rng.data_ptr()),
      static_cast<float *>(probability.data_ptr()), static_cast<int64_t *>(tokens.data_ptr()),
      static_cast<float *>(selected_probability.data_ptr()), allow_invalid);
  check_cuda(cudaGetLastError());
}

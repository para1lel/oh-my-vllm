#pragma once

constexpr int GreedyTile = 1024;
constexpr int GreedyBlocks = (Vocabulary + GreedyTile - 1) / GreedyTile;

__global__ void greedy_scores_kernel(const __nv_bfloat16 *logits, const void *history,
                                     const uint8_t *allowed, const float *parameters,
                                     uint64_t *partials, float *probability, bool compact) {
  using Reduce = cub::BlockReduce<uint64_t, 256>;
  __shared__ typename Reduce::TempStorage temporary;
  int row = blockIdx.y;
  if (parameters[row * 7] != 0) {
    asm volatile("trap;");
    return;
  }
  uint64_t maximum = 0;
  for (int offset = threadIdx.x; offset < GreedyTile; offset += 256) {
    int token = blockIdx.x * GreedyTile + offset;
    if (token < Vocabulary) {
      maximum = max(maximum, sample_key(logits, history, allowed, parameters, row, token, compact));
      probability[row * Vocabulary + token] = 0;
    }
  }
  maximum = Reduce(temporary).Reduce(maximum, cuda::maximum<uint64_t>{});
  if (!threadIdx.x)
    partials[row * GreedyBlocks + blockIdx.x] = maximum;
}

__global__ void greedy_select_kernel(const uint64_t *partials, float *probability, int64_t *tokens,
                                     float *selected_probability, bool allow_invalid) {
  using Reduce = cub::BlockReduce<uint64_t, 256>;
  __shared__ typename Reduce::TempStorage temporary;
  int row = blockIdx.x;
  uint64_t key = threadIdx.x < GreedyBlocks ? partials[row * GreedyBlocks + threadIdx.x] : 0;
  key = Reduce(temporary).Reduce(key, cuda::maximum<uint64_t>{});
  if (threadIdx.x)
    return;
  unsigned encoded = unsigned(key >> TokenBits);
  unsigned bits = encoded ^ ((encoded >> 31) ? 0x80000000u : 0xffffffffu);
  if (!isfinite(__uint_as_float(bits))) {
    if (!allow_invalid) {
      asm volatile("trap;");
      return;
    }
    tokens[row] = -1;
    selected_probability[row] = 0;
    return;
  }
  int token = int(TokenMask - (key & TokenMask));
  probability[row * Vocabulary + token] = 1;
  tokens[row] = token;
  selected_probability[row] = 1;
}

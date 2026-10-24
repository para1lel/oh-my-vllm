#include "../../runtime/cuda/contracts.cuh"
#include "blend.cuh"
#include "lookup.cuh"
#include "shared_weights.cuh"

#include <algorithm>
#include <cstdint>

__device__ bool engram_valid(bool condition) {
  if (!condition)
    asm volatile("trap;");
  return condition;
}

__device__ bool engram_multiplier(int64_t value, int64_t vocabulary_size) {
  return engram_valid(value > 0 && (value & 1) &&
                      value <= INT64_MAX / (vocabulary_size > 1 ? vocabulary_size - 1 : 1));
}

__device__ int64_t engram_input(const int64_t *tokens, const int64_t *history, int position,
                                int history_rows) {
  return position < history_rows ? history[position] : tokens[position - history_rows];
}

__global__ void engram_hash_kernel(const int64_t *tokens, const int64_t *history,
                                   const int64_t *multipliers, const int64_t *primes,
                                   const int64_t *offsets, int64_t *hashes, int64_t *next_history,
                                   int rows, int history_rows, int layers, int heads,
                                   int max_ngram_size, int tail_rows, int64_t pad_id,
                                   int64_t vocabulary_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int columns = (max_ngram_size - 1) * heads;
  if (index < layers) {
    for (int shift = 0; shift < max_ngram_size; ++shift)
      if (!engram_multiplier(multipliers[index * max_ngram_size + shift], vocabulary_size))
        return;
    int64_t end = 0;
    for (int column = 0; column < columns; ++column) {
      int64_t prime = primes[index * columns + column];
      if (!engram_valid(prime >= 2 && offsets[index * columns + column] == end &&
                        end <= INT64_MAX - prime))
        return;
      end += prime;
    }
  }
  if (index < tail_rows) {
    int64_t token =
        engram_input(tokens, history, history_rows + rows - tail_rows + index, history_rows);
    if (!engram_valid(token >= -1 && token < vocabulary_size))
      return;
    next_history[index] = token;
  }
  if (index >= rows * layers * columns)
    return;
  int column = index % columns;
  int layer = (index / columns) % layers;
  int row = index / (layers * columns);
  int ngram = column / heads + 2;
  int64_t running = 0;
  bool blocked = false;
  for (int shift = 0; shift < ngram; ++shift) {
    int position = history_rows + row - shift;
    int64_t token = position < 0 ? -1 : engram_input(tokens, history, position, history_rows);
    if (!engram_valid(token >= -1 && token < vocabulary_size))
      return;
    blocked |= token == -1;
    int64_t multiplier = multipliers[layer * max_ngram_size + shift];
    if (!engram_multiplier(multiplier, vocabulary_size))
      return;
    running ^= (blocked ? pad_id : token) * multiplier;
  }
  int64_t prime = primes[layer * columns + column];
  int64_t offset = offsets[layer * columns + column];
  if (!engram_valid(prime >= 2 && offset >= 0 && offset <= INT64_MAX - (prime - 1)))
    return;
  hashes[index] = running % prime + offset;
}

void engram_hash(TensorView tokens, TensorView history, TensorView multipliers, TensorView primes,
                 TensorView offsets, TensorView hashes, TensorView next_history, int64_t pad_id,
                 int64_t vocabulary_size) {
  TVM_FFI_ICHECK(tokens.device().device_type == kDLCUDA && tokens.ndim() == 1 &&
                 tokens.size(0) >= 0 && tokens.size(0) <= 384);
  TVM_FFI_ICHECK(multipliers.ndim() == 2 && multipliers.size(0) >= 1 && multipliers.size(0) <= 2 &&
                 multipliers.size(1) >= 2 && multipliers.size(1) <= 4);
  int rows = tokens.size(0), layers = multipliers.size(0);
  int max_ngram_size = multipliers.size(1);
  TVM_FFI_ICHECK(history.ndim() == 1 && history.size(0) >= 0 && history.size(0) < max_ngram_size);
  TVM_FFI_ICHECK(primes.ndim() == 2 && primes.size(0) == layers && primes.size(1) > 0 &&
                 primes.size(1) % (max_ngram_size - 1) == 0);
  int64_t heads = primes.size(1) / (max_ngram_size - 1);
  TVM_FFI_ICHECK(heads >= 1 && heads <= 8);
  TVM_FFI_ICHECK(vocabulary_size >= 1 && vocabulary_size <= 99092 && pad_id >= 0 &&
                 pad_id < vocabulary_size);
  int history_rows = history.size(0), columns = primes.size(1);
  int tail_rows = std::min(max_ngram_size - 1, history_rows + rows);
  tensor_contract(tokens, tokens, kDLInt, 64, {rows});
  tensor_contract(history, tokens, kDLInt, 64, {history_rows});
  tensor_contract(multipliers, tokens, kDLInt, 64, {layers, max_ngram_size});
  tensor_contract(primes, tokens, kDLInt, 64, {layers, columns});
  tensor_contract(offsets, tokens, kDLInt, 64, {layers, columns});
  tensor_contract(hashes, tokens, kDLInt, 64, {rows, layers, columns});
  tensor_contract(next_history, tokens, kDLInt, 64, {tail_rows});
  for (auto output : {hashes, next_history})
    for (auto input : {tokens, history, multipliers, primes, offsets})
      if (tvm::ffi::GetDataSize(output) && tvm::ffi::GetDataSize(input))
        disjoint(input, output);
  if (tvm::ffi::GetDataSize(hashes) && tvm::ffi::GetDataSize(next_history))
    disjoint(hashes, next_history);
  tvm::ffi::CUDADeviceGuard guard(tokens.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, tokens.device().device_id));
  int count = std::max({rows * layers * columns, tail_rows, layers});
  engram_hash_kernel<<<(count + 127) / 128, 128, 0, stream>>>(
      static_cast<const int64_t *>(tokens.data_ptr()),
      static_cast<const int64_t *>(history.data_ptr()),
      static_cast<const int64_t *>(multipliers.data_ptr()),
      static_cast<const int64_t *>(primes.data_ptr()),
      static_cast<const int64_t *>(offsets.data_ptr()), static_cast<int64_t *>(hashes.data_ptr()),
      static_cast<int64_t *>(next_history.data_ptr()), rows, history_rows, layers, heads,
      max_ngram_size, tail_rows, pad_id, vocabulary_size);
  check_cuda(cudaGetLastError());
}

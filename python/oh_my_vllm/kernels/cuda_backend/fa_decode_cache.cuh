#pragma once
#include "fa_attention.cuh"
#include <cutlass/cuda_host_adapter.hpp>
#include <map>
#include <tuple>

__device__ inline int fa_encoded_index(int row, int column) {
  return (column / 128) * 8192 + row * 128 + ((column % 128) ^ ((row & 7) * 16));
}

__global__ void fa_encode_cache_kernel(const __nv_bfloat16 *cache, __nv_fp8_e4m3 *encoded,
                                       float *scales, const int32_t *slots, const float *key_norm,
                                       int64_t tokens) {
  int row = blockIdx.x, head = threadIdx.x / 32, lane = threadIdx.x & 31;
  int slot = slots[row];
  if (slot < 0 || slot >= tokens) {
    asm volatile("trap;");
    return;
  }
  int64_t group = ((int64_t(slot / 784) * 2 + head / 4) * 784 + slot % 784) * 4 + head % 4;
  float values[8], maximum = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    values[j] = __bfloat162float(cache[group * 256 + lane + j * 32]);
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
  float scale = fmaxf(maximum, 1e-10f) / 448.f;
  if (head < 4) {
    float gain = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      gain = fmaxf(gain, fabsf(key_norm[lane + j * 32] + 1.f));
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      gain = fmaxf(gain, __shfl_xor_sync(0xffffffff, gain, offset));
    // RMS gives norm <= 16; 32 also covers BF16 rounding and the two RoPE terms.
    scale = fmaxf(32.f * gain, 1e-10f) / 448.f;
    if (!isfinite(maximum) || maximum > scale * 448.f) {
      asm volatile("trap;");
      return;
    }
  }
  if (lane == 0)
    scales[((int64_t(slot / 784) * 2 + head / 4) * 4 + head % 4) * 784 + slot % 784] = scale;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    encoded[group * 256 + lane + j * 32] = __nv_fp8_e4m3(values[j] / scale);
}

inline void fa_cache_encode(TensorView cache, TensorView encoded, TensorView scales,
                            TensorView slots, TensorView key_norm) {
  TVM_FFI_ICHECK(cache.ndim() == 5 && cache.size(0) > 0 && cache.size(0) <= INT32_MAX / 784);
  TVM_FFI_ICHECK(slots.ndim() == 1 && slots.size(0) > 0 && slots.size(0) <= 32768);
  int blocks = cache.size(0);
  tensor_contract(cache, cache, kDLBfloat, 16, {blocks, 2, 784, 4, 256});
  tensor_contract(encoded, cache, kDLFloat8_e4m3fn, 8, {blocks, 2, 784, 4, 256});
  tensor_contract(scales, cache, kDLFloat, 32, {blocks, 2, 4, 784});
  tensor_contract(slots, cache, kDLInt, 32, {slots.size(0)});
  tensor_contract(key_norm, cache, kDLFloat, 32, {256});
  disjoint(key_norm, encoded);
  disjoint(key_norm, scales);
  disjoint(cache, encoded);
  disjoint(cache, scales);
  disjoint(slots, encoded);
  disjoint(slots, scales);
  disjoint(encoded, scales);
  tvm::ffi::CUDADeviceGuard guard(cache.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, cache.device().device_id));
  fa_encode_cache_kernel<<<slots.size(0), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(cache.data_ptr()),
      static_cast<__nv_fp8_e4m3 *>(encoded.data_ptr()), static_cast<float *>(scales.data_ptr()),
      static_cast<const int32_t *>(slots.data_ptr()),
      static_cast<const float *>(key_norm.data_ptr()), int64_t(blocks) * 784);
  check_cuda(cudaGetLastError());
}

inline const CUtensorMap &fa_encoded_tensor_map(TensorView cache) {
  using Key = std::tuple<int, uintptr_t, int64_t>;
  static thread_local std::map<Key, CUtensorMap> maps;
  Key key{cache.device().device_id, reinterpret_cast<uintptr_t>(cache.data_ptr()), cache.size(0)};
  auto found = maps.find(key);
  if (found == maps.end()) {
    CUtensorMap map{};
    uint64_t dimensions[5] = {128, 784, 2, 4, uint64_t(cache.size(0)) * 2};
    uint64_t strides[4] = {1024, 128, 256, 802816};
    uint32_t box[5] = {128, 64, 2, 1, 1}, elements[5] = {1, 1, 1, 1, 1};
    auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, cache.data_ptr(), dimensions, strides, box,
        elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    found = maps.emplace(key, map).first;
  }
  return found->second;
}

inline const CUtensorMap &fa_scale_tensor_map(TensorView scales, int width = 64) {
  using Key = std::tuple<int, uintptr_t, int64_t, int>;
  static thread_local std::map<Key, CUtensorMap> maps;
  Key key{scales.device().device_id, reinterpret_cast<uintptr_t>(scales.data_ptr()), scales.size(0),
          width};
  auto found = maps.find(key);
  if (found == maps.end()) {
    CUtensorMap map{};
    uint64_t dimensions[3] = {784, 4, uint64_t(scales.size(0)) * 2};
    uint64_t strides[2] = {784 * 4, 784 * 4 * 4};
    uint32_t box[3] = {uint32_t(width), 1, 1}, elements[3] = {1, 1, 1};
    auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, scales.data_ptr(), dimensions, strides, box,
        elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    found = maps.emplace(key, map).first;
  }
  return found->second;
}

__device__ inline void fa_native_group_sync() {
  int barrier = threadIdx.x / 128 + 1;
  asm volatile("bar.sync %0, 128;" ::"r"(barrier) : "memory");
}

template <int Operand>
__device__ inline void fa_stage_encoded_kv(const __nv_fp8_e4m3 *cache, const float *scales,
                                           const int32_t *tables, int sequence, int table_width,
                                           int64_t pages, int base, int last, int head,
                                           const CUtensorMap &map, const CUtensorMap &scale_map,
                                           uint64_t *ready, unsigned char *storage) {
  auto *destination = storage + Operand * 16384;
  auto *scale_destination = reinterpret_cast<float *>(storage - 512);
  int scale_offset = base & 3, scale_count = scale_offset ? 68 : 64;
  bool full = base + 64 <= last && base % 784 + 64 <= 784;
  if (full) {
    if ((threadIdx.x % 128) == 0) {
      int page = tables[sequence * table_width + base / 784];
      if (page < 0 || page >= pages) {
        asm volatile("trap;");
        return;
      }
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
          ready, 16384 + (Operand == 1 ? scale_count * 4 : 0));
      cute::SM90_TMA_LOAD_5D::copy(&map, ready, uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                   destination, 0, base % 784, 0, head, page * 2 + Operand);
      if (Operand == 1)
        cute::SM90_TMA_LOAD_3D::copy(
            &scale_map, ready, uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL), scale_destination,
            (base - scale_offset) % 784, head, page * 2 + Operand);
    }
  } else {
    for (int row = (threadIdx.x % 128); Operand == 1 && row < 64; row += 128) {
      int position = min(base + row, last - 1);
      int page = tables[sequence * table_width + position / 784];
      if (page < 0 || page >= pages) {
        asm volatile("trap;");
        return;
      }
      int64_t index = ((int64_t(page) * 2 + Operand) * 4 + head) * 784 + position % 784;
      unsigned address = __cvta_generic_to_shared(scale_destination + scale_offset + row);
      int bytes = base + row < last ? 4 : 0;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(address),
                   "l"(scales + index), "r"(bytes)
                   : "memory");
    }
    for (int i = (threadIdx.x % 128); i < 64 * 16; i += 128) {
      int row = i / 16, column = i % 16 * 16;
      int position = min(base + row, last - 1);
      int page = tables[sequence * table_width + position / 784];
      if (page < 0 || page >= pages) {
        asm volatile("trap;");
        return;
      }
      int64_t offset =
          (((int64_t(page) * 2 + Operand) * 784 + position % 784) * 4 + head) * 256 + column;
      unsigned address = __cvta_generic_to_shared(destination + fa_encoded_index(row, column));
      int bytes = base + row < last ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address),
                   "l"(cache + offset), "r"(bytes)
                   : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    fa_native_group_sync();
    if ((threadIdx.x % 128) == 0)
      cutlass::arch::ClusterBarrier::arrive(ready);
  }
}

#pragma once

#include "fa_attention.cuh"
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <map>
#include <tuple>

__device__ inline void fa_tmem_load32(uint32_t address, float (&values)[32]) {
  uint32_t words[32];
  cute::SM100_TMEM_LOAD_32dp32b32x::copy(
      address, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
      words[8], words[9], words[10], words[11], words[12], words[13], words[14], words[15],
      words[16], words[17], words[18], words[19], words[20], words[21], words[22], words[23],
      words[24], words[25], words[26], words[27], words[28], words[29], words[30], words[31]);
#pragma unroll
  for (int i = 0; i < 32; ++i)
    values[i] = __uint_as_float(words[i]);
}

__device__ inline void fa_tmem_store32(uint32_t address, const float (&values)[32]) {
  cute::SM100_TMEM_STORE_32dp32b32x::copy(
      __float_as_uint(values[0]), __float_as_uint(values[1]), __float_as_uint(values[2]),
      __float_as_uint(values[3]), __float_as_uint(values[4]), __float_as_uint(values[5]),
      __float_as_uint(values[6]), __float_as_uint(values[7]), __float_as_uint(values[8]),
      __float_as_uint(values[9]), __float_as_uint(values[10]), __float_as_uint(values[11]),
      __float_as_uint(values[12]), __float_as_uint(values[13]), __float_as_uint(values[14]),
      __float_as_uint(values[15]), __float_as_uint(values[16]), __float_as_uint(values[17]),
      __float_as_uint(values[18]), __float_as_uint(values[19]), __float_as_uint(values[20]),
      __float_as_uint(values[21]), __float_as_uint(values[22]), __float_as_uint(values[23]),
      __float_as_uint(values[24]), __float_as_uint(values[25]), __float_as_uint(values[26]),
      __float_as_uint(values[27]), __float_as_uint(values[28]), __float_as_uint(values[29]),
      __float_as_uint(values[30]), __float_as_uint(values[31]), address);
}

#include "fa_prefill_pipeline.cuh"

struct FAPrefillPipelineTensorMaps {
  alignas(128) CUtensorMap key;
  alignas(128) CUtensorMap value;
};

static inline const FAPrefillPipelineTensorMaps &fa_prefill_pipeline_tensor_maps(TensorView cache) {
  using CacheKey = std::tuple<int, uintptr_t, int64_t>;
  static thread_local std::map<CacheKey, FAPrefillPipelineTensorMaps> pipeline_maps;
  CacheKey key{cache.device().device_id, reinterpret_cast<uintptr_t>(cache.data_ptr()),
               cache.size(0)};
  auto found = pipeline_maps.find(key);
  if (found == pipeline_maps.end()) {
    FAPrefillPipelineTensorMaps maps{};
    uint64_t key_dimensions[5] = {64, 784, 4, 4, uint64_t(cache.size(0)) * 2};
    uint64_t key_strides[4] = {2048, 128, 512, 1605632};
    uint64_t value_dimensions[5] = {64, 8, 4, 4, uint64_t(cache.size(0)) * 196};
    uint64_t value_strides[4] = {2048, 128, 512, 16384};
    uint32_t key_box[5] = {64, 64, 4, 1, 1};
    uint32_t value_box[5] = {64, 8, 4, 1, 8};
    uint32_t elements[5] = {1, 1, 1, 1, 1};
    auto encode = [&](CUtensorMap *descriptor, const uint64_t *dimensions, const uint64_t *strides,
                      const uint32_t *box) {
      auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, cache.data_ptr(), dimensions, strides,
          box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    };
    encode(&maps.key, key_dimensions, key_strides, key_box);
    encode(&maps.value, value_dimensions, value_strides, value_box);
    found = pipeline_maps.emplace(key, maps).first;
  }
  return found->second;
}

static inline int64_t fa_prefill_pipeline_l2_bytes(int device) {
  static thread_local std::map<int, int> capacities;
  auto found = capacities.find(device);
  if (found == capacities.end()) {
    int bytes = 0;
    check_cuda(cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, device));
    TVM_FFI_ICHECK(bytes > 0);
    found = capacities.emplace(device, bytes).first;
  }
  return found->second;
}

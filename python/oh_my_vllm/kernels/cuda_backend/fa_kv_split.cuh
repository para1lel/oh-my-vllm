#pragma once

#include "common.cuh"
#include "fa_input.cuh"
#include "launch.cuh"
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <map>
#include <mutex>
#include <tuple>

template <int Rows>
__global__ void fa_kv_split_projection_kernel(const __nv_fp8_e4m3 *input, const float *input_scale,
                                              const float *weight_scale, float *output,
                                              const __grid_constant__ CUtensorMap weight_map) {
  static_assert(Rows == 4 || Rows == 16);
  constexpr int Capacity = Rows == 4 ? 8 : 16;
  constexpr int TmemColumns = 4 * Capacity;
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using MMA = decltype(make_tiled_mma(SM100_MMA_F8F6F4_SS<Element, Element, float, 64, Capacity,
                                                          UMMA::Major::K, UMMA::Major::K>{}));
  using WeightLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(MMA{}, Shape<_64, _512>{})));
  using InputLayout =
      decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{},
                                       partition_shape_B(MMA{}, Shape<Int<Capacity>, _512>{})));
  static_assert(cosize_v<WeightLayout> == 32768 && cosize_v<InputLayout> == Capacity * 512);
  extern __shared__ __align__(128) unsigned char kv_split_storage[];
  auto *weights = reinterpret_cast<Element *>(kv_split_storage);
  auto *inputs = weights + 2 * 32768;
  auto *barriers = reinterpret_cast<uint64_t *>(inputs + Capacity * 512);
  auto *tensor_memory = reinterpret_cast<uint32_t *>(barriers + 3);
  auto input_tile = make_tensor(make_smem_ptr(inputs), InputLayout{});
  int thread = threadIdx.x, warp = thread / 32;
  int first_column = blockIdx.x * 64, split = blockIdx.y;
  TMEM::Allocator1Sm allocator;
  if (warp == 0) {
    allocator.allocate(TmemColumns, tensor_memory);
    allocator.release_allocation_lock();
  }
  if (thread == 0) {
    for (int index = 0; index < 3; ++index)
      initialize_barrier(barriers[index], 1);
    cutlass::arch::fence_barrier_init();
  }
  for (int index = thread; index < Capacity * 32; index += 128) {
    int row = index / 32, column = index % 32 * 16;
    *reinterpret_cast<uint4 *>(&input_tile(make_coord(row, column % 32), 0, column / 32)) =
        make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  auto prefetch = [&](int tile) {
    int groups = tile == 0 ? 4 : 1;
    if (thread == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barriers + 1 + tile,
                                                                     groups * 8192);
#pragma unroll
      for (int sub = 0; sub < 4; ++sub)
        if (sub < groups)
          SM90_TMA_LOAD_2D::copy(&weight_map, barriers + 1 + tile,
                                 uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                 weights + tile * 32768 + sub * 8192,
                                 (split * 5 + tile * 4 + sub) * 128, first_column);
    }
  };
  prefetch(0);
  prefetch(1);
  MMA mma;
  auto input_fragment = mma.get_slice(0).make_fragment_B(input_tile);
  auto partial = partition_fragment_C(mma, Shape<_64, Int<Capacity>>{});
  uint32_t address = *tensor_memory + uint32_t(warp * 32) * 65536u;
  float accumulated[Rows] = {};
  for (int tile = 0; tile < 2; ++tile) {
    int group = split * 5 + tile * 4, groups = tile == 0 ? 4 : 1;
    auto weight_tile = make_tensor(make_smem_ptr(weights + tile * 32768), WeightLayout{});
    auto weight_fragment = mma.get_slice(0).make_fragment_A(weight_tile);
    float factors[4][Rows];
#pragma unroll
    for (int sub = 0; sub < 4; ++sub)
      if (sub < groups)
#pragma unroll
        for (int row = 0; row < Rows; ++row)
          factors[sub][row] = input_scale[row * 40 + group + sub] *
                              weight_scale[int(blockIdx.x / 2) * 40 + group + sub];
    int chunks = groups * 8;
    for (int index = thread; index < Rows * chunks; index += 128) {
      int row = index / chunks, column = index % chunks * 16;
      unsigned destination =
          __cvta_generic_to_shared(&input_tile(make_coord(row, column % 32), 0, column / 32));
      const auto *source = input + int64_t(row) * 5120 + group * 128 + column;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(destination), "l"(source)
                   : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    wait_barrier(barriers[1 + tile], 0);
    cutlass::arch::fence_view_async_shared();
    if (warp == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int sub = 0; sub < 4; ++sub)
        if (sub < groups) {
          partial.data() = *tensor_memory + sub * Capacity;
          mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            gemm(mma, weight_fragment(_, _, sub * 4 + k), input_fragment(_, _, sub * 4 + k),
                 partial);
            mma.accumulate_ = UMMA::ScaleOut::One;
          }
        }
      cutlass::arch::umma_arrive(barriers);
    }
    wait_barrier(*barriers, tile & 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
    for (int sub = 0; sub < 4; ++sub)
      if (sub < groups) {
        uint32_t values[Rows];
        if constexpr (Rows == 4)
          SM100_TMEM_LOAD_32dp32b4x::copy(address + sub * Capacity, values[0], values[1], values[2],
                                          values[3]);
        else {
          SM100_TMEM_LOAD_32dp32b8x::copy(address + sub * Capacity, values[0], values[1], values[2],
                                          values[3], values[4], values[5], values[6], values[7]);
          SM100_TMEM_LOAD_32dp32b8x::copy(address + sub * Capacity + 8, values[8], values[9],
                                          values[10], values[11], values[12], values[13],
                                          values[14], values[15]);
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row)
          accumulated[row] =
              fmaf(__uint_as_float(values[row]), factors[sub][row], accumulated[row]);
      }
    cutlass::arch::fence_view_async_tmem_load();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
  }
#pragma unroll
  for (int row = 0; row < Rows; ++row)
    if ((thread & 31) < 16)
      output[(int64_t(split) * Rows + row) * 5120 + first_column + warp * 16 + (thread & 31)] =
          accumulated[row];
  __syncthreads();
  if (warp == 0)
    allocator.free(*tensor_memory, TmemColumns);
}

template <int Rows>
static inline void fa_kv_split_projection_launch(TensorView input, TensorView input_scale,
                                                 TensorView weight, TensorView weight_scale,
                                                 TensorView partial, cudaStream_t stream) {
  constexpr int Capacity = Rows == 4 ? 8 : 16;
  constexpr int Shared = ((65536 + Capacity * 512 + 28 + 127) / 128) * 128;
  static_assert(Rows == 4 || Rows == 16);
  TVM_FFI_ICHECK(input.size(0) == Rows && weight.size(0) == 2048 &&
                 int64_t(partial.numel()) >= int64_t(8) * Rows * 5120);
  static std::once_flag configured;
  std::call_once(configured, [] {
    check_cuda(cudaFuncSetAttribute(fa_kv_split_projection_kernel<Rows>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, Shared));
    check_cuda(cudaFuncSetAttribute(fa_kv_split_projection_kernel<Rows>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  });
  using WeightKey = std::tuple<int, uintptr_t>;
  static thread_local std::map<WeightKey, CUtensorMap> weight_maps;
  WeightKey key{weight.device().device_id, reinterpret_cast<uintptr_t>(weight.data_ptr())};
  auto found = weight_maps.find(key);
  if (found == weight_maps.end()) {
    CUtensorMap descriptor{};
    uint64_t dimensions[2] = {5120, 2048}, strides[1] = {5120};
    uint32_t box[2] = {128, 64}, elements[2] = {1, 1};
    auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, weight.data_ptr(), dimensions, strides, box,
        elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    found = weight_maps.emplace(key, descriptor).first;
  }
  launch_kernel(fa_kv_split_projection_kernel<Rows>, dim3(32, 8), dim3(128), Shared, stream,
                static_cast<const __nv_fp8_e4m3 *>(input.data_ptr()),
                static_cast<const float *>(input_scale.data_ptr()),
                static_cast<const float *>(weight_scale.data_ptr()),
                static_cast<float *>(partial.data_ptr()), found->second);
  check_cuda(cudaGetLastError());
}

__device__ __forceinline__ uint4 fa_kv_pack_eight(const __nv_bfloat16 (&values)[8]) {
  uint32_t words[4];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair)
    words[pair] = uint32_t(__bfloat16_as_ushort(values[pair * 2])) |
                  (uint32_t(__bfloat16_as_ushort(values[pair * 2 + 1])) << 16);
  return make_uint4(words[0], words[1], words[2], words[3]);
}

__global__ void fa_kv_split_heads_kernel(const float *partial, __nv_bfloat16 *packed,
                                         const float *k_norm, const int64_t *positions,
                                         const int32_t *slots, __nv_bfloat16 *cache, int rows,
                                         int64_t cache_tokens) {
  int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
  __shared__ __align__(16) __nv_bfloat16 merged_key[256];
  float keys[8] = {}, value_sums[8] = {};
  int column = head * 256 + lane * 8;
#pragma unroll
  for (int split = 0; split < 8; ++split) {
    int64_t index = (int64_t(split) * rows + row) * 5120 + column;
    float4 key_low = *reinterpret_cast<const float4 *>(partial + index);
    float4 key_high = *reinterpret_cast<const float4 *>(partial + index + 4);
    float4 value_low = *reinterpret_cast<const float4 *>(partial + index + 1024);
    float4 value_high = *reinterpret_cast<const float4 *>(partial + index + 1028);
    float key_values[8] = {key_low.x,  key_low.y,  key_low.z,  key_low.w,
                           key_high.x, key_high.y, key_high.z, key_high.w};
    float value_values[8] = {value_low.x,  value_low.y,  value_low.z,  value_low.w,
                             value_high.x, value_high.y, value_high.z, value_high.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      keys[j] += key_values[j];
      value_sums[j] += value_values[j];
    }
  }
  __nv_bfloat16 key_bits[8], value_bits[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    key_bits[j] = __float2bfloat16_rn(__fmaf_rn(0.f, 0.f, __fmul_rn(1.f, keys[j])));
    value_bits[j] = __float2bfloat16_rn(__fmaf_rn(0.f, 0.f, __fmul_rn(1.f, value_sums[j])));
  }
  uint4 packed_key = fa_kv_pack_eight(key_bits), packed_value = fa_kv_pack_eight(value_bits);
  *reinterpret_cast<uint4 *>(packed + int64_t(row) * 2048 + column) = packed_key;
  *reinterpret_cast<uint4 *>(packed + int64_t(row) * 2048 + column + 1024) = packed_value;
  *reinterpret_cast<uint4 *>(merged_key + lane * 8) = packed_key;
  __syncwarp();
  float values[8], total = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    values[j] = __bfloat162float(merged_key[lane + j * 32]);
    total += values[j] * values[j];
  }
  float inverse = rsqrtf(warp_sum(total) / 256.f + 1e-6f);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    values[j] = fa_bf16(values[j] * inverse * (k_norm[lane + j * 32] + 1.f));
  int axis = lane % 3;
  if ((axis == 1 && lane >= 33) || (axis == 2 && lane >= 30))
    axis = 0;
  float angle =
      float(positions[int64_t(axis) * rows + row]) * powf(10000000.f, -float(lane) / 32.f);
  float cosine = fa_bf16(cosf(angle)), sine = fa_bf16(sinf(angle));
  float a = values[0], b = values[1];
  values[0] = fa_bf16(a * cosine - b * sine);
  values[1] = fa_bf16(b * cosine + a * sine);
  int slot = slots[row];
  if (slot < 0 || slot >= cache_tokens) {
    asm volatile("trap;");
    return;
  }
  int64_t key_base = ((int64_t(slot / 784) * 1568 + slot % 784) * 4 + head) * 256;
  *reinterpret_cast<uint4 *>(cache + key_base + int64_t(784) * 4 * 256 + lane * 8) = packed_value;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int dim = lane + j * 32;
    cache[key_base + dim] = __float2bfloat16_rn(values[j]);
  }
}

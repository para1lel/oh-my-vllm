#pragma once

#include "../../../primitives/cuda/common.cuh"
#include "../../../runtime/cuda/launch.cuh"
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/cuda_host_adapter.hpp>
#include <cutlass/arch/barrier.h>
#include <map>
#include <tuple>

template <int Rows, bool Fused>
constexpr int FASmallInputSharedBytes =
    (((Fused ? 2 * 32768 + 10 * 4096 + 24 + 4 + 80 + Rows * 40 * 4 : 3 * 32768 + 4096 + 32 + 4) +
      127) /
     128) *
    128;

template <int Rows, bool Fused = false, class Residual = __nv_bfloat16>
__global__ __launch_bounds__(Fused ? 640 : 128, Fused ? 2 : 1) void fa_small_input_kernel(
    __nv_fp8_e4m3 *input, float *input_scale, const float *weight_scale, __nv_bfloat16 *output,
    const __grid_constant__ CUtensorMap weight_map, int columns, const __nv_bfloat16 *x,
    const Residual *residual, const float *norm, __nv_bfloat16 *summed) {
  static_assert(Rows == 1 || Rows == 4 || Rows == 5);
  constexpr int WeightBuffers = Fused ? 2 : 3, InputTiles = Fused ? 10 : 1;
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using MMA = decltype(make_tiled_mma(
      SM100_MMA_F8F6F4_SS<Element, Element, float, 64, 8, UMMA::Major::K, UMMA::Major::K>{}));
  using WeightLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(MMA{}, Shape<_64, _512>{})));
  using InputLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_B(MMA{}, Shape<_8, _512>{})));
  static_assert(cosize_v<WeightLayout> == 32768 && cosize_v<InputLayout> == 4096);
  extern __shared__ __align__(128) unsigned char fa_input_storage[];
  auto *weights = reinterpret_cast<Element *>(fa_input_storage);
  auto *inputs = weights + WeightBuffers * 32768;
  auto *barriers = reinterpret_cast<uint64_t *>(inputs + InputTiles * 4096);
  auto *tensor_memory = reinterpret_cast<uint32_t *>(barriers + WeightBuffers + 1);
  auto *norm_partials = reinterpret_cast<float *>(tensor_memory + 1);
  auto *private_scales = norm_partials + 20;
  int thread = threadIdx.x, warp = thread / 32, lane = thread & 31;
  int first_column = blockIdx.x * 64;
  TMEM::Allocator1Sm allocator;
  if (warp == 0) {
    allocator.allocate(32, tensor_memory);
    allocator.release_allocation_lock();
  }
  if (thread == 0) {
    for (int index = 0; index < WeightBuffers + 1; ++index)
      initialize_barrier(barriers[index], 1);
    cutlass::arch::fence_barrier_init();
  }
  for (int index = thread; index < InputTiles * 256; index += blockDim.x) {
    int tile = index / 256, row = index % 256 / 32, column = index % 32 * 16;
    auto input_tile = make_tensor(make_smem_ptr(inputs + tile * 4096), InputLayout{});
    *reinterpret_cast<uint4 *>(&input_tile(make_coord(row, column % 32), 0, column / 32)) =
        make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  auto prefetch = [&](int group, int slot) {
    if (thread == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barriers + 1 + slot, 32768);
#pragma unroll
      for (int sub = 0; sub < 4; ++sub)
        SM90_TMA_LOAD_2D::copy(
            &weight_map, barriers + 1 + slot, uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST),
            weights + slot * 32768 + sub * 8192, (group + sub) * 128, first_column);
    }
  };
  prefetch(0, 0);
  prefetch(4, 1);
  if constexpr (Fused) {
    static_assert(Rows == 1);
    int column = thread * 8;
    float data[8], skip[8];
    load_float_eight(x + column, data);
    load_float_eight(residual + column, skip);
    float values[8], total = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float value = data[j] + skip[j];
      values[j] = value;
      total += value * value;
      if (blockIdx.x == 0)
        summed[column + j] = __float2bfloat16_rn(value);
    }
    total = warp_sum(total);
    if (lane == 0)
      norm_partials[warp] = total;
    __syncthreads();
    total = warp_sum(lane < 20 ? norm_partials[lane] : 0.f);
    float inverse = rsqrtf(total / 5120.f + 1e-6f);
    float4 low = *reinterpret_cast<const float4 *>(norm + column);
    float4 high = *reinterpret_cast<const float4 *>(norm + column + 4);
    float gains[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
    float maximum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      values[j] = __bfloat162float(__float2bfloat16_rn(values[j] * inverse * (gains[j] + 1.f)));
      maximum = fmaxf(maximum, fabsf(values[j]));
    }
    unsigned mask = lane < 16 ? 0xffffu : 0xffff0000u;
    maximum = __uint_as_float(__reduce_max_sync(mask, __float_as_uint(maximum)));
    float divisor = fmaxf(maximum, 1e-10f) / 448.f;
    if (lane % 16 == 0) {
      private_scales[column / 128] = divisor;
      if (blockIdx.x == 0)
        input_scale[column / 128] = divisor;
    }
    uint32_t words[2];
#pragma unroll
    for (int j = 0; j < 8; j += 4)
      words[j / 4] = __nv_fp8x4_e4m3(make_float4(values[j] / divisor, values[j + 1] / divisor,
                                                 values[j + 2] / divisor, values[j + 3] / divisor))
                         .__x;
    uint2 quantized = make_uint2(words[0], words[1]);
    int tile = column / 512, local = column % 512;
    auto input_tile = make_tensor(make_smem_ptr(inputs + tile * 4096), InputLayout{});
    *reinterpret_cast<uint2 *>(&input_tile(make_coord(0, local % 32), 0, local / 32)) = quantized;
    if (blockIdx.x == 0)
      *reinterpret_cast<uint2 *>(input + column) = quantized;
    __syncthreads();
    if (thread >= 128)
      return;
  }
  auto consumer_sync = [] {
    if constexpr (Fused)
      asm volatile("bar.sync 1, 128;" ::: "memory");
    else
      __syncthreads();
  };
  MMA mma;
  auto partial = partition_fragment_C(mma, Shape<_64, _8>{});
  uint32_t address = *tensor_memory + uint32_t(warp * 32) * 65536u;
  float accumulated[Rows] = {};
  for (int group = 0; group < 40; group += 4) {
    int tile = group / 4, slot = tile % WeightBuffers;
    if constexpr (!Fused) {
      if (group + 8 < 40)
        prefetch(group + 8, (slot + 2) % 3);
    }
    auto weight_tile = make_tensor(make_smem_ptr(weights + slot * 32768), WeightLayout{});
    auto weight_fragment = mma.get_slice(0).make_fragment_A(weight_tile);
    auto input_tile = make_tensor(make_smem_ptr(inputs + (Fused ? tile * 4096 : 0)), InputLayout{});
    auto input_fragment = mma.get_slice(0).make_fragment_B(input_tile);
    float factors[4][Rows];
#pragma unroll
    for (int sub = 0; sub < 4; ++sub)
#pragma unroll
      for (int row = 0; row < Rows; ++row)
        factors[sub][row] =
            (Fused ? private_scales[row * 40 + group + sub] : input_scale[row * 40 + group + sub]) *
            weight_scale[int(blockIdx.x / 2) * 40 + group + sub];
    if constexpr (!Fused) {
      for (int index = thread; index < Rows * 32; index += 128) {
        int row = index / 32, column = index % 32 * 16;
        unsigned destination =
            __cvta_generic_to_shared(&input_tile(make_coord(row, column % 32), 0, column / 32));
        const auto *source = input + int64_t(row) * 5120 + group * 128 + column;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(destination), "l"(source)
                     : "memory");
      }
      asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }
    wait_barrier(barriers[1 + slot], (tile / WeightBuffers) & 1);
    // Every consumer must finish this wait before the slot can change phase again.
    consumer_sync();
    cutlass::arch::fence_view_async_shared();
    if (warp == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        partial.data() = *tensor_memory + sub * 8;
        mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          gemm(mma, weight_fragment(_, _, sub * 4 + k), input_fragment(_, _, sub * 4 + k), partial);
          mma.accumulate_ = UMMA::ScaleOut::One;
        }
      }
      cutlass::arch::umma_arrive(barriers);
    }
    wait_barrier(*barriers, tile & 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if constexpr (Fused) {
      if (group + 8 < 40)
        prefetch(group + 8, slot);
    }
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      uint32_t values[8];
      if constexpr (Rows == 1)
        SM100_TMEM_LOAD_32dp32b1x::copy(address + sub * 8, values[0]);
      else if constexpr (Rows == 4)
        SM100_TMEM_LOAD_32dp32b4x::copy(address + sub * 8, values[0], values[1], values[2],
                                        values[3]);
      else
        SM100_TMEM_LOAD_32dp32b8x::copy(address + sub * 8, values[0], values[1], values[2],
                                        values[3], values[4], values[5], values[6], values[7]);
#pragma unroll
      for (int row = 0; row < Rows; ++row)
        accumulated[row] = fmaf(__uint_as_float(values[row]), factors[sub][row], accumulated[row]);
    }
    cutlass::arch::fence_view_async_tmem_load();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    consumer_sync();
  }
#pragma unroll
  for (int row = 0; row < Rows; ++row)
    if (lane < 16) {
      float value = __fmaf_rn(0.f, 0.f, __fmul_rn(1.f, accumulated[row]));
      output[int64_t(row) * columns + first_column + warp * 16 + lane] = __float2bfloat16_rn(value);
    }
  consumer_sync();
  if (warp == 0)
    allocator.free(*tensor_memory, 32);
}

template <int Rows, bool Fused = false, class Residual = __nv_bfloat16>
static inline void
fa_small_input_launch(TensorView input, TensorView input_scale, TensorView weight,
                      TensorView weight_scale, TensorView output, int columns, cudaStream_t stream,
                      const __nv_bfloat16 *x = nullptr, const Residual *residual = nullptr,
                      const float *norm = nullptr, __nv_bfloat16 *summed = nullptr) {
  TVM_FFI_ICHECK(columns == 14336 || columns == 12288);
  if constexpr (Fused)
    TVM_FFI_ICHECK(x && residual && norm && summed);
  constexpr int Shared = FASmallInputSharedBytes<Rows, Fused>;
  using WeightKey = std::tuple<int, uintptr_t, int>;
  static thread_local std::map<WeightKey, CUtensorMap> weight_maps;
  WeightKey key{weight.device().device_id, reinterpret_cast<uintptr_t>(weight.data_ptr()), columns};
  auto found = weight_maps.find(key);
  if (found == weight_maps.end()) {
    check_cuda(cudaFuncSetAttribute(fa_small_input_kernel<Rows, Fused, Residual>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, Shared));
    check_cuda(cudaFuncSetAttribute(fa_small_input_kernel<Rows, Fused, Residual>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUtensorMap descriptor{};
    uint64_t dimensions[2] = {5120, uint64_t(columns)}, strides[1] = {5120};
    uint32_t box[2] = {128, 64}, elements[2] = {1, 1};
    auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, weight.data_ptr(), dimensions, strides, box,
        elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    found = weight_maps.emplace(key, descriptor).first;
  }
  launch_kernel(fa_small_input_kernel<Rows, Fused, Residual>, dim3(columns / 64),
                dim3(Fused ? 640 : 128), Shared, stream,
                static_cast<__nv_fp8_e4m3 *>(input.data_ptr()),
                static_cast<float *>(input_scale.data_ptr()),
                static_cast<const float *>(weight_scale.data_ptr()),
                static_cast<__nv_bfloat16 *>(output.data_ptr()), found->second, columns, x,
                residual, norm, summed);
  check_cuda(cudaGetLastError());
}

#pragma once

#include "common.cuh"
#include "launch.cuh"
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/cuda_host_adapter.hpp>
#include <cutlass/arch/barrier.h>
#include <map>
#include <tuple>

template <int Rows>
__global__ void paired_up_kernel(const __nv_fp8_e4m3 *input, const float *input_scale,
                                 const float *weight_scale, __nv_bfloat16 *gate_up,
                                 __nv_fp8_e4m3 *product, float *product_scale,
                                 const __grid_constant__ CUtensorMap weight_map) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using MMA = decltype(make_tiled_mma(
      SM100_MMA_F8F6F4_SS<Element, Element, float, 128, 16, UMMA::Major::K, UMMA::Major::K>{}));
  using WeightLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(MMA{}, Shape<_128, _256>{})));
  using InputLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_B(MMA{}, Shape<_16, _256>{})));
  static_assert(cosize_v<WeightLayout> == 32768 && cosize_v<InputLayout> == 4096);
  extern __shared__ __align__(128) unsigned char storage[];
  auto *weights = reinterpret_cast<Element *>(storage);
  auto *inputs = weights + 3 * 65536;
  auto *barriers = reinterpret_cast<uint64_t *>(inputs + 4096);
  auto *tensor_memory = reinterpret_cast<uint32_t *>(barriers + 4);
  auto input_tile = make_tensor(make_smem_ptr(inputs), InputLayout{});
  int thread = threadIdx.x, warp = thread / 32, lane = thread & 31;
  int first_column = blockIdx.x * 128;
  TMEM::Allocator1Sm allocator;
  if (warp == 0) {
    allocator.allocate(64, tensor_memory);
    allocator.release_allocation_lock();
  }
  if (thread == 0) {
    for (int index = 0; index < 4; ++index)
      initialize_barrier(barriers[index], 1);
    cutlass::arch::fence_barrier_init();
  }
  for (int index = thread; index < 256; index += 128) {
    int row = index / 16, column = index % 16 * 16;
    *reinterpret_cast<uint4 *>(&input_tile(make_coord(row, column % 32), 0, column / 32)) =
        make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  auto prefetch = [&](int group, int slot) {
    if (thread == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barriers + 1 + slot, 65536);
#pragma unroll
      for (int sub = 0; sub < 2; ++sub)
#pragma unroll
        for (int side = 0; side < 2; ++side)
          SM90_TMA_LOAD_2D::copy(&weight_map, barriers + 1 + slot,
                                 uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST),
                                 weights + slot * 65536 + side * 32768 + sub * 16384,
                                 (group + sub) * 128, first_column + side * 17408);
    }
  };
  prefetch(0, 0);
  prefetch(2, 1);
  MMA mma;
  auto input_fragment = mma.get_slice(0).make_fragment_B(input_tile);
  auto partial = partition_fragment_C(mma, Shape<_128, _16>{});
  uint32_t address = *tensor_memory + uint32_t(warp * 32) * 65536u;
  float accumulated[2][Rows] = {};
  for (int group = 0; group < 40; group += 2) {
    int tile = group / 2, slot = tile % 3;
    if (group + 4 < 40)
      prefetch(group + 4, (slot + 2) % 3);
    auto gate_tile = make_tensor(make_smem_ptr(weights + slot * 65536), WeightLayout{});
    auto up_tile = make_tensor(make_smem_ptr(weights + slot * 65536 + 32768), WeightLayout{});
    auto gate_fragment = mma.get_slice(0).make_fragment_A(gate_tile);
    auto up_fragment = mma.get_slice(0).make_fragment_A(up_tile);
    float factors[2][2][Rows];
#pragma unroll
    for (int sub = 0; sub < 2; ++sub)
#pragma unroll
      for (int side = 0; side < 2; ++side)
#pragma unroll
        for (int row = 0; row < Rows; ++row)
          factors[sub][side][row] = input_scale[row * 40 + group + sub] *
                                    weight_scale[(int(blockIdx.x) + side * 136) * 40 + group + sub];
    for (int index = thread; index < Rows * 16; index += 128) {
      int row = index / 16, column = index % 16 * 16;
      unsigned destination =
          __cvta_generic_to_shared(&input_tile(make_coord(row, column % 32), 0, column / 32));
      const auto *source = input + int64_t(row) * 5120 + group * 128 + column;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(destination), "l"(source)
                   : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    wait_barrier(barriers[1 + slot], (tile / 3) & 1);
    cutlass::arch::fence_view_async_shared();
    if (warp == 0) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub)
#pragma unroll
        for (int side = 0; side < 2; ++side) {
          partial.data() = *tensor_memory + (sub * 2 + side) * 16;
          mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            if (side == 0)
              gemm(mma, gate_fragment(_, _, sub * 4 + k), input_fragment(_, _, sub * 4 + k),
                   partial);
            else
              gemm(mma, up_fragment(_, _, sub * 4 + k), input_fragment(_, _, sub * 4 + k), partial);
            mma.accumulate_ = UMMA::ScaleOut::One;
          }
        }
      cutlass::arch::umma_arrive(barriers);
    }
    wait_barrier(*barriers, tile & 1);
#pragma unroll
    for (int sub = 0; sub < 2; ++sub)
#pragma unroll
      for (int side = 0; side < 2; ++side) {
        uint32_t values[16];
        if constexpr (Rows == 1)
          SM100_TMEM_LOAD_32dp32b1x::copy(address + (sub * 2 + side) * 16, values[0]);
        else if constexpr (Rows == 4)
          SM100_TMEM_LOAD_32dp32b4x::copy(address + (sub * 2 + side) * 16, values[0], values[1],
                                          values[2], values[3]);
        else
          SM100_TMEM_LOAD_32dp32b8x::copy(address + (sub * 2 + side) * 16, values[0], values[1],
                                          values[2], values[3], values[4], values[5], values[6],
                                          values[7]);
#pragma unroll
        for (int row = 0; row < Rows; ++row)
          accumulated[side][row] =
              fmaf(__uint_as_float(values[row]), factors[sub][side][row], accumulated[side][row]);
      }
    cutlass::arch::fence_view_async_tmem_load();
    __syncthreads();
  }
  float products[Rows];
  __shared__ float maxima[Rows][4];
#pragma unroll
  for (int row = 0; row < Rows; ++row) {
    float gate = __bfloat162float(__float2bfloat16_rn(accumulated[0][row]));
    float up = __bfloat162float(__float2bfloat16_rn(accumulated[1][row]));
    gate_up[int64_t(row) * 34816 + first_column + thread] = __float2bfloat16_rn(gate);
    gate_up[int64_t(row) * 34816 + 17408 + first_column + thread] = __float2bfloat16_rn(up);
    float exponent = expf(-fabsf(gate));
    float sigmoid = (gate >= 0.f ? 1.f : exponent) / (1.f + exponent);
    products[row] = __bfloat162float(__float2bfloat16_rn(gate * sigmoid * up));
    unsigned maximum = __reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(products[row])));
    if (lane == 0)
      maxima[row][warp] = __uint_as_float(maximum);
  }
  __syncthreads();
#pragma unroll
  for (int row = 0; row < Rows; ++row) {
    float maximum = lane < 4 ? maxima[row][lane] : 0.f;
    maximum = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(maximum)));
    float divisor = fmaxf(maximum, 1e-10f) / 448.f;
    if (thread == 0)
      product_scale[row * 136 + blockIdx.x] = divisor;
    product[int64_t(row) * 17408 + first_column + thread] = __nv_fp8_e4m3(products[row] / divisor);
  }
  __syncthreads();
  if (warp == 0)
    allocator.free(*tensor_memory, 64);
}

template <int Rows>
inline void paired_up_launch(TensorView input, TensorView input_scale, TensorView weight,
                             TensorView weight_scale, TensorView gate_up, TensorView product,
                             TensorView product_scale, cudaStream_t stream) {
  using WeightKey = std::tuple<int, uintptr_t>;
  static thread_local std::map<WeightKey, CUtensorMap> weight_maps;
  WeightKey key{weight.device().device_id, reinterpret_cast<uintptr_t>(weight.data_ptr())};
  auto found = weight_maps.find(key);
  if (found == weight_maps.end()) {
    check_cuda(cudaFuncSetAttribute(paired_up_kernel<Rows>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, 200832));
    CUtensorMap descriptor{};
    uint64_t dimensions[2] = {5120, 34816}, strides[1] = {5120};
    uint32_t box[2] = {128, 128}, elements[2] = {1, 1};
    auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, weight.data_ptr(), dimensions, strides, box,
        elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    found = weight_maps.emplace(key, descriptor).first;
  }
  launch_kernel(paired_up_kernel<Rows>, dim3(136), dim3(128), 200832, stream,
                static_cast<const __nv_fp8_e4m3 *>(input.data_ptr()),
                static_cast<const float *>(input_scale.data_ptr()),
                static_cast<const float *>(weight_scale.data_ptr()),
                static_cast<__nv_bfloat16 *>(gate_up.data_ptr()),
                static_cast<__nv_fp8_e4m3 *>(product.data_ptr()),
                static_cast<float *>(product_scale.data_ptr()), found->second);
  check_cuda(cudaGetLastError());
}

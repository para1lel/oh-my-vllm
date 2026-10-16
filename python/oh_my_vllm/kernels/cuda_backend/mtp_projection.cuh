#pragma once
#include "common.cuh"
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/cuda_host_adapter.hpp>
#include <cutlass/arch/barrier.h>
#include <map>
#include <tuple>

template <int Capacity>
__global__ void mtp_projection_kernel(const __nv_bfloat16 *input, float *output, int rows,
                                      const __grid_constant__ CUtensorMap weight_map) {
  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using MMA = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<Element, Element, float, 64, Capacity,
                                                           UMMA::Major::K, UMMA::Major::K>{}));
  using WeightLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(MMA{}, Shape<_64, _256>{})));
  using InputLayout =
      decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{},
                                       partition_shape_B(MMA{}, Shape<Int<Capacity>, _256>{})));
  static_assert(cosize_v<WeightLayout> == 16384);
  extern __shared__ __align__(128) unsigned char storage[];
  auto *weights = reinterpret_cast<Element *>(storage);
  auto *inputs = weights + 2 * 16384;
  auto *barriers = reinterpret_cast<uint64_t *>(inputs + cosize_v<InputLayout>);
  auto *tensor_memory = reinterpret_cast<uint32_t *>(barriers + 3);
  auto input_tile = make_tensor(make_smem_ptr(inputs), InputLayout{});
  int thread = threadIdx.x, warp = thread / 32;
  TMEM::Allocator1Sm allocator;
  if (warp == 0) {
    allocator.allocate(32, tensor_memory);
    allocator.release_allocation_lock();
  }
  if (thread == 0) {
    for (int index = 0; index < 3; ++index)
      initialize_barrier(barriers[index], 1);
    cutlass::arch::fence_barrier_init();
  }
  for (int index = thread; index < Capacity * 32; index += 128) {
    int row = index / 32, column = index % 32 * 8;
    *reinterpret_cast<uint4 *>(&input_tile(make_coord(row, column % 16), 0, column / 16)) =
        make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  auto prefetch = [&](int tile, int slot) {
    if (thread == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barriers + 1 + slot, 32768);
#pragma unroll
      for (int sub = 0; sub < 4; ++sub)
        SM90_TMA_LOAD_2D::copy(
            &weight_map, barriers + 1 + slot, uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST),
            weights + slot * 16384 + sub * 4096, int(blockIdx.y) * 2560 + tile * 256 + sub * 64,
            int(blockIdx.x) * 64);
    }
  };
  prefetch(0, 0);
  prefetch(1, 1);
  MMA mma;
  auto input_fragment = mma.get_slice(0).make_fragment_B(input_tile);
  auto partial = partition_fragment_C(mma, Shape<_64, Int<Capacity>>{});
  partial.data() = *tensor_memory;
  mma.accumulate_ = UMMA::ScaleOut::Zero;
  for (int tile = 0; tile < 10; ++tile) {
    int slot = tile & 1;
    auto weight_tile = make_tensor(make_smem_ptr(weights + slot * 16384), WeightLayout{});
    auto weight_fragment = mma.get_slice(0).make_fragment_A(weight_tile);
    for (int index = thread; index < rows * 32; index += 128) {
      int row = index / 32, column = index % 32 * 8;
      unsigned destination =
          __cvta_generic_to_shared(&input_tile(make_coord(row, column % 16), 0, column / 16));
      const auto *source =
          input + int64_t(row) * 10240 + int(blockIdx.y) * 2560 + tile * 256 + column;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(destination), "l"(source)
                   : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    wait_barrier(barriers[1 + slot], (tile / 2) & 1);
    cutlass::arch::fence_view_async_shared();
    if (warp == 0) {
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        gemm(mma, weight_fragment(_, _, k), input_fragment(_, _, k), partial);
        mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(barriers);
    }
    wait_barrier(*barriers, tile & 1);
    // All warps must pass the old barrier phase before the slot is reused.
    __syncthreads();
    if (tile + 2 < 10)
      prefetch(tile + 2, slot);
  }
  uint32_t address = *tensor_memory + uint32_t(warp * 32) * 65536u;
#pragma unroll
  for (int first = 0; first < Capacity; first += 8) {
    uint32_t values[8];
    SM100_TMEM_LOAD_32dp32b8x::copy(address + first, values[0], values[1], values[2], values[3],
                                    values[4], values[5], values[6], values[7]);
#pragma unroll
    for (int index = 0; index < 8; ++index)
      if (first + index < rows && (thread & 31) < 16)
        output[(int64_t(blockIdx.y) * rows + first + index) * 5120 + int(blockIdx.x) * 64 +
               warp * 16 + (thread & 31)] = __uint_as_float(values[index]);
  }
  cutlass::arch::fence_view_async_tmem_load();
  __syncthreads();
  if (warp == 0)
    allocator.free(*tensor_memory, 32);
}

inline void mtp_projection_launch(TensorView input, TensorView weight, TensorView partials,
                                  int rows, cudaStream_t stream) {
  using Key = std::tuple<int, uintptr_t>;
  static thread_local std::map<Key, CUtensorMap> maps;
  Key key{weight.device().device_id, reinterpret_cast<uintptr_t>(weight.data_ptr())};
  auto found = maps.find(key);
  if (found == maps.end()) {
    check_cuda(cudaFuncSetAttribute(mtp_projection_kernel<8>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    65536 + 8 * 512 + 128));
    check_cuda(cudaFuncSetAttribute(mtp_projection_kernel<16>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    65536 + 16 * 512 + 128));
    check_cuda(cudaFuncSetAttribute(mtp_projection_kernel<32>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    65536 + 32 * 512 + 128));
    CUtensorMap descriptor{};
    uint64_t dimensions[2] = {10240, 5120}, strides[1] = {20480};
    uint32_t box[2] = {64, 64}, elements[2] = {1, 1};
    auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, weight.data_ptr(), dimensions, strides,
        box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    found = maps.emplace(key, descriptor).first;
  }
  auto launch = [&](auto capacity) {
    constexpr int Capacity = decltype(capacity)::value;
    constexpr int Shared = 65536 + Capacity * 256 * 2 + 128;
    mtp_projection_kernel<Capacity><<<dim3(80, 4), 128, Shared, stream>>>(
        static_cast<const __nv_bfloat16 *>(input.data_ptr()),
        static_cast<float *>(partials.data_ptr()), rows, found->second);
    check_cuda(cudaGetLastError());
  };
  if (rows <= 8)
    launch(std::integral_constant<int, 8>{});
  else if (rows <= 16)
    launch(std::integral_constant<int, 16>{});
  else
    launch(std::integral_constant<int, 32>{});
}

#pragma once

#include "gdn_kernels.cuh"
#include "launch.cuh"
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/cuda_host_adapter.hpp>
#include <cutlass/arch/barrier.h>
#include <map>
#include <tuple>

struct GDNInputConvolution {
  const __nv_bfloat16 *weight;
  __nv_bfloat16 *state;
  const int *starts;
  const int *reads;
  const int *writes;
  __nv_bfloat16 *output;
  int batch;
  int slots;
};

template <int Rows, bool FuseConvolution, bool Streaming = false>
__global__ void gdn_input_kernel(const __nv_fp8_e4m3 *input, const float *input_scale,
                                 const float *weight_scale, __nv_bfloat16 *output,
                                 const __grid_constant__ CUtensorMap weight_map,
                                 GDNInputConvolution convolution) {
  static_assert(Rows == 1 || Rows == 4 || Rows == 5);
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using MMA = decltype(make_tiled_mma(
      SM100_MMA_F8F6F4_SS<Element, Element, float, 64, 8, UMMA::Major::K, UMMA::Major::K>{}));
  using WeightLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(MMA{}, Shape<_64, _512>{})));
  using InputLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_B(MMA{}, Shape<_8, _512>{})));
  static_assert(cosize_v<WeightLayout> == 32768 && cosize_v<InputLayout> == 4096);
  extern __shared__ __align__(128) unsigned char storage[];
  auto *weights = reinterpret_cast<Element *>(storage);
  auto *inputs = weights + 3 * 32768;
  auto *barriers = reinterpret_cast<uint64_t *>(inputs + 4096);
  auto *tensor_memory = reinterpret_cast<uint32_t *>(barriers + 4);
  auto input_tile = make_tensor(make_smem_ptr(inputs), InputLayout{});
  int thread = threadIdx.x, warp = thread / 32;
  int first_column = blockIdx.x * 64;
  TMEM::Allocator1Sm allocator;
  if (warp == 0) {
    allocator.allocate(32, tensor_memory);
    allocator.release_allocation_lock();
  }
  if (thread == 0) {
    for (int index = 0; index < 4; ++index)
      initialize_barrier(barriers[index], 1);
    cutlass::arch::fence_barrier_init();
  }
  for (int index = thread; index < 256; index += 128) {
    int row = index / 32, column = index % 32 * 16;
    *reinterpret_cast<uint4 *>(&input_tile(make_coord(row, column % 32), 0, column / 32)) =
        make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  auto prefetch = [&](int group, int slot) {
    if (thread == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barriers + 1 + slot, 32768);
#pragma unroll
      for (int sub = 0; sub < 4; ++sub)
        SM90_TMA_LOAD_2D::copy(&weight_map, barriers + 1 + slot,
                               uint64_t(Streaming ? cute::TMA::CacheHintSm90::EVICT_FIRST
                                                  : cute::TMA::CacheHintSm90::EVICT_NORMAL),
                               weights + slot * 32768 + sub * 8192, (group + sub) * 128,
                               first_column);
    }
  };
  prefetch(0, 0);
  prefetch(4, 1);
  MMA mma;
  auto input_fragment = mma.get_slice(0).make_fragment_B(input_tile);
  auto partial = partition_fragment_C(mma, Shape<_64, _8>{});
  uint32_t address = *tensor_memory + uint32_t(warp * 32) * 65536u;
  float accumulated[Rows] = {};
  for (int group = 0; group < 40; group += 4) {
    int tile = group / 4, slot = tile % 3;
    if (group + 8 < 40)
      prefetch(group + 8, (slot + 2) % 3);
    auto weight_tile = make_tensor(make_smem_ptr(weights + slot * 32768), WeightLayout{});
    auto weight_fragment = mma.get_slice(0).make_fragment_A(weight_tile);
    float factors[4][Rows];
#pragma unroll
    for (int sub = 0; sub < 4; ++sub)
#pragma unroll
      for (int row = 0; row < Rows; ++row)
        factors[sub][row] = input_scale[row * 40 + group + sub] *
                            weight_scale[int(blockIdx.x / 2) * 40 + group + sub];
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
    wait_barrier(barriers[1 + slot], (tile / 3) & 1);
    cutlass::arch::fence_view_async_shared();
    if (warp == 0) {
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
    __syncthreads();
  }
#pragma unroll
  for (int row = 0; row < Rows; ++row)
    if ((thread & 31) < 16)
      output[int64_t(row) * 16384 + first_column + warp * 16 + (thread & 31)] =
          __float2bfloat16_rn(accumulated[row]);
  if constexpr (FuseConvolution) {
    static_assert(Rows == 1);
    if (first_column < 10240 && (thread & 31) < 16) {
      int sequence = 0;
      gdn_require(convolution.starts[0] == 0 && convolution.starts[convolution.batch] == 1);
      while (sequence + 1 < convolution.batch && 0 >= convolution.starts[sequence + 1])
        ++sequence;
      int first = convolution.starts[sequence], last = convolution.starts[sequence + 1];
      int source = convolution.reads[sequence], target = convolution.writes[0];
      gdn_require(first >= 0 && first <= 0 && 0 < last && last <= 1);
      gdn_require(last - first <= 5);
      gdn_require(source >= 0 && source < convolution.slots && target >= -1 &&
                  target < convolution.slots);
      if (target >= 0)
        for (int request = 0; request < convolution.batch; ++request)
          gdn_require(target != convolution.reads[request]);
      int channel = first_column + warp * 16 + (thread & 31);
      __nv_bfloat16 projected = __float2bfloat16_rn(accumulated[0]);
      float sum = 0;
#pragma unroll
      for (int tap = 0; tap < 4; ++tap) {
        int position = tap - 3;
        auto value =
            position >= first
                ? projected
                : convolution.state[(int64_t(source) * 10240 + channel) * 3 + position - first + 3];
        sum += __bfloat162float(value) * __bfloat162float(convolution.weight[channel * 4 + tap]);
        if (target >= 0 && tap > 0)
          convolution.state[(int64_t(target) * 10240 + channel) * 3 + tap - 1] = value;
      }
      convolution.output[channel] = __float2bfloat16_rn(sum * gdn_sigmoid(sum));
    }
  }
  __syncthreads();
  if (warp == 0)
    allocator.free(*tensor_memory, 32);
}

#include "gdn_input_pipeline.cuh"

template <int Rows, bool FuseConvolution, bool Streaming = false, bool Pipeline = false>
static inline void gdn_input_launch(TensorView input, TensorView input_scale, TensorView weight,
                                    TensorView weight_scale, TensorView output,
                                    GDNInputConvolution convolution, cudaStream_t stream) {
  auto kernel = gdn_input_kernel<Rows, FuseConvolution, Streaming>;
  if constexpr (Pipeline)
    kernel = gdn_input_pipeline_kernel<Rows, FuseConvolution, Streaming>;
  constexpr int SharedBytes = Pipeline ? 106624 : 102528;
  constexpr int Threads = Pipeline ? 192 : 128;
  using WeightKey = std::tuple<int, uintptr_t>;
  static thread_local std::map<WeightKey, CUtensorMap> weight_maps;
  WeightKey key{weight.device().device_id, reinterpret_cast<uintptr_t>(weight.data_ptr())};
  auto found = weight_maps.find(key);
  if (found == weight_maps.end()) {
    check_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
    CUtensorMap descriptor{};
    uint64_t dimensions[2] = {5120, 16384}, strides[1] = {5120};
    uint32_t box[2] = {128, 64}, elements[2] = {1, 1};
    auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, weight.data_ptr(), dimensions, strides, box,
        elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        Streaming ? CU_TENSOR_MAP_L2_PROMOTION_NONE : CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    found = weight_maps.emplace(key, descriptor).first;
  }
  launch_kernel(kernel, dim3(256), dim3(Threads), SharedBytes, stream,
                static_cast<const __nv_fp8_e4m3 *>(input.data_ptr()),
                static_cast<const float *>(input_scale.data_ptr()),
                static_cast<const float *>(weight_scale.data_ptr()),
                static_cast<__nv_bfloat16 *>(output.data_ptr()), found->second, convolution);
  check_cuda(cudaGetLastError());
}

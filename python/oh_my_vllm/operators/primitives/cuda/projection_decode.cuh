#pragma once

#include "common.cuh"
#include "../../runtime/cuda/launch.cuh"

#include <cute/tensor.hpp>
#include <map>
#include <tuple>

#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/cuda_host_adapter.hpp>
#include <cutlass/arch/barrier.h>

template <int FixedRows = 0>
__global__ void decode_projection_kernel(const __nv_fp8_e4m3 *activation,
                                         const __nv_fp8_e4m3 *weight, const float *activation_scale,
                                         const float *weight_scale, float *output, int rows,
                                         int columns, int inner, int splits,
                                         const __grid_constant__ CUtensorMap weight_map) {
  static_assert(FixedRows == 0 || FixedRows == 1);
  constexpr int Capacity = FixedRows == 1 ? 1 : 8;
  if constexpr (FixedRows == 1)
    rows = 1;
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using MMA = decltype(make_tiled_mma(
      SM100_MMA_F8F6F4_SS<Element, Element, float, 64, 8, UMMA::Major::K, UMMA::Major::K>{}));
  using WeightLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(MMA{}, Shape<_64, _512>{})));
  using InputLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_B(MMA{}, Shape<_8, _512>{})));
  extern __shared__ __align__(128) unsigned char storage[];
  auto *weight_storage = reinterpret_cast<Element *>(storage);
  auto *input_storage = weight_storage + cosize_v<WeightLayout>;
  auto *barriers = reinterpret_cast<uint64_t *>(input_storage + cosize_v<InputLayout>);
  auto *tensor_memory = reinterpret_cast<uint32_t *>(barriers + 2);
  auto weight_tile = make_tensor(make_smem_ptr(weight_storage), WeightLayout{});
  auto input_tile = make_tensor(make_smem_ptr(input_storage), InputLayout{});
  int thread = threadIdx.x, warp = thread / 32, first_column = blockIdx.x * 64;
  TMEM::Allocator1Sm allocator;
  if (warp == 0) {
    allocator.allocate(32, tensor_memory);
    allocator.release_allocation_lock();
  }
  if (thread == 0) {
    initialize_barrier(*barriers, 1);
    initialize_barrier(barriers[1], 1);
    cutlass::arch::fence_barrier_init();
  }
  for (int i = thread; i < 8 * 32; i += 128) {
    int row = i / 32, column = (i % 32) * 16;
    *reinterpret_cast<uint4 *>(&input_tile(make_coord(row, column % 32), 0, column / 32)) =
        make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  MMA mma;
  auto weight_fragment = mma.get_slice(0).make_fragment_A(weight_tile);
  auto input_fragment = mma.get_slice(0).make_fragment_B(input_tile);
  auto partial_fragment = partition_fragment_C(mma, Shape<_64, _8>{});
  partial_fragment.data() = *tensor_memory;
  uint32_t fragment_address = *tensor_memory + uint32_t(warp * 32) * 65536u;
  float accumulated[Capacity] = {};
  int barrier_phase = 0, scale_groups = inner / 128, split_groups = scale_groups / splits;
  int first_group = blockIdx.y * split_groups;
  for (int local_group = 0; local_group < split_groups; local_group += 4) {
    int group = first_group + local_group;
    int valid_groups = min(4, split_groups - local_group);
    if (thread == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barriers + 1,
                                                                     valid_groups * 8192);
#pragma unroll
      for (int sub = 0; sub < 4; ++sub)
        if (sub < valid_groups)
          SM90_TMA_LOAD_2D::copy(&weight_map, barriers + 1,
                                 uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST),
                                 weight_storage + sub * 8192, (group + sub) * 128, first_column);
    }
    float scale_products[4][Capacity] = {};
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      bool valid = local_group + sub < split_groups;
      float weight_factor =
          valid ? weight_scale[int64_t(blockIdx.x / 2) * scale_groups + group + sub] : 0.f;
#pragma unroll
      for (int row = 0; row < Capacity; ++row)
        if (row < rows && valid)
          scale_products[sub][row] =
              weight_factor * activation_scale[row * scale_groups + group + sub];
    }
    for (int i = thread; i < rows * 32; i += 128) {
      int row = i / 32, column = (i % 32) * 16;
      unsigned destination =
          __cvta_generic_to_shared(&input_tile(make_coord(row, column % 32), 0, column / 32));
      int bytes = local_group * 128 + column < split_groups * 128 ? 16 : 0;
      const auto *source = activation + int64_t(row) * inner + (bytes ? group * 128 + column : 0);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(destination), "l"(source),
                   "r"(bytes)
                   : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    wait_barrier(barriers[1], barrier_phase);
    cutlass::arch::fence_view_async_shared();
    if (warp == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        if (sub >= valid_groups)
          continue;
        partial_fragment.data() = *tensor_memory + sub * 8;
        mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          gemm(mma, weight_fragment(_, _, sub * 4 + k), input_fragment(_, _, sub * 4 + k),
               partial_fragment);
          mma.accumulate_ = UMMA::ScaleOut::One;
        }
      }
      cutlass::arch::umma_arrive(barriers);
    }
    wait_barrier(*barriers, barrier_phase);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    barrier_phase ^= 1;
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      if (sub >= valid_groups)
        continue;
      uint32_t values[Capacity];
      if constexpr (FixedRows == 1)
        SM100_TMEM_LOAD_32dp32b1x::copy(fragment_address + sub * 8, values[0]);
      else
        SM100_TMEM_LOAD_32dp32b8x::copy(fragment_address + sub * 8, values[0], values[1], values[2],
                                        values[3], values[4], values[5], values[6], values[7]);
#pragma unroll
      for (int row = 0; row < Capacity; ++row) {
        if (row < rows && (thread & 31) < 16) {
          float scale = scale_products[sub][row];
          accumulated[row] = fmaf(__uint_as_float(values[row]), scale, accumulated[row]);
        }
      }
    }
    cutlass::arch::fence_view_async_tmem_load();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
  }
#pragma unroll
  for (int row = 0; row < Capacity; ++row)
    if (row < rows && (thread & 31) < 16)
      output[(int64_t(blockIdx.y) * rows + row) * columns + first_column + warp * 16 +
             (thread & 31)] = accumulated[row];
  __syncthreads();
  if (warp == 0) {
    allocator.free(*tensor_memory, 32);
  }
}

template <int FixedRows = 0>
inline void decode_projection_rows_launch(TensorView activation, TensorView weight,
                                          TensorView activation_scale, TensorView weight_scale,
                                          TensorView partials, int rows, int splits,
                                          cudaStream_t stream) {
  static_assert(FixedRows == 0 || FixedRows == 1);
  TVM_FFI_ICHECK(FixedRows == 0 || rows == FixedRows);
  using WeightKey = std::tuple<int, uintptr_t, int64_t, int64_t>;
  static thread_local std::map<WeightKey, CUtensorMap> weight_maps;
  int columns = weight.size(0), inner = weight.size(1);
  WeightKey key{weight.device().device_id, reinterpret_cast<uintptr_t>(weight.data_ptr()), columns,
                inner};
  auto found = weight_maps.find(key);
  if (found == weight_maps.end()) {
    check_cuda(cudaFuncSetAttribute(decode_projection_kernel<FixedRows>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, 36992));
    CUtensorMap descriptor{};
    uint64_t dimensions[2] = {uint64_t(inner), uint64_t(columns)}, strides[1] = {uint64_t(inner)};
    uint32_t box[2] = {128, 64}, elements[2] = {1, 1};
    auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, weight.data_ptr(), dimensions, strides, box,
        elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    found = weight_maps.emplace(key, descriptor).first;
  }
  const auto *a = static_cast<const __nv_fp8_e4m3 *>(activation.data_ptr());
  const auto *w = static_cast<const __nv_fp8_e4m3 *>(weight.data_ptr());
  const auto *sa = static_cast<const float *>(activation_scale.data_ptr());
  const auto *sw = static_cast<const float *>(weight_scale.data_ptr());
  auto *out = static_cast<float *>(partials.data_ptr());
  launch_kernel(decode_projection_kernel<FixedRows>, dim3(columns / 64, splits), dim3(128), 36992,
                stream, a, w, sa, sw, out, rows, columns, inner, splits, found->second);
  check_cuda(cudaGetLastError());
}

inline void decode_projection_launch(TensorView activation, TensorView weight,
                                     TensorView activation_scale, TensorView weight_scale,
                                     TensorView partials, int rows, int splits,
                                     cudaStream_t stream) {
  if (rows == 1)
    decode_projection_rows_launch<1>(activation, weight, activation_scale, weight_scale, partials,
                                     rows, splits, stream);
  else
    decode_projection_rows_launch<0>(activation, weight, activation_scale, weight_scale, partials,
                                     rows, splits, stream);
}

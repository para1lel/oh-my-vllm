#pragma once

#include "half_k32.cuh"
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <map>
#include <set>
#include <tuple>

namespace encoded_projection {

constexpr int tcgen_narrow_rows = 128, tcgen_narrow_columns = 128, tcgen_narrow_slots = 2,
              tcgen_narrow_raw_stages = 3;
constexpr int tcgen_narrow_half_elements = tcgen_narrow_columns * 32;
constexpr int tcgen_narrow_raw_bytes = (tcgen_narrow_rows + tcgen_narrow_columns) * 128;
constexpr int tcgen_narrow_data_bytes =
    tcgen_narrow_slots * (4096 + tcgen_narrow_half_elements) * 2 +
    tcgen_narrow_raw_stages * tcgen_narrow_raw_bytes;
constexpr int tcgen_narrow_shared_bytes = 1024 + tcgen_narrow_data_bytes + 132 * (6144 / 32);

__device__ __forceinline__ void tcgen_narrow_load32(uint32_t address, float (&values)[32]) {
  auto &words = *reinterpret_cast<uint32_t (*)[32]>(values);
  cute::SM100_TMEM_LOAD_32dp32b32x::copy(
      address, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
      words[8], words[9], words[10], words[11], words[12], words[13], words[14], words[15],
      words[16], words[17], words[18], words[19], words[20], words[21], words[22], words[23],
      words[24], words[25], words[26], words[27], words[28], words[29], words[30], words[31]);
}

__device__ inline int tcgen_narrow_raw_address(int row, int column) {
  return row * 128 + (column ^ ((row & 7) * 16));
}

using namespace cute;

struct TcgenNarrowLayout {
  using Element = cutlass::half_t;
  using MMA = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<Element, Element, float, tcgen_narrow_rows, tcgen_narrow_columns,
                           UMMA::Major::K, UMMA::Major::K>{}));
  using InputLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW64_Atom<Element>{}, partition_shape_A(MMA{}, Shape<_128, _32>{})));
  using WeightLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW64_Atom<Element>{}, partition_shape_B(MMA{}, Shape<_128, _32>{})));
  static_assert(cosize_v<InputLayout> == 4096 &&
                cosize_v<WeightLayout> == tcgen_narrow_half_elements);
};

template <class ScaleLayout>
__device__ __forceinline__ void
tcgen_narrow_initialize(uint64_t *loaded, uint64_t *raw_empty, uint64_t *converted,
                        uint64_t *half_empty, uint64_t *computed, uint64_t *tmem_empty,
                        uint8_t *scales_a, uint8_t *scales_b, const uint8_t *input_scales,
                        const uint8_t *weight_scales, int thread, int groups, int first_row,
                        int first_column, int rows, int outputs, ScaleLayout weight_layout) {
  if (thread == 0) {
    for (int slot = 0; slot < tcgen_narrow_slots; ++slot) {
      initialize_barrier(converted[slot], 128);
      initialize_barrier(half_empty[slot], 1);
      initialize_barrier(computed[slot], 1);
      initialize_barrier(tmem_empty[slot], 512);
    }
    for (int slot = 0; slot < tcgen_narrow_raw_stages; ++slot) {
      initialize_barrier(loaded[slot], 1);
      initialize_barrier(raw_empty[slot], 128);
    }
    cutlass::arch::fence_barrier_init();
  }
  for (int index = thread; index < tcgen_narrow_rows * groups; index += blockDim.x) {
    int row = index / groups, group = index % groups;
    scales_a[group * tcgen_narrow_rows + row] =
        first_row + row < rows ? input_scales[int64_t(first_row + row) * groups + group] : 127;
  }
  for (int index = thread; index < (tcgen_narrow_columns / 32) * groups; index += blockDim.x) {
    int column = index / groups, group = index % groups;
    scales_b[group * (tcgen_narrow_columns / 32) + column] =
        first_column + column * 32 < outputs
            ? weight_scales[weight_layout(make_coord(first_column + column * 32, group * 32, 0))]
            : 127;
  }
}

__device__ __forceinline__ void
tcgen_narrow_load_raw(uint8_t *raw, uint64_t *loaded, uint64_t *raw_empty,
                      const CUtensorMap &input_map, const CUtensorMap &weight_map, int thread,
                      int batches, int first_row, int first_column) {
  for (int batch = 0; batch < batches; ++batch) {
    int slot = batch % tcgen_narrow_raw_stages, phase = (batch / tcgen_narrow_raw_stages) & 1;
    if (batch >= tcgen_narrow_raw_stages)
      wait_barrier(raw_empty[slot], phase ^ 1);
    if (thread == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(loaded + slot,
                                                                     tcgen_narrow_raw_bytes);
      SM90_TMA_LOAD_2D::copy(&input_map, loaded + slot, 0, raw + slot * tcgen_narrow_raw_bytes,
                             batch * 128, first_row);
      SM90_TMA_LOAD_2D::copy(&weight_map, loaded + slot, 0,
                             raw + slot * tcgen_narrow_raw_bytes + 16384, batch * 128,
                             first_column);
    }
  }
}

__device__ __forceinline__ void tcgen_narrow_compute(TcgenNarrowLayout::Element *inputs,
                                                     TcgenNarrowLayout::Element *weights,
                                                     uint64_t *converted, uint64_t *half_empty,
                                                     uint64_t *computed, uint64_t *tmem_empty,
                                                     const uint32_t *tensor_memory, int groups) {
  using namespace cute;
  using Element = TcgenNarrowLayout::Element;
  using MMA = TcgenNarrowLayout::MMA;
  using InputLayout = TcgenNarrowLayout::InputLayout;
  using WeightLayout = TcgenNarrowLayout::WeightLayout;
  MMA mma;
  auto partial = partition_fragment_C(mma, Shape<_128, _128>{});
  for (int group = 0; group < groups; ++group) {
    int slot = group % tcgen_narrow_slots, phase = (group / tcgen_narrow_slots) & 1;
    if (group >= tcgen_narrow_slots)
      wait_barrier(tmem_empty[slot], phase ^ 1);
    wait_barrier(converted[slot], phase);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    auto a = make_tensor(make_smem_ptr(inputs + slot * 4096), InputLayout{});
    auto b =
        make_tensor(make_smem_ptr(weights + slot * tcgen_narrow_half_elements), WeightLayout{});
    auto af = mma.get_slice(0).make_fragment_A(a);
    auto bf = mma.get_slice(0).make_fragment_B(b);
    partial.data() = *tensor_memory + slot * tcgen_narrow_columns;
    mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      gemm(mma, af(_, _, k), bf(_, _, k), partial);
      mma.accumulate_ = UMMA::ScaleOut::One;
    }
    cutlass::arch::umma_arrive(computed + slot);
    cutlass::arch::umma_arrive(half_empty + slot);
  }
}

__device__ __forceinline__ void
tcgen_narrow_convert(const uint8_t *raw, TcgenNarrowLayout::Element *inputs,
                     TcgenNarrowLayout::Element *weights, uint64_t *loaded, uint64_t *raw_empty,
                     uint64_t *converted, uint64_t *half_empty, int thread, int groups) {
  using namespace cute;
  using Element = TcgenNarrowLayout::Element;
  using MMA = TcgenNarrowLayout::MMA;
  using InputLayout = TcgenNarrowLayout::InputLayout;
  using WeightLayout = TcgenNarrowLayout::WeightLayout;
  int converter = thread - 128;
  for (int group = 0; group < groups; ++group) {
    int batch = group / 4, raw_slot = batch % tcgen_narrow_raw_stages;
    int slot = group % tcgen_narrow_slots, phase = (group / tcgen_narrow_slots) & 1;
    if (group >= tcgen_narrow_slots)
      wait_barrier(half_empty[slot], phase ^ 1);
    wait_barrier(loaded[raw_slot], (batch / tcgen_narrow_raw_stages) & 1);
    auto a = make_tensor(make_smem_ptr(inputs + slot * 4096), InputLayout{});
    auto b =
        make_tensor(make_smem_ptr(weights + slot * tcgen_narrow_half_elements), WeightLayout{});
    for (int index = converter * 8; index < (tcgen_narrow_rows + tcgen_narrow_columns) * 32;
         index += 128 * 8) {
      int row = index / 32, column = index % 32;
      bool activation = row < tcgen_narrow_rows;
      int local_row = activation ? row : row - tcgen_narrow_rows;
      auto *source = raw + raw_slot * tcgen_narrow_raw_bytes + (activation ? 0 : 16384) +
                     tcgen_narrow_raw_address(local_row, (group % 4) * 32 + column);
      uint64_t packed = *reinterpret_cast<const uint64_t *>(source);
      uint32_t p0 = half_pair(packed), p1 = half_pair(packed >> 16);
      uint32_t p2 = half_pair(packed >> 32), p3 = half_pair(packed >> 48);
      int reordered = column / 2;
      // Each K32 dot uses the released low K4 pairs, then its high K4 pairs.
      if (activation) {
        *reinterpret_cast<uint64_t *>(&a(make_coord(local_row, reordered), 0, 0)) =
            uint64_t(p0) | (uint64_t(p2) << 32);
        *reinterpret_cast<uint64_t *>(&a(make_coord(local_row, reordered), 0, 1)) =
            uint64_t(p1) | (uint64_t(p3) << 32);
      } else {
        *reinterpret_cast<uint64_t *>(&b(make_coord(local_row, reordered), 0, 0)) =
            uint64_t(p0) | (uint64_t(p2) << 32);
        *reinterpret_cast<uint64_t *>(&b(make_coord(local_row, reordered), 0, 1)) =
            uint64_t(p1) | (uint64_t(p3) << 32);
      }
    }
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::ClusterBarrier::arrive(converted + slot);
    if (group % 4 == 3 || group + 1 == groups)
      cutlass::arch::ClusterBarrier::arrive(raw_empty + raw_slot);
  }
}

template <class Output>
__device__ __forceinline__ void
tcgen_narrow_accumulate_output(uint8_t *raw, const uint8_t *scales_a, const uint8_t *scales_b,
                               Output *output, uint64_t *computed, uint64_t *tmem_empty,
                               const uint32_t *tensor_memory, int thread, int groups, int first_row,
                               int first_column, int rows, int outputs) {
  cutlass::arch::warpgroup_reg_alloc<104>();
  float total[32] = {};
  int row = (thread - 256) % tcgen_narrow_rows, segment = (thread - 256) / tcgen_narrow_rows;
  uint32_t address = *tensor_memory + uint32_t(row / 32 * 32) * 65536u + segment * 32;
  for (int group = 0; group < groups; ++group) {
    int slot = group % tcgen_narrow_slots, phase = (group / tcgen_narrow_slots) & 1;
    wait_barrier(computed[slot], phase);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    float sa = fp8_scale(scales_a[group * tcgen_narrow_rows + row]);
    float sb = fp8_scale(scales_b[group * (tcgen_narrow_columns / 32) + segment]);
    float values[32];
    tcgen_narrow_load32(address + slot * tcgen_narrow_columns, values);
    cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
    for (int i = 0; i < 32; ++i)
      total[i] = __fmaf_rn(__fmul_rn(values[i], sa), sb, total[i]);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    cutlass::arch::ClusterBarrier::arrive(tmem_empty + slot);
  }
  asm volatile("barrier.sync 1, 512;" ::: "memory");
  auto *result = reinterpret_cast<Output *>(raw);
#pragma unroll
  for (int column = 0; column < 32; ++column) {
    int offset = (segment * 32 + column) * tcgen_narrow_rows + row;
    if constexpr (std::is_same_v<Output, float>)
      result[offset] = total[column];
    else
      result[offset] = __float2bfloat16_rn(total[column]);
  }
  asm volatile("barrier.sync 1, 512;" ::: "memory");
  for (int index = thread - 256; index < tcgen_narrow_rows * tcgen_narrow_columns; index += 512) {
    int output_row = index / tcgen_narrow_columns, column = index % tcgen_narrow_columns;
    if (first_row + output_row < rows && first_column + column < outputs)
      output[int64_t(first_row + output_row) * outputs + first_column + column] =
          result[column * tcgen_narrow_rows + output_row];
  }
}

template <class Output, class ScaleLayout>
__global__ __launch_bounds__(768, 1) void tcgen_narrow_k32_kernel(
    const uint8_t *input_scales, const uint8_t *weight_scales, Output *output, int rows,
    int outputs, int width, ScaleLayout weight_layout,
    const __grid_constant__ CUtensorMap input_map, const __grid_constant__ CUtensorMap weight_map) {
  using namespace cute;
  using Element = TcgenNarrowLayout::Element;
  using MMA = TcgenNarrowLayout::MMA;
  using InputLayout = TcgenNarrowLayout::InputLayout;
  using WeightLayout = TcgenNarrowLayout::WeightLayout;
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *loaded = reinterpret_cast<uint64_t *>(storage);
  auto *raw_empty = loaded + tcgen_narrow_raw_stages;
  auto *converted = raw_empty + tcgen_narrow_raw_stages;
  auto *half_empty = converted + tcgen_narrow_slots;
  auto *computed = half_empty + tcgen_narrow_slots;
  auto *tmem_empty = computed + tcgen_narrow_slots;
  auto *tensor_memory = reinterpret_cast<uint32_t *>(tmem_empty + tcgen_narrow_slots);
  auto *inputs = reinterpret_cast<Element *>(storage + 1024);
  auto *weights = inputs + tcgen_narrow_slots * 4096;
  auto *raw =
      reinterpret_cast<uint8_t *>(weights + tcgen_narrow_slots * tcgen_narrow_half_elements);
  auto *scales_a = raw + tcgen_narrow_raw_stages * tcgen_narrow_raw_bytes;
  auto *scales_b = scales_a + (width / 32) * tcgen_narrow_rows;
  int thread = threadIdx.x, warp = thread / 32;
  int first_row = blockIdx.x * tcgen_narrow_rows, first_column = blockIdx.y * tcgen_narrow_columns;
  int groups = width / 32, batches = (width + 127) / 128;
  TMEM::Allocator1Sm allocator;
  if (warp == 0) {
    allocator.allocate(tcgen_narrow_slots * tcgen_narrow_columns, tensor_memory);
    allocator.release_allocation_lock();
  }
  tcgen_narrow_initialize(loaded, raw_empty, converted, half_empty, computed, tmem_empty, scales_a,
                          scales_b, input_scales, weight_scales, thread, groups, first_row,
                          first_column, rows, outputs, weight_layout);
  __syncthreads();
  if (thread < 256) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (warp == 0) {
      tcgen_narrow_load_raw(raw, loaded, raw_empty, input_map, weight_map, thread, batches,
                            first_row, first_column);
    } else if (warp == 1) {
      tcgen_narrow_compute(inputs, weights, converted, half_empty, computed, tmem_empty,
                           tensor_memory, groups);
    } else if (thread >= 128) {
      tcgen_narrow_convert(raw, inputs, weights, loaded, raw_empty, converted, half_empty, thread,
                           groups);
    }
  } else {
    tcgen_narrow_accumulate_output<Output>(raw, scales_a, scales_b, output, computed, tmem_empty,
                                           tensor_memory, thread, groups, first_row, first_column,
                                           rows, outputs);
  }
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    allocator.free(*tensor_memory, tcgen_narrow_slots * tcgen_narrow_columns);
  }
}

struct TcgenNarrowMaps {
  alignas(128) CUtensorMap input;
  alignas(128) CUtensorMap weight;
};

using TcgenNarrowMapKey = std::tuple<int, uintptr_t, uintptr_t, int, int, int>;

template <class Output, class ScaleLayout>
void launch_tcgen_narrow_k32(const uint8_t *input, const uint8_t *weight,
                             const uint8_t *input_scales, const uint8_t *weight_scales,
                             Output *output, int rows, int outputs, int width,
                             ScaleLayout weight_layout, cudaStream_t stream) {
  int device;
  check_cuda(cudaGetDevice(&device));
  static thread_local std::set<int> configured;
  if (!configured.count(device)) {
    cudaStreamCaptureStatus capture;
    check_cuda(cudaStreamIsCapturing(stream, &capture));
    TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone)
        << "K32 prefill kernels must be prepared before capture";
    check_cuda(cudaFuncSetAttribute(tcgen_narrow_k32_kernel<Output, ScaleLayout>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    tcgen_narrow_shared_bytes));
    configured.insert(device);
  }
  static thread_local std::map<TcgenNarrowMapKey, TcgenNarrowMaps> descriptors;
  TcgenNarrowMapKey key{device,
                        reinterpret_cast<uintptr_t>(input),
                        reinterpret_cast<uintptr_t>(weight),
                        rows,
                        outputs,
                        width};
  auto found = descriptors.find(key);
  if (found == descriptors.end()) {
    TcgenNarrowMaps maps{};
    auto encode = [&](CUtensorMap *descriptor, const uint8_t *pointer, int count, int tile) {
      uint64_t dimensions[2] = {uint64_t(width), uint64_t(count)};
      uint64_t strides[1] = {uint64_t(width)};
      uint32_t box[2] = {128, uint32_t(tile)}, elements[2] = {1, 1};
      auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t *>(pointer), dimensions,
          strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    };
    encode(&maps.input, input, rows, tcgen_narrow_rows);
    encode(&maps.weight, weight, outputs, tcgen_narrow_columns);
    found = descriptors.emplace(key, maps).first;
  }
  int shared_bytes = ((1024 + tcgen_narrow_data_bytes + 132 * (width / 32) + 127) / 128) * 128;
  tcgen_narrow_k32_kernel<Output, ScaleLayout>
      <<<dim3((rows + 127) / 128, (outputs + tcgen_narrow_columns - 1) / tcgen_narrow_columns), 768,
         shared_bytes, stream>>>(input_scales, weight_scales, output, rows, outputs, width,
                                 weight_layout, found->second.input, found->second.weight);
}

} // namespace encoded_projection

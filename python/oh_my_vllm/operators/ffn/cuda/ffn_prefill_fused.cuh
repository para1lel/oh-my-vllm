#pragma once

#include "../../primitives/cuda/common.cuh"
#include "../../runtime/cuda/dependent.cuh"
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <map>
#include <tuple>

namespace ffn_prefill_fused4 {

constexpr int SharedBytes = 217600;

__device__ inline void load32(uint32_t address, float (&values)[32]) {
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

template <bool Grouped, bool Programmatic = false>
__global__ __launch_bounds__(640, 1) void upper_kernel(
    const float *input_scales, const float *weight_scales, __nv_fp8_e4m3 *product,
    float *product_scales, int rows, int columns, int inner,
    const __grid_constant__ CUtensorMap input_map, const __grid_constant__ CUtensorMap weight_map) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using MMA = decltype(make_tiled_mma(
      SM100_MMA_F8F6F4_SS<Element, Element, float, 128, 256, UMMA::Major::K, UMMA::Major::K>{}));
  using InputLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(MMA{}, Shape<_128, _128>{})));
  using WeightLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_B(MMA{}, Shape<_256, _128>{})));
  static_assert(cosize_v<InputLayout> == 16384 && cosize_v<WeightLayout> == 32768);
  extern __shared__ __align__(128) unsigned char storage[];
  auto *loaded = reinterpret_cast<uint64_t *>(storage);
  auto *computed = loaded + 4;
  auto *ab_empty = computed + 2;
  auto *tmem_empty = ab_empty + 4;
  auto *tensor_memory = reinterpret_cast<uint32_t *>(tmem_empty + 2);
  auto *inputs = reinterpret_cast<Element *>(storage + 128);
  auto *weights = inputs + 65536;
  auto *scales_a = reinterpret_cast<float *>(weights + 131072);
  auto *scales_b = scales_a + 128 * (inner / 128);
  int thread = threadIdx.x, warp = thread / 32;
  int first_row = (Grouped ? blockIdx.z * 16 + blockIdx.x : blockIdx.x) * 128;
  if constexpr (Grouped) {
    if (first_row >= rows)
      return;
  }
  int first_column = blockIdx.y * 128, groups = inner / 128;
  TMEM::Allocator1Sm allocator;
  if (warp == 0) {
    allocator.allocate(512, tensor_memory);
    allocator.release_allocation_lock();
  }
  if (thread == 0) {
    for (int slot = 0; slot < 4; ++slot) {
      initialize_barrier(loaded[slot], 1);
      initialize_barrier(ab_empty[slot], 1);
    }
    for (int slot = 0; slot < 2; ++slot) {
      initialize_barrier(computed[slot], 1);
      initialize_barrier(tmem_empty[slot], 512);
    }
    cutlass::arch::fence_barrier_init();
  }
  auto load_weight_scales = [&]() {
    for (int index = thread * 4; index < 2 * groups; index += 640 * 4) {
      unsigned destination = __cvta_generic_to_shared(scales_b + index);
      const float *source =
          weight_scales + int64_t(blockIdx.y + (index / groups) * 136) * groups + index % groups;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(destination), "l"(source)
                   : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
  };
  if constexpr (Programmatic) {
    load_weight_scales();
    if (thread == 0) {
      for (int slot = 0; slot < 4; ++slot) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(loaded + slot, 49152);
        SM90_TMA_LOAD_2D::copy(&weight_map, loaded + slot, 0, weights + slot * 32768, slot * 128,
                               first_column);
        SM90_TMA_LOAD_2D::copy(&weight_map, loaded + slot, 0, weights + slot * 32768 + 16384,
                               slot * 128, first_column + 17408);
      }
    }
    wait_for_producer();
    launch_consumer();
  }
  for (int index = thread * 4; index < 128 * groups; index += 640 * 4) {
    int row = index / groups, group = index % groups;
    float4 values =
        first_row + row < rows
            ? *reinterpret_cast<const float4 *>(input_scales + int64_t(first_row) * groups + index)
            : make_float4(0.f, 0.f, 0.f, 0.f);
    float unpacked[4] = {values.x, values.y, values.z, values.w};
#pragma unroll
    for (int item = 0; item < 4; ++item)
      scales_a[(group + item) * 128 + (row ^ ((group + item) & 31))] = unpacked[item];
  }
  if constexpr (!Programmatic)
    load_weight_scales();
  __syncthreads();
  if (thread < 128) {
    // Keep both roles within the CTA's initial 640 * 96 register allocation.
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (warp == 0) {
      for (int group = 0; group < groups; ++group) {
        int slot = group & 3, phase = (group / 4) & 1;
        if (group >= 4)
          wait_barrier(ab_empty[slot], phase ^ 1);
        asm volatile("" ::: "memory");
        if (thread == 0) {
          if (!Programmatic || group >= 4)
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(loaded + slot, 49152);
          SM90_TMA_LOAD_2D::copy(&input_map, loaded + slot, 0, inputs + slot * 16384, group * 128,
                                 first_row);
          if (!Programmatic || group >= 4) {
            SM90_TMA_LOAD_2D::copy(&weight_map, loaded + slot, 0, weights + slot * 32768,
                                   group * 128, first_column);
            SM90_TMA_LOAD_2D::copy(&weight_map, loaded + slot, 0, weights + slot * 32768 + 16384,
                                   group * 128, first_column + 17408);
          }
        }
      }
      for (int group = max(0, groups - 4); group < groups; ++group)
        wait_barrier(ab_empty[group & 3], (group / 4) & 1);
    } else if (warp == 1) {
      MMA mma;
      auto partial = partition_fragment_C(mma, Shape<_128, _256>{});
      for (int group = 0; group < groups; ++group) {
        int slot = group & 3, phase = (group / 4) & 1;
        int tmem_slot = group & 1, tmem_phase = (group / 2) & 1;
        if (group >= 2)
          wait_barrier(tmem_empty[tmem_slot], tmem_phase ^ 1);
        wait_barrier(loaded[slot], phase);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        auto input_tile = make_tensor(make_smem_ptr(inputs + slot * 16384), InputLayout{});
        auto weight_tile = make_tensor(make_smem_ptr(weights + slot * 32768), WeightLayout{});
        auto input_fragment = mma.get_slice(0).make_fragment_A(input_tile);
        auto weight_fragment = mma.get_slice(0).make_fragment_B(weight_tile);
        partial.data() = *tensor_memory + tmem_slot * 256;
        mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          gemm(mma, input_fragment(_, _, k), weight_fragment(_, _, k), partial);
          mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(computed + tmem_slot);
        cutlass::arch::umma_arrive(ab_empty + slot);
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<112>();
    float accumulated[64];
#pragma unroll
    for (int i = 0; i < 64; ++i)
      accumulated[i] = 0.f;
    int row = (thread - 128) % 128, segment = (thread - 128) / 128;
    uint32_t address = *tensor_memory + uint32_t(row / 32 * 32) * 65536u + segment * 64;
    for (int group = 0; group < groups; ++group) {
      int tmem_slot = group & 1, tmem_phase = (group / 2) & 1;
      wait_barrier(computed[tmem_slot], tmem_phase);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      float factor =
          scales_a[group * 128 + (row ^ (group & 31))] * scales_b[(segment / 2) * groups + group];
#pragma unroll
      for (int part = 0; part < 2; ++part) {
        float partial[32];
        load32(address + tmem_slot * 256 + part * 32, partial);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int i = 0; i < 32; i += 2) {
          int column = part * 32 + i;
          float2 total = make_float2(accumulated[column], accumulated[column + 1]);
          float2 value = make_float2(partial[i], partial[i + 1]), updated;
          SM100_1x2x1_F32F32F32F32::fma(updated, factor, value, total);
          accumulated[column] = updated.x;
          accumulated[column + 1] = updated.y;
        }
      }
      cutlass::arch::fence_view_async_tmem_load();
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      cutlass::arch::ClusterBarrier::arrive(tmem_empty + tmem_slot);
    }
    asm volatile("barrier.sync 1, 512;" ::: "memory");
    auto *result = reinterpret_cast<__nv_bfloat16 *>(inputs);
#pragma unroll
    for (int column = 0; column < 64; column += 8) {
      uint32_t words[4];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        __nv_bfloat162 converted = __floats2bfloat162_rn(accumulated[column + pair * 2] + 0.f,
                                                         accumulated[column + pair * 2 + 1] + 0.f);
        words[pair] = *reinterpret_cast<uint32_t *>(&converted);
      }
      *reinterpret_cast<uint4 *>(result + row * 264 + segment * 64 + column) =
          make_uint4(words[0], words[1], words[2], words[3]);
    }
  }
  __syncthreads();
  auto *result = reinterpret_cast<__nv_bfloat16 *>(inputs);
  for (int row = warp; row < 128; row += 20) {
    if (first_row + row < rows) {
      int column = (thread & 31) * 4;
      float values[4];
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        float gate = __bfloat162float(result[row * 264 + column + item]);
        float up = __bfloat162float(result[row * 264 + 128 + column + item]);
        float exponent = expf(-fabsf(gate));
        float sigmoid = (gate >= 0 ? 1.f : exponent) / (1.f + exponent);
        values[item] = __bfloat162float(__float2bfloat16_rn(gate * sigmoid * up));
      }
      quantize_four(make_float4(values[0], values[1], values[2], values[3]),
                    product + int64_t(first_row + row) * 17408 + first_column + column,
                    product_scales + int64_t(first_row + row) * 136 + blockIdx.y, 1e-30f);
    }
  }
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    allocator.free(*tensor_memory, 512);
  }
}

struct Maps {
  alignas(128) CUtensorMap input;
  alignas(128) CUtensorMap weight;
};

using MapKey = std::tuple<int, uintptr_t, uintptr_t, int, int, int>;

struct DeviceState {
  int multiprocessors = 0;
  bool configured = false;
  std::map<MapKey, Maps> descriptors;
};

static thread_local std::map<int, DeviceState> device_states;

static DeviceState &prepare_device(int device, cudaStream_t stream) {
  auto &state = device_states[device];
  if (!state.configured) {
    cudaStreamCaptureStatus capture;
    check_cuda(cudaStreamIsCapturing(stream, &capture));
    TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone)
        << "prefill kernels must be prepared before capture";
    check_cuda(
        cudaDeviceGetAttribute(&state.multiprocessors, cudaDevAttrMultiProcessorCount, device));
    TVM_FFI_ICHECK(state.multiprocessors > 0);
    check_cuda(cudaFuncSetAttribute(upper_kernel<false>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
    check_cuda(cudaFuncSetAttribute(upper_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    SharedBytes));
    check_cuda(cudaFuncSetAttribute(upper_kernel<false, true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
    state.configured = true;
  }
  return state;
}

static void upper(const __nv_fp8_e4m3 *input, const __nv_fp8_e4m3 *weight,
                  const float *input_scales, const float *weight_scales, __nv_fp8_e4m3 *product,
                  float *product_scales, int rows, int columns, int inner, cudaStream_t stream,
                  bool programmatic = false) {
  int device;
  check_cuda(cudaGetDevice(&device));
  auto &state = prepare_device(device, stream);
  auto &descriptors = state.descriptors;
  bool grouped = !programmatic && (rows + 127) / 128 >= state.multiprocessors;
  auto kernel = programmatic ? upper_kernel<false, true>
                : grouped    ? upper_kernel<true>
                             : upper_kernel<false>;
  MapKey key{device,
             reinterpret_cast<uintptr_t>(input),
             reinterpret_cast<uintptr_t>(weight),
             rows,
             columns,
             inner};
  auto found = descriptors.find(key);
  if (found == descriptors.end()) {
    Maps maps{};
    auto encode = [&](CUtensorMap *descriptor, const __nv_fp8_e4m3 *pointer, int count, int tile) {
      uint64_t dimensions[2] = {uint64_t(inner), uint64_t(count)}, strides[1] = {uint64_t(inner)};
      uint32_t box[2] = {128, uint32_t(tile)}, elements[2] = {1, 1};
      auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<__nv_fp8_e4m3 *>(pointer),
          dimensions, strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    };
    encode(&maps.input, input, rows, 128);
    encode(&maps.weight, weight, columns, 128);
    found = descriptors.emplace(key, maps).first;
  }
  int shared_bytes = ((128 + 196608 + 130 * (inner / 128) * 4 + 127) / 128) * 128;
  int row_tiles = (rows + 127) / 128;
  dim3 grid(row_tiles, columns / 256);
  if (grouped)
    grid = dim3(row_tiles < 16 ? row_tiles : 16, columns / 256, (row_tiles + 15) / 16);
  if (programmatic)
    launch_dependent_kernel(kernel, grid, dim3(640), shared_bytes, stream, input_scales,
                            weight_scales, product, product_scales, rows, columns, inner,
                            found->second.input, found->second.weight);
  else
    launch_kernel(kernel, grid, dim3(640), shared_bytes, stream, input_scales, weight_scales,
                  product, product_scales, rows, columns, inner, found->second.input,
                  found->second.weight);
  check_cuda(cudaGetLastError());
}

} // namespace ffn_prefill_fused4

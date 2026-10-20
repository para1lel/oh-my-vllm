#pragma once

#include "gdn_wy_common.cuh"
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace gdn_wy {

using NativeElement = cutlass::bfloat16_t;
using NativeProjection = decltype(cute::make_tiled_mma(
    cute::SM100_MMA_F16BF16_SS<NativeElement, NativeElement, float, 128, 64, cute::UMMA::Major::K,
                               cute::UMMA::Major::K>{}));
using NativeStateLayout = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<NativeElement>{},
    cute::partition_shape_A(NativeProjection{}, cute::Shape<cute::_128, cute::_128>{})));
using NativeWeightLayout = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<NativeElement>{},
    cute::partition_shape_B(NativeProjection{}, cute::Shape<cute::_64, cute::_128>{})));

using NativeCoefficientLayout = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<NativeElement>{},
    cute::partition_shape_B(NativeProjection{}, cute::Shape<cute::_64, cute::_64>{})));

template <class Layout>
__device__ inline auto native_tensor(NativeElement *pointer, Layout layout) {
  return cute::make_tensor(cute::make_smem_ptr(pointer), layout);
}

template <class Tensor>
__device__ inline auto &native_element(Tensor &tensor, int row, int column) {
  return tensor(cute::make_coord(row, column % 16), 0, column / 16);
}

__device__ inline void native_copy16(void *target, const void *source, bool valid = true) {
  uint32_t address = uint32_t(__cvta_generic_to_shared(target));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(source),
               "r"(valid ? 16 : 0)
               : "memory");
}

__device__ inline uint32_t native_pair(float a, float b) {
  __nv_bfloat162_raw pair = __floats2bfloat162_rn(a, b);
  return uint32_t(pair.x) | uint32_t(pair.y) << 16;
}

__device__ inline void native_load32(uint32_t address, float (&values)[32]) {
  uint32_t words[32];
  cute::SM100_TMEM_LOAD_32dp32b32x::copy(
      address, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
      words[8], words[9], words[10], words[11], words[12], words[13], words[14], words[15],
      words[16], words[17], words[18], words[19], words[20], words[21], words[22], words[23],
      words[24], words[25], words[26], words[27], words[28], words[29], words[30], words[31]);
  cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
  for (int i = 0; i < 32; ++i)
    values[i] = __uint_as_float(words[i]);
}

__device__ inline void carry_load32(uint32_t address, float (&values)[32]) {
  uint32_t words[32];
  cute::SM100_TMEM_LOAD_16dp32b32x::copy(
      address, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
      words[8], words[9], words[10], words[11], words[12], words[13], words[14], words[15],
      words[16], words[17], words[18], words[19], words[20], words[21], words[22], words[23],
      words[24], words[25], words[26], words[27], words[28], words[29], words[30], words[31]);
  cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
  for (int i = 0; i < 32; ++i)
    values[i] = __uint_as_float(words[i]);
}

} // namespace gdn_wy

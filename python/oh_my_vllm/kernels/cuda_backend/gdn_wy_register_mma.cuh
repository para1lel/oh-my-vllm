#pragma once

#include <cuda_bf16.h>
#include <type_traits>

namespace gdn_wy_register_mma {
struct accumulator {};

struct matrix_a {};

struct matrix_b {};

struct row_major {};

struct col_major {};

template <class Role, int M, int N, int K, class Type, class Layout = row_major> struct fragment {
  static_assert(M == 16 && N == 16 && K == 16);
  static constexpr int num_elements = 8;
  Type x[num_elements];
};

__device__ inline unsigned shared_address(const void *pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

template <class Role, class Layout>
__device__ inline void load_matrix_sync(fragment<Role, 16, 16, 16, __nv_bfloat16, Layout> &operand,
                                        const __nv_bfloat16 *pointer, int stride) {
  int lane = threadIdx.x % 32;
  unsigned address = shared_address(pointer + (lane % 16) * stride + (lane / 16) * 8);
  auto *registers = reinterpret_cast<unsigned *>(operand.x);
  if constexpr (std::is_same_v<Role, matrix_b> && std::is_same_v<Layout, row_major>)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]), "=r"(registers[3])
                 : "r"(address));
  else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]), "=r"(registers[3])
                 : "r"(address));
}

__device__ inline void fill_fragment(fragment<accumulator, 16, 16, 16, float> &value,
                                     float initial) {
  for (int index = 0; index < 8; ++index)
    value.x[index] = initial;
}

template <class Layout>
__device__ inline void mma_sync(fragment<accumulator, 16, 16, 16, float> &result,
                                const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> &a,
                                const fragment<matrix_b, 16, 16, 16, __nv_bfloat16, Layout> &b,
                                const fragment<accumulator, 16, 16, 16, float> &initial) {
  const auto *ar = reinterpret_cast<const unsigned *>(a.x);
  const auto *br = reinterpret_cast<const unsigned *>(b.x);
  for (int half = 0; half < 2; ++half) {
    int b0 = std::is_same_v<Layout, col_major> ? half : half * 2;
    int b1 = std::is_same_v<Layout, col_major> ? half + 2 : half * 2 + 1;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(result.x[half * 4]), "=f"(result.x[half * 4 + 1]),
                   "=f"(result.x[half * 4 + 2]), "=f"(result.x[half * 4 + 3])
                 : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br[b0]), "r"(br[b1]),
                   "f"(initial.x[half * 4]), "f"(initial.x[half * 4 + 1]),
                   "f"(initial.x[half * 4 + 2]), "f"(initial.x[half * 4 + 3]));
  }
}
} // namespace gdn_wy_register_mma

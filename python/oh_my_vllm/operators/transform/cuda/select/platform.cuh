#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/kernel_launch.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <vector>
#include <type_traits>
#include "../../../runtime/cuda/contracts.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#define SELECT_ENABLE_SM100
#endif

#define SELECT_ASSERT(condition, ...) TVM_FFI_ICHECK(condition)
#define SELECT_CUDA_CHECK(expression) check_cuda(expression)

namespace select_platform {

using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

template <typename T> CUTE_HOST_DEVICE constexpr T ceil_div(T value, T divisor) {
  return (value + divisor - 1) / divisor;
}

template <typename T> CUTE_HOST_DEVICE constexpr T ceil(T value, T divisor) {
  return ceil_div(value, divisor) * divisor;
}

CUTE_DEVICE void trap() { asm volatile("trap;"); }

CUTE_DEVICE void st_shared(void *pointer, __int128_t value) {
  asm volatile("st.shared.b128 [%0], %1;" ::"r"(cute::cast_smem_ptr_to_uint(pointer)), "q"(value));
}

CUTE_DEVICE __int128_t ld_shared(const void *pointer) {
  __int128_t value;
  asm volatile("ld.shared.b128 %0, [%1];"
               : "=q"(value)
               : "r"(cute::cast_smem_ptr_to_uint(pointer)));
  return value;
}

template <typename T>
std::vector<uint64_t> make_stride_helper(const std::vector<T> &strides, size_t element_size) {
  std::vector<uint64_t> bytes;
  for (auto stride : strides)
    bytes.push_back(uint64_t(stride) * element_size);
  return bytes;
}

inline CUtensorMap make_tensor_map(const std::vector<uint64_t> &dimensions,
                                   const std::vector<uint64_t> &strides,
                                   const std::vector<uint32_t> &box, void *pointer,
                                   CUtensorMapDataType type, CUtensorMapSwizzle swizzle,
                                   CUtensorMapL2promotion promotion) {
  void *entry = nullptr;
  cudaDriverEntryPointQueryResult query;
  check_cuda(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry, 12000,
                                              cudaEnableDefault, &query));
  TVM_FFI_ICHECK(query == cudaDriverEntryPointSuccess);
  std::vector<uint32_t> elements(dimensions.size(), 1);
  CUtensorMap result;
  auto encode = reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(entry);
  TVM_FFI_ICHECK(encode(&result, type, dimensions.size(), pointer, dimensions.data(),
                        strides.data(), box.data(), elements.data(), CU_TENSOR_MAP_INTERLEAVE_NONE,
                        swizzle, promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
  return result;
}

CUTE_DEVICE void load128(const void *pointer, void *output) {
  auto &v = *reinterpret_cast<ulonglong2 *>(output);
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u64 {%0,%1}, [%2];"
               : "=l"(v.x), "=l"(v.y)
               : "l"(pointer));
}

CUTE_DEVICE void load256(const void *pointer, void *output) {
  auto *v = reinterpret_cast<ulonglong2 *>(output);
  asm volatile("ld.global.nc.L1::no_allocate.L2::evict_last.L2::256B.v4.u64 {%0,%1,%2,%3}, [%4];"
               : "=l"(v[0].x), "=l"(v[0].y), "=l"(v[1].x), "=l"(v[1].y)
               : "l"(pointer));
}

CUTE_DEVICE void store128(void *pointer, const void *input) {
  auto v = *reinterpret_cast<const ulonglong2 *>(input);
  asm volatile("st.global.L1::no_allocate.v2.u64 [%0], {%1,%2};" ::"l"(pointer), "l"(v.x),
               "l"(v.y));
}

CUTE_DEVICE void store256(void *pointer, const void *input) {
  if (reinterpret_cast<uintptr_t>(pointer) % 32 != 0) {
    store128(pointer, input);
    store128(static_cast<char *>(pointer) + 16, static_cast<const char *>(input) + 16);
    return;
  }
  auto *v = reinterpret_cast<const ulonglong2 *>(input);
  asm volatile(
      "st.global.L1::no_allocate.L2::evict_first.v4.u64 [%0], {%1,%2,%3,%4};" ::"l"(pointer),
      "l"(v[0].x), "l"(v[0].y), "l"(v[1].x), "l"(v[1].y));
}

} // namespace select_platform

#define SELECT_LDG_128(input, output, ...) select_platform::load128(input, output)
#define SELECT_LDG_256(input, output, ...) select_platform::load256(input, output)
#define SELECT_STG_128(output, input, ...) select_platform::store128(output, input)
#define SELECT_STG_256(output, input, ...) select_platform::store256(output, input)

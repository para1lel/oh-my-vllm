#pragma once

#include "fa_attention.cuh"
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <map>
#include <tuple>

__device__ inline void fa_tmem_load32(uint32_t address, float (&values)[32]) {
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

__device__ inline void fa_tmem_store32(uint32_t address, const float (&values)[32]) {
  cute::SM100_TMEM_STORE_32dp32b32x::copy(
      __float_as_uint(values[0]), __float_as_uint(values[1]), __float_as_uint(values[2]),
      __float_as_uint(values[3]), __float_as_uint(values[4]), __float_as_uint(values[5]),
      __float_as_uint(values[6]), __float_as_uint(values[7]), __float_as_uint(values[8]),
      __float_as_uint(values[9]), __float_as_uint(values[10]), __float_as_uint(values[11]),
      __float_as_uint(values[12]), __float_as_uint(values[13]), __float_as_uint(values[14]),
      __float_as_uint(values[15]), __float_as_uint(values[16]), __float_as_uint(values[17]),
      __float_as_uint(values[18]), __float_as_uint(values[19]), __float_as_uint(values[20]),
      __float_as_uint(values[21]), __float_as_uint(values[22]), __float_as_uint(values[23]),
      __float_as_uint(values[24]), __float_as_uint(values[25]), __float_as_uint(values[26]),
      __float_as_uint(values[27]), __float_as_uint(values[28]), __float_as_uint(values[29]),
      __float_as_uint(values[30]), __float_as_uint(values[31]), address);
}

inline constexpr int FAPrefillSharedBytes = (128 * 256 + 2 * 64 * 256 + 128 * 64) * 2 + 128;

__global__ void fa_prefill_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *cache,
                                  const int32_t *tables, const int32_t *lengths,
                                  const int32_t *starts, float *partial, float *lse,
                                  int table_width, int tiles, int max_queries, int query_rows,
                                  int cache_pages, int first,
                                  const __grid_constant__ CUtensorMap key_map,
                                  const __grid_constant__ CUtensorMap value_map) {
  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using QKMma = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<Element, Element, float, 128, 64, UMMA::Major::K, UMMA::Major::K>{}));
  using PVMma = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<Element, Element, float, 128, 256, UMMA::Major::K, UMMA::Major::MN>{}));
  using QLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(QKMma{}, Shape<_128, _256>{})));
  using KLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{},
                                                   partition_shape_B(QKMma{}, Shape<_64, _256>{})));
  using PLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{},
                                                   partition_shape_A(PVMma{}, Shape<_128, _64>{})));
  using VLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_MN_SW128_Atom<Element>{},
                                                   partition_shape_B(PVMma{}, Shape<_256, _64>{})));
  static_assert((cosize_v<QLayout> + cosize_v<KLayout> + cosize_v<VLayout> +
                 cosize_v<PLayout>)*sizeof(Element) +
                    2 * sizeof(uint64_t) + sizeof(uint32_t) <=
                FAPrefillSharedBytes);
  static_assert(128 * 33 * sizeof(float) <= cosize_v<KLayout> * sizeof(Element));
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *query_storage = reinterpret_cast<Element *>(storage);
  auto *key_storage = query_storage + cosize_v<QLayout>;
  auto *value_storage = key_storage + cosize_v<KLayout>;
  auto *probability_storage = value_storage + cosize_v<VLayout>;
  auto *barrier = reinterpret_cast<uint64_t *>(probability_storage + cosize_v<PLayout>);
  auto *tmem_base = reinterpret_cast<uint32_t *>(barrier + 2);
  auto sQ = make_tensor(make_smem_ptr(query_storage), QLayout{});
  auto sK = make_tensor(make_smem_ptr(key_storage), KLayout{});
  auto sV = make_tensor(make_smem_ptr(value_storage), VLayout{});
  auto sP = make_tensor(make_smem_ptr(probability_storage), PLayout{});
  int tid = threadIdx.x, warp = tid / 32;
  int seq = blockIdx.x / tiles, qbase = (blockIdx.x % tiles) * 128, kh = blockIdx.y;
  int start = starts[seq], end = starts[seq + 1];
  if (start < 0 || end > query_rows || end <= start || end - start > max_queries ||
      starts[0] != 0 || starts[gridDim.x / tiles] != query_rows) {
    asm volatile("trap;");
    return;
  }
  if (start + qbase / 6 >= end)
    return;
  int last = lengths[min(end - 1, start + (qbase + 127) / 6)];
  if (last < first || last > table_width * 784) {
    asm volatile("trap;");
    return;
  }
  TMEM::Allocator1Sm allocator;
  if (warp == 0)
    allocator.allocate(512, tmem_base);
  if (tid == 0) {
    initialize_barrier(*barrier, 1);
    initialize_barrier(barrier[1], 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  uint32_t lane_base = *tmem_base + uint32_t(warp * 32) * 65536u;
  float zeros[32] = {};
#pragma unroll
  for (int c = 0; c < 256; c += 32)
    fa_tmem_store32(lane_base + 64 + c, zeros);
  cutlass::arch::fence_view_async_tmem_store();
  for (int i = tid; i < 128 * 32; i += 128) {
    int r = i / 32, c = (i % 32) * 8;
    int row = start + (qbase + r) / 6, head = kh * 6 + (qbase + r) % 6;
    uint4 value =
        row < end ? *reinterpret_cast<const uint4 *>(query + (int64_t(row) * 24 + head) * 256 + c)
                  : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4 *>(&sQ(make_coord(r, c % 16), 0, c / 16)) = value;
  }
  __syncthreads();
  QKMma qk;
  PVMma pv;
  auto qfrag = qk.get_slice(0).make_fragment_A(sQ);
  auto kfrag = qk.get_slice(0).make_fragment_B(sK);
  auto pfrag = pv.get_slice(0).make_fragment_A(sP);
  auto vfrag = pv.get_slice(0).make_fragment_B(sV);
  auto score = partition_fragment_C(qk, Shape<_128, _64>{});
  auto output = partition_fragment_C(pv, Shape<_128, _256>{});
  score.data() = *tmem_base;
  output.data() = *tmem_base + 64;
  int phase = 0, transport_phase = 0;
  float maximum = -INFINITY, denominator = 0;
  int row = start + (qbase + tid) / 6;
  int length = row < end ? lengths[row] : 0;
  if (length < 0 || length > last) {
    asm volatile("trap;");
    return;
  }
  for (int base = first; base < last; base += 64) {
    if (first % 16 == 0 && base + 64 <= last) {
      if (tid == 0) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barrier + 1, 65536);
#pragma unroll
        for (int token = 0; token < 64; token += 16) {
          int position = base + token;
          int page = tables[seq * table_width + position / 784];
          if (page < 0 || page >= cache_pages) {
            asm volatile("trap;");
            return;
          }
#pragma unroll
          for (int block = 0; block < 4; ++block)
            SM90_TMA_LOAD_5D::copy(
                &key_map, barrier + 1, uint64_t(TMA::CacheHintSm90::EVICT_NORMAL),
                key_storage + block * 4096 + token * 64, 0, position % 784, block, kh, page * 2);
        }
#pragma unroll
        for (int token = 0; token < 64; token += 8) {
          int position = base + token;
          int page = tables[seq * table_width + position / 784];
          if (page < 0 || page >= cache_pages) {
            asm volatile("trap;");
            return;
          }
          SM90_TMA_LOAD_5D::copy(
              &value_map, barrier + 1, uint64_t(TMA::CacheHintSm90::EVICT_NORMAL),
              value_storage + token * 256, 0, position % 784, 0, kh, page * 2 + 1);
        }
      }
      wait_barrier(barrier[1], transport_phase);
      transport_phase ^= 1;
      __syncthreads();
    } else {
      for (int i = tid; i < 64 * 32; i += 128) {
        int r = i / 32, c = (i % 32) * 8, pos = base + r;
        int logical = min(pos, last - 1), page = tables[seq * table_width + logical / 784];
        if (page < 0 || page >= cache_pages) {
          asm volatile("trap;");
          return;
        }
        int64_t offset = ((int64_t(page) * 1568 + logical % 784) * 4 + kh) * 256 + c;
        unsigned kd = __cvta_generic_to_shared(&sK(make_coord(r, c % 16), 0, c / 16));
        unsigned vd = __cvta_generic_to_shared(&sV(make_coord(c, r % 16), 0, r / 16));
        int bytes = pos < last ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(kd), "l"(cache + offset),
                     "r"(bytes)
                     : "memory");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(vd),
                     "l"(cache + offset + int64_t(784) * 4 * 256), "r"(bytes)
                     : "memory");
      }
      asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }
    cutlass::arch::fence_view_async_shared();
    if (warp == 0) {
      qk.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        gemm(qk, qfrag(_, _, k), kfrag(_, _, k), score);
        qk.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(barrier);
    }
    wait_barrier(*barrier, phase);
    phase ^= 1;
    float values[64], next_max = maximum;
#pragma unroll
    for (int c = 0; c < 64; c += 32) {
      float block[32];
      fa_tmem_load32(lane_base + c, block);
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        float s =
            (row < end && base + c + j < length) ? block[j] * 0.09016844005556021f : -INFINITY;
        values[c + j] = s;
        next_max = fmaxf(next_max, s);
      }
    }
    float safe = next_max == -INFINITY ? 0.f : next_max;
    float alpha = fa_exp2(maximum - safe), sum = 0;
#pragma unroll
    for (int c = 0; c < 64; ++c) {
      float probability = fa_exp2(values[c] - safe);
      sum += probability;
      sP(make_coord(tid, c % 16), 0, c / 16) = Element(probability);
    }
    maximum = next_max;
    denominator = denominator * alpha + sum;
    if (__any_sync(0xffffffff, alpha != 1.f)) {
#pragma unroll 1
      for (int c = 0; c < 256; c += 32) {
        float block[32];
        fa_tmem_load32(lane_base + 64 + c, block);
#pragma unroll
        for (int j = 0; j < 32; ++j)
          block[j] *= alpha;
        fa_tmem_store32(lane_base + 64 + c, block);
      }
    }
    cutlass::arch::fence_view_async_tmem_store();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    if (warp == 0) {
      pv.accumulate_ = UMMA::ScaleOut::One;
#pragma unroll
      for (int k = 0; k < 4; ++k)
        gemm(pv, pfrag(_, _, k), vfrag(_, _, k), output);
      cutlass::arch::umma_arrive(barrier);
    }
    wait_barrier(*barrier, phase);
    phase ^= 1;
    __syncthreads();
  }
  int head = kh * 6 + (qbase + tid) % 6;
  float inverse = denominator > 0 ? 1.f / denominator : 1.f;
  auto *staged = reinterpret_cast<float *>(key_storage);
#pragma unroll
  for (int c = 0; c < 256; c += 32) {
    float block[32];
    fa_tmem_load32(lane_base + 64 + c, block);
#pragma unroll
    for (int j = 0; j < 32; ++j)
      staged[tid * 33 + j] = block[j] * inverse;
    __syncthreads();
    for (int i = tid; i < 128 * 32; i += 128) {
      int r = i / 32, column = i % 32, target_row = start + (qbase + r) / 6;
      int target_head = kh * 6 + (qbase + r) % 6;
      if (target_row < end)
        partial[(int64_t(target_row) * 24 + target_head) * 256 + c + column] =
            staged[r * 33 + column];
    }
    __syncthreads();
  }
  if (row < end)
    lse[int64_t(row) * 24 + head] = maximum + log2f(denominator);
  cutlass::arch::fence_view_async_tmem_load();
  __syncthreads();
  if (warp == 0) {
    allocator.release_allocation_lock();
    allocator.free(*tmem_base, 512);
  }
}

struct FAPrefillTensorMaps {
  alignas(128) CUtensorMap key;
  alignas(128) CUtensorMap value;
};

inline const FAPrefillTensorMaps &fa_prefill_tensor_maps(TensorView cache) {
  using CacheKey = std::tuple<int, uintptr_t, int64_t>;
  static thread_local std::map<CacheKey, FAPrefillTensorMaps> descriptors;
  CacheKey key{cache.device().device_id, reinterpret_cast<uintptr_t>(cache.data_ptr()),
               cache.size(0)};
  auto found = descriptors.find(key);
  if (found == descriptors.end()) {
    FAPrefillTensorMaps maps{};
    uint64_t dimensions[5] = {64, 784, 4, 4, uint64_t(cache.size(0)) * 2};
    uint64_t strides[4] = {2048, 128, 512, 1605632};
    uint32_t key_box[5] = {64, 16, 1, 1, 1};
    uint32_t value_box[5] = {64, 8, 4, 1, 1};
    uint32_t elements[5] = {1, 1, 1, 1, 1};
    auto encode = [&](CUtensorMap *descriptor, const uint32_t *box) {
      auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, cache.data_ptr(), dimensions, strides,
          box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    };
    encode(&maps.key, key_box);
    encode(&maps.value, value_box);
    found = descriptors.emplace(key, maps).first;
  }
  return found->second;
}

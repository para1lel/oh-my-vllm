#pragma once

namespace fa_ring_stages {
using namespace cute;
using Barrier = cutlass::arch::ClusterBarrier;
using Element = cutlass::bfloat16_t;
using Barrier = cutlass::arch::ClusterBarrier;
using QKMma = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_SS<Element, Element, float, 128, 128, UMMA::Major::K, UMMA::Major::K>{}));
using PVMma = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_TS<Element, Element, float, 128, 128, UMMA::Major::K, UMMA::Major::MN>{}));
using QLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{},
                                                 partition_shape_A(QKMma{}, Shape<_128, _256>{})));
using KLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{},
                                                 partition_shape_B(QKMma{}, Shape<_128, _128>{})));
using PLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{},
                                                 partition_shape_A(PVMma{}, Shape<_128, _128>{})));
using VLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_MN_SW128_Atom<Element>{},
                                                 partition_shape_B(PVMma{}, Shape<_128, _128>{})));
static_assert((cosize_v<QLayout> + 2 * cosize_v<KLayout> + 2 * cosize_v<VLayout> +
               0 * cosize_v<PLayout>)*sizeof(Element) +
                  384 * sizeof(float) + 128 <=
              FARingSharedBytes);

struct Shared {
  Element *query_storage, *key_storage, *value_storage, *probability_storage;
  float *alpha_storage, *inverse_storage;
  uint64_t *barriers, *key_ready, *value_ready, *consumed, *score_ready, *score_free;
  uint64_t *probability_ready, *corrected;
  uint32_t *tmem_base;
  uint64_t *key_consumed, *value_consumed, *remote_key_ready, *remote_pv_ready;

  __device__ explicit Shared(unsigned char *storage) {
    query_storage = reinterpret_cast<Element *>(storage);
    key_storage = query_storage + cosize_v<QLayout>;
    value_storage = key_storage + 2 * cosize_v<KLayout>;
    probability_storage = value_storage + 2 * cosize_v<VLayout>;
    alpha_storage = reinterpret_cast<float *>(probability_storage);
    inverse_storage = alpha_storage + 256;
    barriers = reinterpret_cast<uint64_t *>(inverse_storage + 128);
    key_ready = barriers;
    value_ready = barriers + 2;
    consumed = barriers + 4;
    score_ready = barriers + 6;
    score_free = barriers + 8;
    probability_ready = barriers + 10;
    corrected = barriers + 12;
    tmem_base = reinterpret_cast<uint32_t *>(barriers + 22);
    key_consumed = barriers + 18;
    value_consumed = barriers + 20;
    remote_key_ready = barriers + 14;
    remote_pv_ready = barriers + 16;
  }
};

struct Work {
  const __nv_bfloat16 *query, *cache;
  const int32_t *tables, *lengths, *starts;
  float *partial, *lse;
  int table_width, tiles, max_queries, query_rows, cache_pages, first;
  const CUtensorMap &key_map, &value_map;
  int64_t l2_bytes;
  const CUtensorMap &key16_map, &value16_map;
  __nv_bfloat16 *attended;
  int tid = 0, warp = 0, lane = 0, seq = 0, qbase = 0, kh = 0;
  int start = 0, end = 0, last = 0, length = 0, trips = 0;
  bool split_key_value = false;
  int rank = 0, cluster = 0, cluster_qbase = 0;
};

__device__ __forceinline__ bool prepare(Work &w) {
  w.tid = threadIdx.x;
  w.warp = w.tid / 32;
  w.lane = w.tid % 32;
  w.rank = 0;
  w.cluster = blockIdx.x;
  w.seq = w.cluster / w.tiles;
  w.cluster_qbase = (w.cluster % w.tiles) * 128;
  w.qbase = w.cluster_qbase + w.rank * 128;
  w.kh = blockIdx.y;
  w.start = w.starts[w.seq];
  w.end = w.starts[w.seq + 1];
  if (w.start < 0 || w.end > w.query_rows || w.end <= w.start || w.end - w.start > w.max_queries ||
      w.starts[0] != 0 || w.starts[gridDim.x / w.tiles] != w.query_rows) {
    asm volatile("trap;");
    return false;
  }
  if (w.start + w.cluster_qbase / 6 >= w.end)
    return false;
  w.last = w.lengths[min(w.end - 1, w.start + (w.cluster_qbase + 127) / 6)];
  if (w.last < w.first || w.last > w.table_width * 784) {
    asm volatile("trap;");
    return false;
  }
  constexpr int64_t bytes_per_token = 2 * 4 * 256 * int64_t(sizeof(Element));
  int64_t kv_bytes = (int64_t(w.lengths[w.end - 1]) - w.first) * bytes_per_token;
  w.split_key_value = kv_bytes > w.l2_bytes;
  w.length = 0;
  if (w.tid < 256) {
    int row = w.start + (w.qbase + (w.warp % 4) * 32 + (w.warp / 4) * 16 + w.lane % 16) / 6;
    w.length = row < w.end ? w.lengths[row] : 0;
    if (w.length < 0 || w.length > w.last) {
      asm volatile("trap;");
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ void initialize_query(Shared &s, const Work &w) {
  auto sQ = make_tensor(make_smem_ptr(s.query_storage), QLayout{});
  if (w.tid < 128) {
    for (int i = w.tid; i < 128 * 32; i += 128) {
      int r = i / 32, c = (i % 32) * 8;
      int row = w.start + (w.qbase + r) / 6, head = w.kh * 6 + (w.qbase + r) % 6;
      uint4 value =
          row < w.end
              ? *reinterpret_cast<const uint4 *>(w.query + (int64_t(row) * 24 + head) * 256 + c)
              : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4 *>(&sQ(make_coord(r, c % 16), 0, c / 16)) = value;
    }
    cutlass::arch::fence_view_async_shared();
  } else if (w.tid >= 256 && w.tid < 384) {
    uint32_t address = *s.tmem_base + uint32_t((w.tid - 256) / 32 * 32) * 65536u + 256;
    float zeros[32] = {};
#pragma unroll
    for (int c = 0; c < 256; c += 32)
      fa_ring_tmem_store32(address + c, zeros);
    cutlass::arch::fence_view_async_tmem_store();
    fa_ring_before_sync();
  }
  __syncthreads();
  fa_ring_after_sync();
}

} // namespace fa_ring_stages

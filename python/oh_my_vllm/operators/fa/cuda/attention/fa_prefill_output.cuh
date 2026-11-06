#pragma once

namespace fa_prefill_stages {
template <bool ExportBFloat16> __device__ __forceinline__ void export_output(Shared &s, Work &w) {
  if constexpr (ExportBFloat16) {
    auto *staged = reinterpret_cast<float *>(s.query_storage);
    if (w.tid >= 128 && w.tid < 256) {
      int owner = w.tid - 128;
      uint32_t address = *s.tmem_base + uint32_t(owner / 32 * 32) * 65536u + 128;
      float inverse = s.inverse_storage[owner];
#pragma unroll
      for (int c = 0; c < 256; c += 32) {
        float block[32];
        fa_tmem_load32(address + c, block);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int j = 0; j < 32; ++j)
          staged[owner * 257 + c + j] = block[j] * inverse;
      }
      fa_prefill_before_sync();
    }
    __syncthreads();
    for (int group = w.warp; group < 256; group += 10) {
      int owner = group / 2, half = group % 2;
      int row = w.start + (w.qbase + owner) / 6, head = w.kh * 6 + (w.qbase + owner) % 6;
      if (row >= w.end)
        continue;
      int64_t head_row = int64_t(row) * 24 + head;
      float normalizer = w.lane == 0 ? w.lse[head_row] : 0.f;
      normalizer = __shfl_sync(0xffffffff, normalizer, 0);
      float maximum = fmaxf(-INFINITY, normalizer);
      float safe = maximum == -INFINITY ? 0.f : maximum;
      float weight = fa_exp2(normalizer - safe);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float value = 0.f + staged[owner * 257 + half * 128 + w.lane * 4 + j] * weight;
        value = weight > 0 ? value / weight : 0.f;
        value = __bfloat162float(__float2bfloat16_rn(value));
        w.attended[head_row * 256 + half * 128 + w.lane * 4 + j] = __float2bfloat16_rn(value);
      }
    }
  } else if (w.tid >= 128 && w.tid < 256) {
    int owner = w.tid - 128;
    uint32_t address = *s.tmem_base + uint32_t(owner / 32 * 32) * 65536u + 128;
    float inverse = s.inverse_storage[owner];
    auto *staged = reinterpret_cast<float *>(s.key_storage);
#pragma unroll
    for (int c = 0; c < 256; c += 32) {
      float block[32];
      fa_tmem_load32(address + c, block);
      cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
      for (int j = 0; j < 32; ++j)
        staged[owner * 33 + j] = block[j] * inverse;
      asm volatile("barrier.sync 1, 128;" ::: "memory");
      for (int i = owner; i < 128 * 32; i += 128) {
        int r = i / 32, column = i % 32, target_row = w.start + (w.qbase + r) / 6;
        int head = w.kh * 6 + (w.qbase + r) % 6;
        if (target_row < w.end)
          w.partial[((int64_t(target_row) * 24 + head) * w.splits + w.split) * 256 + c + column] =
              staged[r * 33 + column];
      }
      asm volatile("barrier.sync 1, 128;" ::: "memory");
    }
    fa_prefill_before_sync();
  }
}

} // namespace fa_prefill_stages

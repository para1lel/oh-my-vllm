#pragma once

namespace fa_ring_stages {
template <bool ExportBFloat16>
__device__ __forceinline__ void export_output(Shared &s, Work &w, unsigned char *storage) {
  auto *staged = reinterpret_cast<float *>(storage);
  auto *packed_stage = reinterpret_cast<uint32_t *>(storage);
  if (w.tid >= 256 && w.tid < 384) {
    int owner = w.tid - 256;
    uint32_t address = *s.tmem_base + uint32_t(owner / 32 * 32) * 65536u + 256;
    float inverse = s.inverse_storage[owner];
    bool finite = true;
    if constexpr (ExportBFloat16) {
      int row = w.start + (w.qbase + owner) / 6, head = w.kh * 6 + (w.qbase + owner) % 6;
      finite = row < w.end && isfinite(w.lse[int64_t(row) * 24 + head]);
    }
#pragma unroll
    for (int c = 0; c < 256; c += 32) {
      float values[32];
      fa_ring_tmem_load32(address + c, values);
      cutlass::arch::fence_view_async_tmem_load();
      if constexpr (ExportBFloat16) {
#pragma unroll
        for (int j = 0; j < 32; j += 2) {
          float x = finite ? 0.f + values[j] * inverse : 0.f;
          float y = finite ? 0.f + values[j + 1] * inverse : 0.f;
          __nv_bfloat162_raw pair = __floats2bfloat162_rn(x, y);
          packed_stage[owner * 129 + (c + j) / 2] = uint32_t(pair.x) | (uint32_t(pair.y) << 16);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 32; ++j)
          staged[owner * 257 + c + j] = values[j] * inverse;
      }
    }
    fa_ring_before_sync();
  }
  __syncthreads();
  if constexpr (ExportBFloat16) {
    for (int i = w.tid; i < 128 * 128; i += 480) {
      int r = i / 128, column = i % 128, target_row = w.start + (w.qbase + r) / 6;
      int head = w.kh * 6 + (w.qbase + r) % 6;
      if (target_row < w.end) {
        int64_t offset = (int64_t(target_row) * 24 + head) * 128 + column;
        reinterpret_cast<uint32_t *>(w.attended)[offset] = packed_stage[r * 129 + column];
      }
    }
  } else {
    for (int i = w.tid; i < 128 * 256; i += 480) {
      int r = i / 256, column = i % 256, target_row = w.start + (w.qbase + r) / 6;
      int head = w.kh * 6 + (w.qbase + r) % 6;
      if (target_row < w.end) {
        int64_t offset = (int64_t(target_row) * 24 + head) * 256 + column;
        w.partial[offset] = staged[r * 257 + column];
      }
    }
  }
}

} // namespace fa_ring_stages

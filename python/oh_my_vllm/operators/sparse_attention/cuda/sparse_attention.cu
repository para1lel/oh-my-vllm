#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <climits>
#include <cstdint>

#include "../../runtime/cuda/contracts.cuh"

namespace {

constexpr int SharedBytes = (16 * 512 + 64 * 512 + 16 * 64) * 2 + 2 * 8 * 16 * 4 + 64 * 4;

__device__ __forceinline__ int address(int row, int column, int stride) {
  return row * stride + (column ^ ((row & 7) * 8));
}

__device__ __forceinline__ void load_a(unsigned (&a)[4], const __nv_bfloat16 *data, int inner,
                                       int lane, int stride) {
  unsigned pointer =
      __cvta_generic_to_shared(data + address(lane % 16, inner + (lane / 16) * 8, stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(pointer));
}

__device__ __forceinline__ void load_key(unsigned (&b)[2], const __nv_bfloat16 *data, int inner,
                                         int warp, int lane) {
  unsigned pointer = __cvta_generic_to_shared(
      data + address(warp * 8 + lane % 8, inner + ((lane / 8) % 2) * 8, 512));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(pointer));
}

__device__ __forceinline__ void load_value(unsigned (&b)[2], const __nv_bfloat16 *data, int inner,
                                           int column, int lane) {
  unsigned pointer = __cvta_generic_to_shared(data + address(inner + lane % 16, column, 512));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(pointer));
}

__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ float fp8(uint8_t value) {
  return __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(value, __NV_E4M3)));
}

__device__ void reject(__nv_bfloat16 *output, uint8_t *status) {
  for (int i = threadIdx.x; i < 16 * 512; i += blockDim.x)
    output[i] = __float2bfloat16_rn(0.f);
  if (threadIdx.x == 0)
    *status = 1;
}

__global__ void attend_kernel(const __nv_bfloat16 *query, const float *sink, const int32_t *indices,
                              const uint8_t *window, const uint8_t *window_scales,
                              const uint8_t *compressed, const uint8_t *compressed_scales,
                              __nv_bfloat16 *output, uint8_t *status, int heads, int slots,
                              int64_t window_rows, int64_t compressed_rows) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *kv = q + 16 * 512;
  auto *p = kv + 64 * 512;
  auto *maxima = reinterpret_cast<float *>(p + 16 * 64);
  auto *sums = maxima + 8 * 16;
  auto *ids = reinterpret_cast<int32_t *>(sums + 8 * 16);
  int row = blockIdx.x, head = blockIdx.y * 16;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int fr = lane / 4, fc = (lane % 4) * 2;
  output += (int64_t(row) * heads + head) * 512;
  status += row * (heads / 16) + blockIdx.y;
  bool invalid = false;
  for (int i = threadIdx.x; i < 16 * 512; i += blockDim.x) {
    auto value = query[(int64_t(row) * heads + head) * 512 + i];
    invalid |= !isfinite(__bfloat162float(value));
    q[address(i / 512, i % 512, 512)] = value;
  }
  if (threadIdx.x < 16)
    invalid |= !isfinite(sink[head + threadIdx.x]);
  if (__syncthreads_or(invalid)) {
    reject(output, status);
    return;
  }
  float maximum[2] = {sink[head + fr], sink[head + fr + 8]};
  float denominator[2] = {1.f, 1.f};
  float accumulated[8][4] = {};
  for (int base = 0; base < slots; base += 64) {
    if (threadIdx.x < 64) {
      int id = base + threadIdx.x < slots ? indices[row * slots + base + threadIdx.x] : -1;
      invalid = id < -1 || int64_t(id) >= window_rows + compressed_rows;
      ids[threadIdx.x] = invalid ? -1 : id;
    }
    if (__syncthreads_or(invalid)) {
      reject(output, status);
      return;
    }
    for (int i = threadIdx.x; i < 64 * 512; i += blockDim.x) {
      int id = ids[i / 512], channel = i % 512;
      float restored = 0.f;
      if (id >= 0 && id < window_rows) {
        uint8_t code = window[int64_t(id) * 512 + channel];
        uint8_t scale = window_scales[int64_t(id) * 16 + channel / 32];
        invalid |= (code & 127) == 127 || scale == 255;
        restored = ldexpf(fp8(code), int(scale) - 127);
      } else if (id >= 0) {
        int64_t index = id - window_rows;
        uint8_t packed = compressed[index * 256 + channel / 2];
        uint8_t code = (packed >> ((channel % 2) * 4)) & 15;
        uint8_t scale = compressed_scales[index * 32 + channel / 16];
        invalid |= (scale & 128) != 0 || scale == 0 || scale == 127;
        float value = __half2float(static_cast<__half>(__nv_cvt_fp4_to_halfraw(code, __NV_E2M1)));
        restored = __fmul_rn(value, fp8(scale));
      }
      auto value = __float2bfloat16_rn(restored);
      invalid |= !isfinite(__bfloat162float(value));
      kv[address(i / 512, channel, 512)] = value;
    }
    if (__syncthreads_or(invalid)) {
      reject(output, status);
      return;
    }
    float scores[4] = {};
    for (int inner = 0; inner < 512; inner += 16) {
      unsigned a[4], b[2];
      load_a(a, q, inner, lane, 512);
      load_key(b, kv, inner, warp, lane);
      mma(scores, a, b);
    }
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      float local = -INFINITY;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int slot = warp * 8 + fc + j;
        float value = scores[rg * 2 + j] * 0.044194173824159216f;
        invalid |= ids[slot] != -1 && !isfinite(value);
        value = ids[slot] != -1 ? value : -INFINITY;
        scores[rg * 2 + j] = value;
        local = fmaxf(local, value);
      }
      local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, 1));
      local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, 2));
      if (lane % 4 == 0)
        maxima[warp * 16 + fr + rg * 8] = local;
    }
    if (__syncthreads_or(invalid)) {
      reject(output, status);
      return;
    }
    float alpha[2];
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      int r = fr + rg * 8;
      float next = maximum[rg];
#pragma unroll
      for (int w = 0; w < 8; ++w)
        next = fmaxf(next, maxima[w * 16 + r]);
      alpha[rg] = expf(maximum[rg] - next);
      maximum[rg] = next;
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float probability = expf(scores[rg * 2 + j] - next);
        sum += probability;
        p[address(r, warp * 8 + fc + j, 64)] = __float2bfloat16_rn(probability);
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      sum += __shfl_xor_sync(0xffffffff, sum, 2);
      if (lane % 4 == 0)
        sums[warp * 16 + r] = sum;
    }
    __syncthreads();
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      denominator[rg] *= alpha[rg];
#pragma unroll
      for (int w = 0; w < 8; ++w)
        denominator[rg] += sums[w * 16 + fr + rg * 8];
    }
#pragma unroll
    for (int tile = 0; tile < 8; ++tile)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        accumulated[tile][j] *= alpha[j / 2];
    for (int inner = 0; inner < 64; inner += 16) {
      unsigned a[4];
      load_a(a, p, inner, lane, 64);
#pragma unroll
      for (int tile = 0; tile < 8; ++tile) {
        unsigned b[2];
        load_value(b, kv, inner, (tile * 8 + warp) * 8, lane);
        mma(accumulated[tile], a, b);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int tile = 0; tile < 8; ++tile)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      auto value = __float2bfloat16_rn(accumulated[tile][j] / denominator[j / 2]);
      invalid |= !isfinite(__bfloat162float(value));
      output[(fr + (j / 2) * 8) * 512 + (tile * 8 + warp) * 8 + fc + j % 2] = value;
    }
  if (__syncthreads_or(invalid))
    reject(output, status);
  else if (threadIdx.x == 0)
    *status = 0;
}

} // namespace

void attend(TensorView query, TensorView sink, TensorView indices, TensorView window,
            TensorView window_scales, TensorView compressed, TensorView compressed_scales,
            TensorView output, TensorView status) {
  TVM_FFI_ICHECK(query.ndim() == 3 && indices.ndim() == 2 && window.ndim() == 2 &&
                 compressed.ndim() == 2);
  int64_t rows = query.size(0), heads = query.size(1), slots = indices.size(1);
  int64_t window_rows = window.size(0), compressed_rows = compressed.size(0);
  TVM_FFI_ICHECK(rows >= 0 && rows <= 384 && (heads == 16 || heads == 64) && slots >= 0 &&
                 slots <= 640 && window_rows >= 0 && compressed_rows >= 0 &&
                 window_rows <= INT_MAX && compressed_rows <= INT_MAX - window_rows);
  tensor_contract(query, query, kDLBfloat, 16, {rows, heads, 512});
  tensor_contract(sink, query, kDLFloat, 32, {heads});
  tensor_contract(indices, query, kDLInt, 32, {rows, slots});
  tensor_contract(window, query, kDLUInt, 8, {window_rows, 512});
  tensor_contract(window_scales, query, kDLUInt, 8, {window_rows, 16});
  tensor_contract(compressed, query, kDLUInt, 8, {compressed_rows, 256});
  tensor_contract(compressed_scales, query, kDLUInt, 8, {compressed_rows, 32});
  tensor_contract(output, query, kDLBfloat, 16, {rows, heads, 512});
  tensor_contract(status, query, kDLUInt, 8, {rows, heads / 16});
  for (auto input : {query, sink, indices, window, window_scales, compressed, compressed_scales}) {
    disjoint(input, output);
    disjoint(input, status);
  }
  disjoint(output, status);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(query.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, query.device().device_id));
  check_cuda(cudaFuncSetAttribute(attend_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  SharedBytes));
  attend_kernel<<<dim3(rows, heads / 16), 256, SharedBytes, stream>>>(
      static_cast<const __nv_bfloat16 *>(query.data_ptr()),
      static_cast<const float *>(sink.data_ptr()), static_cast<const int32_t *>(indices.data_ptr()),
      static_cast<const uint8_t *>(window.data_ptr()),
      static_cast<const uint8_t *>(window_scales.data_ptr()),
      static_cast<const uint8_t *>(compressed.data_ptr()),
      static_cast<const uint8_t *>(compressed_scales.data_ptr()),
      static_cast<__nv_bfloat16 *>(output.data_ptr()), static_cast<uint8_t *>(status.data_ptr()),
      heads, slots, window_rows, compressed_rows);
  check_cuda(cudaGetLastError());
}

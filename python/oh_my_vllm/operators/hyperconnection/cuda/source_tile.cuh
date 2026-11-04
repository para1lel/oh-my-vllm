#pragma once

__global__ __launch_bounds__(128,
                             4) void source_partial_kernel(const float *input, const float *weight,
                                                           float *partials, int rows, int chunk) {
  __shared__ __align__(16) float storage[2][96 * 32];
  int r = threadIdx.x / 8, o = threadIdx.x % 8;
  int row_base = blockIdx.y * 64;
  int first = blockIdx.x * chunk, end = min(first + chunk, FlatWidth);
  float total[4][3] = {};
  released_stage<32>(storage[0], input, weight, row_base, 0, rows, Mixes, FlatWidth, FlatWidth,
                     first);
  int buffer = 0;
  for (int base = first; base < end; base += 32) {
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    if (base + 32 < end)
      released_stage<32>(storage[buffer ^ 1], input, weight, row_base, 0, rows, Mixes, FlatWidth,
                         FlatWidth, base + 32);
#pragma unroll 8
    for (int k = 0; k < 32; ++k) {
      float a[4], b[3];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        a[i] = storage[buffer][(r + i * 16) * 32 + (k ^ ((r & 7) * 4))];
#pragma unroll
      for (int j = 0; j < 3; ++j)
        b[j] = storage[buffer][(64 + o + j * 8) * 32 + (k ^ (o * 4))];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
          total[i][j] = __fmaf_rn(a[i], b[j], total[i][j]);
    }
    buffer ^= 1;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = row_base + r + i * 16;
    if (row < rows)
#pragma unroll
      for (int j = 0; j < 3; ++j)
        partials[(blockIdx.x * rows + row) * Mixes + o + j * 8] = total[i][j];
  }
}

inline void multiply_source_chunks(const float *input, const float *weight, float *output, int rows,
                                   int chunk, void *workspace, cudaStream_t stream) {
  TVM_FFI_ICHECK(chunk > 0 && chunk % 32 == 0);
  int chunks = chunk_count(FlatWidth, chunk);
  int capacity = int(BlasBytes / (chunks * Mixes * sizeof(float))) / 64 * 64;
  TVM_FFI_ICHECK(capacity >= 64);
  auto *partials = static_cast<float *>(workspace);
  check_cuda(cudaFuncSetAttribute(source_partial_kernel,
                                  cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  for (int first = 0; first < rows; first += capacity) {
    int count = min(capacity, rows - first);
    source_partial_kernel<<<dim3(chunks, (count + 63) / 64), 128, 0, stream>>>(
        input + int64_t(first) * FlatWidth, weight, partials, count, chunk);
    check_cuda(cudaGetLastError());
    int values = count * Mixes;
    chunk_reduce_kernel<<<(values + 255) / 256, 256, 0, stream>>>(partials, output + first * Mixes,
                                                                  values, chunks, Mixes);
    check_cuda(cudaGetLastError());
  }
}

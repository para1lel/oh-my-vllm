#pragma once

template <int Columns>
__device__ __forceinline__ void
released_stage(float *shared, const float *input, const float *weight, int row_base, int out_base,
               int rows, int outputs, int width, int input_stride, int base) {
#pragma unroll
  for (int index = threadIdx.x; index < 64 * 8; index += 128) {
    int row = index / 8, k = (index % 8) * 4;
    int address = row * 32 + (k ^ ((row & 7) * 4));
    unsigned destination = __cvta_generic_to_shared(shared + address);
    const float *source =
        input + (row_base + row < rows ? row_base + row : 0) * input_stride + base + k;
    int bytes = row_base + row < rows ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(destination), "l"(source), "r"(bytes));
  }
#pragma unroll
  for (int index = threadIdx.x; index < Columns * 8; index += 128) {
    int row = index / 8, k = (index % 8) * 4;
    int address = (64 + row) * 32 + (k ^ ((row & 7) * 4));
    unsigned destination = __cvta_generic_to_shared(shared + address);
    const float *source =
        weight + (out_base + row < outputs ? out_base + row : 0) * width + base + k;
    int bytes = out_base + row < outputs ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(destination), "l"(source), "r"(bytes));
  }
  asm volatile("cp.async.commit_group;");
}

// A warp owns four rows and eight output lanes. The XOR layout permits aligned
// asynchronous copies and broadcasts each scalar shared read without bank conflicts.
template <int Columns>
__global__ __launch_bounds__(128, 4) void released_whole_kernel(const float *input,
                                                                const float *weight, float *output,
                                                                int rows, int outputs, int width,
                                                                int input_stride,
                                                                int output_stride) {
  __shared__ __align__(16) float storage[2][(64 + Columns) * 32];
  int r = threadIdx.x / 8, o = threadIdx.x % 8;
  int row_base = blockIdx.y * 64, out_base = blockIdx.x * Columns;
  float total[4][Columns / 8] = {};
  released_stage<Columns>(storage[0], input, weight, row_base, out_base, rows, outputs, width,
                          input_stride, 0);
  int buffer = 0;
  for (int base = 0; base < width; base += 32) {
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    if (base + 32 < width)
      released_stage<Columns>(storage[buffer ^ 1], input, weight, row_base, out_base, rows, outputs,
                              width, input_stride, base + 32);
#pragma unroll 8
    for (int k = 0; k < 32; ++k) {
      float a[4], b[Columns / 8];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        a[i] = storage[buffer][(r + i * 16) * 32 + (k ^ ((r & 7) * 4))];
#pragma unroll
      for (int j = 0; j < Columns / 8; ++j)
        b[j] = storage[buffer][(64 + o + j * 8) * 32 + (k ^ (o * 4))];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < Columns / 8; ++j)
          total[i][j] = __fmaf_rn(a[i], b[j], total[i][j]);
    }
    buffer ^= 1;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = row_base + r + i * 16;
#pragma unroll
    for (int j = 0; j < Columns / 8; ++j) {
      int out = out_base + o + j * 8;
      if (row < rows && out < outputs)
        output[row * output_stride + out] = __fadd_rn(0.f, total[i][j]);
    }
  }
}

#include "common.cuh"
#include "groupwise.cuh"

__global__ void merge_partials_kernel(const float *partials, __nv_bfloat16 *output, int count,
                                      int splits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    float value = 0;
    for (int split = 0; split < splits; ++split)
      value += partials[split * count + index];
    output[index] = __float2bfloat16_rn(value);
  }
}

template <int Threads>
__global__ void residual_rms_quant_kernel(const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
                                          const float *weight, __nv_bfloat16 *summed,
                                          __nv_fp8_e4m3 *data, float *scales) {
  constexpr int Chunks = 5120 / (Threads * 8);
  int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  float values[Chunks][8], total = 0;
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int column = chunk * Threads * 8 + threadIdx.x * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      auto value = __float2bfloat16_rn(__bfloat162float(x[row * 5120 + column + j]) +
                                       __bfloat162float(residual[row * 5120 + column + j]));
      summed[row * 5120 + column + j] = value;
      values[chunk][j] = __bfloat162float(value);
      total += values[chunk][j] * values[chunk][j];
    }
  }
  __shared__ float partial[Threads / 32];
  total = warp_sum(total);
  if (lane == 0)
    partial[warp] = total;
  __syncthreads();
  total = warp_sum(lane < Threads / 32 ? partial[lane] : 0);
  float inverse = rsqrtf(total / 5120.f + 1e-6f);
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int column = chunk * Threads * 8 + threadIdx.x * 8;
    float maximum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      values[chunk][j] = __bfloat162float(
          __float2bfloat16_rn(values[chunk][j] * inverse * (weight[column + j] + 1.f)));
      maximum = fmaxf(maximum, fabsf(values[chunk][j]));
    }
    unsigned mask = lane < 16 ? 0xffffu : 0xffff0000u;
    maximum = __uint_as_float(__reduce_max_sync(mask, __float_as_uint(maximum)));
    float divisor = fmaxf(maximum, 1e-10f) / 448.f;
    if (lane % 16 == 0)
      scales[row * 40 + column / 128] = divisor;
#pragma unroll
    for (int j = 0; j < 8; j += 4)
      *reinterpret_cast<uint32_t *>(data + row * 5120 + column + j) =
          __nv_fp8x4_e4m3(make_float4(values[chunk][j] / divisor, values[chunk][j + 1] / divisor,
                                      values[chunk][j + 2] / divisor,
                                      values[chunk][j + 3] / divisor))
              .__x;
  }
}

__global__ void silu_product_quant_kernel(const __nv_bfloat16 *gate_up, __nv_fp8_e4m3 *data,
                                          float *scales, int groups) {
  int group = blockIdx.x * 8 + threadIdx.x / 32;
  if (group >= groups)
    return;
  int row = group / 136, column = (group % 136) * 128 + (threadIdx.x & 31) * 4;
  float values[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float gate = __bfloat162float(gate_up[row * 34816 + column + j]);
    float up = __bfloat162float(gate_up[row * 34816 + 17408 + column + j]);
    float exponent = expf(-fabsf(gate));
    float sigmoid = (gate >= 0 ? 1.f : exponent) / (1.f + exponent);
    values[j] = __bfloat162float(__float2bfloat16_rn(gate * sigmoid * up));
  }
  quantize_four(make_float4(values[0], values[1], values[2], values[3]),
                data + row * 17408 + column, scales + group);
}

void ffn(TensorView x, TensorView residual, TensorView norm, TensorView gate_up_weight,
         TensorView gate_up_scale, TensorView down_weight, TensorView down_scale, TensorView output,
         TensorView summed, TensorView quantized_input, TensorView input_scale, TensorView gate_up,
         TensorView quantized_product, TensorView product_scale, TensorView partials) {
  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == 5120 && x.size(0) > 0 && x.size(0) <= 32768);
  int rows = x.size(0);
  for (auto tensor : {x, residual, summed, output})
    tensor_contract(tensor, x, kDLBfloat, 16, {rows, 5120});
  tensor_contract(norm, x, kDLFloat, 32, {5120});
  tensor_contract(gate_up_weight, x, kDLFloat8_e4m3fn, 8, {34816, 5120});
  tensor_contract(gate_up_scale, x, kDLFloat, 32, {272, 40});
  tensor_contract(down_weight, x, kDLFloat8_e4m3fn, 8, {5120, 17408});
  tensor_contract(down_scale, x, kDLFloat, 32, {40, 136});
  tensor_contract(quantized_input, x, kDLFloat8_e4m3fn, 8, {rows, 5120});
  tensor_contract(input_scale, x, kDLFloat, 32, {rows, 40});
  tensor_contract(gate_up, x, kDLBfloat, 16, {rows, 34816});
  tensor_contract(quantized_product, x, kDLFloat8_e4m3fn, 8, {rows, 17408});
  tensor_contract(product_scale, x, kDLFloat, 32, {rows, 136});
  int splits = rows < 8 ? 8 : 2;
  tensor_contract(partials, x, kDLFloat, 32,
                  {rows <= 32 ? splits : 1, rows <= 32 ? rows : 1, 5120});
  auto tensors = {x,
                  residual,
                  norm,
                  gate_up_weight,
                  gate_up_scale,
                  down_weight,
                  down_scale,
                  output,
                  summed,
                  quantized_input,
                  input_scale,
                  gate_up,
                  quantized_product,
                  product_scale,
                  partials};
  int left = 0;
  for (auto a : tensors) {
    int right = 0;
    for (auto b : tensors)
      if (right++ > left)
        disjoint(a, b);
    ++left;
  }
  tvm::ffi::CUDADeviceGuard guard(x.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  auto norm_launch = [&](auto threads) {
    residual_rms_quant_kernel<decltype(threads)::value>
        <<<rows, decltype(threads)::value, 0, stream>>>(

            static_cast<const __nv_bfloat16 *>(x.data_ptr()),
            static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
            static_cast<const float *>(norm.data_ptr()),
            static_cast<__nv_bfloat16 *>(summed.data_ptr()),
            static_cast<__nv_fp8_e4m3 *>(quantized_input.data_ptr()),
            static_cast<float *>(input_scale.data_ptr()));
  };
  if (rows <= 32)
    norm_launch(std::integral_constant<int, 640>{});
  else
    norm_launch(std::integral_constant<int, 320>{});
  check_cuda(cudaGetLastError());
  if (rows <= 32)
    groupwise<128, 32, 256, 128, 1, true>(
        static_cast<const __nv_fp8_e4m3 *>(gate_up_weight.data_ptr()),
        static_cast<const __nv_fp8_e4m3 *>(quantized_input.data_ptr()),
        static_cast<const float *>(gate_up_scale.data_ptr()),
        static_cast<const float *>(input_scale.data_ptr()), gate_up.data_ptr(), 34816, rows, 5120,
        stream);
  else
    groupwise(static_cast<const __nv_fp8_e4m3 *>(quantized_input.data_ptr()),
              static_cast<const __nv_fp8_e4m3 *>(gate_up_weight.data_ptr()),
              static_cast<const float *>(input_scale.data_ptr()),
              static_cast<const float *>(gate_up_scale.data_ptr()),
              static_cast<__nv_bfloat16 *>(gate_up.data_ptr()), rows, 34816, 5120, stream);
  silu_product_quant_kernel<<<(rows * 136 + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(gate_up.data_ptr()),
      static_cast<__nv_fp8_e4m3 *>(quantized_product.data_ptr()),
      static_cast<float *>(product_scale.data_ptr()), rows * 136);
  check_cuda(cudaGetLastError());
  if (rows <= 32) {
    auto run = [&](auto count) {
      groupwise<64, 128, 128, 1, 128, false, decltype(count)::value, float>(
          static_cast<const __nv_fp8_e4m3 *>(quantized_product.data_ptr()),
          static_cast<const __nv_fp8_e4m3 *>(down_weight.data_ptr()),
          static_cast<const float *>(product_scale.data_ptr()),
          static_cast<const float *>(down_scale.data_ptr()), partials.data_ptr(), rows, 5120,
          17408 / splits, stream);
    };
    if (splits == 8)
      run(std::integral_constant<int, 8>{});
    else
      run(std::integral_constant<int, 2>{});
    merge_partials_kernel<<<(rows * 5120 + 255) / 256, 256, 0, stream>>>(
        static_cast<const float *>(partials.data_ptr()),
        static_cast<__nv_bfloat16 *>(output.data_ptr()), rows * 5120, splits);
    check_cuda(cudaGetLastError());
  } else
    groupwise(static_cast<const __nv_fp8_e4m3 *>(quantized_product.data_ptr()),
              static_cast<const __nv_fp8_e4m3 *>(down_weight.data_ptr()),
              static_cast<const float *>(product_scale.data_ptr()),
              static_cast<const float *>(down_scale.data_ptr()),
              static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, 5120, 17408, stream);
}

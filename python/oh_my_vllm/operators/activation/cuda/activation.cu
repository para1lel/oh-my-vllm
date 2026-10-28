#include <cuda_bf16.h>
#include "../../runtime/cuda/contracts.cuh"

namespace {

__device__ float activate(float gate, float up, float route, bool clipped) {
  if (clipped) {
    gate = fminf(gate, 10.f);
    up = fminf(fmaxf(up, -10.f), 10.f);
  }
  return ((gate / (1.f + expf(-gate))) * up) * route;
}

__global__ void swiglu_kernel(const __nv_bfloat162 *gate, const __nv_bfloat162 *up,
                              const float *routes, __nv_bfloat162 *output, int pairs, int count,
                              bool clipped, bool routed) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  float2 g = __bfloat1622float2(gate[index]);
  float2 u = __bfloat1622float2(up[index]);
  float route = routed ? routes[index / pairs] : 1.f;
  output[index] =
      __floats2bfloat162_rn(activate(g.x, u.x, route, clipped), activate(g.y, u.y, route, clipped));
}

} // namespace

void swiglu(TensorView gate, TensorView up, TensorView route, TensorView output, int64_t limit,
            bool routed) {
  TVM_FFI_ICHECK(gate.ndim() == 2 && gate.size(0) >= 0 && gate.size(0) <= 32768);
  int rows = gate.size(0), width = gate.size(1);
  TVM_FFI_ICHECK(((width == 576 || width == 2304) && limit == 10) ||
                 (width == 2816 && limit == 0 && !routed));
  tensor_contract(gate, gate, kDLBfloat, 16, {rows, width});
  tensor_contract(up, gate, kDLBfloat, 16, {rows, width});
  tensor_contract(route, gate, kDLFloat, 32, {routed ? rows : 0});
  tensor_contract(output, gate, kDLBfloat, 16, {rows, width});
  disjoint(gate, output);
  disjoint(up, output);
  disjoint(route, output);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(gate.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gate.device().device_id));
  int pairs = width / 2, count = rows * pairs;
  swiglu_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat162 *>(gate.data_ptr()),
      static_cast<const __nv_bfloat162 *>(up.data_ptr()),
      static_cast<const float *>(route.data_ptr()),
      static_cast<__nv_bfloat162 *>(output.data_ptr()), pairs, count, limit != 0, routed);
  check_cuda(cudaGetLastError());
}

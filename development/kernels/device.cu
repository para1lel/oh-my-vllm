#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

using tvm::ffi::Shape;
using tvm::ffi::String;
using tvm::ffi::Tensor;
using tvm::ffi::TensorView;

void cuda_check(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

struct DeviceAllocator {
  void AllocData(DLTensor *tensor) {
    tvm::ffi::CUDADeviceGuard guard(tensor->device.device_id);
    cuda_check(cudaMalloc(&tensor->data, tvm::ffi::GetDataSize(*tensor)));
  }

  void FreeData(DLTensor *tensor) {
    tvm::ffi::CUDADeviceGuard guard(tensor->device.device_id);
    cudaFree(tensor->data);
  }
};

Tensor allocate(Shape shape, String dtype, int64_t device) {
  TVM_FFI_ICHECK(device >= 0 && device <= INT32_MAX);
  for (auto extent : shape)
    TVM_FFI_ICHECK(extent > 0);
  return Tensor::FromNDAlloc(DeviceAllocator{}, shape, tvm::ffi::StringToDLDataType(dtype),
                             DLDevice{kDLCUDA, static_cast<int>(device)});
}

void copy_bytes(TensorView destination, TensorView source) {
  TVM_FFI_ICHECK(destination.IsContiguous() && source.IsContiguous());
  TVM_FFI_ICHECK(tvm::ffi::GetDataSize(destination) == tvm::ffi::GetDataSize(source));
  auto device =
      destination.device().device_type == kDLCUDA ? destination.device() : source.device();
  TVM_FFI_ICHECK(device.device_type == kDLCUDA);
  for (auto tensor : {source, destination})
    TVM_FFI_ICHECK(
        tensor.device().device_type == kDLCPU ||
        (tensor.device().device_type == kDLCUDA && tensor.device().device_id == device.device_id));
  tvm::ffi::CUDADeviceGuard guard(device.device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device.device_id));
  cuda_check(cudaMemcpyAsync(destination.data_ptr(), source.data_ptr(),
                             tvm::ffi::GetDataSize(source), cudaMemcpyDefault, stream));
  cuda_check(cudaStreamSynchronize(stream));
}

void synchronize(int64_t device) {
  tvm::ffi::CUDADeviceGuard guard(static_cast<int>(device));
  cuda_check(cudaDeviceSynchronize());
}

int64_t l2_bytes() {
  tvm::ffi::CUDADeviceGuard guard(0);
  int bytes = 0;
  cuda_check(cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, 0));
  return bytes;
}

__global__ void evict_kernel(const uint32_t *input, uint32_t *output, int64_t elements) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    uint32_t value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(input + index));
    output[index] = value + 1;
  }
}

void evict(TensorView input, TensorView output) {
  TVM_FFI_ICHECK(input.IsContiguous() && output.IsContiguous());
  TVM_FFI_ICHECK(input.dtype().code == kDLUInt && input.dtype().bits == 32 &&
                 input.dtype().lanes == 1);
  TVM_FFI_ICHECK(output.dtype() == input.dtype() && output.numel() == input.numel());
  TVM_FFI_ICHECK(input.device().device_type == kDLCUDA && output.device().device_type == kDLCUDA &&
                 input.device().device_id == output.device().device_id);
  TVM_FFI_ICHECK(input.data_ptr() != output.data_ptr());
  tvm::ffi::CUDADeviceGuard guard(input.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, input.device().device_id));
  evict_kernel<<<(input.numel() + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint32_t *>(input.data_ptr()), static_cast<uint32_t *>(output.data_ptr()),
      input.numel());
  cuda_check(cudaGetLastError());
}

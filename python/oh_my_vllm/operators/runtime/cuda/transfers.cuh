#pragma once

#include <algorithm>
#include <limits>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <vector>

__global__ void upload_metadata_kernel(uint8_t *destination, const uint8_t *source, size_t bytes) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  if ((reinterpret_cast<uintptr_t>(destination) | reinterpret_cast<uintptr_t>(source)) % 16 == 0) {
    for (size_t i = index; i < bytes / 16; i += stride)
      reinterpret_cast<uint4 *>(destination)[i] = reinterpret_cast<const uint4 *>(source)[i];
    for (size_t i = bytes / 16 * 16 + index; i < bytes; i += stride)
      destination[i] = source[i];
  } else {
    for (size_t i = index; i < bytes; i += stride)
      destination[i] = source[i];
  }
}

inline void upload_bytes(tvm::ffi::TensorView destination, tvm::ffi::TensorView source) {
  TVM_FFI_ICHECK(source.IsContiguous() && destination.IsContiguous());
  TVM_FFI_ICHECK(source.device().device_type == kDLCPU &&
                 destination.device().device_type == kDLCUDA);
  auto bytes = tvm::ffi::GetDataSize(source);
  TVM_FFI_ICHECK(bytes == tvm::ffi::GetDataSize(destination));
  if (bytes == 0)
    return;
  tvm::ffi::CUDADeviceGuard guard(destination.device().device_id);
  cudaPointerAttributes attributes;
  check_cuda(cudaPointerGetAttributes(&attributes, source.data_ptr()));
  TVM_FFI_ICHECK(attributes.type == cudaMemoryTypeHost && attributes.devicePointer);
  auto stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, destination.device().device_id));
  auto blocks = std::min<size_t>(256, (bytes + 4095) / 4096);
  // Metadata must not queue behind a large H2D transfer on the copy engine.
  upload_metadata_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<uint8_t *>(destination.data_ptr()),
      static_cast<const uint8_t *>(attributes.devicePointer), bytes);
  check_cuda(cudaGetLastError());
}

inline void copy_batch(tvm::ffi::Array<tvm::ffi::Tensor> destinations,
                       tvm::ffi::Array<tvm::ffi::Tensor> sources, int64_t stream_handle) {
  TVM_FFI_ICHECK(!sources.empty() && sources.size() == destinations.size());
  auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
  TVM_FFI_ICHECK(stream && stream != cudaStreamLegacy);
  auto first = sources[0].device();
  bool to_host = first.device_type == kDLCUDA;
  auto device = to_host ? first : destinations[0].device();
  TVM_FFI_ICHECK(device.device_type == kDLCUDA);
  tvm::ffi::CUDADeviceGuard guard(device.device_id);
  cudaStreamCaptureStatus capture;
  check_cuda(cudaStreamIsCapturing(stream, &capture));
  TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone);
  std::vector<void *> dst;
  std::vector<const void *> src;
  std::vector<size_t> sizes;
  std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
  for (size_t i = 0; i < sources.size(); ++i) {
    auto source = sources[i];
    auto destination = destinations[i];
    TVM_FFI_ICHECK(source.IsContiguous() && destination.IsContiguous());
    auto bytes = tvm::ffi::GetDataSize(source);
    TVM_FFI_ICHECK(bytes && bytes == tvm::ffi::GetDataSize(destination));
    auto host = to_host ? destination : source;
    auto gpu = to_host ? source : destination;
    TVM_FFI_ICHECK(host.device().device_type == kDLCPU && gpu.device().device_type == kDLCUDA &&
                   gpu.device().device_id == device.device_id);
    cudaPointerAttributes attributes;
    check_cuda(cudaPointerGetAttributes(&attributes, host.data_ptr()));
    TVM_FFI_ICHECK(attributes.type == cudaMemoryTypeHost)
        << "asynchronous host copies require pinned memory";
    auto address = reinterpret_cast<uintptr_t>(destination.data_ptr());
    TVM_FFI_ICHECK(address <= std::numeric_limits<uintptr_t>::max() - bytes);
    ranges.emplace_back(address, address + bytes);
    auto source_address = reinterpret_cast<uintptr_t>(source.data_ptr());
    TVM_FFI_ICHECK(source_address <= std::numeric_limits<uintptr_t>::max() - bytes);
    if (!sizes.empty() && reinterpret_cast<uintptr_t>(dst.back()) + sizes.back() == address &&
        reinterpret_cast<uintptr_t>(src.back()) + sizes.back() == source_address) {
      sizes.back() += bytes;
    } else {
      dst.push_back(destination.data_ptr());
      src.push_back(source.data_ptr());
      sizes.push_back(bytes);
    }
  }
  std::sort(ranges.begin(), ranges.end());
  for (size_t i = 1; i < ranges.size(); ++i)
    TVM_FFI_ICHECK(ranges[i - 1].second <= ranges[i].first) << "batch copy destinations overlap";
  cudaMemcpyAttributes attributes{};
  attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attributes.flags = cudaMemcpyFlagPreferOverlapWithCompute;
  size_t first_attribute = 0;
  check_cuda(cudaMemcpyBatchAsync(dst.data(), src.data(), sizes.data(), sizes.size(), &attributes,
                                  &first_attribute, 1, stream));
}

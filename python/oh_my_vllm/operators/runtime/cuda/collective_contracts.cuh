#pragma once

#include <climits>
#include <cstdint>
#include <limits>
#include <vector>
#include <nccl.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/string.h>

using tvm::ffi::Array;
using tvm::ffi::Bytes;
using tvm::ffi::Shape;
using tvm::ffi::String;
using tvm::ffi::Tensor;

inline void collective_bounds(int64_t world, int64_t rank) {
  TVM_FFI_ICHECK(world > 0 && world <= INT_MAX && rank >= 0 && rank < world)
      << "invalid collective world or rank";
}

inline ncclDataType_t collective_dtype(DLDataType dtype) {
  TVM_FFI_ICHECK(dtype.lanes == 1);
  if (dtype.code == kDLFloat && dtype.bits == 32)
    return ncclFloat32;
  if (dtype.code == kDLBfloat && dtype.bits == 16)
    return ncclBfloat16;
  if (dtype.code == kDLFloat && dtype.bits == 16)
    return ncclFloat16;
  if (dtype.code == kDLInt && dtype.bits == 32)
    return ncclInt32;
  if (dtype.code == kDLInt && dtype.bits == 64)
    return ncclInt64;
  if (dtype.code == kDLUInt && dtype.bits == 8)
    return ncclUint8;
  TVM_FFI_THROW(ValueError) << "unsupported collective dtype";
  return ncclNumTypes;
}

inline int64_t collective_tensor(const Tensor &tensor) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA && tensor.device().device_id == 0)
      << "collective tensors must use rank-local cuda:0";
  collective_dtype(tensor.dtype());
  TVM_FFI_ICHECK(tensor.ndim() == 1 && tensor.IsContiguous() && tensor.size(0) >= 0 &&
                 tensor.byte_offset() == 0)
      << "collective buffers must be contiguous flat tensors";
  uint64_t count = tensor.size(0);
  uint64_t width = tensor.dtype().bits / 8;
  TVM_FFI_ICHECK(count <= uint64_t(INT64_MAX) / width);
  auto pointer = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK((pointer || count == 0) && pointer % width == 0 &&
                 count * width <= std::numeric_limits<uintptr_t>::max() - pointer)
      << "collective buffer bounds or alignment are invalid";
  return count;
}

inline void collective_disjoint(const Tensor &left, const Tensor &right) {
  uint64_t a_bytes = uint64_t(left.size(0)) * (left.dtype().bits / 8);
  uint64_t b_bytes = uint64_t(right.size(0)) * (right.dtype().bits / 8);
  auto a = reinterpret_cast<uintptr_t>(left.data_ptr());
  auto b = reinterpret_cast<uintptr_t>(right.data_ptr());
  TVM_FFI_ICHECK(a_bytes == 0 || b_bytes == 0 || a + a_bytes <= b || b + b_bytes <= a)
      << "collective buffers overlap";
}

inline int64_t collective_staging_bytes(const Tensor &tensor) {
  // NCCL LL/LL128 loaders read enclosing 4/16-byte spans.
  uint64_t bytes = uint64_t(tensor.size(0)) * (tensor.dtype().bits / 8);
  auto pointer = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  if (bytes == 0 || (pointer % 16 == 0 && bytes % 16 == 0))
    return 0;
  TVM_FFI_ICHECK(bytes <= uint64_t(INT64_MAX) - 15) << "staging round-up exceeds capacity";
  return (bytes + 15) / 16 * 16;
}

Shape collective_check(int64_t world, int64_t rank, String kind, Array<Tensor> tensors,
                       Shape directions, Shape peers) {
  collective_bounds(world, rank);
  bool routing = kind == "routing";
  TVM_FFI_ICHECK(routing || kind == "all_reduce" || kind == "all_gather" ||
                 kind == "reduce_scatter")
      << "unknown collective operation";
  if (routing) {
    TVM_FFI_ICHECK(tensors.size() == directions.size() && tensors.size() == peers.size() &&
                   tensors.size() <= uint64_t(world) * 2)
        << "invalid routing table";
    for (size_t i = 0; i < tensors.size(); ++i) {
      TVM_FFI_ICHECK((directions[i] == 0 || directions[i] == 1) && peers[i] >= 0 &&
                     peers[i] < world)
          << "invalid route direction or peer";
      for (size_t j = 0; j < i; ++j)
        TVM_FFI_ICHECK(directions[i] != directions[j] || peers[i] != peers[j]) << "duplicate route";
    }
  } else {
    TVM_FFI_ICHECK(tensors.size() == 2 && directions.empty() && peers.empty());
  }
  std::vector<int64_t> staging;
  for (size_t i = 0; i < tensors.size(); ++i) {
    collective_tensor(tensors[i]);
    staging.push_back(collective_staging_bytes(tensors[i]));
    TVM_FFI_ICHECK(tensors[i].dtype() == tensors[0].dtype()) << "collective dtypes differ";
    for (size_t j = 0; j < i; ++j)
      collective_disjoint(tensors[i], tensors[j]);
  }
  if (!routing) {
    int64_t input = tensors[0].size(0), output = tensors[1].size(0);
    if (kind == "all_reduce") {
      TVM_FFI_ICHECK(input == output) << "all-reduce counts differ";
    } else if (kind == "all_gather") {
      TVM_FFI_ICHECK(input <= INT64_MAX / world && output == input * world)
          << "all-gather counts differ";
    } else {
      TVM_FFI_ICHECK(output <= INT64_MAX / world && input == output * world)
          << "reduce-scatter counts differ";
    }
  } else {
    int64_t sent = -1, received = -1;
    for (size_t i = 0; i < tensors.size(); ++i)
      if (peers[i] == rank)
        (directions[i] == 0 ? sent : received) = tensors[i].size(0);
    TVM_FFI_ICHECK(sent == received) << "self-route send and receive counts differ";
  }
  return Shape(staging);
}

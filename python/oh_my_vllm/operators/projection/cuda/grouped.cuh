#pragma once

#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <tvm/ffi/container/array.h>

namespace grouped_projection {

using namespace cute;
using Problem = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using MmaTile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, MmaTile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, cutlass::bfloat16_t,
    cutlass::layout::RowMajor *, 8, cutlass::bfloat16_t, cutlass::layout::RowMajor *, 8,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, ElementA,
    cutlass::layout::RowMajor *, 16, ElementB, cutlass::layout::ColumnMajor *, 128, float, MmaTile,
    Cluster, cutlass::gemm::collective::StageCountAutoCarveout<sizeof(Epilogue::SharedStorage)>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Problem, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using Scale = cutlass::detail::Sm1xxBlockScaledConfig<32>;

struct Metadata {
  Shape<int, int, int> *shapes;
  const ElementA::DataType **a;
  const ElementB::DataType **b;
  const ElementA::ScaleFactorType **sa, **sb;
  cutlass::bfloat16_t **d;
  Kernel::InternalStrideA *stride_a;
  Kernel::InternalStrideB *stride_b;
  Kernel::InternalStrideD *stride_d;
  Mainloop::InternalLayoutSFA *layout_a;
  Mainloop::InternalLayoutSFB *layout_b;
  int bytes = 0;

  template <class T> CUTLASS_HOST_DEVICE T *take(void *base, int groups) {
    bytes = (bytes + alignof(T) - 1) / alignof(T) * alignof(T);
    auto result = reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(base) + bytes);
    bytes += groups * sizeof(T);
    return result;
  }

  CUTLASS_HOST_DEVICE Metadata(void *base, int groups) {
    shapes = take<Shape<int, int, int>>(base, groups);
    a = take<const ElementA::DataType *>(base, groups);
    b = take<const ElementB::DataType *>(base, groups);
    sa = take<const ElementA::ScaleFactorType *>(base, groups);
    sb = take<const ElementA::ScaleFactorType *>(base, groups);
    d = take<cutlass::bfloat16_t *>(base, groups);
    stride_a = take<Kernel::InternalStrideA>(base, groups);
    stride_b = take<Kernel::InternalStrideB>(base, groups);
    stride_d = take<Kernel::InternalStrideD>(base, groups);
    layout_a = take<Mainloop::InternalLayoutSFA>(base, groups);
    layout_b = take<Mainloop::InternalLayoutSFB>(base, groups);
  }
};

Gemm::Arguments arguments(Metadata metadata, int groups, int sm_count) {
  Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                       {groups, metadata.shapes, nullptr},
                       {metadata.a, metadata.stride_a, metadata.b, metadata.stride_b, metadata.sa,
                        metadata.layout_a, metadata.sb, metadata.layout_b},
                       {{1.f, 0.f}, nullptr, metadata.stride_d, metadata.d, metadata.stride_d}};
  args.hw_info.sm_count = sm_count;
  check_cuda(cudaGetDevice(&args.hw_info.device_id));
  return args;
}

CUTLASS_DEVICE int scale_start(int offset, int group, int width) {
  return (((offset + 127) / 128) + group) * 128 * (width / 32);
}

__global__ void prepare_metadata(const int32_t *counts, const int32_t *offsets,
                                 const uint64_t *weights, const uint8_t *input,
                                 const uint8_t *scales, cutlass::bfloat16_t *output,
                                 uint8_t *status, Metadata meta, int capacity, int groups,
                                 int outputs, int width) {
  int invalid = offsets[0] != 0;
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    int64_t first = offsets[group], last = offsets[group + 1];
    invalid |= counts[group] < 0 || counts[group] > 384 || first < 0 || last < 0 ||
               first > capacity || last > capacity || last - first != counts[group];
  }
  invalid = __syncthreads_or(invalid);
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    int rows = invalid ? 0 : counts[group], first = invalid ? 0 : offsets[group];
    auto shape = make_shape(rows, outputs, width, 1);
    meta.shapes[group] = make_shape(rows, outputs, width);
    meta.a[group] = reinterpret_cast<const ElementA::DataType *>(input + first * width);
    meta.b[group] = reinterpret_cast<const ElementB::DataType *>(weights[2 * group]);
    meta.sa[group] = reinterpret_cast<const ElementA::ScaleFactorType *>(
        scales + scale_start(first, group, width));
    meta.sb[group] = reinterpret_cast<const ElementA::ScaleFactorType *>(weights[2 * group + 1]);
    meta.d[group] = output + first * outputs;
    meta.stride_a[group] =
        cutlass::make_cute_packed_stride(Kernel::InternalStrideA{}, make_shape(rows, width, 1));
    meta.stride_b[group] =
        cutlass::make_cute_packed_stride(Kernel::InternalStrideB{}, make_shape(outputs, width, 1));
    meta.stride_d[group] =
        cutlass::make_cute_packed_stride(Kernel::InternalStrideD{}, make_shape(rows, outputs, 1));
    meta.layout_a[group] = Scale::tile_atom_to_shape_SFA(shape);
    meta.layout_b[group] = Scale::tile_atom_to_shape_SFB(shape);
    status[group] = invalid ? 1 : 0;
  }
}

__global__ void pack_group_scales(const uint8_t *input, const int32_t *counts,
                                  const int32_t *offsets, const uint8_t *status, Metadata meta,
                                  int width) {
  int group = blockIdx.x;
  if (status[group])
    return;
  auto output = const_cast<ElementA::ScaleFactorType *>(meta.sa[group]);
  int groups_k = width / 32;
  for (int i = threadIdx.x; i < counts[group] * groups_k; i += blockDim.x) {
    int row = i / groups_k, column = i % groups_k;
    reinterpret_cast<uint8_t *>(output)[meta.layout_a[group](make_coord(row, column * 32, 0))] =
        input[(offsets[group] + row) * groups_k + column];
  }
}

} // namespace grouped_projection

tvm::ffi::Array<int64_t> grouped_sizes(int64_t groups) {
  using namespace grouped_projection;
  TVM_FFI_ICHECK(groups >= 1 && groups <= 384);
  int device, sm_count;
  check_cuda(cudaGetDevice(&device));
  check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  cudaStreamCaptureStatus capture;
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
  check_cuda(cudaStreamIsCapturing(stream, &capture));
  TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone);
  if constexpr (Kernel::SharedStorageSize >= (48 << 10))
    check_cuda(cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Kernel::SharedStorageSize));
  Metadata meta(nullptr, groups);
  return {meta.bytes, int64_t(Gemm::get_workspace_size(arguments(meta, groups, sm_count))),
          sm_count};
}

template <bool Prepared>
void project_grouped_impl(TensorView encoded, TensorView scales, TensorView counts,
                          TensorView offsets, TensorView weights, TensorView output,
                          TensorView status, TensorView padded_input, TensorView packed_scales,
                          TensorView metadata, TensorView scratch, int64_t width,
                          int64_t sm_count) {
  namespace gp = grouped_projection;
  TVM_FFI_ICHECK(encoded.ndim() == 2 && output.ndim() == 2 && counts.ndim() == 1);
  int64_t capacity = encoded.size(0), outputs = output.size(1), groups = counts.size(0);
  encoded_projection::dimensions(0, outputs, width, 4);
  TVM_FFI_ICHECK(capacity >= 0 && capacity <= 2304 && groups >= 1 && groups <= 384 &&
                 capacity <= groups * 384 && sm_count > 0 && sm_count <= 1024);
  int padded = encoded_projection::round128(width);
  gp::Metadata meta(metadata.data_ptr(), groups);
  int scale_capacity = (encoded_projection::round128(capacity) + groups * 128) * (padded / 32);
  tensor_contract(encoded, encoded, kDLUInt, 8, {capacity, Prepared ? padded : width});
  if constexpr (Prepared)
    tensor_contract(scales, encoded, kDLUInt, 8, {scale_capacity});
  else {
    tensor_contract(scales, encoded, kDLUInt, 8, {capacity, width / 32});
    tensor_contract(padded_input, encoded, kDLUInt, 8, {capacity, padded});
    tensor_contract(packed_scales, encoded, kDLUInt, 8, {scale_capacity});
  }
  tensor_contract(counts, encoded, kDLInt, 32, {groups});
  tensor_contract(offsets, encoded, kDLInt, 32, {groups + 1});
  tensor_contract(weights, encoded, kDLUInt, 64, {groups, 2});
  tensor_contract(output, encoded, kDLBfloat, 16, {capacity, outputs});
  tensor_contract(status, encoded, kDLUInt, 8, {groups});
  tensor_contract(metadata, encoded, kDLUInt, 8, {meta.bytes});
  tvm::ffi::CUDADeviceGuard guard(encoded.device().device_id);
  auto args = gp::arguments(meta, groups, sm_count);
  tensor_contract(scratch, encoded, kDLUInt, 8, {int64_t(gp::Gemm::get_workspace_size(args))});
  TensorView writes[] = {output, status, metadata, scratch, packed_scales, padded_input};
  int write_count = Prepared ? 4 : (width == padded ? 5 : 6);
  for (int i = 0; i < write_count; ++i) {
    for (auto input : {encoded, scales, counts, offsets, weights})
      encoded_projection::separate(writes[i], input);
    for (int j = 0; j < i; ++j)
      encoded_projection::separate(writes[i], writes[j]);
  }
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, encoded.device().device_id));
  check_cuda(cudaMemsetAsync(output.data_ptr(), 0, tvm::ffi::GetDataSize(output), stream));
  if constexpr (!Prepared)
    check_cuda(cudaMemsetAsync(packed_scales.data_ptr(), 127, tvm::ffi::GetDataSize(packed_scales),
                               stream));
  auto input = Prepared || width == padded ? encoded : padded_input;
  auto input_scales = Prepared ? scales : packed_scales;
  gp::prepare_metadata<<<1, 256, 0, stream>>>(static_cast<const int32_t *>(counts.data_ptr()),
                                              static_cast<const int32_t *>(offsets.data_ptr()),
                                              static_cast<const uint64_t *>(weights.data_ptr()),
                                              static_cast<const uint8_t *>(input.data_ptr()),
                                              static_cast<const uint8_t *>(input_scales.data_ptr()),
                                              static_cast<cutlass::bfloat16_t *>(output.data_ptr()),
                                              static_cast<uint8_t *>(status.data_ptr()), meta,
                                              capacity, groups, outputs, padded);
  check_cuda(cudaGetLastError());
  if (!capacity)
    return;
  if constexpr (!Prepared) {
    if (width != padded)
      encoded_projection::launch_padding(encoded, padded_input, capacity, width, padded, stream);
    gp::pack_group_scales<<<groups, 256, 0, stream>>>(
        static_cast<const uint8_t *>(scales.data_ptr()),
        static_cast<const int32_t *>(counts.data_ptr()),
        static_cast<const int32_t *>(offsets.data_ptr()),
        static_cast<const uint8_t *>(status.data_ptr()), meta, width);
  }
  check_cuda(cudaGetLastError());
  encoded_projection::check_cutlass(gp::Gemm::can_implement(args));
  encoded_projection::check_cutlass(
      gp::Kernel::initialize_workspace(args, scratch.data_ptr(), stream));
  auto params = gp::Kernel::to_underlying_arguments(args, scratch.data_ptr());
  encoded_projection::check_cutlass(gp::Gemm::run(params, stream));
}

void project_grouped(TensorView encoded, TensorView scales, TensorView counts, TensorView offsets,
                     TensorView weights, TensorView output, TensorView status,
                     TensorView padded_input, TensorView packed_scales, TensorView metadata,
                     TensorView scratch, int64_t width, int64_t sm_count) {
  project_grouped_impl<false>(encoded, scales, counts, offsets, weights, output, status,
                              padded_input, packed_scales, metadata, scratch, width, sm_count);
}

void project_grouped_prepared(TensorView encoded, TensorView packed_scales, TensorView counts,
                              TensorView offsets, TensorView weights, TensorView output,
                              TensorView status, TensorView metadata, TensorView scratch,
                              int64_t width, int64_t sm_count) {
  project_grouped_impl<true>(encoded, packed_scales, counts, offsets, weights, output, status,
                             encoded, packed_scales, metadata, scratch, width, sm_count);
}

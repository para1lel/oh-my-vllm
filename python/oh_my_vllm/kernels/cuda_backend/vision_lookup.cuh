#pragma once

#include "common.cuh"
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>

struct VisionLookupArguments {
  const cutlass::bfloat16_t *table = nullptr;
};

template <class Value> struct VisionLookupGelu;

template <> struct VisionLookupGelu<float> {
  static constexpr bool kIsHeavy = true;
  using Arguments = VisionLookupArguments;

  CUTLASS_HOST_DEVICE float operator()(float value, const Arguments &args = Arguments()) const {
    cutlass::NumericConverter<cutlass::bfloat16_t, float> convert;
    return float(args.table[convert(value).raw()]);
  }
};

template <int Count> struct VisionLookupGelu<cutlass::Array<float, Count>> {
  static constexpr bool kIsHeavy = true;
  using Arguments = VisionLookupArguments;

  CUTLASS_HOST_DEVICE cutlass::Array<float, Count>
  operator()(const cutlass::Array<float, Count> &input, const Arguments &args = Arguments()) const {
    cutlass::Array<float, Count> output;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < Count; ++index)
      output[index] = VisionLookupGelu<float>{}(input[index], args);
    return output;
  }
};

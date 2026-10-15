#pragma once

#include "common.cuh"

__device__ inline float vision_cubic(float distance) {
  float x = fabsf(distance);
  if (x < 1.f)
    return ((1.5f * x - 2.5f) * x) * x + 1.f;
  if (x < 2.f)
    return ((-0.5f * x + 2.5f) * x - 4.f) * x + 2.f;
  return 0.f;
}

__device__ inline float vision_pixel(const uint8_t *image, int channel, int y, int x,
                                     int source_height, int source_width, int height, int width) {
  if (height == source_height && width == source_width)
    return image[(int64_t(y) * width + x) * 3 + channel];
  float scale_y = float(source_height) / height, scale_x = float(source_width) / width;
  float center_y = (y + .5f) * scale_y - .5f, center_x = (x + .5f) * scale_x - .5f;
  float filter_y = fmaxf(scale_y, 1.f), filter_x = fmaxf(scale_x, 1.f);
  int first_y = max(0, int(ceilf(center_y - 2.f * filter_y)));
  int last_y = min(source_height - 1, int(floorf(center_y + 2.f * filter_y)));
  int first_x = max(0, int(ceilf(center_x - 2.f * filter_x)));
  int last_x = min(source_width - 1, int(floorf(center_x + 2.f * filter_x)));
  float total = 0, denominator = 0;
  for (int sy = first_y; sy <= last_y; ++sy) {
    float wy = vision_cubic((sy - center_y) / filter_y);
    for (int sx = first_x; sx <= last_x; ++sx) {
      float weight = wy * vision_cubic((sx - center_x) / filter_x);
      total += weight * image[(int64_t(sy) * source_width + sx) * 3 + channel];
      denominator += weight;
    }
  }
  return fminf(255.f, fmaxf(0.f, nearbyintf(total / denominator)));
}

__global__ void vision_patch_kernel(const uint8_t *images, __nv_bfloat16 *patches, int rows,
                                    int frames, int source_height, int source_width,
                                    int grid_height, int grid_width) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= int64_t(rows) * 1536)
    return;
  int patch = index / 1536, component = index % 1536;
  int tokens = grid_height * grid_width, groups = (frames + 1) / 2;
  int item = patch / (tokens * groups), group = (patch / tokens) % groups;
  int local = patch % tokens, block = local / 4;
  int py = (block / (grid_width / 2)) * 2 + (local % 4) / 2;
  int px = (block % (grid_width / 2)) * 2 + local % 2;
  int channel = component / 512, temporal = (component / 256) % 2;
  int y = py * 16 + (component % 256) / 16, x = px * 16 + component % 16;
  int frame = min(group * 2 + temporal, frames - 1);
  const uint8_t *image =
      images + (int64_t(item) * frames + frame) * source_height * source_width * 3;
  float pixel = vision_pixel(image, channel, y, x, source_height, source_width, grid_height * 16,
                             grid_width * 16);
  patches[index] = __float2bfloat16_rn(pixel * (2.f / 255.f) - 1.f);
}

__global__ void vision_position_kernel(const float *projected, const __nv_bfloat16 *bias,
                                       const __nv_bfloat16 *position, __nv_bfloat16 *output,
                                       int rows, int grid_height, int grid_width) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= int64_t(rows) * 1152)
    return;
  int row = index / 1152, column = index % 1152;
  int local = row % (grid_height * grid_width), block = local / 4;
  int py = (block / (grid_width / 2)) * 2 + (local % 4) / 2;
  int px = (block % (grid_width / 2)) * 2 + local % 2;
  float y = py * (47.f / (grid_height - 1)), x = px * (47.f / (grid_width - 1));
  int y0 = int(y), x0 = int(x), y1 = min(y0 + 1, 47), x1 = min(x0 + 1, 47);
  float dy = y - y0, dx = x - x0;
  float embedded =
      __bfloat162float(position[(y0 * 48 + x0) * 1152 + column]) * (1.f - dy) * (1.f - dx) +
      __bfloat162float(position[(y0 * 48 + x1) * 1152 + column]) * (1.f - dy) * dx +
      __bfloat162float(position[(y1 * 48 + x0) * 1152 + column]) * dy * (1.f - dx) +
      __bfloat162float(position[(y1 * 48 + x1) * 1152 + column]) * dy * dx;
  float patch =
      __bfloat162float(__float2bfloat16_rn(projected[index] + __bfloat162float(bias[column])));
  output[index] = __float2bfloat16_rn(patch + __bfloat162float(__float2bfloat16_rn(embedded)));
}

#pragma once

// Generated from the released CUDA 13.0 cuBLAS image in torch 2.13.0+cu130.
// Generator: classify_chunks.cu (external artifact)
// Row counts: measured 1..32768; N=1 uses the dot order.
struct ChunkRange {
  int first;
  int last;
  int chunk;
};

constexpr ChunkRange kChunkRanges[] = {
    {2, 5, 192},        {6, 7, 224},         {8, 16, 192},       {17, 128, 160},
    {129, 192, 448},    {193, 256, 288},     {257, 384, 448},    {385, 512, 576},
    {513, 572, 736},    {573, 639, 352},     {640, 640, 736},    {641, 726, 864},
    {727, 768, 448},    {769, 896, 512},     {897, 992, 576},    {993, 1024, 224},
    {1025, 1152, 256},  {1153, 1280, 288},   {1281, 1408, 320},  {1409, 1536, 352},
    {1537, 1664, 384},  {1665, 1792, 416},   {1793, 2048, 448},  {2049, 2176, 480},
    {2177, 2304, 2560}, {2305, 2432, 544},   {2433, 2560, 576},  {2561, 2688, 2944},
    {2689, 2944, 640},  {2945, 3072, 3424},  {3073, 3328, 736},  {3329, 3456, 768},
    {3457, 3584, 800},  {3585, 3712, 832},   {3713, 3840, 864},  {3841, 4096, 896},
    {4097, 4224, 960},  {4225, 4480, 992},   {4481, 4736, 1024}, {4737, 4864, 1088},
    {4865, 5179, 1152}, {5180, 5184, 20480}, {5185, 5186, 1152}, {5187, 32768, 20480},
};

inline int released_chunk(int rows) {
  if (rows <= 1)
    return 0;
  int low = 0, high = int(sizeof(kChunkRanges) / sizeof(kChunkRanges[0])) - 1;
  while (low <= high) {
    int middle = (low + high) / 2;
    if (rows < kChunkRanges[middle].first)
      high = middle - 1;
    else if (rows > kChunkRanges[middle].last)
      low = middle + 1;
    else
      return kChunkRanges[middle].chunk;
  }
  // A hole in the table must stop the call, not reuse the last layout.
  return 0;
}

// Measured from the same CUDA 13.0 cuBLAS image for the routed-expert router
// (M=384, K=5120) and the ratio-based compressor (M=512, K=5120).
constexpr ChunkRange kRouterChunkRanges[] = {
    {2, 5, 160},     {6, 7, 96},      {8, 16, 128},    {17, 32, 160},    {33, 64, 128},
    {65, 127, 160},  {128, 128, 96},  {129, 256, 128}, {257, 384, 160},  {385, 461, 224},
    {462, 512, 288}, {513, 572, 384}, {573, 639, 288}, {640, 640, 384},  {641, 726, 448},
    {727, 768, 320}, {769, 896, 384}, {897, 992, 448}, {993, 1024, 224}, {1025, 32768, 5120},
};

constexpr ChunkRange kDsparkChunkRanges[] = {
    {2, 5, 128},       {6, 7, 96},        {8, 16, 128},      {17, 28, 64},        {29, 32, 96},
    {33, 53, 64},      {54, 64, 96},      {65, 81, 160},     {82, 128, 96},       {129, 192, 128},
    {193, 256, 160},   {257, 384, 128},   {385, 512, 160},   {513, 640, 192},     {641, 768, 224},
    {769, 896, 256},   {897, 1024, 288},  {1025, 1152, 320}, {1153, 1280, 384},   {1281, 1536, 224},
    {1537, 1792, 256}, {1793, 2048, 288}, {2049, 2304, 320}, {2305, 2432, 352},   {2433, 2688, 384},
    {2689, 2816, 416}, {2817, 3072, 448}, {3073, 3168, 480}, {3169, 32768, 5120},
};

constexpr ChunkRange kCompressorChunkRanges[] = {
    {2, 5, 128},        {6, 7, 96},        {8, 16, 128},        {17, 32, 160},
    {33, 64, 64},       {65, 256, 160},    {257, 384, 224},     {385, 512, 288},
    {513, 640, 384},    {641, 726, 448},   {727, 768, 224},     {769, 827, 256},
    {828, 896, 512},    {897, 1248, 5120}, {1249, 1280, 736},   {1281, 1408, 864},
    {1409, 1473, 5120}, {1474, 1536, 864}, {1537, 32768, 5120},
};

inline int find_chunk(const ChunkRange *ranges, int count, int rows) {
  int low = 0, high = count - 1;
  while (low <= high) {
    int middle = (low + high) / 2;
    if (rows < ranges[middle].first)
      high = middle - 1;
    else if (rows > ranges[middle].last)
      low = middle + 1;
    else
      return ranges[middle].chunk;
  }
  return 0;
}

// Zero means that the shape or row count has no measured released layout.
inline int project_chunk(int rows, int outputs, int width) {
  if (rows < 2 || rows > 32768 || width != 5120)
    return 0;
  if (outputs == 384)
    return find_chunk(kRouterChunkRanges,
                      int(sizeof(kRouterChunkRanges) / sizeof(kRouterChunkRanges[0])), rows);
  if (outputs == 128)
    return find_chunk(kDsparkChunkRanges,
                      int(sizeof(kDsparkChunkRanges) / sizeof(kDsparkChunkRanges[0])), rows);
  if (outputs == 512)
    return find_chunk(kCompressorChunkRanges,
                      int(sizeof(kCompressorChunkRanges) / sizeof(kCompressorChunkRanges[0])),
                      rows);
  return 0;
}

def validate_cache(codes, scales):
    shape = tuple(codes.shape)
    if (
        len(shape) != 5
        or not 1 < shape[0] <= (2**31 - 1) // 128
        or shape[1:] != (2, 128, 4, 256)
        or str(codes.dtype).removeprefix("torch.") != "uint8"
        or tuple(scales.shape) != shape[:-1]
        or str(scales.dtype).removeprefix("torch.") != "float32"
    ):
        raise ValueError("FP8 FA requires raw uint8 codes and per-vector FP32 scales")

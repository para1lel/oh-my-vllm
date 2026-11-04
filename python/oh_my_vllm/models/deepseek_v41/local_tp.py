"""Keep released MP4 row-projection rounding on each dense owner."""


class LocalRowProjection:
    def __init__(self, partitions):
        self.partitions = partitions
        self.clear()

    def clear(self):
        self.binding = None
        self.weights = ()

    def _bind(self, weight, scale):
        binding = (
            id(weight),
            weight.data_ptr(),
            weight.device,
            tuple(weight.shape),
            id(scale),
            None if scale is None else scale.data_ptr(),
        )
        if binding == self.binding:
            return
        width = weight.size(1) // self.partitions
        if width * self.partitions != weight.size(1):
            raise ValueError("row projection must divide into complete MP4 shards")
        scales = None if scale is None else scale.size(1) // self.partitions
        if scale is not None and scales * self.partitions != scale.size(1):
            raise ValueError("row projection shards must preserve scale blocks")
        shards = []
        for rank in range(self.partitions):
            shard = weight[:, rank * width : (rank + 1) * width].contiguous()
            if scale is not None:
                shard.scale = scale[:, rank * scales : (rank + 1) * scales].contiguous()
            shards.append(shard)
        self.weights = tuple(shards)
        self.binding = binding

    def __call__(self, inputs, weight, scale, project):
        self._bind(weight, scale)
        width = inputs.size(-1) // self.partitions
        if width * self.partitions != inputs.size(-1):
            raise ValueError("row projection input must divide into MP4 shards")
        pairs = []
        for rank, shard in enumerate(self.weights):
            rows = inputs[..., rank * width : (rank + 1) * width].contiguous()
            # MP4 rounds each projection before the FP32 rank reduction.
            partial = project(rows, shard).float()
            if rank % 2 == 0:
                pairs.append(partial)
            else:
                pairs[-1].add_(partial)
        output = pairs[0]
        for pair in pairs[1:]:
            output.add_(pair)
        return output

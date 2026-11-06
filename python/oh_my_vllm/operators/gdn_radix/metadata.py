from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class ChunkLayout:
    metadata: tuple
    offsets: tuple

    @property
    def chunks(self):
        return len(self.metadata)


def make_layout(schedule):
    writes = schedule.validate()
    metadata, offsets = [], [0]
    for first, last in pairwise(schedule.sequence_rows):
        base, begin = len(metadata), first
        while begin < last:
            end = min(begin + 64, last)
            for row in range(begin, end):
                if writes[row] >= 0:
                    end = row + 1
                    break
            metadata.append((begin, end, writes[end - 1], 0))
            begin = end
        count = len(metadata) - base
        if count:
            begin, end, target, _ = metadata[base]
            metadata[base] = (begin, end, target, count)
        offsets.append(len(metadata))
    return ChunkLayout(tuple(metadata), tuple(offsets))

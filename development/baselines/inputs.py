from dataclasses import dataclass, field
from itertools import accumulate


@dataclass
class CoreInputs:
    case: object
    tensors: dict
    checkpoint_rows: tuple = ()
    checkpoint_slots: tuple = ()
    checkpoint_sequences: tuple = ()
    sequence_offsets: tuple = ()
    metadata: dict = field(default_factory=dict)

    def __getattr__(self, name):
        tensors = self.__dict__.get("tensors", {})
        if name in tensors:
            return tensors[name]
        raise AttributeError(name)


@dataclass
class Prepared:
    run: object
    reset: object
    outputs: dict
    metadata: dict
    owner: object = None


def make_inputs(case):
    import torch

    seed = int(case.identity[:8], 16)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tensors = {}

    def normal(shape, scale=0.2, dtype=torch.bfloat16):
        return (
            torch.randn(shape, device="cuda", generator=generator).mul_(scale).to(dtype)
        )

    def integers(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    offsets = (0, *accumulate(case.queries))
    tensors["sequence_offsets"] = integers(offsets)
    if case.operator == "fa_attention":
        pages = tuple((length + 783) // 784 for length in case.contexts)
        page_offsets = (0, *accumulate(pages))
        order = torch.randperm(sum(pages), device="cuda", generator=generator) + 1
        tensors.update(
            q=normal((case.rows, 24, 256)),
            cache=normal((sum(pages) + 2, 2, 784, 4, 256), 0.05),
            page_offsets=integers(page_offsets),
            page_indices=order.to(torch.int32),
            last_page_lengths=integers(
                tuple((length - 1) % 784 + 1 for length in case.contexts)
            ),
        )
        return CoreInputs(
            case,
            tensors,
            sequence_offsets=offsets,
            metadata={"seed": seed, "page_size": 784, "cache_layout": "NHD"},
        )
    checkpoint_rows, checkpoint_slots, checkpoint_sequences = [], [], []
    for sequence, (start, length, position) in enumerate(
        zip(offsets, case.queries, case.positions, strict=False)
    ):
        for token in range(length):
            if case.verify or (position + token + 1) % 784 == 0 or token == length - 1:
                checkpoint_rows.append(start + token)
                checkpoint_sequences.append(sequence)
                checkpoint_slots.append(case.batch + len(checkpoint_slots))
    slots = case.batch + len(checkpoint_rows) + 2
    state = torch.full(
        (slots, 48, 128, 128), -0.125, dtype=torch.bfloat16, device="cuda"
    )
    state[: case.batch] = normal((case.batch, 48, 128, 128), 0.01)
    tensors.update(
        q=normal((case.rows, 16, 128)),
        k=normal((case.rows, 16, 128)),
        v=normal((case.rows, 48, 128)),
        a=normal((case.rows, 48)),
        b=normal((case.rows, 48)),
        A_log=torch.linspace(-2, 0, 48, device="cuda"),
        dt_bias=torch.linspace(-2, -1, 48, device="cuda"),
        state_pool=state,
        read_slots=integers(tuple(range(case.batch))),
        checkpoint_rows=integers(checkpoint_rows),
        checkpoint_slots=integers(checkpoint_slots),
    )
    return CoreInputs(
        case,
        tensors,
        tuple(checkpoint_rows),
        tuple(checkpoint_slots),
        tuple(checkpoint_sequences),
        offsets,
        {"seed": seed, "state_layout": "HV,V,K", "nonzero_initial_state": True},
    )

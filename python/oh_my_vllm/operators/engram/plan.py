from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    rows: int
    history_rows: int
    layers: int
    heads: int
    max_ngram_size: int
    pad_id: int
    vocabulary_size: int

    @property
    def next_history_rows(self):
        return min(self.max_ngram_size - 1, self.history_rows + self.rows)

    @property
    def input_shapes(self):
        columns = (self.max_ngram_size - 1) * self.heads
        return (
            (self.rows,),
            (self.history_rows,),
            (self.layers, self.max_ngram_size),
            (self.layers, columns),
            (self.layers, columns),
        )

    @property
    def storage(self):
        return (
            ((self.rows, self.layers, self.input_shapes[3][1]), "int64"),
            ((self.next_history_rows,), "int64"),
        )

    @property
    def reuse_key(self):
        return (
            self.rows,
            self.history_rows,
            self.layers,
            self.heads,
            self.max_ngram_size,
            self.pad_id,
            self.vocabulary_size,
        )


def make_plan(
    *,
    rows,
    pad_id,
    vocabulary_size=99092,
    layers=2,
    heads=8,
    max_ngram_size=4,
    history_rows=0,
):
    for value, lower, upper in (
        (rows, 0, 384),
        (layers, 1, 2),
        (heads, 1, 8),
        (max_ngram_size, 2, 4),
        (vocabulary_size, 1, 99092),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError("Engram dimensions exceed supported bounds")
    if (
        type(history_rows) is not int
        or not 0 <= history_rows < max_ngram_size
        or type(pad_id) is not int
        or not 0 <= pad_id < vocabulary_size
    ):
        raise ValueError("Invalid Engram history or compressed padding ID")
    return Plan(
        rows, history_rows, layers, heads, max_ngram_size, pad_id, vocabulary_size
    )

import unittest
from dataclasses import replace


class Engine(unittest.TestCase):
    def test_protocol_rejects_unsafe_graphs_and_metadata(self):
        from oh_my_vllm.kernels.tile_plan import (
            Capacity,
            StepMetadata,
            Task,
            compile_graph,
        )

        capacity = Capacity(
            tasks=4, edges=4, queue=2, rows=16, batch=2, pages=8, slots=8
        )
        tasks = (
            Task("scale", (0,), 1, 0, 16, factor=2),
            Task("scale", (0,), 2, 0, 16, factor=3),
            Task("add", (1, 2), 3, 0, 16, depends=(0, 1)),
            Task("copy", (3,), 1, 0, 16, depends=(2,)),
        )
        graph = compile_graph(tasks, capacity, buffers=4)
        self.assertEqual(graph.roots, (0, 1))
        self.assertEqual(graph.successors, ((2,), (2,), (3,), ()))
        valid = StepMetadata(
            epoch=1,
            query_lengths=(1, 2),
            valid_rows=3,
            page_tables=((7,), (2,)),
            read_slots=(0, 1),
            write_slots=(2, 3, 4),
        )
        valid.validate(capacity, previous_epoch=0)
        for changed in (
            replace(valid, epoch=0),
            replace(valid, valid_rows=4),
            replace(valid, query_lengths=(17,)),
            replace(valid, read_slots=(8, 1)),
            replace(valid, write_slots=(2, 2, 4)),
            replace(valid, page_tables=((8,), (2,))),
        ):
            with self.subTest(metadata=changed), self.assertRaises(ValueError):
                changed.validate(capacity, previous_epoch=0)
        with self.assertRaises(ValueError):
            valid.validate(capacity, previous_epoch=1)
        for changed in (
            (replace(tasks[0], depends=(3,)), *tasks[1:]),
            (*tasks[:2], replace(tasks[2], depends=(0,)), tasks[3]),
            (*tasks[:3], replace(tasks[3], depends=())),
            (replace(tasks[0], count=17), *tasks[1:]),
            (replace(tasks[0], inputs=(4,)), *tasks[1:]),
            (replace(tasks[0], depends=(1, 1)), *tasks[1:]),
        ):
            with self.subTest(tasks=changed), self.assertRaises(ValueError):
                compile_graph(changed, capacity, buffers=4)
        for changed in (
            replace(capacity, queue=1),
            replace(capacity, tasks=3),
            replace(capacity, edges=2),
        ):
            with self.subTest(capacity=changed), self.assertRaises(ValueError):
                compile_graph(tasks, changed, buffers=4)


if __name__ == "__main__":
    unittest.main()

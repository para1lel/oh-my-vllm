import copy
import json
import unittest
from pathlib import Path

from oh_my_vllm.operators.runtime.distributed import CollectivePlan


class CollectivePlanChecks(unittest.TestCase):
    def test_ordered_plan_binds_counts_and_rejects_invalid_storage(self):
        fixture = json.loads(
            Path(__file__)
            .with_name("fixtures")
            .joinpath("collectives.json")
            .read_text()
        )
        plan = CollectivePlan.from_wire(fixture["plan"], world_size=4)
        self.assertEqual(plan.sha256, fixture["sha256"])
        for rank in range(4):
            self.assertEqual(
                [list(pair) for pair in plan.buffer_counts(rank)],
                fixture["buffer_counts"][rank],
            )
        with self.assertRaises(ValueError):
            plan.buffer_counts(4)
        with self.assertRaises(ValueError):
            CollectivePlan.from_wire(fixture["plan"], world_size=1)
        for case in range(14):
            value = copy.deepcopy(fixture["plan"])
            ops = value["operations"]
            if case == 0:
                value["version"] = 2
            elif case == 1:
                value["world_size"] = 0
            elif case in (2, 3, 4):
                ops[case - 2]["count"] = (1 << 64) - 1
            elif case == 5:
                ops[3]["counts"] = [[0, 0]]
            elif case == 6:
                ops[3]["counts"][0][0] = 1 << 64
            elif case == 7:
                ops[3]["counts"][1][0] = ((1 << 63) - 1) // 4
            elif case == 8:
                value["operations"] = [ops[0]] * 4097
            elif case == 9:
                ops[0]["dtype"] = "f64"
            elif case == 10:
                ops[0]["extra"] = 0
            else:
                ops[0]["count"] = [True, -1, 0.5][case - 11]
            with self.subTest(case=case), self.assertRaises(ValueError):
                CollectivePlan.from_wire(value, world_size=4)
        for case in range(3):
            value = copy.deepcopy(fixture["plan"])
            if case == 0:
                value["operations"][:2] = reversed(value["operations"][:2])
            elif case == 1:
                value["operations"][0]["dtype"] = "f16"
            else:
                value["operations"][3]["counts"][0][2] = 1
            changed = CollectivePlan.from_wire(value, world_size=4)
            self.assertNotEqual(changed.sha256, fixture["sha256"])
        fixture["plan"]["operations"][0]["count"] = 999
        self.assertEqual(plan.buffer_counts(0)[0], (8, 8))


if __name__ == "__main__":
    unittest.main()

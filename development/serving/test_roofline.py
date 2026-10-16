import unittest

from .roofline import lower_bound


class Roofline(unittest.TestCase):
    def test_prefill_splitting_and_other_requests_do_not_raise_the_bound(self):
        identity = {"operation": "identity", "config": {"num_speculative_tokens": 0}}

        def target(first, count):
            return {
                "operation": "target",
                "verify": False,
                "requests": [{"id": 1, "start": first, "count": count, "prompt": 16}],
            }

        def commit(position):
            return {
                "operation": "commit",
                "requests": [
                    {
                        "id": 1,
                        "tokens": int(position == 16),
                        "accepted": 0,
                        "proposed": 0,
                        "position": position,
                    }
                ],
            }

        complete = [identity, target(0, 16), commit(16)]
        split = [identity]
        for first in range(0, 16, 4):
            split.extend([target(first, 4), commit(first + 4)])
        split.append(
            {
                "operation": "target",
                "verify": True,
                "requests": [{"id": 2, "start": 100000, "count": 20, "prompt": 16}],
            }
        )
        expected = lower_bound(complete, {1})
        actual = lower_bound(split, {1})
        self.assertEqual(actual["work"], expected["work"])
        self.assertEqual(actual["seconds"], expected["seconds"])
        self.assertGreater(actual["seconds"], 0)
        work = actual["work"]
        self.assertEqual(
            actual["seconds"],
            max(
                work["hbm_bytes"] / 7.7e12,
                work["fp8_flops"] / 4.5e15 + work["bf16_flops"] / 2.25e15,
                work["fp32_flops"] / 75e12,
            ),
        )
        with self.assertRaises(ValueError):
            lower_bound([*complete, target(0, 16)], {1})
        with self.assertRaises(ValueError):
            lower_bound(complete, {1}, expected_tokens=8192)
        mtp = [{"operation": "identity", "config": {"num_speculative_tokens": 4}}]
        mtp.append(
            {
                "operation": "target",
                "verify": False,
                "requests": [{"id": 1, "start": 0, "count": 4, "prompt": 4}],
            }
        )
        mtp.append(
            {"operation": "context", "requests": [{"id": 1, "start": 1, "count": 4}]}
        )
        mtp.extend(
            {
                "operation": "draft",
                "stage": stage,
                "requests": [{"id": 1, "position": 4 + stage}],
            }
            for stage in range(4)
        )
        mtp.append(
            {
                "operation": "commit",
                "requests": [
                    {"id": 1, "tokens": 1, "accepted": 0, "proposed": 4, "position": 4}
                ],
            }
        )
        with self.assertRaises(ValueError):
            lower_bound(mtp, {1}, expected_tokens=2)
        unused = {
            "operation": "target",
            "verify": True,
            "requests": [{"id": 1, "start": 4, "count": 1, "prompt": 4}],
        }
        with self.assertRaises(ValueError):
            lower_bound([*mtp, unused], {1}, expected_tokens=6)

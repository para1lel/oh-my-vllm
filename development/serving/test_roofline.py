import unittest

from .roofline import lower_bound


class Roofline(unittest.TestCase):
    def test_prefill_chunks_keep_the_actual_operator_calls(self):
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
        self.assertGreater(actual["work"]["hbm_bytes"], expected["work"]["hbm_bytes"])
        self.assertEqual(actual["work"]["fp8_flops"], expected["work"]["fp8_flops"])
        self.assertGreater(actual["seconds"], expected["seconds"])
        self.assertEqual(
            [
                item["rows"]
                for item in actual["components"]
                if item["operation"] == "sample"
            ],
            [1],
        )
        self.assertGreater(actual["seconds"], 0)
        work = actual["work"]
        self.assertEqual(
            actual["seconds"],
            sum(item["seconds"] for item in actual["components"]),
        )
        self.assertGreaterEqual(actual["seconds"], work["hbm_bytes"] / 7.7e12)
        large = [
            identity,
            dict(
                operation="target",
                verify=False,
                requests=[dict(id=1, start=0, count=32768, prompt=32768)],
            ),
            dict(
                operation="commit",
                requests=[dict(id=1, tokens=1, accepted=0, proposed=0, position=32768)],
            ),
        ]
        large_bound = lower_bound(large, {1}, 1)
        self.assertGreater(
            large_bound["seconds"], large_bound["global_resource_seconds"]
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

    def test_actual_batches_are_not_combined_by_request_round(self):
        identity = {"operation": "identity", "config": {"num_speculative_tokens": 0}}
        targets = [dict(id=request, start=0, count=16, prompt=16) for request in (1, 2)]
        commits = [
            dict(id=request, tokens=1, accepted=0, proposed=0, position=16)
            for request in (1, 2)
        ]
        together = [
            identity,
            dict(operation="target", verify=False, requests=targets),
            dict(operation="commit", requests=commits),
        ]
        separate = [identity]
        for target, commit in zip(targets, commits, strict=True):
            separate.extend(
                [
                    dict(operation="target", verify=False, requests=[target]),
                    dict(operation="commit", requests=[commit]),
                ]
            )
        combined = lower_bound(together, {1, 2}, 1)
        split = lower_bound(separate, {1, 2}, 1)
        self.assertGreater(split["seconds"], combined["seconds"])
        self.assertEqual(
            [
                item["rows"]
                for item in split["components"]
                if item["operation"] == "sample"
            ],
            [1, 1],
        )
        ragged = [
            identity,
            dict(
                operation="target",
                verify=False,
                requests=[targets[0], dict(id=2, start=0, count=4, prompt=8)],
            ),
            dict(
                operation="commit",
                requests=[
                    commits[0],
                    dict(id=2, tokens=0, accepted=0, proposed=0, position=4),
                ],
            ),
            dict(
                operation="target",
                verify=False,
                requests=[dict(id=2, start=4, count=4, prompt=8)],
            ),
            dict(
                operation="commit",
                requests=[dict(id=2, tokens=1, accepted=0, proposed=0, position=8)],
            ),
        ]
        result = lower_bound(ragged, {1, 2}, 1)
        samples = [
            item for item in result["components"] if item["operation"] == "sample"
        ]
        self.assertEqual([item["request_ids"] for item in samples], [[1], [2]])
        attention = next(
            item for item in result["components"] if item["operation"] == "fa_attention"
        )
        self.assertEqual(attention["work_per_call"]["bf16_flops"], 4 * 146 * 24 * 256)

    def test_mtp_keeps_verification_rows_and_actual_proposals(self):
        events = [
            dict(operation="identity", config=dict(num_speculative_tokens=4)),
            dict(
                operation="target",
                verify=False,
                requests=[dict(id=1, start=0, count=4, prompt=4)],
            ),
            dict(operation="context", requests=[dict(id=1, start=1, count=4)]),
            dict(operation="draft", stage=0, requests=[dict(id=1, position=4)]),
            dict(operation="draft", stage=1, requests=[dict(id=1, position=5)]),
            dict(
                operation="commit",
                requests=[dict(id=1, tokens=1, accepted=0, proposed=2, position=4)],
            ),
            dict(
                operation="target",
                verify=True,
                requests=[dict(id=1, start=4, count=3, prompt=4)],
            ),
            dict(operation="context", requests=[dict(id=1, start=5, count=2)]),
            dict(
                operation="commit",
                requests=[dict(id=1, tokens=3, accepted=2, proposed=0, position=7)],
            ),
        ]
        result = lower_bound(events, {1}, 4)
        self.assertEqual(
            result["acceptance"][1], dict(tokens=4, accepted=2, proposed=2)
        )
        drafts = [
            item for item in result["components"] if item["operation"] == "mtp_draft"
        ]
        self.assertEqual([item["queries"] for item in drafts], [[(4, 1)], [(5, 1)]])
        self.assertEqual(
            drafts[0]["work_per_call"]["bf16_flops"],
            2 * 248320 * 5120 + 4 * 4 * 24 * 256,
        )
        self.assertEqual(
            drafts[1]["work_per_call"]["bf16_flops"],
            2 * 248320 * 5120 + 2 * 5120 * 10240 + 4 * 5 * 24 * 256,
        )
        samples = [
            item for item in result["components"] if item["operation"] == "sample"
        ]
        self.assertEqual([item["rows"] for item in samples], [1, 3])
        self.assertEqual(samples[-1]["work_per_call"]["fp64_flops"], 6)

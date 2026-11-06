import unittest
from copy import deepcopy
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from oh_my_vllm.models.qwen38.cache import Cache
from oh_my_vllm.models.qwen38.request import Request

from tests.constraints_fixtures import (
    AcceptDevice,
    ChainDevice,
    ChainDraft,
    ChainSample,
    PackedAcceptance,
)
from tests.qwen_radix_physical_fixtures import Device as CacheDevice
from tests.qwen_radix_physical_fixtures import config, reference, row


class ConstraintsDraftChain(unittest.TestCase):
    def test_draft_chain_keeps_stage_order_private_probabilities_and_fallback(
        self,
    ):
        from oh_my_vllm.models.qwen38.draft_chain import can_chain
        from oh_my_vllm.models.qwen38.speculation import Speculation

        request = self._chain_request_factory()

        d = ChainDevice()
        cache = Cache(
            CacheDevice(),
            config(mamba_blocks=4, num_gpu_blocks=256, num_speculative_tokens=4),
        )
        events = []
        trace = SimpleNamespace(
            record=lambda *args, **kwargs: events.append((args, kwargs))
        )
        speculation = Speculation(d, None, cache, None, trace)
        with (
            patch("oh_my_vllm.models.qwen38.speculation.Draft", ChainDraft),
            patch("oh_my_vllm.models.qwen38.sampling.prepare", ChainSample),
        ):
            first = request()
            item = row(end=2, sampling="target")
            item.update(request_id=1, token_ids=first.history[:2])
            batch = cache.batch([item], [first], False)
            endpoints = {0: (np.full((1, 5120), 2, np.float32), None)}
            first.proposal_offset = 17
            speculation.propose(batch, endpoints, True)
            self.assertEqual(first.drafts, [3, 7, 15, 31])
            self.assertEqual(d.downloads, 1)
            self.assertEqual(first.proposal_offset, 17)
            self.assertEqual(
                events,
                [
                    (
                        ("draft",),
                        {"stage": stage, "requests": [{"id": 1, "position": position}]},
                    )
                    for stage, position in enumerate((2, 3, 4, 5))
                ],
            )
            self.assertTrue(first.drafts_greedy)
            self.assertIsNone(first.proposals)
            saved = first.drafts.copy()
            second = request()
            batch.requests = [second]
            endpoints[0] = (np.full((1, 5120), 4, np.float32), None)
            speculation.propose(batch, endpoints, True)
            self.assertEqual(second.drafts, [5, 11, 23, 47])
            self.assertEqual(d.downloads, 2)
            self.assertEqual(first.drafts, saved)
            self._check_stochastic_chain_probabilities(
                request, batch, d, speculation, endpoints
            )
            self._check_chain_fallbacks(
                request, batch, d, speculation, endpoints, can_chain, cache
            )
            speculation.close()

    def test_accept_packs_greedy_flag_into_torch_drafts_metadata(self):
        from oh_my_vllm.models.qwen38.speculation import Speculation

        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={},
            decode_bytes=lambda tokens: bytes(tokens),
        )
        options = {
            "max_tokens": 32,
            "min_tokens": 16,
            "stop": [],
            "sampling": {"temperature": 0, "repetition_guard": False},
        }
        first = Request(tokenizer, [4, 4], options)
        first.append([5])
        first.state_position = 2
        device = AcceptDevice()
        cache = Cache(
            CacheDevice(),
            config(mamba_blocks=4, num_gpu_blocks=256, num_speculative_tokens=4),
        )
        trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        speculation = Speculation(device, None, cache, None, trace)
        item = row(end=2, sampling="target")
        item.update(request_id=1, token_ids=first.history[:2])
        batch = cache.batch([item], [first], False)
        cache.complete_exports(batch)
        cache.commit(first, batch, 0, None)
        endpoints = {0: (np.full((1, 5120), 2, np.float32), None)}
        with (
            patch("oh_my_vllm.models.qwen38.speculation.Draft", ChainDraft),
            patch("oh_my_vllm.models.qwen38.sampling.prepare", ChainSample),
        ):
            speculation.propose(batch, endpoints, True)
        acceptance = PackedAcceptance(torch.zeros((1, 7), dtype=torch.int64))
        speculation.acceptance[1] = acceptance
        proposed = list(first.drafts)
        item = row(
            2,
            7,
            sampling="verify",
            deepest=2,
            contiguous=2,
            resume=reference(2, first.state_slot),
            exports=[
                {**reference(end, slot), "role": "private_endpoint"}
                for end, slot in zip(range(3, 8), range(6, 11), strict=True)
            ],
        )
        item.update(request_id=1, token_ids=[5, *proposed])
        item["sampling"]["drafts"] = len(proposed)
        batch = cache.batch([item], [first], True)
        sampler = SimpleNamespace(
            greedy=True, token_rows=None, select=lambda *args, **kwargs: (None, None)
        )
        speculation.accept(batch, None, None, sampler, False)
        self.assertIs(type(first.drafts_greedy), bool)
        packed = acceptance.copied["drafts"]
        self.assertIsInstance(packed, np.ndarray)
        self.assertEqual(packed.dtype, np.int64)
        self.assertEqual(
            packed.tolist(),
            [[*proposed, *([-1] * (4 - len(proposed))), 1]],
        )
        speculation.close()

    def test_stochastic_chain_requires_prefix_independent_masks(self):
        from oh_my_vllm.models.qwen38.draft_chain import can_chain

        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={},
        )
        request = Request(
            tokenizer,
            [4],
            {
                "max_tokens": 4096,
                "ignore_eos": True,
                "sampling": {"temperature": 1, "presence_penalty": 0.5},
            },
        )
        cache = Cache(
            CacheDevice(),
            config(mamba_blocks=4, num_gpu_blocks=256, num_speculative_tokens=4),
        )
        item = row(end=1, sampling="target")
        item["token_ids"] = request.history.copy()
        batch = cache.batch([item], [request], False)
        request.state_position = 1
        self.assertTrue(can_chain(batch, [0], True, cache.blocks))
        for history in (
            [1] * 60,
            [1] * 61,
            [1, 2] * 31,
            [1, 2] * 30 + [3],
            list(range(128)),
            list(range(23)) * 3,
        ):
            request.history = [4, *history]
            eligible = can_chain(batch, [0], True, cache.blocks)
            if history == [1] * 61 or history == [1, 2] * 31:
                self.assertFalse(eligible)
            if eligible:
                initial = request.mask()
                for length in range(4):
                    for prefix in product((1, 2, 3), repeat=length):
                        np.testing.assert_array_equal(
                            request.mask(
                                generated=request.generated + length, prefix=prefix
                            ),
                            initial,
                        )
        request.history = [4, *range(128)]
        self.assertTrue(can_chain(batch, [0], True, cache.blocks))
        request.minimum = request.generated + 2
        self.assertFalse(can_chain(batch, [0], True, cache.blocks))
        request.minimum = 0
        request.ignore_eos = False
        self.assertFalse(can_chain(batch, [0], True, cache.blocks))

    def _check_chain_fallbacks(
        self, request, batch, d, speculation, endpoints, can_chain, cache
    ):
        for update in (
            {"frequency_penalty": 0.2},
            {"repetition_penalty": 1.2},
        ):
            rejected = request()
            rejected.options["sampling"].update(update)
            batch.requests = [rejected]
            before = d.downloads
            speculation.propose(batch, endpoints, True)
            with self.subTest(update=update):
                self.assertEqual(d.downloads - before, 4)
        for field, value in (("thinking", True), ("minimum", 4), ("limit", 5)):
            rejected = request()
            setattr(rejected, field, value)
            batch.requests = [rejected]
            before = d.downloads
            speculation.propose(batch, endpoints, True)
            with self.subTest(field=field):
                self.assertGreater(d.downloads - before, 1)
        rejected = request()
        batch.requests = [rejected]
        before = d.downloads
        speculation.propose(batch, endpoints, False)
        self.assertEqual(d.downloads - before, 4)
        rejected.options["stop"] = ["x"]
        before = d.downloads
        speculation.propose(batch, endpoints, True)
        self.assertEqual(d.downloads - before, 4)
        rejected.options["stop"] = []
        rejected.constraint.matcher = object()
        self.assertFalse(can_chain(batch, [0], True, cache.blocks))
        rejected.constraint.matcher = None
        for table in ([], [0], [cache.blocks + 1]):
            batch.metadata = (
                *batch.metadata[:2],
                np.asarray([table], np.int32),
                *batch.metadata[3:],
            )
            with self.subTest(table=table):
                self.assertFalse(can_chain(batch, [0], True, cache.blocks))

    def _check_stochastic_chain_probabilities(
        self, request, batch, d, speculation, endpoints
    ):
        for seed, offset in ((7, 17), (19, 31)):
            stochastic = request(dict(temperature=0.7, presence_penalty=0.5, seed=seed))
            stochastic.proposal_offset = offset
            batch.requests = [stochastic]
            before = d.downloads
            speculation.propose(batch, endpoints, True)
            self.assertEqual(d.downloads - before, 1)
            self.assertEqual(stochastic.proposal_offset, offset + 4)
            self.assertFalse(stochastic.drafts_greedy)
            expected = np.zeros((4, 248320), np.float32)
            expected[np.arange(4), [5, 11, 23, 47]] = 1
            np.testing.assert_array_equal(stochastic.proposals, expected)
            endpoints[0] = (np.full((1, 5120), 2, np.float32), None)
            batch.requests = [request()]
            speculation.propose(batch, endpoints, True)
            np.testing.assert_array_equal(stochastic.proposals, expected)
            endpoints[0] = (np.full((1, 5120), 4, np.float32), None)

    def _chain_request_factory(self):
        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={},
            decode_bytes=lambda tokens: bytes(tokens),
        )
        options = {
            "max_tokens": 32,
            "min_tokens": 16,
            "stop": [],
            "sampling": {"temperature": 0, "repetition_guard": False},
        }

        def request(sampling=None):
            configured = deepcopy(options)
            configured["sampling"].update(sampling or {})
            value = Request(tokenizer, [4, 4], configured)
            value.append([5])
            value.state_position = 2
            return value

        return request

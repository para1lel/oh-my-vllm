import unittest
from copy import deepcopy
from types import SimpleNamespace

from .accounting import reference_seconds
from .qwen_fp8_math import (
    ba_precision,
    pass_resources,
    resident_allowance,
    triangular_inverse,
    wy_chunk,
)
from .qwen_fp8_roofline import lower_bound, target_components
from .qwen_fp8_schedule import canonical_passes, prefill_allocation
from .qwen_fp8_trace import (
    FORMAT,
    LAYOUT,
    VERSION,
    Blocked,
    digest,
)
from .test_qwen_fp8_contract import (
    GENERATION,
    GENERATION_SHA256,
    fixture_dispatch,
    wire_result,
    wire_row,
)
from .test_qwen_fp8_contract import (
    validate_fixture as validate,
)
from .workload import PROFILE, request_body


def state_for_schedule(prompt=2, output=2, mtp=False, parser_cut=False):
    history = list(range(10, 10 + prompt + output))
    first = dict(
        row=dict(
            forward_start=0,
            forward_end=prompt,
            token_ids=history[:prompt],
            sampling=dict(kind="target"),
        ),
        tokens=history[prompt : prompt + 1],
        result={},
    )
    steps = [first]
    generated = 1
    while generated < output:
        count = min(5 if mtp else 1, output - generated)
        start = prompt + generated - 1
        dispatch = max(count, 3 if parser_cut else count)
        inputs = history[start : min(len(history), start + dispatch)]
        inputs += [200] * (dispatch - len(inputs))
        steps.append(
            dict(
                row=dict(
                    forward_start=start,
                    forward_end=start + dispatch,
                    token_ids=inputs,
                    sampling=dict(kind="verify" if mtp else "target"),
                ),
                tokens=history[prompt + generated : prompt + generated + count],
                result={},
            )
        )
        generated += count
    return SimpleNamespace(
        manifest=dict(ordinal=0, input_tokens=prompt, output_tokens=output),
        history=history,
        accepted_end=prompt + output,
        steps=steps,
    )


def fixture_artifacts():
    sha = "a" * 64
    native = {
        name: dict(source_id=sha, source_fingerprint=sha, binary_id=sha)
        for name in [
            "device",
            "ffn",
            "fa_fp8",
            "gdn_radix",
            "mtp_fp8",
            "sample",
            "vision",
        ]
    }
    artifacts = dict(
        source=sha,
        binary=sha,
        native=native,
        checkpoint=dict(config=sha, index=sha, shard=sha),
        tokenizer=dict.fromkeys(
            ["tokenizer.json", "generation_config.json", "chat_template.jinja"], sha
        ),
        device=dict(
            name="B200",
            uuid="GPU-fixture",
            multiprocessor_count=148,
            l2_cache_bytes=126 * 1024**2,
        ),
        execution=dict(mode="fixture"),
        accounting=dict(version=VERSION),
    )
    artifacts["tokenizer"]["generation_config.json"] = GENERATION_SHA256
    return artifacts


def fixture_identity(artifacts):
    identity = dict(
        operation="identity",
        **artifacts,
        artifact_id=digest(artifacts),
        config=dict(
            tensor_parallel_size=1,
            max_num_seqs=4,
            max_num_batched_tokens=32768,
            max_model_len=262144,
            worker_epoch=1,
            block_size=128,
            num_speculative_tokens=0,
        ),
        cache_memory=dict(
            layout_id=LAYOUT,
            format_fingerprint=FORMAT,
            usable_pages=64,
            usable_checkpoints=1,
        ),
    )
    return identity


def fixture_events(identity):
    events = [identity]
    history = [10, 11, 12, 13]
    for step, (start, end, emitted) in enumerate(
        ((0, 2, [12]), (2, 3, [13]), (3, 4, [])), 1
    ):
        final = step == 3
        if final:
            events.append(
                dict(operation="accepted", requests=[dict(id=7, accepted_end=4)])
            )
        row = wire_row(
            start,
            end,
            history[start:end],
            "finalize" if final else "advance",
            "none" if final else "target",
        )
        events.append(
            dict(
                operation="accounting_call",
                worker_epoch=1,
                step_id=step,
                validated=True,
                scheduled=[row],
                outputs=[wire_result(row, emitted)],
                dispatches=fixture_dispatch(row, step),
            )
        )
    events.append(
        dict(
            operation="complete",
            requests=[
                dict(
                    id=7,
                    status="completed",
                    accepted_end=4,
                    execution=dict(worker_epoch=1, step_id=3, request_id=7, lease_id=1),
                    token_sha256=digest(history),
                )
            ],
        )
    )
    return events


def archived_fixture():
    body = request_body([dict(role="user", content="CPU fixture")], 2, 0)
    request = dict(
        ordinal=0,
        id=7,
        response_id="chatcmpl_fixture_7",
        input_tokens=2,
        output_tokens=2,
        seed=0,
        prompt_token_ids=[10, 11],
        body=body,
        body_sha256=digest(body),
        tools=dict(complete=0, partial=0),
    )
    artifacts = fixture_artifacts()
    manifest = dict(
        artifact_id=digest(artifacts),
        mode="ordinary",
        workload=PROFILE,
        requests=[request],
        sampling=dict(GENERATION),
        timer=dict(clock="perf_counter", started=1, finished=2),
    )
    events = fixture_events(fixture_identity(artifacts))
    responses = [
        dict(
            id=request["response_id"],
            usage=dict(
                prompt_tokens=2,
                completion_tokens=2,
                total_tokens=4,
                prompt_tokens_details=dict(cached_tokens=0),
            ),
            choices=[dict(finish_reason="length", message=dict(content="fixture"))],
        )
    ]
    return events, manifest, artifacts, responses


class KnownResourceFormulas(unittest.TestCase):
    def test_batch_shares_weights_and_new_KV_is_not_an_HBM_read(self):
        state = state_for_schedule()
        logical = canonical_passes({7: state}, "ordinary")[0]
        work, objects, _ = pass_resources(logical, target_components)
        logical["requests"].append(dict(logical["requests"][0], id=8))
        batched, shared, _ = pass_resources(logical, target_components)
        self.assertEqual(objects, shared)
        self.assertEqual(work["hbm_bytes"], 0)
        self.assertEqual(batched["fp8_flops"], 2 * work["fp8_flops"])
        self.assertIn("vocabulary", objects)

    def test_inverse_zero_diagonal_and_padded_rows_have_no_credit(self):
        self.assertEqual(triangular_inverse(1)["bf16_flops"], 0)
        self.assertEqual(triangular_inverse(2)["fp32_flops"], 0)
        self.assertEqual(triangular_inverse(3)["fp32_flops"], 96)
        self.assertEqual(triangular_inverse(8)["fp32_flops"], 5376)
        self.assertEqual(triangular_inverse(9)["bf16_flops"], 2688)
        self.assertLess(wy_chunk(0, 64)["bf16_flops"], wy_chunk(64, 64)["bf16_flops"])
        self.assertEqual(wy_chunk(0, 64)["bf16_flops"], 172515328)
        self.assertEqual(wy_chunk(0, 64)["fp32_flops"], 623616)
        with self.assertRaises(ValueError):
            triangular_inverse(65)

    def test_precision_is_selected_from_complete_modeled_shape(self):
        self.assertEqual(ba_precision(2, 1, True), "bf16")
        self.assertEqual(ba_precision(5, 5, True), "fp32")
        self.assertEqual(ba_precision(33, 17, False), "bf16")
        for shape in ((None, 1, True), (True, 1, False), (2, 3, True)):
            with self.assertRaises(ValueError):
                ba_precision(*shape)

    def test_tensor_precision_times_add_and_other_resources_take_max(self):
        work = dict(
            hbm_bytes=7.7e12,
            fp8_flops=4.5e15,
            bf16_flops=2.25e15,
            tf32_flops=0,
            fp32_flops=75e12,
            fp64_flops=0,
        )
        self.assertEqual(reference_seconds(work), 2)

    def test_resident_profile_needs_bound_recorded_hardware(self):
        self.assertEqual(
            resident_allowance(
                dict(
                    name="B200", multiprocessor_count=148, l2_cache_bytes=126 * 1024**2
                )
            ),
            1 << 30,
        )
        for device in (
            {},
            dict(name="B200", multiprocessor_count=200, l2_cache_bytes=1),
        ):
            with self.assertRaises(ValueError):
                resident_allowance(device)


class CanonicalUsefulSchedule(unittest.TestCase):
    def test_fair_allocations_use_native_budget_without_checkpoint_alignment(self):
        pending = [(i, 0, 32768) for i in range(4)]
        self.assertEqual(list(prefill_allocation(pending, 32768).values()), [8192] * 4)
        self.assertEqual(list(prefill_allocation(pending, 32764).values()), [8191] * 4)
        self.assertEqual(prefill_allocation([(0, 0, 2), (1, 0, 100)], 10), {0: 2, 1: 8})

    def test_exact_end_is_unique_including_saved_parser_cut(self):
        for mtp, cut in ((False, False), (True, False), (True, True)):
            state = state_for_schedule(output=3, mtp=mtp, parser_cut=cut)
            passes = canonical_passes({7: state}, "mtp4" if mtp else "ordinary")
            rows = [r for p in passes for r in p["requests"]]
            self.assertEqual(sum(r["end"] - r["start"] for r in rows), 5)
            self.assertEqual(sum(r["head_count"] for r in rows), 3)
            self.assertEqual(sum(r["context_count"] for r in rows), 4 if mtp else 0)
            closing = [p for p in passes if p["phase"] == "exact_end"]
            self.assertEqual(bool(closing), not cut)

    def test_physical_padding_replay_and_launch_metadata_cannot_add_credit(self):
        state = state_for_schedule(output=6, mtp=True)
        expected = canonical_passes({7: state}, "mtp4")
        state.replays = 100
        state.padding = 10000
        state.cow_bytes = 10**12
        self.assertEqual(canonical_passes({7: state}, "mtp4"), expected)


class StrictArchivedEvidence(unittest.TestCase):
    def test_forged_resume_export_valid_end_or_loaded_hash_is_rejected(self):
        for mutate in (
            lambda a: a[0][2]["scheduled"][0]["model_state"]["resume"].update(slot=99),
            lambda a: a[0][1]["outputs"][0]["validated_exports"][0].update(end=1),
            lambda a: a[0][-2]["outputs"][0]["committed_page_valid_ends"][0].update(
                valid_end=5
            ),
            lambda a: a[2]["native"]["fa_fp8"].update(binary_id="unknown"),
            lambda a: a[0][1]["scheduled"][0].update(prefill_token_ids=[11, 10]),
            lambda a: a[0][1]["scheduled"][0]["model_state"].pop("endpoint_source"),
            lambda a: a[0][1]["scheduled"][0]["model_state"].pop("private_checkpoints"),
            lambda a: a[0][1]["scheduled"][0]["model_state"].pop("page_copies"),
            lambda a: a[0][1]["scheduled"][0]["model_state"][
                "private_checkpoints"
            ].pop(),
            lambda a: a[0][1]["scheduled"][0]["model_state"]["exports"][0].update(
                version=9
            ),
            lambda a: a[0][1]["scheduled"][0]["model_state"].update(
                endpoint_source=dict(kind="cached_hidden", slot=99999, version=999)
            ),
        ):
            args = deepcopy(archived_fixture())
            mutate(args)
            with self.subTest(mutate=mutate), self.assertRaises((Blocked, ValueError)):
                validate(*args)

    def test_valid_wire_trace_closes_exact_accepted_state(self):
        args = archived_fixture()
        states = validate(*args)
        self.assertTrue(states[7].complete)
        self.assertEqual(states[7].history, [10, 11, 12, 13])

    def test_missing_current_trace_fields_and_identity_are_blocked(self):
        args = archived_fixture()
        args[0][1] = dict(operation="target", requests=[dict(id=7, start=0, count=2)])
        result = lower_bound(*args)
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["seconds"])
        self.assertIsNone(result["efficiency"])
        args = archived_fixture()
        args[0][0]["source"] = "changed"
        with self.assertRaises(Blocked):
            validate(*args)

    def test_parser_usage_and_finalization_must_match(self):
        for mutate in (
            lambda a: a[0][-1]["requests"][0].update(accepted_end=3),
            lambda a: a[0][-2]["outputs"][0].update(token_ids=[99]),
            lambda a: a[0][-2]["outputs"][0].update(computed_end=3),
            lambda a: a[3][0]["usage"].update(completion_tokens=1),
            lambda a: a[0][-1]["requests"][0].update(token_sha256="changed"),
            lambda a: a[0][1].update(validated=False),
        ):
            args = deepcopy(archived_fixture())
            mutate(args)
            with self.subTest(mutate=mutate), self.assertRaises((Blocked, ValueError)):
                validate(*args)

    def test_no_minimum_of_independent_prefixes_can_invent_checkpoint(self):
        args = archived_fixture()
        args[0][1]["scheduled"][0]["model_state"].update(
            checkpoint_hit_end=16384, kv_contiguous_end=12288
        )
        with self.assertRaises(Blocked):
            validate(*args)

    def test_saved_hidden_sampling_uses_forward_arithmetic_once(self):
        args = archived_fixture()
        expected = canonical_passes(validate(*args), "ordinary")
        call = args[0][2]
        call["scheduled"][0]["sampling"] = dict(kind="none")
        call["outputs"][0].update(token_ids=[], accepted_end=3)
        row = wire_row(3, 3, [], action="head_only")
        row["model_state"]["endpoint_source"] = dict(
            kind="saved_hidden", end=3, slot=2, version=1
        )
        head = dict(
            call,
            step_id=3,
            scheduled=[row],
            outputs=[wire_result(row, [13])],
            dispatches=[],
        )
        args[0].insert(3, head)
        args[0][-2]["step_id"] = args[0][-1]["requests"][0]["execution"]["step_id"] = 4
        args[0][-2]["dispatches"][0]["step_id"] = 4
        actual = canonical_passes(validate(*args), "ordinary")
        self.assertEqual(actual, expected)
        for field, value in (("slot", 99999), ("version", 999), ("end", 2)):
            broken = deepcopy(args)
            broken[0][3]["scheduled"][0]["model_state"]["endpoint_source"][field] = (
                value
            )
            with self.subTest(field=field), self.assertRaises(Blocked):
                validate(*broken)
        actual[1]["requests"][0].pop("ba_spans")
        with self.assertRaises((ValueError, KeyError)):
            pass_resources(actual[1], target_components)

    def test_target_cannot_sample_an_old_prompt_endpoint_after_generation(self):
        args = archived_fixture()
        row = wire_row(0, 2, [10, 11])
        result = wire_result(row, [13])
        result["accepted_end"] = 4
        args[0][2].update(
            scheduled=[row], outputs=[result], dispatches=fixture_dispatch(row, 2)
        )
        final = wire_row(0, 4, [10, 11, 12, 13], "finalize", "none")
        args[0][-2].update(
            scheduled=[final],
            outputs=[wire_result(final, [])],
            dispatches=fixture_dispatch(final, 3),
        )
        with self.assertRaisesRegex(Blocked, "prefill/replay"):
            validate(*args)

    def test_parser_cut_still_requires_finalize_and_allows_zero_row_export(self):
        args = archived_fixture()
        accepted = args[0].pop(3)
        row = args[0][3]["scheduled"][0]
        row.update(purpose="advance", sampling=dict(kind="target"))
        args[0][3]["outputs"] = [wire_result(row, [14])]
        args[0].insert(4, accepted)
        with self.assertRaisesRegex(Blocked, "finalize"):
            validate(*args)
        row = wire_row(4, 4, [], "finalize", "none", "export_only")
        row["model_state"]["endpoint_source"] = dict(
            kind="saved_hidden", end=4, slot=2, version=1
        )
        final = dict(
            args[0][3],
            step_id=4,
            scheduled=[row],
            outputs=[wire_result(row, [])],
            dispatches=[],
        )
        args[0].insert(5, final)
        args[0][-1]["requests"][0]["execution"]["step_id"] = 4
        self.assertTrue(validate(*args)[7].complete)

    def test_duplicate_unknown_ids_and_missing_final_success_fail(self):
        for mutate in (
            lambda a: a[0].pop(),
            lambda a: a[0].insert(2, deepcopy(a[0][1])),
            lambda a: a[0][1]["scheduled"][0].update(request_id=99),
            lambda a: a[1]["requests"][0].update(body_sha256="changed"),
            lambda a: a[3].append(deepcopy(a[3][0])),
        ):
            args = deepcopy(archived_fixture())
            mutate(args)
            with self.subTest(mutate=mutate), self.assertRaises((Blocked, ValueError)):
                validate(*args)

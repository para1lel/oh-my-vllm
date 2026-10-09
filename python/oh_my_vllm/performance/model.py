"""Canonical Qwen/DSpark equations bound to host-recorded effective shapes.

Parameter transfer nodes have no activation dependency. Shared-resource work
also constrains the bound. Intermediates have ideal on-chip reuse; persistent
incoming reads and final live writes use independent capacity credits.
"""

from collections import defaultdict

from .dag import Dag, Node


class Model:
    def __init__(self, hardware, weights, *, angle_seen=None):
        self.hardware = hardware
        self.weights = weights
        self.graph = Dag(hardware)
        self.reads = {}
        self.epoch = 0
        self.phase = None
        self.angles = {}
        self.angle_seen = set() if angle_seen is None else angle_seen
        self.relaxed_native = defaultdict(int)

    def rope_angles(self, family, positions, count):
        """Reuse position/frequency angles across layers and shared requests.

        Fixed frequencies can be prepared before request timing. Position
        multiplication and FP64 range reduction remain request-dependent.
        Unpublished FP64 rounding throughput has zero cost, a stated relaxation.
        """
        nodes = []
        for position in sorted(set(positions)):
            key = family, position, count
            if key not in self.angles:
                fresh = key not in self.angle_seen
                self.angle_seen.add(key)
                self.angles[key] = self.node(
                    family + f":angles:{position}",
                    simt_fp64=count * 4 * fresh,
                    sfu=count * 2 * fresh,
                    simt_fp32=count * 2 * fresh * (family == "dspark"),
                    convert_fp32_bf16=count * 2 * fresh * (family == "dspark"),
                )
            nodes.append(self.angles[key])
        return self.node(family + ":angle_lookup", tuple(nodes))

    def embedding(self, ids, *, markov=False, parents=()):
        """Union actual rows of the shared embedding, including draft aliases."""
        prefix = "dspark.markov_w1" if markov else "target.embedding"
        width = 256 if markov else 5120
        rows = tuple(
            self.read(f"{prefix}:row:{token}", width * 2, parents)
            for token in sorted(set(ids))
        )
        return self.node(prefix + ":lookup", tuple(dict.fromkeys((*rows, *parents))))

    def node(self, name, parents=(), **work):
        # A valid fused implementation can pair BF16 output roundings. The
        # public packed conversion/resource mapping is unresolved; retain its
        # value count as recognized work with zero normative service time.
        rounding = work.pop("convert_fp32_bf16", 0)
        self.relaxed_native["fp32_to_bf16_rounding_values"] += rounding
        return self.graph.add(Node(name, tuple(parents), work))

    def read(self, name, size=None, parents=()):
        if size is None:
            size = self.weights[name]["bytes"]
        key = (self.epoch, name)
        if key in self.reads:
            node, old_size = self.reads[key]
            if old_size != size:
                raise ValueError("inconsistent buffer range identity")
            return node
        node = self.graph.add(Node("read:" + name, tuple(parents), external_bytes=size))
        self.reads[key] = node, size
        return node

    def norm(self, name, parent, rows, width, *, residual=False, gate=False):
        weight = self.read(name)
        if name.startswith("dspark."):
            self.relaxed_native["bf16_multiply"] += rows * width
            self.relaxed_native["fp32_to_bf16_packed_pairs"] += rows * width // 2
            return self.node(
                name + ":normalize",
                (parent, weight),
                simt_fp32=rows * (3 * width + 1),
                sfu=rows,
                convert_bf16_fp32=rows * width,
            )
        packed_residual = residual and width == 5120
        if packed_residual:
            # The public packed-BF16 resource mapping is unresolved. Recognize the work,
            # give it zero time, and avoid charging nonexistent FP32 casts.
            self.relaxed_native["bf16_add"] += rows * width
        floating_residual = residual and not packed_residual
        return self.node(
            name + ":normalize",
            (parent, weight),
            simt_fp32=rows
            * (4 * width + 1 + width * int(floating_residual) + 4 * width * int(gate)),
            sfu=rows * (1 + 2 * width * int(gate)),
            convert_bf16_fp32=rows * width * (1 + int(floating_residual) + int(gate)),
            convert_fp32_bf16=rows * width * (1 + int(floating_residual)),
        )

    def linear(self, name, parent, rows):
        # Packed project Linear has a .weight key. DSpark has raw matrices.
        weight_name = name + ".weight" if name + ".weight" in self.weights else name
        spec = self.weights[weight_name]
        if len(spec["shape"]) != 2 or rows <= 0:
            raise ValueError("unsupported matrix shape")
        n, k = spec["shape"]
        dtype = spec["dtype"]
        weights = [self.read(weight_name)]
        activation = parent
        if dtype == "torch.float8_e4m3fn":
            scale = name + ".scale"
            weights.append(self.read(scale))
            activation = self.node(
                name + ":quantize",
                (parent,),
                # Direct scaled rounding needs one multiplication per value
                # and one scale multiplication per group. Refinement FMAs and
                # explicit clips are implementation overhead, not a floor.
                simt_fp32=rows * k + rows * (k // 128),
                sfu=rows * (k // 128),
                convert_bf16_fp32=rows * k,
            )
            self.relaxed_native["fp32_to_fp8"] += rows * k
            # REDUX can perform several logical comparisons in one instruction.
            # Its published throughput/resource mapping is unresolved.
            self.relaxed_native["fp8_group_max_comparisons"] += rows * k
            # One product of the two scales and one scaled accumulation per
            # output and K group; padding and repeated CTA loads add no work.
            groups = k // 128
            scale_work = rows * n * (2 * groups - 1) + rows * (n // 128) * groups
            resource = "tensor_fp8"
        elif dtype == "torch.bfloat16":
            resource = "tensor_bf16"
            scale_work = 0
        else:
            raise ValueError(f"unsupported matrix precision: {dtype}")
        return self.node(
            name + ":matmul",
            (activation, *weights),
            **{resource: 2 * rows * n * k, "simt_fp32": scale_work},
        )

    def attention(
        self,
        name,
        parent,
        queries,
        contexts,
        *,
        heads,
        kv_heads,
        dim,
        tables,
        incoming_contexts=None,
    ):
        if len(queries) != len(contexts):
            raise ValueError("attention sequence metadata mismatch")
        pairs = sum(
            q * c + q * (q + 1) // 2 for q, c in zip(queries, contexts, strict=True)
        )
        # Each retained incoming KV range is read once. New K/V can stay on chip.
        cached = self.read(
            name + ":incoming_kv",
            unique_context_tokens(
                contexts if incoming_contexts is None else incoming_contexts, tables
            )
            * 2
            * kv_heads
            * dim
            * 2,
        )
        qk = self.node(
            name + ":qk", (parent, cached), tensor_bf16=2 * heads * dim * pairs
        )
        softmax = self.node(
            name + ":softmax", (qk,), simt_fp32=3 * heads * pairs, sfu=heads * pairs
        )
        self.relaxed_native["attention_max_comparisons"] += heads * pairs
        pv = self.node(name + ":pv", (softmax,), tensor_bf16=2 * heads * dim * pairs)
        # Unnormalized probabilities can enter PV. Divide only the output
        # vectors, instead of adding a normalization multiply to every pair.
        return self.node(
            name + ":normalize_output",
            (pv,),
            simt_fp32=heads * sum(queries) * dim,
            sfu=heads * sum(queries),
        )

    def target_layer(self, prefix, parent, requests, *, state_bytes, mtp=False):
        rows = sum(r["query"] for r in requests)
        queries = [r["query"] for r in requests]
        contexts = [r["context"] for r in requests]
        normalized = self.norm(
            prefix + ".input_norm",
            parent,
            rows,
            5120,
            residual=not (mtp or prefix == "target.layers.0"),
        )
        if prefix + ".qkv.weight" in self.weights:
            projected = self.linear(prefix + ".qkv", normalized, rows)
            angles = self.rope_angles(
                "target_rope",
                [
                    position
                    for r in requests
                    for position in range(
                        r.get("position_start", r["context"]),
                        r.get("position_start", r["context"]) + r["query"],
                    )
                ],
                32,
            )
            prepared = self.node(
                prefix + ":norm_rope",
                (
                    projected,
                    self.read(prefix + ".q_norm"),
                    self.read(prefix + ".k_norm"),
                    angles,
                ),
                simt_fp32=rows * 28 * (4 * 256 + 1 + 6 * 32),
                sfu=rows * 28,
            )
            attended = self.attention(
                prefix,
                prepared,
                queries,
                contexts,
                heads=24,
                kv_heads=4,
                dim=256,
                tables=[r["fa_block_table"] for r in requests],
                incoming_contexts=[
                    r.get("incoming_context", r["context"]) for r in requests
                ],
            )
            gated = self.node(
                prefix + ":output_gate",
                (attended, projected),
                # Positive sigmoid permits reciprocal directly. The final
                # BF16 product stays in the recognized relaxed operation set.
                simt_fp32=rows * 6144 * 2,
                sfu=rows * 6144 * 2,
            )
            self.relaxed_native["bf16_output_gate_product"] += rows * 6144
            branch = self.linear(prefix + ".out", gated, rows)
        else:
            projected = self.linear(prefix + ".qkvz", normalized, rows)
            conv_history = self.read(
                prefix + ":conv_history",
                len({r["state_source"] for r in requests if r["state_source"] > 0})
                * 3
                * 10240
                * 2,
            )
            convolved = self.node(
                prefix + ":convolution",
                (projected, self.read(prefix + ".conv"), conv_history),
                simt_fp32=rows * 10240 * 8,
                sfu=rows * 10240 * 2,
            )
            gates = self.linear(prefix + ".ba", normalized, rows)
            gates = self.node(
                prefix + ":delta_gates",
                (gates, self.read(prefix + ".a_log"), self.read(prefix + ".dt_bias")),
                # a > 20 can bypass softplus exp/log. Precompute -exp(a_log).
                # Positive sigmoid needs only exp scaling and denominator add.
                simt_fp32=rows * 48 * 4,
                # The softplus branch can bypass exp/log for a > 20. Without
                # observed branch counts, only mandatory sigmoid SFU is charged.
                sfu=rows * 48 * 2,
            )
            qk = self.node(
                prefix + ":qk_normalize",
                (convolved,),
                # Prefill L2 normalization has no learned gamma. Recurrent
                # normalization also scales the sixteen Q heads by 1/sqrt(128).
                simt_fp32=rows * 4096 * 3
                + sum(r["query"] for r in requests if not r.get("prefill", False)) * 16,
                sfu=rows * 32,
            )
            state = self.read(
                prefix + ":incoming_state",
                len({r["state_source"] for r in requests if r["state_source"] > 0})
                * 48
                * 128
                * 128
                * state_bytes,
            )
            prefill_rows = sum(r["query"] for r in requests if r.get("prefill", False))
            recurrent_rows = rows - prefill_rows
            prefill = [r for r in requests if r.get("prefill", False)]
            recurrence = self.node(
                prefix + ":recurrent",
                (qk, gates, state),
                simt_fp32=recurrent_rows * 48 * (7 * 128 * 128 + 2 * 128),
                sfu=recurrent_rows * 48,
            )
            tails = [recurrence]
            for sequence, request in enumerate(prefill):
                previous = state
                for begin in range(0, request["query"], 64):
                    n = min(64, request["query"] - begin)
                    label = f"{prefix}:prefill:{sequence}:{begin}"
                    # Shared raw QK/KK scores use16 key heads; gates and
                    # recurrent state use48 value heads.
                    scan = sum(
                        max(0, min(32, n - start) - offset)
                        for start in range(0, n, 32)
                        for offset in (1, 2, 4, 8, 16)
                    ) + max(0, n - 32)
                    transfers = n * (n - 1) // 2
                    gamma = self.node(
                        label + ":gate_prefix",
                        (gates,),
                        simt_fp32=48 * (n + scan),
                        sfu=48 * n * 3,
                        shuffle=48 * scan,
                    )
                    transfer = self.node(
                        label + ":gate_transfers",
                        (gamma,),
                        simt_fp32=48 * transfers,
                        sfu=48 * transfers,
                    )
                    kk = self.node(
                        label + ":kk", (qk,), tensor_bf16=16 * 128 * n * (n - 1)
                    )
                    matrix = self.node(
                        label + ":beta_scaled_matrix",
                        (kk, transfer, gates),
                        simt_fp32=48 * transfers * 2,
                    )
                    inverse_simt = sum(
                        2 * m * (m - 1) * (m - 2) // 6
                        for m in (min(8, n - o) for o in range(0, n, 8))
                    )
                    inverse_tensor = 2 * n * (n - 1) * (n - 2) // 6 - inverse_simt
                    inverse = self.node(
                        label + ":triangular_inverse",
                        (matrix,),
                        simt_fp32=48 * inverse_simt,
                        tensor_bf16=48 * inverse_tensor,
                    )
                    ks = self.node(
                        label + ":ks",
                        (qk, previous),
                        tensor_bf16=48 * 2 * n * 128 * 128,
                    )
                    residual = self.node(
                        label + ":residual", (ks, gates), simt_fp32=48 * n * 128 * 2
                    )
                    updates = self.node(
                        label + ":updates",
                        (
                            self.node(
                                label + ":inverse_beta_columns",
                                (inverse, gates),
                                simt_fp32=48 * transfers,
                            ),
                            residual,
                        ),
                        tensor_bf16=48 * 128 * n * (n + 1),
                    )
                    qk_scores = self.node(
                        label + ":qk", (qk,), tensor_bf16=16 * 128 * n * (n + 1)
                    )
                    qs = self.node(
                        label + ":qs",
                        (qk, previous),
                        tensor_bf16=48 * 2 * n * 128 * 128,
                    )
                    output = self.node(
                        label + ":output",
                        (
                            qs,
                            self.node(
                                label + ":output_score_transfers",
                                (qk_scores, transfer),
                                simt_fp32=48 * transfers,
                            ),
                            updates,
                            gates,
                        ),
                        tensor_bf16=48 * 128 * n * (n + 1),
                        simt_fp32=48 * n * 128 * 3,
                    )
                    decayed_updates = self.node(
                        label + ":decayed_updates",
                        (updates, transfer),
                        simt_fp32=48 * n * 128,
                        convert_fp32_bf16=48 * n * 128,
                    )
                    update_state = self.node(
                        label + ":kv",
                        (qk, decayed_updates),
                        tensor_bf16=48 * 2 * n * 128 * 128,
                    )
                    previous = self.node(
                        label + ":state",
                        (previous, update_state, gates),
                        simt_fp32=48 * 128 * 128 * 2,
                    )
                    tails.append(output)
                tails.append(previous)
            delta = self.node(prefix + ":delta", tuple(tails))
            gated = self.norm(prefix + ".gate_norm", delta, rows * 48, 128, gate=True)
            branch = self.linear(prefix + ".out", gated, rows)
        post = self.norm(prefix + ".post_norm", branch, rows, 5120, residual=True)
        packed = self.linear(prefix + ".gate_up", post, rows)
        self.relaxed_native["bf16_multiply"] += rows * 17408
        self.relaxed_native["fp32_to_bf16_packed_pairs"] += rows * 17408 // 2
        activated = self.node(
            prefix + ":silu",
            (packed,),
            simt_fp32=rows * 17408 * 3,
            sfu=rows * 17408 * 2,
        )
        down = self.linear(prefix + ".down", activated, rows)
        return self.node(prefix + ":residual_tuple", (parent, branch, down))

    def target(self, requests, state_dtype):
        if state_dtype not in ("torch.float32", "torch.bfloat16"):
            raise ValueError("unknown recurrent state precision")
        self.target_requests = requests
        rows = sum(r["query"] for r in requests)
        unique_ids = set().union(*(set(r["unique_token_ids"]) for r in requests))
        embedded = self.embedding(unique_ids)
        parent = embedded
        taps = []
        for layer in range(64):
            parent = self.target_layer(
                f"target.layers.{layer}",
                parent,
                requests,
                state_bytes=4 if state_dtype == "torch.float32" else 2,
            )
            if "dspark.fc" in self.weights and layer in (5, 19, 33, 47, 61):
                self.relaxed_native["bf16_add"] += rows * 5120
                taps.append(self.node(f"target:feature:{layer}", (parent,)))
        if taps:
            self.features = self.node("target:features", tuple(taps))
        parent = self.norm("target.norm", parent, rows, 5120, residual=True)
        samples = sum(r["sample_rows"] for r in requests)
        if samples:
            logits = self.linear("target.head", parent, samples)
            parent = self.node(
                "target:greedy", (logits,), simt_fp32=samples * (248320 - 1)
            )
        return parent

    def bounds(self, endpoint):
        path = self.graph.relaxed_path(endpoint)

        pending, seen = [endpoint], set()
        while pending:
            index = pending.pop()
            if index not in seen:
                seen.add(index)
                pending.extend(self.graph.nodes[index].parents)
        counts = {}
        for index in seen:
            for resource, work in self.graph.nodes[index].work.items():
                counts[resource] = counts.get(resource, 0) + work
        return {
            "critical_path_s": path,
            "work": counts,
            "relaxed_native_operations": dict(self.relaxed_native),
            "incoming_ranges": {
                f"{epoch}:{name}": size
                for (epoch, name), (node, size) in self.reads.items()
                if node in seen
            },
        }

    def dspark_norm_rope(self, name, parents, rows, heads, angles):
        self.relaxed_native["bf16_multiply"] += rows * heads * 128 * 3
        self.relaxed_native["bf16_add"] += rows * heads * 128
        self.relaxed_native["fp32_to_bf16_packed_pairs"] += rows * heads * 64
        return self.node(
            name,
            (*parents, angles),
            simt_fp32=rows * heads * (3 * 128 + 1),
            sfu=rows * heads,
            convert_bf16_fp32=rows * heads * 128,
        )

    def dspark_inject(self, operation, parent):
        rows = operation["rows"]
        angles = self.rope_angles(
            "dspark",
            [
                position
                for start, count in zip(
                    operation["contexts"], operation["queries"], strict=True
                )
                for position in range(start, start + count)
            ],
            64,
        )
        projected = self.linear("dspark.fc", parent, rows)
        context = self.norm("dspark.hidden_norm", projected, rows, 5120)
        tails = []
        for layer in range(5):
            prefix = f"dspark.layers.{layer}"
            k = self.linear(prefix + ".k", context, rows)
            v = self.linear(prefix + ".v", context, rows)
            prepared = self.dspark_norm_rope(
                prefix + ":inject_rope",
                (k, self.read(prefix + ".k_norm")),
                rows,
                8,
                angles,
            )
            tails.append(self.node(prefix + ":append", (prepared, v)))
        return self.node("dspark:context_ready", tuple(tails))

    def dspark_backbone(self, operation, parent):
        contexts = operation["contexts"]
        batch, count = len(contexts), 7
        rows = batch * count
        angles = self.rope_angles(
            "dspark",
            [p for context in contexts for p in range(context, context + count)],
            64,
        )
        embedded = self.embedding(operation["anchors"])
        hidden = self.node("dspark:anchors", (embedded, parent))
        for layer in range(5):
            prefix = f"dspark.layers.{layer}"
            normalized = self.norm(prefix + ".input_norm", hidden, rows, 5120)
            q = self.linear(prefix + ".q", normalized, rows)
            k = self.linear(prefix + ".k", normalized, rows)
            v = self.linear(prefix + ".v", normalized, rows)
            prepared = self.dspark_norm_rope(
                prefix + ":norm_rope",
                (q, k, self.read(prefix + ".q_norm"), self.read(prefix + ".k_norm")),
                rows,
                40,
                angles,
            )
            cached = self.read(
                prefix + ":context_kv",
                unique_context_tokens(
                    operation["incoming_contexts"], operation["tables"]
                )
                * 2
                * 8
                * 128
                * 2,
            )
            pairs = sum(count * (context + count) for context in contexts)
            qk = self.node(
                prefix + ":parallel_qk",
                (prepared, cached),
                tensor_bf16=2 * 32 * 128 * pairs,
            )
            softmax = self.node(
                prefix + ":softmax", (qk,), simt_fp32=3 * 32 * pairs, sfu=32 * pairs
            )
            self.relaxed_native["attention_max_comparisons"] += 32 * pairs
            attended = self.node(
                prefix + ":parallel_pv", (softmax, v), tensor_bf16=2 * 32 * 128 * pairs
            )
            attended = self.node(
                prefix + ":normalize_output",
                (attended,),
                simt_fp32=32 * rows * 128,
                sfu=32 * rows,
            )
            out = self.linear(prefix + ".o", attended, rows)
            self.relaxed_native["bf16_add"] += rows * 5120
            residual = self.node(prefix + ":attention_residual", (hidden, out))
            normalized = self.norm(prefix + ".post_norm", residual, rows, 5120)
            gate = self.linear(prefix + ".gate", normalized, rows)
            up = self.linear(prefix + ".up", normalized, rows)
            self.relaxed_native["bf16_multiply"] += rows * 17408
            self.relaxed_native["fp32_to_bf16_packed_pairs"] += rows * 17408 // 2
            activated = self.node(
                prefix + ":silu",
                (gate, up),
                simt_fp32=rows * 17408 * 3,
                sfu=rows * 17408 * 2,
            )
            down = self.linear(prefix + ".down", activated, rows)
            self.relaxed_native["bf16_add"] += rows * 5120
            hidden = self.node(prefix + ":mlp_residual", (residual, down))
        hidden = self.norm("dspark.norm", hidden, rows, 5120)
        base = self.linear("target.head", hidden, rows)
        # The conditional Markov chain has seven genuine token dependencies.
        previous = base
        for position in range(7):
            previous_ids = (
                operation["anchors"]
                if position == 0
                else [row[position - 1] for row in operation["candidates"]]
            )
            latent = self.embedding(previous_ids, markov=True, parents=(previous,))
            markov = self.linear("dspark.markov_w2", latent, batch)
            self.relaxed_native["bf16_add"] += batch * 248320
            logits = self.node(
                f"dspark:markov_add:{position}",
                (markov, base, previous),
            )
            confidence_input = self.node(
                f"dspark:confidence_input:{position}", (hidden, latent)
            )
            confidence = self.linear(
                "dspark.confidence_weight", confidence_input, batch
            )
            confidence = self.node(
                f"dspark:confidence:{position}",
                (confidence, latent),
                simt_fp32=batch * 3,
                sfu=batch,
            )
            previous = self.node(
                f"dspark:greedy:{position}",
                (logits, confidence),
                simt_fp32=batch * (248320 - 1),
            )
        return previous

    def mtp_forward(self, operation, parent):
        """One true draft epoch; parameter traffic has a new retention boundary."""
        self.epoch += 1
        requests = [
            {
                "query": q,
                "context": max(0, c - 1),
                "position_start": c,
                "prefill": q > 1,
                "state_source": 0,
                "fa_block_table": table,
                "incoming_context": incoming,
            }
            for q, c, table, incoming in zip(
                operation["queries"],
                operation["contexts"],
                operation["tables"],
                operation.get(
                    "incoming_contexts", [max(0, c - 1) for c in operation["contexts"]]
                ),
                strict=True,
            )
        ]
        rows = sum(operation["queries"])
        if self.phase == "prefill":
            for request in requests:
                request["incoming_context"] = 0
        embedded = self.embedding(operation["unique_token_ids"])
        embedded = self.norm("target.mtp_embedding_norm", embedded, rows, 5120)
        normalized = self.norm("target.mtp_hidden_norm", parent, rows, 5120)
        joined = self.node("mtp:joined", (embedded, normalized))
        projected = self.linear("target.mtp_fc", joined, rows)
        result = self.target_layer(
            "target.mtp", projected, requests, state_bytes=2, mtp=True
        )
        return self.norm("target.mtp_norm", result, rows, 5120, residual=True)

    def mtp_logits(self, parent, rows):
        logits = self.linear("target.head", parent, rows)
        return self.node("mtp:greedy", (logits,), simt_fp32=rows * (248320 - 1))

    def proposal_chain(self, operation, parent):
        """The first head follows committed MTP forward; three updates follow."""
        rows = operation["rows"]
        parent = self.mtp_logits(parent, rows)
        for index in range(1, 4):
            parent = self.mtp_forward(
                {
                    "queries": [1] * rows,
                    "contexts": [c + index for c in operation["contexts"]],
                    "unique_token_ids": [row[index - 1] for row in operation["tokens"]],
                    "tables": operation["tables"],
                },
                parent,
            )
            parent = self.mtp_logits(parent, rows)
        return parent


def unique_context_tokens(contexts, tables):
    """Union physical page ranges; shared prefix reads count once per layer."""
    if len(contexts) != len(tables):
        raise ValueError("cache table metadata mismatch")
    pages = {}
    for count, table in zip(contexts, tables, strict=True):
        if type(count) is not int or count < 0 or len(table) * 784 < count:
            raise ValueError("invalid effective cache length")
        for index in range((count + 783) // 784):
            page = table[index]
            if type(page) is not int or page <= 0:
                raise ValueError("invalid physical page")
            pages[page] = max(pages.get(page, 0), min(784, count - index * 784))
    return sum(pages.values())

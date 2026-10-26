# DeepSeek-V4.1-Flash Model and Checkpoint Contract

Audit date: 2026-10-26. Scope: read-only inspection of the released model directory and the current `oh-my-vllm` contract. This report freezes observed facts. It does not define an adapter. `model-root` means the supplied local model directory. `repo-root` means this checkout.

## Facts

### Identity and files

- Model identity: `DeepseekV41ForCausalLM`, model type `deepseek_v41`, Transformers version `5.6.0`. The public config uses `bfloat16` as the base dtype.
- Model config SHA256: `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.
- Inference config SHA256: `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809`.
- The checkpoint has 48 safetensors files. The index has `96,085` tensor names and declares `510,286,023,000` data bytes. A read-only header scan matched all index names and declared bytes.
- Header dtype totals are BF16 `3,952,883,712` bytes, F32 `169,229,128`, FP8 E4M3 `204,015,223,296`, E8M0 scale `23,563,015,184`, and I8 `278,585,671,680`. Full SHA256 digests for the 48 large shards were not computed.
- Index SHA256: `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.
- The index places vision and aligner tensors in shard 1.
- It places the shared embedding in shard 2 and backbone layers 0–39 in shards 3–42.
- It places the output norm and head in shard 43.
- It places DSpark MTP stages 0–2 in shards 44–46 and Engram tables in shards 47–48.

Evidence: [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) and [index](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/model.safetensors.index.json). Local: `model-root/config.json:2-7`, `model-root/model.safetensors.index.json`, `metadata.total_size`, `weight_map`, and all `model-00001-of-00048.safetensors` through `model-00048-of-00048.safetensors`.

### Shape and layer structure

- Vocabulary size is `129,280`. Hidden size is `5,120`. The backbone has 40 layers, 64 query heads, one key/value latent head, head dimension `512`, and MoE intermediate size `2,304`.
- Query RoPE uses `64` dimensions. Query low-rank size is `1,280`. Output projection uses 8 groups and low-rank size `1,024`.
- RMSNorm epsilon is `1e-20`. SwiGLU limit is `10.0`. Word embeddings are untied.
- Vision has 32 layers, hidden size `1,024`, 16 heads, intermediate size `2,816`, patch size `14`, downsample ratio `3`, and at most `1,024` language-model image tokens. The aligner maps `9,216` inputs to `5,120`.
- The public quantization config says dynamic FP8, weight blocks `[32, 32]`, E8M0 scales, and FP4 experts.
- The inference config restates this as `dtype: "fp8"` and `expert_dtype: "fp4"`, and uses `compress_rope_theta: 160000`.
- It has 43 compression entries: 40 backbone entries followed by three zero entries for DSpark. Backbone ratios are `0, 0`, then ratio `2` for layers 2–19, then ratio `1` for layers 20–39.

Evidence: [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json), [inference config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/config.json), and [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py). Local: `model-root/config.json:22-169`, `model-root/inference/config.json:1-65`, `model-root/inference/model.py:44-136,1183-1223`, `model-root/model.safetensors.index.json#/weight_map`, `vision.*`, `aligner.*`, `embed.weight`, and `head.weight`.

### Attention, sparse lookup, and precision

- All layers have a causal sliding-window cache of 128 tokens.
- KV source layers are 2, 8, 14, and 20. They create shared compressed KV. Index source layers are 2, 8, 14, 20, 24, 28, 32, and 36.
- Ratio-2 compressors pool groups with a learned softmax gate. Ratio-1 compressors use one projected token. The compressed KV path uses FP4 E2M1 with one E4M3 scale per 16 channels. The indexer uses FP4 with E8M0 scales over 32 channels.
- The indexer keeps up to 512 compressed positions. Layer 20 selects up to `2,048` candidate blocks of 8 positions. Later indexers score inside that candidate mask.
- The README names the three CSA2 modes Full, Reindex, and Reuse. The source expresses the same sharing through `compress_ratios`, `kv_source_layers`, `index_source_layers`, and shared runtime state. It does not make those mode names an adapter API.
- The source code shares compressed KV, index keys, and Top-K indices through process-global runtime slots. It does not implement a separate CED encoder/decoder cache path.
- Dense weights use FP8 E4M3 with E8M0 scales. The source sets 32 as the FP8 weight and activation block size. Routed expert weights are stored as I8 plus scales in the Hugging Face checkpoint and convert to packed FP4 E2M1 by default.
- The source converts `wo_a` to BF16 before the grouped output einsum. It quantizes KV and compressed KV, then writes the dequantized result into the reference cache buffers.

Evidence: [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py), [kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py), and [converter](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/convert.py). Local: `model-root/config.json:43-127`, `model-root/inference/config.json:26-52`, `model-root/inference/model.py:429-610,613-790`, `model-root/inference/kernel.py:14-20,98-204,310-403,477-591`, `model-root/inference/convert.py:13-53,154-182`, `layers.2.attn.compressor.*`, `layers.24.attn.indexer.*`, and `layers.0.attn.*`.

### MoE and Hyper-Connections

- Each backbone layer has 384 routed experts, 6 active experts, and one shared expert. The routed gate uses `sqrtsoftplus`, normalizes Top-K weights, and applies route scale `1.5`. Vision tokens have a separate gate bias.
- Each DSpark MTP layer has 128 routed experts and 3 active experts. It also has one shared expert.
- Hyper-Connections expands the residual stream to 4 copies. Each attention and FFN sublayer computes pre, post, and combination coefficients. The combination matrix uses 20 Sinkhorn iterations and epsilon `1e-6`.
- Layer weights include `hc_attn_fn`, `hc_attn_base`, `hc_attn_scale`, `hc_ffn_fn`, `hc_ffn_base`, and `hc_ffn_scale`. For the released shape, each `hc_*_fn` has shape `[24, 20480]`.

Evidence: [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) and [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py). Local: `model-root/config.json:52-57,128-130,145-155`, `model-root/inference/model.py:792-904,907-994`, `layers.0.ffn.gate.*`, `layers.0.ffn.experts.*`, `layers.0.hc_*`, and `mtp.0.ffn.*`.

### Engram

- Engram runs at layers 1 and 14. It uses maximum n-gram size 4, 8 heads, head dimension 256, tokenizer vocabulary input size `16,000,000`, and compressed tokenizer vocabulary size `99,092`.
- Table rows are `384,006,168` and `384,016,682`. The two FP8 table weights are `98,305,579,008` and `98,308,270,592` bytes. Their E8M0 scale tensors are `3,072,049,344` and `3,072,133,456` bytes.
- All 12 Engram tensors occupy `203,073,076,240` data bytes.
- The total includes `196,928,422,400` FP8 E4M3 bytes, `6,144,490,000` E8M0 bytes, and `163,840` BF16 bytes.
- Exact index names and header metadata are:

| Index name | Shard | dtype | shape | data bytes |
|---|---|---|---|---:|
| `layers.1.engram.embed.weight` / `.scale` | 47 | F8_E4M3 / F8_E8M0 | `[384006168,256]` / `[384006168,8]` | `98,305,579,008` / `3,072,049,344` |
| `layers.1.engram.wkv.weight` / `.scale` | 47 | F8_E4M3 / F8_E8M0 | `[25600,6144]` / `[800,192]` | `157,286,400` / `153,600` |
| `layers.1.engram.q_weight` / `.k_weight` | 47 | BF16 / BF16 | `[4,5120]` / `[4,5120]` | `40,960` / `40,960` |
| `layers.14.engram.embed.weight` / `.scale` | 48 | F8_E4M3 / F8_E8M0 | `[384016682,256]` / `[384016682,8]` | `98,308,270,592` / `3,072,133,456` |
| `layers.14.engram.wkv.weight` / `.scale` | 48 | F8_E4M3 / F8_E8M0 | `[25600,6144]` / `[800,192]` | `157,286,400` / `153,600` |
| `layers.14.engram.q_weight` / `.k_weight` | 48 | BF16 / BF16 | `[4,5120]` / `[4,5120]` | `40,960` / `40,960` |

- `NgramHashState` builds a normalized token map from the tokenizer and asserts the result is exactly `99,092` entries. Image spans are excluded from n-grams. `ParallelEngramEmbedding` row-shards tables by world size and combines lookup results with all-reduce.
- `ParallelEngramEmbedding` loads local FP8 rows and E8M0 scales, dequantizes them to BF16, and all-reduces lookup values.
- The reference loader creates these tables as model parameters on the CUDA device. It defines no pinned host-memory residency or transfer contract.
- The converter row-shards Engram rows with ceiling division. It pads weight rows with zero and scale rows with one. It writes `model{rank}-mp{world_size}.safetensors`.
- `generate.py` loads that exact per-rank file after the model builds the same tokenizer-derived hash map.

Evidence: [Engram source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/engram.py), [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py), and [converter](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/convert.py). Local: `model-root/config.json:131-144`, `model-root/inference/engram.py:17-61,87-184`, `model-root/inference/model.py:296-365`, `model-root/inference/convert.py:135-141`, `model-root/model.safetensors.index.json#/weight_map/layers.1.engram.*`, `#/weight_map/layers.14.engram.*`, `model-00047-of-00048.safetensors`, and `model-00048-of-00048.safetensors`.

### DSpark

- DSpark has block size 5 and noise token ID `128799`.
- It targets layers 37–39, uses Markov rank 256, and has three MTP layers.
- It has 128 routed experts and activates 3 experts.
- Stage 0 has `mtp.0.main_proj` and `main_norm`. Stage 2 has `markov_head.embed`, `markov_head.head`, `norm`, and `confidence_head.proj`. MTP embedding and output head are tied to the backbone and are skipped by the converter.
- MTP data is in shards 44–46.
- It contributes `2,652,643,800`, `2,573,913,560`, and `2,706,317,272` bytes for stages 0–2.
- Examples are `mtp.0.main_proj.weight` FP8 E4M3 `[5120,15360]` and its E8M0 scale `[160,480]`.
- Stage-2 Markov embedding/head is BF16 `[129280,256]`. The confidence projection is BF16 `[1,5376]`.
- The source implements draft forward and confidence calculation. `generate.py` uses plain autoregressive generation and does not call `forward_spec`.

Evidence: [DSpark source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py), [generation source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/generate.py), and [converter](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/convert.py). Local: `model-root/config.json:145-155`, `model-root/inference/model.py:1032-1156,1274-1282`, `model-root/inference/generate.py:28-87,124-133`, `model-root/inference/convert.py:108-110`, `model-root/model.safetensors.index.json#/weight_map/mtp.0.*`, `#/weight_map/mtp.1.*`, `#/weight_map/mtp.2.*`, and `model-00044-of-00048.safetensors` through `model-00046-of-00048.safetensors`.

### Tokenizer and chat template

- Tokenizer identity: BPE tokenizer JSON SHA256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`, tokenizer config SHA256 `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547`.
- Base BPE vocabulary has `128,000` entries. Added tokens extend the model vocabulary to `129,280`. BOS is ID 0, EOS and pad are ID 1 and 2. Tokenizer config disables automatic BOS and EOS insertion.
- `model_max_length` is `1,048,576`.
- Relevant IDs include System `128,799`, User `128,803`, and Assistant `128,804`.
- They include `<think>` `128,821` and `</think>` `128,822`.
- They include DSML `128,825`, image placeholder `129,264`, and task tokens for action, query, authority, domain, title, and URL reading.
- `chat_template.jinja` supports chat and thinking modes, numeric reasoning effort 1–100, tool calls, tool results, images, mid-conversation system messages, and task tokens. Its SHA256 is `d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3`.
- `tokenizer_config.json` names the pad token content as the EOS string, while `config.json` and `tokenizer.json` assign pad ID 2 to `<｜▁pad▁｜>`. This is an identity/encoding conflict to resolve before freezing prompts.

Evidence: [tokenizer](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/tokenizer.json), [template](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/chat_template.jinja), and [encoding](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/encoding.py). Local: `model-root/tokenizer.json`, `model-root/tokenizer_config.json:1-33`, `model-root/chat_template.jinja:1-252`, and `model-root/encoding/encoding.py:29-51,419-521`.

### Context and parallelism

- Public config and tokenizer advertise `1,048,576` positions. RoPE uses YaRN factor 16 with original context `65,536`. The README recommends a 1M context window and at least 256K output tokens.
- The reference config omits `max_seq_len`, so `ModelArgs` defaults to `4,096`. Interactive generation overrides it to `65,536`. The released reference runner therefore does not prove 1M execution.
- The reference runtime uses one tensor-parallel rank per converted safetensors file. It checks routed expert counts and sharded dimensions against world size. MP4 satisfies the observed divisibility rules. MP8 is the documented example.
- With MP4, local counts are 96 backbone routed experts, 32 MTP routed experts, 16 attention heads, and 2 output groups per rank. The source checks embedding and row/column-parallel dimensions. Engram rows use padded ceiling shards.
- No model source implements separate TEP4 or DEP4 axes. Current `oh-my-vllm` shared validation names `tp4`, `tep4`, and `dep4`, but the repository has only the Qwen adapter. A DSV4 adapter must define its own rank ownership, cache groups, and request placement before any of these strategies can be claimed.

Evidence: [model README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md), [inference README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/README.md), [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py), [runner](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/run.sh), and [repo contract](https://github.com/para1lel/oh-my-vllm/blob/nightly/crates/model-contract/src/lib.rs). Local: `model-root/config.json:43-50`, `model-root/tokenizer_config.json:21-22`, `model-root/README.md:45-57,168-175`, `model-root/inference/model.py:152-279,854-904,1187-1191`, `model-root/inference/README.md:15-35`, `model-root/inference/run.sh:14-53`, `repo-root/crates/model-contract/src/lib.rs:102-125`, `repo-root/crates/model-contract/src/validation.rs:141-165`, and `repo-root/docs/adapters.md:24-56`.

## Evidence Paths

| Contract area | Primary evidence |
|---|---|
| Public model identity | `model-root/config.json` |
| Runtime shape and forward path | `model-root/inference/config.json`, `inference/model.py` |
| Tensor names, shard placement, and declared size | `model-root/model.safetensors.index.json` |
| Tensor dtypes and shapes | The 48 safetensors headers named by the index |
| Conversion and per-rank loading | `model-root/inference/convert.py`, `inference/generate.py` |
| Token identity and prompt format | `model-root/tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `encoding/encoding.py` |
| Current service contract | `repo-root/docs/adapters.md`, `repo-root/crates/model-contract/src/lib.rs`, `repo-root/crates/model-contract/src/validation.rs` |

## Open Questions

- Which source is authoritative for CED, 20+20 layer roles, SWA Bounded Replay, and the 890-byte KV claim? The README states them. The released config and inference source do not provide those interfaces.
- Should a production adapter store FP4/FP8 KV bytes, or keep the reference code's dequantized BF16 cache representation?
- Is 1M context a required serving limit, or is `65,536` the supported limit until a long-context cache and scheduler contract exists?
- Is DSV4 limited to TP4, or can MP8 and other tensor-parallel sizes be accepted after memory validation?
- Are TEP4 and DEP4 intended for this model? The released runtime has no expert-parallel or data-parallel implementation.
- What exact tokenizer and checkpoint identity protocol should bind Engram hash tables? The model source asserts tokenizer-derived compressed vocabulary size but does not verify file digests.
- Which pinned host-memory pool, transfer lifetime, and device lookup cache should hold the Engram rows in this service?
- What speculative acceptance loop, confidence threshold, and scheduler state contract should provide DSpark? The repository contains the forward path but no serving loop.
- The README says no Jinja chat template is included, but `chat_template.jinja` is present. Which prompt renderer should be frozen?
- Why does DSpark call token ID `128799` a noise token when the tokenizer maps that ID to `<｜System｜>`?
- Should the pad contract follow `config.json`/`tokenizer.json` ID 2 or the EOS string in `tokenizer_config.json`?

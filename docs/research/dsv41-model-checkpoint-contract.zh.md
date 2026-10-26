# DeepSeek-V4.1-Flash 模型和 checkpoint 契约

审计日期: 2026-10-26. 范围: 只读检查发布的模型目录和当前 `oh-my-vllm` 契约. 本报告冻结已观察到的事实. 它不定义适配器. `model-root` 表示给定的本地模型目录. `repo-root` 表示当前 checkout.

## 事实

### 身份和文件

- 模型身份: `DeepseekV41ForCausalLM`, 模型类型 `deepseek_v41`, Transformers 版本 `5.6.0`. 公共配置使用 `bfloat16` 作为基础 dtype.
- 模型配置 SHA256: `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.
- inference 配置 SHA256: `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809`.
- checkpoint 有 48 个 safetensors 文件. 索引有 `96,085` 个张量名称, 并声明 `510,286,023,000` 个数据字节. 只读 header 扫描匹配全部索引名称和声明的数据字节.
- Header dtype 汇总为 BF16 `3,952,883,712` 字节, F32 `169,229,128` 字节, FP8 E4M3 `204,015,223,296` 字节, E8M0 scale `23,563,015,184` 字节, 以及 I8 `278,585,671,680` 字节. 没有计算 48 个大 shard 的完整 SHA256 digest.
- 索引 SHA256: `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.
- 索引把 vision 和 aligner 张量放在 shard 1, 共享 embedding 放在 shard 2, backbone layer 0–39 放在 shard 3–42, output norm 和 head 放在 shard 43, DSpark MTP stage 0–2 放在 shard 44–46, 两个 Engram 表放在 shard 47–48.

证据: [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) 和 [index](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/model.safetensors.index.json). 本地: `model-root/config.json:2-7`, `model-root/model.safetensors.index.json`, `metadata.total_size`, `weight_map`, 以及全部 `model-00001-of-00048.safetensors` 到 `model-00048-of-00048.safetensors`.

### 形状和层结构

- 词表大小是 `129,280`. hidden size 是 `5,120`. backbone 有 40 层, 64 个 query head, 1 个 key/value latent head, head dimension 是 `512`, MoE intermediate size 是 `2,304`.
- Query RoPE 使用 `64` 个维度. Query low-rank size 是 `1,280`. Output projection 使用 8 组和 `1,024` 的 low-rank size.
- RMSNorm epsilon 是 `1e-20`. SwiGLU limit 是 `10.0`. Word embedding 不共享.
- Vision 有 32 层, hidden size `1,024`, 16 个 head, intermediate size `2,816`, patch size `14`, downsample ratio `3`, 最多 `1,024` 个语言模型 image token. Aligner 把 `9,216` 个输入映射到 `5,120`.
- Public quantization config 声明 dynamic FP8, weight block 是 `[32, 32]`, scale 是 E8M0, expert dtype 是 FP4. Inference config 用 `dtype: "fp8"` 和 `expert_dtype: "fp4"` 重述这一点, 使用 `compress_rope_theta: 160000`, 并有 43 个 compression entry: 前 40 个是 backbone entry, 后 3 个零值 entry 属于 DSpark stage. Backbone ratio 是 `0, 0`, layer 2–19 使用 ratio `2`, layer 20–39 使用 ratio `1`.

证据: [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json), [inference config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/config.json) 和 [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py). 本地: `model-root/config.json:22-169`, `model-root/inference/config.json:1-65`, `model-root/inference/model.py:44-136,1183-1223`, `model-root/model.safetensors.index.json#/weight_map`, `vision.*`, `aligner.*`, `embed.weight` 和 `head.weight`.

### 注意力, 稀疏检索和精度

- 每层都有 128 token 的 causal sliding-window cache.
- KV source layer 是 2, 8, 14, 20. 它们生成共享的 compressed KV. Index source layer 是 2, 8, 14, 20, 24, 28, 32, 36.
- Ratio-2 compressor 使用带学习 gate 的 softmax 对 token 组进行池化. Ratio-1 compressor 使用单个投影 token. Compressed KV 路径使用 FP4 E2M1, 每 16 个 channel 使用一个 E4M3 scale. Indexer 使用 FP4, 每 32 个 channel 使用 E8M0 scale.
- Indexer 每个 query 最多保留 512 个 compressed position. Layer 20 最多选择 `2,048` 个, 每个包含 8 个 position 的 candidate block. 后续 indexer 只在该 candidate mask 内评分.
- README 把三个 CSA2 mode 命名为 Full, Reindex 和 Reuse. 源码通过 `compress_ratios`, `kv_source_layers`, `index_source_layers` 和共享 runtime state 表达相同的共享关系, 但没有把这些 mode 名称暴露为 adapter API.
- 源码通过进程级 runtime slot 共享 compressed KV, index key 和 Top-K index. 它没有实现独立的 CED encoder/decoder cache 路径.
- Dense weight 使用 FP8 E4M3 和 E8M0 scale. 源码把 32 设为 FP8 weight 和 activation block size. Hugging Face checkpoint 中的 routed expert weight 以 I8 加 scale 存储, 默认转换为 packed FP4 E2M1.
- 源码在 grouped output einsum 前把 `wo_a` 转换为 BF16. 它先量化 KV 和 compressed KV, 再把反量化结果写入 reference cache buffer.

证据: [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py), [kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py) 和 [converter](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/convert.py). 本地: `model-root/config.json:43-127`, `model-root/inference/config.json:26-52`, `model-root/inference/model.py:429-610,613-790`, `model-root/inference/kernel.py:14-20,98-204,310-403,477-591`, `model-root/inference/convert.py:13-53,154-182`, `layers.2.attn.compressor.*`, `layers.24.attn.indexer.*` 和 `layers.0.attn.*`.

### MoE 和 Hyper-Connections

- 每个 backbone layer 有 384 个 routed expert, 激活 6 个 expert, 以及 1 个 shared expert. Routed gate 使用 `sqrtsoftplus`, 归一化 Top-K weight, 并使用 route scale `1.5`. Vision token 使用独立的 gate bias.
- 每个 DSpark MTP layer 有 128 个 routed expert 和 3 个 active expert. 它也有 1 个 shared expert.
- Hyper-Connections 把 residual stream 扩展为 4 个副本. 每个 attention 和 FFN sublayer 计算 pre, post 和 combination coefficient. Combination matrix 使用 20 次 Sinkhorn iteration 和 epsilon `1e-6`.
- Layer weight 包含 `hc_attn_fn`, `hc_attn_base`, `hc_attn_scale`, `hc_ffn_fn`, `hc_ffn_base` 和 `hc_ffn_scale`. 发布形状下每个 `hc_*_fn` 的形状是 `[24, 20480]`.

证据: [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) 和 [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py). 本地: `model-root/config.json:52-57,128-130,145-155`, `model-root/inference/model.py:792-904,907-994`, `layers.0.ffn.gate.*`, `layers.0.ffn.experts.*`, `layers.0.hc_*` 和 `mtp.0.ffn.*`.

### Engram

- Engram 运行在 layer 1 和 14. 它使用最大 n-gram size 4, 8 个 head, head dimension 256, tokenizer vocabulary input size `16,000,000`, 以及 compressed tokenizer vocabulary size `99,092`.
- Table row 数是 `384,006,168` 和 `384,016,682`. 两个 FP8 table weight 的大小是 `98,305,579,008` 和 `98,308,270,592` 字节. 它们的 E8M0 scale tensor 大小是 `3,072,049,344` 和 `3,072,133,456` 字节.
- 全部 12 个 Engram tensor 占用 `203,073,076,240` 个数据字节: `196,928,422,400` 个 FP8 E4M3 字节, `6,144,490,000` 个 E8M0 字节, 以及 `163,840` 个 BF16 字节.
- 精确的 index name 和 header metadata 如下:

| Index name | Shard | dtype | shape | data bytes |
|---|---|---|---|---:|
| `layers.1.engram.embed.weight` / `.scale` | 47 | F8_E4M3 / F8_E8M0 | `[384006168,256]` / `[384006168,8]` | `98,305,579,008` / `3,072,049,344` |
| `layers.1.engram.wkv.weight` / `.scale` | 47 | F8_E4M3 / F8_E8M0 | `[25600,6144]` / `[800,192]` | `157,286,400` / `153,600` |
| `layers.1.engram.q_weight` / `.k_weight` | 47 | BF16 / BF16 | `[4,5120]` / `[4,5120]` | `40,960` / `40,960` |
| `layers.14.engram.embed.weight` / `.scale` | 48 | F8_E4M3 / F8_E8M0 | `[384016682,256]` / `[384016682,8]` | `98,308,270,592` / `3,072,133,456` |
| `layers.14.engram.wkv.weight` / `.scale` | 48 | F8_E4M3 / F8_E8M0 | `[25600,6144]` / `[800,192]` | `157,286,400` / `153,600` |
| `layers.14.engram.q_weight` / `.k_weight` | 48 | BF16 / BF16 | `[4,5120]` / `[4,5120]` | `40,960` / `40,960` |

- `NgramHashState` 从 tokenizer 构建 normalized token map, 并断言结果正好有 `99,092` 个 entry. Image span 不参与 n-gram. `ParallelEngramEmbedding` 按 world size 对 table row 分片, 并使用 all-reduce 合并 lookup 结果.
- `ParallelEngramEmbedding` 加载本地 FP8 row 和 E8M0 scale, 将它们反量化为 BF16, 再对 lookup value 执行 all-reduce.
- Reference loader 把这些 table 创建为 CUDA device 上的 model parameter. 它没有定义 pinned host-memory residency 或 transfer contract.
- Converter 使用 ceiling division 对 Engram row 分片. 它用零填充 weight row, 用一填充 scale row. 它写出 `model{rank}-mp{world_size}.safetensors`, `generate.py` 在模型构建相同的 tokenizer 派生 hash map 后, 加载这个精确的 per-rank 文件.

证据: [Engram source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/engram.py), [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py) 和 [converter](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/convert.py). 本地: `model-root/config.json:131-144`, `model-root/inference/engram.py:17-61,87-184`, `model-root/inference/model.py:296-365`, `model-root/inference/convert.py:135-141`, `model-root/model.safetensors.index.json#/weight_map/layers.1.engram.*`, `#/weight_map/layers.14.engram.*`, `model-00047-of-00048.safetensors` 和 `model-00048-of-00048.safetensors`.

### DSpark

- DSpark 的 block size 是 5, noise token ID 是 `128799`, target layer 是 37–39, Markov rank 是 256, MTP layer 数是 3, routed expert 数是 128, active expert 数是 3.
- Stage 0 有 `mtp.0.main_proj` 和 `main_norm`. Stage 2 有 `markov_head.embed`, `markov_head.head`, `norm` 和 `confidence_head.proj`. MTP embedding 和 output head 与 backbone 绑定, converter 会跳过它们.
- MTP 数据在 shard 44–46, stage 0–2 分别占用 `2,652,643,800`,`2,573,913,560` 和 `2,706,317,272` 字节. 示例包括 `mtp.0.main_proj.weight` 的 FP8 E4M3 `[5120,15360]`, 对应的 E8M0 scale `[160,480]`, 以及 stage 2 的 Markov embedding/head BF16 `[129280,256]`, confidence projection 是 BF16 `[1,5376]`.
- 源码实现了 draft forward 和 confidence 计算. `generate.py` 使用普通 autoregressive generation, 不调用 `forward_spec`.

证据: [DSpark source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py), [generation source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/generate.py) 和 [converter](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/convert.py). 本地: `model-root/config.json:145-155`, `model-root/inference/model.py:1032-1156,1274-1282`, `model-root/inference/generate.py:28-87,124-133`, `model-root/inference/convert.py:108-110`, `model-root/model.safetensors.index.json#/weight_map/mtp.0.*`, `#/weight_map/mtp.1.*`, `#/weight_map/mtp.2.*` 和 `model-00044-of-00048.safetensors` 到 `model-00046-of-00048.safetensors`.

### Tokenizer 和 chat template

- Tokenizer identity: BPE tokenizer JSON SHA256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`, tokenizer config SHA256 `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547`.
- Base BPE vocabulary 有 `128,000` 个 entry. Added token 把模型词表扩展到 `129,280`. BOS 是 ID 0, EOS 和 pad 是 ID 1 和 2. Tokenizer config 禁止自动插入 BOS 和 EOS.
- `model_max_length` 是 `1,048,576`. 相关 ID 包括 System `128,799`, User `128,803`, Assistant `128,804`, `<think>` `128,821`, `</think>` `128,822`, DSML `128,825`, image placeholder `129,264`, 以及 action, query, authority, domain, title 和 URL reading task token.
- `chat_template.jinja` 支持 chat 和 thinking mode, 1–100 的 numeric reasoning effort, tool call, tool result, image, conversation 中间的 system message 和 task token. 它的 SHA256 是 `d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3`.
- `tokenizer_config.json` 把 pad token content 写成 EOS 字符串, 但 `config.json` 和 `tokenizer.json` 把 pad ID 2 指向 `<｜▁pad▁｜>`. 在冻结 prompt 前必须解决这个 identity/encoding 冲突.

证据: [tokenizer](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/tokenizer.json), [template](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/chat_template.jinja) 和 [encoding](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/encoding.py). 本地: `model-root/tokenizer.json`, `model-root/tokenizer_config.json:1-33`, `model-root/chat_template.jinja:1-252` 和 `model-root/encoding/encoding.py:29-51,419-521`.

### Context 和并行

- Public config 和 tokenizer 宣布 `1,048,576` 个 position. RoPE 使用 YaRN factor 16, 原始 context 是 `65,536`. README 建议 1M context window 和至少 256K output token.
- Reference config 没有 `max_seq_len`, 因此 `ModelArgs` 使用默认值 `4,096`. Interactive generation 把它覆盖为 `65,536`. 因此发布的 reference runner 不能证明 1M execution.
- Reference runtime 为每个 converted safetensors 文件使用一个 tensor-parallel rank. 它要求 routed expert 数和分片维度能被 world size 整除. MP4 满足已观察到的整除规则, 文档示例使用 MP8.
- MP4 下可直接得到的 local count 是每 rank 96 个 backbone routed expert, 32 个 MTP routed expert, 16 个 attention head 和 2 个 output group. 源码检查 embedding 和 row/column-parallel dimension. Engram row 使用带 padding 的 ceiling shard.
- 没有模型源码实现独立的 TEP4 或 DEP4 axis. 当前 `oh-my-vllm` shared validation 命名了 `tp4`, `tep4` 和 `dep4`, 但仓库只有 Qwen adapter. 任何 DSV4 adapter 都必须先定义自己的 rank ownership, cache group 和 request placement, 才能声明支持这些 strategy.

证据: [model README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md), [inference README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/README.md), [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py), [runner](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/run.sh) 和 [repo contract](https://github.com/para1lel/oh-my-vllm/blob/nightly/crates/model-contract/src/lib.rs). 本地: `model-root/config.json:43-50`, `model-root/tokenizer_config.json:21-22`, `model-root/README.md:45-57,168-175`, `model-root/inference/model.py:152-279,854-904,1187-1191`, `model-root/inference/README.md:15-35`, `model-root/inference/run.sh:14-53`, `repo-root/crates/model-contract/src/lib.rs:102-125`, `repo-root/crates/model-contract/src/validation.rs:141-165` 和 `repo-root/docs/adapters.md:24-56`.

## 证据路径

| 契约区域 | 主要证据 |
|---|---|
| Public model identity | `model-root/config.json` |
| Runtime shape 和 forward path | `model-root/inference/config.json`, `inference/model.py` |
| Tensor name, shard placement 和声明大小 | `model-root/model.safetensors.index.json` |
| Tensor dtype 和 shape | 索引列出的 48 个 safetensors header |
| Conversion 和 per-rank loading | `model-root/inference/convert.py`, `inference/generate.py` |
| Token identity 和 prompt format | `model-root/tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `encoding/encoding.py` |
| 当前 service contract | `repo-root/docs/adapters.md`, `repo-root/crates/model-contract/src/lib.rs`, `repo-root/crates/model-contract/src/validation.rs` |

## 待决策问题

- CED, 20+20 layer role, SWA Bounded Replay 和 890-byte KV claim 应以哪个 source 为准? README 声明了这些内容. 发布的 config 和 inference source 没有暴露这些 interface.
- Production adapter 应存储 FP4/FP8 KV byte, 还是保留 reference code 的反量化 BF16 cache 表示?
- 1M context 是必须的 serving limit, 还是在 long-context cache 和 scheduler contract 存在前, `65,536` 才是支持上限?
- DSV4 只需要 TP4, 还是在完成 memory validation 后可以接受 MP8 和其他 tensor-parallel size?
- DSV4 是否需要 TEP4 和 DEP4? 发布的 runtime 没有 expert-parallel 或 data-parallel implementation.
- 哪个精确的 tokenizer 和 checkpoint identity protocol 应绑定 Engram hash table? Model source 只断言 tokenizer 派生的 compressed vocabulary size, 没有验证文件 digest.
- 当前 service 应使用哪个 pinned host-memory pool, transfer lifetime 和 device lookup cache 保存 Engram row?
- DSpark 应暴露什么 speculative acceptance loop, confidence threshold 和 scheduler state contract? 仓库有 forward path, 但没有 serving loop.
- README 说没有 Jinja chat template, 但 `chat_template.jinja` 存在. 应冻结哪一个 prompt renderer?
- 为什么 DSpark 把 token ID `128799` 称为 noise token, 而 tokenizer 把这个 ID 映射为 `<｜System｜>`?
- pad contract 应遵循 `config.json`/`tokenizer.json` 的 ID 2, 还是 `tokenizer_config.json` 中的 EOS 字符串?

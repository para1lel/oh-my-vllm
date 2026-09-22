# 基准测试证据

`2026-09-22-tilelang-acceptance.json` 保留最终 12 组候选实现和冻结基线的完整原始测量，以及历史失败和干扰排除记录。测量实现为干净提交 da75c02。按用户要求，两种指标、两个引擎的稳定性上限从 5% 改为 10%。内嵌的原始 `candidate.comparison` 判定不变，新判定见每组的 `comparison_under_current_policy`，全部 12 组通过。`2026-09-22-tilelang-correctness.json` 记录未改动的完整 GPU 测试、实际模型探针、六组边界、文本/抢占和真实 MTP 服务/agentic 证据，并保留生成回答的局限。不收录临时算子实验记录。

历史 baseline_*、clean_* 和 mtp_* JSON 是已有的 vLLM 服务抓取结果（OpenAI 端点）。其设置、输入数据、计时边界和 MTP 配置与当前配对 EngineCore 协议不同。将其保留为历史数据，不得作为当前验收的分母。

2026-09-19-paired-investigation.json 保留该次会话完成的配对测量，包括失败比例、实际重复测量及可执行文件/源码身份。它明确不是最终验收矩阵。在当时阶段，ordinary bs1 需要因方差重测，多个模式/batch 尚未测量或低于 95%。下述完整验收产物取代该状态。一些早期记录没有捕获 CPU 亲和性掩码，以 null 表示。后续受干扰或不完整的配对被排除，不赋予性能数值。

新测量使用 benchmarks/compare_vllm.py 和 docs/testing.md。不要在此添加 GPU profiling trace。docs/handoff.md 跟踪当前工作。

2026-09-19-acceptance.json 是干净提交 12d227d 上的完整九组矩阵：预热两次、测量三次，每对使用相同单核亲和性。全部通过。文件包含准确命令、协议设置和原始结果。MTP batch2 因方差重测的结果单独保留，不替换原始测量。解释见 docs/acceptance.md。

2026-09-19-batch-text.json 保留普通模式和 MTP/前缀双请求抢占检查的完整生成文本及观测计数。明确记录了 dirty 构建身份；这是语义冒烟测试，不是吞吐数据。

`2026-09-21-independent-stage1.json` 记录独立运行时的第一个里程碑：新环境算子/CPU 测试、短 eager 模型输出及过渡适配器的 MTP4 服务检查。它不是最终迁移验收，也不替换性能基线。

`2026-09-21-independent-stage2.json` 记录独立 Worker/MTP/计算图正确性、FP8 后端缺陷和修复、最终真实服务/探针结果，以及一次明确不用于验收的吞吐诊断。在该阶段，移除默认适配器、真实独立 oh-my-pi 任务和最终受监测九组矩阵仍待完成。

`2026-09-21-independent-stage3.json` 记录移除旧运行时/测试依赖、目标环境完整 GPU/CPU 测试和实际文件/模块/库独立性审计。最终工作负载验收另行记录。

模型命名修正（2026-09-22）：目标是 **Qwen3.8-27B-FP8**。旧原始日志和冻结 JSON 可能含错误的 Qwen3.5 标签或此前的 `qwen3.5-27b-fp8` 服务 ID。这些内容作为历史实际执行记录保留，不代表当前模型名称。当前代码/文档/服务 ID 使用 Qwen3.8。新测量必须使用修正后的名称。

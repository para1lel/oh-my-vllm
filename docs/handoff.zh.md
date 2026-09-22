# 交接记录 — 2026-09-22

## 当前：TileLang 验收和修订后的稳定性门槛

项目自有的七个自定义 kernel 模块均已改用 TileLang。干净测量提交 da75c02 在用户批准的极差/中位数 10% 上限下，通过全部 12 组冻结 EngineCore 吞吐/TTFT 对比；用户于 2026-09-22 将该上限从 5% 调整为 10%。重新判定使用同一批完整原始测量，保留原判定。prefix batch 1 的 TTFT 波动为 5.97%，此前仅因旧门槛失败。最低吞吐比例 95.3098%，最高 TTFT 比例 101.4735%。判定工具输出稳定性上限，CPU 测试覆盖两个引擎、两种指标低于、等于和超过 10% 的情况。本次规则修订没有改动 GPU 推理源码。

未改动的完整 GPU 测试通过 146 项和 12 项子测试，无跳过。普通/MTP 实际模型 FP64 探针、六组总长 262144 token 边界（无 OOM/抢占）、文本/抢占/前缀、12 项 MTP4 约束、生命周期和长上下文服务均通过。真实 oh-my-pi 两种 API 各完成九次成功工具调用，均有正数 MTP 活动。回答生成于最终验收文档更新前，已保留原文和依据准确性方面的局限；下一步复核更新后的文档读取。完整证据见 acceptance.zh.md 和 bench/baseline/2026-09-22-tilelang-*.json。

TileFoundry 仅用于开发，固定个人 fork 6b1b149。离线 TileLang AutoTuner 和成本/显存分析用于选择生产参数。HIR 不能表达 FP64 相位归约，正确性以独立 FP64/模型测试为准。全部 25 份项目自有 Markdown 文档有中文配套版本。以下旧章节是历史进展，不代表当前阻碍。


## 双语文档（用户要求）

全部 25 份项目自有 Markdown 源文档现有同目录 `.zh.md` 译本。agent 继续阅读并以英文原文为准；今后每次 Markdown 修改需在同一次改动中同步中文。AGENTS.md 和 CONTRIBUTING.md 已记录规则。第三方 submodule 和生成的依赖除外。独立翻译审查、完整译本覆盖、本地链接和代码块核对均通过。文档工作不替代 kernel 验收。

最新 kernel 状态：干净 874a54a 的正式 MTP batch4 达到 94.9688%，仍低于 95%。Q/K RMS+RoPE 融合保留中间 BF16 舍入，静态审查通过。其完整 GPU suite 重跑通过 146 项测试和 12 项子测试；首次运行因另一个进程进入所选 GPU，导致进程超时测试失败，不是数值失败。实际普通/MTP 模型探针通过，但生产入口 FP64 检查发现接近位置 262144 时的 RoPE 相位误差。生产修复用 FP64 计算并约化相位，在原容差下通过 packed Q/K、int32/int64 位置和图回放。修复后完整 146+12 suite 及实际普通/MTP FP64 覆盖通过。TileFoundry 分析和代表性 twin 检查通过；文档明确说明其不能表达 FP64 相位，不能验证这个边界。最终 12 组性能和功能验收仍必需。

## 当前任务：TileLang 迁移与 TileFoundry 开发工作流

用户确认开始实现。将七个模块内全部 15 个项目自有 Triton kernel 替换为 TileLang，保留公开行为和已有正确性测试/容差。第三方内部 Triton 不在范围内。TileFoundry 仅用于开发，从上游 main fork 到 para1lel/TileFoundry 的 oh-my-vllm-integration 分支，固定为 3rdparty/TileFoundry。主项目仍在 main，不提上游 PR。取消 fork 的 Transformers 上限，在 oh-my-vllm 源码安装。允许兼容 TVM FFI/protobuf 降级，不维护 TileLang/OR-Tools fork。简单修复已授权，重大修复先讨论。

临时算子实验和记录放在仓库外，最终测试/证据限用户要求及已有文档规定的验收。保留 vLLM 比例门槛，不设相对 Triton 门槛。分析、优化并解决性能失败。用户加强目标：全部 12 组必须通过吞吐、TTFT 和稳定性；只有实测原因解释不算完成。结合 TileFoundry 分析，在所需 shape 上调优。

当前七个生产 kernel 模块均使用 TileLang，无自有 Triton 导入或回退。token 维度不决定算法/布局时采用动态维度；既有 CUDA Graph shape key 不变。TileFoundry fork 6b1b149 含依赖及 HIR sin/cos 适配，已审查并推送。开发专用完整算子 HIR/twin、安装/工作流文档位于 development/kernels 和 docs/tilelang-development.md。

动态长度修复后，完整 GPU pytest 146 项 +12 子测试通过，无跳过；Rust workspace、fmt、硬行宽、clippy 通过。依赖闭包检查和开发安装 dry-run 通过。实际 ordinary FP64 通过。MTP FP64、12 个约束、生命周期及长前缀服务在动态长度修改前通过；最终模型/agentic/边界及正式 12 组仍在进行，暂无最终性能结论。一次诊断与源码编辑重叠，延迟 JIT 检查失败，已丢弃并针对固定源码重跑。临时 MTP driver 把覆盖标志设为 4 而非 1，导致关闭覆盖检查失败；更正后重跑。没有修改容差或测试。算子实验和记录仍在仓库外。

迁移优化保留动态 token/batch，在设备端计算注意力 split 范围。KV gather 显式合并访问放置，一/两序列 GDN 使用 warp 局部归约。审查无正确性阻断。优化后完整 suite 再次 146+12 通过，无跳过；实际普通/MTP FP64 均通过。六组边界在优化前通过，优化后的边界、服务/agentic 随后在 6a6e6a5 通过。首轮正式矩阵七组通过、五组失败：MTP batch 1/2/4 吞吐不足，prefix batch 1/2 TTFT 不稳定。这不是最终验收。后续两序列 GDN 布局通过完整 146+12，继续注意力 shape 调优，最终模型/性能重跑待完成。

下一个注意力里程碑使用带谓词异步 global-to-shared copy、128 线程、64 split。只有 ceildiv 有足够余量时逻辑位置使用 int32，物理缓存偏移始终 int64。审查验证共享内存屏障和尾部补零。重跑全部 146+12 通过；较早一次进程超时测试因 GPU 争用失败，不是数值问题。全文件格式、Rust 检查和 hook 通过。现已离线使用 TileLang AutoTuner，显式合法元数据、CUDA Graph 计时和不变 FP64 检查。工作流要求完整算子和模型重验，不运行在线搜索，不在仓库保留微测试。实际普通/MTP FP64 满足全部必需覆盖，完整性能矩阵仍在进行。

f52a2d4 正式 MTP batch1/2 通过（基线的 97.386%/95.351%），batch4 仍为 92.248%，TTFT/稳定性通过。进一步离线调整 tile1 FP8 量化线程、去掉宽度 5120 的残差 padding、减少大 batch GDN 线程和短范围验证 split。完整 suite 再次 146+12，通过实际 batch4 GDN FP64 输出/状态检查。最终模型/服务/边界及 12 组仍必需。

b868c34 正式 MTP batch4 为 94.6674%，TTFT/稳定性通过，但仍低于 95%。下一里程碑流式累加注意力 merge，避免大 weighted fragment，使长非分组 decode 可采用 128 split/BK32。短行 RMS 使用一个 warp。全部 146+12 GPU 测试和全文件 hook 通过；临时长范围 FP64 覆盖不存在的位置 0、不等长度和更改长度/页表后的图回放。最终验收仍待完成。下列历史性能属于 TileLang 之前的实现，不是本次迁移验收。

## 前次调查：TTFT 与长上下文

用户批准基于最新官方 vLLM main 的新 12 组吞吐/TTFT 基线，以及普通/MTP4 batch 1/2/4 合计 262144-token 边界，见当前需求。新验收待完成，下列结果描述此前任务。冻结上游 SHA：e9f169d16b9408bb9ae44f75072b91a5521d733c。更新 main 前，将旧本地 vLLM telemetry 提交保存为 /data0/shared/dongwu.chen/vllm-before-ttft-20260922.bundle。

基线环境适配完成。最初无法获取确切 cu130 wheel 元数据，因此开始源码构建；官方 wheel 于 02:13 UTC 发布完毕，重查成功，自有源码构建在编译前停止。通过上游 editable 模式安装官方冻结提交 wheel，版本 0.29.1rc1.dev505+ge9f169d16.precompiled，Torch2.13.0+cu130。pip check 和 vllm._custom_ops 导入通过。扩展 hash 见 /tmp/oh-my-vllm-baseline-install-official.log 和 /tmp/oh-my-vllm-ttft-final/baseline-environment.json。

已实现但尚未验收：逐请求 TTFT、可选独立 GDN 池、正式测量/来源检查、修正 Qwen3.8 名称。77 个 Rust 测试通过。为新模型 ID 重建 debug 二进制后，完整 GPU pytest 100 项 +12 子测试通过，无跳过。格式、clippy、全部 hook 通过。131072 输入 batch4 短输出诊断无 OOM/抢占。首次完整 262144 边界暴露有符号 int32 decode 页偏移溢出，乘法前加宽并添加高页 FP64 测试，9 项通过。

六组 ordinary/MTP4 batch 1/2/4 边界完成，输入 258048 + 输出 4096，零抢占。峰值 reserved 显存普通 134951731200 字节、MTP4 140033130496 字节。这些冷诊断不是正式性能测量。实际普通/MTP FP64 保持原容差。12 个 MTP 约束、生命周期及两 API 131099 输入通过；重复请求复用 130928 token。全部自有边界/探针/服务 worker 退出，18015 端口释放。原始边界及日志 hash 见 bench/baseline/2026-09-22-ttft-stage1.json。

两 oh-my-pi API 使用 Qwen3.8 ID 和 MTP4 重跑：Chat/Responses 成功读取 10/8 次，含两份必需源码，工具错误零，耗时 28.02/22.16 秒。GB/GiB 单位错误和其他回答限制保留在 bench/baseline/2026-09-22-ttft-agentic.json。服务/worker 退出，18016 释放，无关 GPU 进程未动。

prefill profile 发现逐行 FP8 量化低效。16 行 tile 在模型诊断中使量化 GPU 时间从 295 降至 51ms。算术不变，小/decode shape 保留旧 kernel。最终阈值调优前完整 suite 103+12，最终边界/FP8 测试 14 项通过。限制及 hash 见 2026-09-22-prefill-quantization.json。优化后普通/MTP 实际 FP64 也通过。最初四次基线因外部 TP4 作业进入 GPU 中止，见 2026-09-22-ttft-discarded.json，结果排除。采集器改为流式写盘，干扰/超时也保留子输出。

当时十一组基线通过审计，剩余在 /tmp/oh-my-vllm-ttft-final 采集/重试。Prefix4 attempt103 吞吐稳定，但 TTFT 极差 13.94%，被拒绝。用户确认遇外部干扰继续等待/重试，当时尚无完整 12 组。候选 MTP attempt0 因复制 FlashInfer 缓存引发高代价重编译，在首次预热取消，未使用测量结果。后续使用现有独立 FlashInfer 缓存及可审计的逐运行 Triton 缓存。

长 prefill 注意力现在压紧活跃 KV，使用独立 ragged TRT-LLM，保留持久 784-token 页和小 query FA2。完整 suite 110+12、两条实际 FP64 路径通过。普通/MTP4 bs4 最大上下文重跑无抢占，峰值 reserved 不变，见 2026-09-22-ragged-prefill.json。最终数值检查含混合长 decode 隔离，3 项通过。完成基线采集、性能修复和审查后才能宣告完成。

prefill 卷积/SiLU tile 和 dual-SM 大投影 GEMM 将诊断 GPU profile 从 1.816 降至 1.721 秒。GPU suite 113+12、实际 MTP target/draft FP64 通过，见 2026-09-22-prefill-tiles.json。首批候选暴露共享 Triton 缓存审计污染，后续必须独立根目录。0ad2bb4 的 131072 输入 batch1 候选两门槛通过，其他首批仅为诊断。MTP 吞吐仍未达标。隔离基线步数诊断 898 对项目 899，指向每步成本而非接受率。临时私有 GEMM tactic 原型触发非法访问，未用于运行时。

packed GDN/卷积/RMS 输入保留 token/head stride，target/draft 元数据合并传输。审查要求每 batch 只转一次 int32 start，避免反复 FlashInfer cast，已实现。GPU suite 120+12、实际 MTP FP64、12 个服务约束、生命周期和 131099 输入前缀通过，服务 18017 退出释放。MTP bs1/2 诊断提升至 305.86/450.65 tok/s，仍低于新门槛，见 2026-09-22-strided-metadata.json。Prefix2 基线因方差及无关 GPU 任务反复进入继续重试，用户授权。

已实现残差 RMS/SiLU 量化融合、独立 CuTe-DSL 词表投影和零拷贝 16-token 子页原生 target decode。784-token 持久布局不变，draft 排除位置零仍用自定义 kernel。最终 suite 138+12。实际 batch2 target/draft 图预热 FP64 通过，与也通过的 eager GDN 探针分开。12 个约束、生命周期和长前缀通过，服务 18018/worker 退出。初始 MTP bs1/2/4 诊断 319.66/473.69/672.44 tok/s，不是正式验收，batch4 仍低于 95%。四序列更大 GDN tile 和八 warp 残差 RMS 通过最终测试/审查；短 MTP4 诊断 674.03 仍低于目标。

六组最大上下文重跑全部零抢占。launch 调优发生在边界 worker 启动之后，证据限制明确记录在 bench/baseline/2026-09-22-native-decode-fusions.json。全部自有诊断/探针/边界/服务 worker 退出，18018 释放。格式、Rust 测试/clippy 和全部 hook 通过，正式采集仍进行。静态审查无阻断，并明确要求区分图预热 FP64 与 eager 覆盖。

Prefix2 从 attempt110+ 暂用 CPU96 单核调查 TTFT 方差。attempt112 仍极差 37.94%，保留所有尝试。候选必须匹配最终接受基线的亲和性。用户再次确认无法预留独占 GPU，继续等待/重试。

全部 12 组更新基线随后采集完成，冻结在 bench/baseline/2026-09-22-refreshed-enginecore.json。Prefix2 attempt114 以 CPU96 单核通过，候选须相同。重复 attempt201 在接受后取消，worker 退出。5166885 上六 ordinary 和 MTP batch1/2 正式两门槛通过；三 prefix 的 TTFT 失败，MTP batch4 被外部 GPU 到达打断。未宣告完整验收，保留尝试在 /tmp/oh-my-vllm-ttft-final。

用户要求的实现/性能对比记录在 docs/performance-gap-2026-09-22.md，含冻结上游链接及 profile hash。draft 注意力是最明确、可隔离的不利 kernel 差异；FP8 GEMM、词表投影和 target 注意力有竞争力。CPU 采样注解含 GPU 等待，不能标成 ZMQ 开销。融合 GDN prefill Q/K 归一化、原生短前缀 target context attention、独立 TRT ragged MTP prefill 已实现并审查。

最新 MTP batch4 诊断 692.012 tok/s，对门槛 693.140，仍不足且非正式。GPU suite 146+12 无跳过。实际 FP64、12 个约束、生命周期、长前缀通过。六组最大上下文无抢占；普通边界早于最终 MTP 专用修改，MTP 边界在修改后重跑。见 2026-09-22-prefill-research-verification.json。全部自有 worker 退出，当时 nvidia-smi 为空，18020 释放。ee2fb63 已推送。

干净 ee2fb63 正式后续完成全部 12 组，十组通过。MTP batch4 中位数 692.1899 tok/s，为 94.8698%，低于 95%；稳定且 TTFT 通过。还需 0.1373% 吞吐，即每完整 batch 减少 32.45ms。Prefix batch1 两比例通过，但两次 TTFT 稳定性失败，最新 44.72–51.16ms、极差 14.20%。Prefix batch2/4、全部 ordinary 和 MTP batch1/2 通过。完整数据及被排除外部干扰尝试见 2026-09-22-performance-gap-candidates.json。自有 worker 退出，当时 nvidia-smi 为空。调查完成，但整体验收未完成。优先 mask draft 注意力，再做 GPU 常驻已接受 token 记账，另调查短前缀时延方差；保留位置零排除及全部容差。

## 前一任务：独立运行时

独立运行时已实现，原冻结 EngineCore 九组全部 >=95%。两种最终 oh-my-pi 以 MTP4 针对对齐后的文档完成。

Rust 拥有 HTTP、调度和逻辑 KV。Python 用项目代码、PyTorch/Triton/FlashInfer、XGrammar 加载运行 Qwen。不安装/导入/链接 vLLM，不依赖旧源码/环境，无旧 runner。使用 scripts/with-env.sh 和 scripts/with-gpu.sh。目标模型、单张 B200、block784、conda oh-my-vllm 不变。requirements/runtime.txt 固定依赖，ADR-006 说明扩展范围。

## 前一任务验证

- GPU pytest 92 项 +12 子测试，无跳过；Rust 75 项通过。
- fmt、硬行宽、clippy、ruff、全部 pre-commit 通过。
- 实际普通/MTP FP64 保留原容差；分组 target/draft 注意力和图 buffer 生命周期/缓存恢复通过。
- 不同城市文本经前缀复用、错峰和重计算抢占通过。12 个真实 MTP 约束和生命周期（混合 batch、续接/删除、断开/释放）通过。
- 六 ordinary/prefix 使用干净 ca5a200，三 MTP 使用干净 5c0848e；两者只改变 MTP 执行。既有图定义 AST 相同，其他非 MTP 源码不变。逐组身份、原始重复和 hash 在最终独立验收产物中。
- 源码/运行时/文件访问审计未发现旧 vLLM 依赖，kernel 缓存属于本项目。未重跑或替换原 EngineCore 基线。

数值和证据见 acceptance.md 及 bench/baseline/2026-09-21-independent-acceptance.json。中间产物保留失败性能、外部 GPU 污染、两次 CPU 重叠排除和生产前已修复的独立原型 buffer 生命周期失败。诊断绝不代替正式重复。

## 前一任务 agentic 验证与清理

Chat/Responses 成功 read 7/18 次，含两必需源码，工具错误零，并有真实后续请求。端到端 26.27/36.68 秒。两答案识别项目及完成状态；Responses 小措辞/计数错误明确保留在 acceptance.md 和 JSON，不声称完美依据文件。

前一独立运行时范围无剩余工作。全部自有 GPU 程序退出，当时 nvidia-smi 无计算进程，18013/18014 释放。没有要求保持服务运行。

仅 main 开发/提交，暂存有意改动，保持 hook 开启和规定署名。不得遗留任务自有 GPU 程序。

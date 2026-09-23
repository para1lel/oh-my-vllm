# 交接记录 — 2026-09-23

[English](handoff.md)。agent 以英文原文为准。

## 当前状态

CUDA 迁移已完成并通过验收。下一项任务是修复整库代码审计发现的问题。

- **Kernel 后端：** B200 上默认使用 CUDA。在 Python 启动前设置
  `OH_MY_VLLM_KERNEL_BACKEND=tilelang` 则改用冻结的 TileLang 对照。运行时身份报告实际使用的
  后端，不会静默回退到 TileLang。
- **职责划分：** Rust 负责服务、调度和逻辑 KV 缓存；Python 负责 GPU 计算。不使用任何 vLLM
  运行时、源码或环境依赖。
- **算子（干净提交 `c36d1c9`，显式选择 CUDA）：** 全部 147 项正式用例通过。每项都通过三轮、
  每轮 20 对交替测量，且单侧 95% bootstrap 下界为正。
- **框架（同一提交）：** 全部 12 组通过。
  - 吞吐为冻结 vLLM 基线的 97.52–125.09%。
  - TTFT 为基线的 76.01–97.50%。
  - 每组都满足 10% 极差规则。
  - 余量最小的是 MTP batch 4，吞吐 97.52%。
- **默认选择变更（`030f60f`）：** 只修改后端选择和身份报告。174 项测试加 31 个子测试的
  套件以及 MTP4 服务约束和生命周期检查，是在 `ef07b2b` 之后、后来成为该提交的工作树上
  通过的（CUDA 源码未变），而不是在 `030f60f` 上单独运行的。
- **其他已通过的检查：**
  - 六组 258048+4096 边界运行完成，无 OOM、无抢占。
  - 真实文本的抢占/前缀检查。
  - 两种 oh-my-pi API。

证据见 [acceptance.zh.md](acceptance.zh.md) 以及
`bench/baseline/2026-09-22-cuda-{operators,framework,features}.json`。

## 待办：修复代码审计问题

对 `030f60f` 的审计记录在 [audit-2026-09-23.zh.md](audit-2026-09-23.zh.md)，包括：

- 4 项 P0，都属于服务可用性。
- 17 项 P1，属于潜在的静默错误或不安全契约。
- 16 项 P2，属于性能、健壮性或证据。
- 14 项 P3，属于测试、证据和可维护性。

审计在默认路径上没有发现产生错误 token 的缺陷，上面的已验收证据依然成立。

按审计给出的批次顺序修复，并在索引中更新每个问题的状态：

1. 服务可用性
2. 证据完整性
3. Kernel 与 worker 契约
4. 调度器加固
5. 经测量的性能改进
6. 清理

每个批次都适用两条规则：

- 修改 `kernels.cu` 设备代码，须重跑其正式算子用例。
- 修改热路径，须重新检查框架各组，才能声称 12 组结果仍然成立。

目前会影响运维的已知问题（详情见审计文档）：

- 单个请求在采样、grammar 或 prepare 阶段出错，就会停止服务引擎；此后的所有请求都返回
  503，直到重启（SRV-01）。
- 用 SIGTERM 或 SIGKILL 杀掉 Rust 进程，会让 Python worker 成为孤儿进程，继续占用 GPU 和
  `with-gpu.sh` 锁（SRV-02）。异常退出后要检查 `nvidia-smi`，只停止自己启动的进程。
- `benchmarks/compare_vllm.py` 是历史九组工具，不执行当前的门槛，应使用
  `benchmarks/ttft.py`（EVD-01）。

## 文档清理（本次变更）

- 新增审计文档。
- 从 `acceptance.md`、`cuda-development.md`、`plan.md` 和 `bench/baseline/README.md` 删除了
  已被取代的里程碑叙述。原始产物仍然保留，并由 `acceptance.md` 中的历史表索引。
- 删除了 `performance-gap-2026-09-22.md`。它描述的 MTP 差距已被后续工作消除，汇总数据仍保留在
  `bench/baseline/2026-09-22-performance-gap.json`。
- 修正了过时内容：
  - 命令改为 `ttft.py` 和完整的 pytest 套件。
  - 协议和抢占的描述与当前代码一致。
  - ADR 状态已更新。
- 同步更新了中文译本。

## 操作规则

- **环境：** 使用 `scripts/with-env.sh`。
- **GPU 选择：** 用 `scripts/with-gpu.sh` 按 UUID 选择空闲 GPU。
- **Git：**
  - 只在 `main` 上开发和提交。
  - 只暂存有意修改的文件。
  - 保留 hooks。
  - 维护中文 Markdown 译本。
  - 带上要求的提交署名。
- **GPU 清理：** 及时停止自己启动的所有 GPU 程序，并确认端口和显存已释放。
- **不进仓库的内容：** 临时调优和原始 GPU trace。正式的汇总证据放在 `bench/baseline`。

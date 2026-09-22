# oh-my-vllm

[English](README.md) · [中文文档索引](docs/README.zh.md)。agent 以英文版为准。

面向单张 B200 GPU 上 Qwen3.8-27B-FP8 的 Rust 优先推理框架。Rust 负责 HTTP 服务、调度和逻辑 KV 缓存；Python 使用独立库运行项目自有的 Qwen 模型以及 GPU 状态/计算 kernel。两端通过 ZMQ DEALER 交换 msgpack。运行、构建和测试统一使用 conda `oh-my-vllm`，不安装或导入 vLLM。

模型位于 `/data0/shared/Qwen3.8-27B-FP8`，包含 16 层全注意力和 48 层 GDN。块大小为 784。参见[架构](docs/architecture.zh.md)、[通信协议](docs/design.zh.md)和[设计决策](docs/decisions/)。

## 快速开始

包装脚本选择框架环境、Python 路径、构建目录和独立 kernel 缓存。GPU 测试等待空闲 B200，并固定其 UUID。

```bash
uv pip install --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python -r requirements/runtime.txt
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/oh-my-vllm-text.ipc --max-tokens 64
```

请使用唯一的 socket 路径。文本脚本加上 `--num-speculative-tokens 4` 即启用 MTP。当前回归包含普通、MTP4 和前缀命中模式，每组完整预热两次、正式测量至少五次。全部 12 组 TileLang 吞吐/TTFT 对比已通过，稳定性采用用户批准的 10% 上限，详见[测试](docs/testing.zh.md)。`benchmarks/compare_vllm.py` 保留此前九组历史对比及其原始配置。

## 本地 OpenAI 服务

```bash
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-serve.ipc --num-gpu-blocks 4200 --mamba-blocks 128 --max-model-len 262144 --num-speculative-tokens 4 serve
```

Chat Completions 和 Responses 地址为 `http://127.0.0.1:8000/v1`，支持思考、工具、JSON 约束和 Responses 存储。[服务文档](docs/serving.zh.md)说明支持的 schema、OMP 配置和验收命令。CPU/客户端测试已通过；真实 MTP4 约束、生命周期、长前缀复用及两种 oh-my-pi 工作流已通过。六组输入输出合计 262144 token 的边界检查通过且无抢占；正式 12 组吞吐/TTFT 验收全部通过。当前结果见[交接记录](docs/handoff.zh.md)。

## 验证与性能

全部 12 组要求吞吐至少达到更新后 vLLM EngineCore 的 95%，TTFT 不超过其 110%，两种指标的极差/中位数均 <=10%。TileLang 实现已通过这些门槛，详见当前验收表。此前独立运行时的九组测试已通过原始冻结基线门槛；这些历史结果不满足新的 TTFT 验收要求。[验收证据](docs/acceptance.zh.md)分别说明当前与历史结果。

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p 'test_*.py'
```

[测试文档](docs/testing.zh.md)覆盖实际模型 FP64 探针、前缀/MTP/抢占、服务和性能检查。[开发文档](docs/development.zh.md)说明固定环境及日志；[性能分析文档](docs/profiling.zh.md)区分主机与 GPU 测量。每个里程碑都需要独立审查并提交。

## 目录结构

- crates/kv-cache：逻辑混合缓存池、前缀缓存和状态预留。
- crates/scheduler：请求、token 预算、分块对齐、抢占和 MTP。
- crates/zmq-worker：HTTP API、CLI、模型进程生命周期和 ZMQ 客户端。
- python/oh_my_vllm/worker：执行、MTP、采样、计算图和桥接。
- python/oh_my_vllm/models 与 kernels：具体模型和独立 GPU 算子。
- tests/probe_worker.py：实际模型 FP64 诊断，不用于吞吐测试。
- benchmarks/ttft.py：独立测量及当前 TTFT/吞吐门槛。
- benchmarks/baseline/enginecore.py：明确隔离的参考数据采集器。
- benchmarks/compare_vllm.py：历史对比和进程管理。

自定义 kernel 使用 TileLang 和仅供开发的 TileFoundry fork。安装和算子语义见[工作流](docs/tilelang-development.zh.md)；全部 12 组迁移性能对比已通过[验收文档](docs/acceptance.zh.md)中的当前规则。

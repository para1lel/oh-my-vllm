---
name: verify-oh-my-vllm
description: 通过 Rust 应用验证 Chat, Responses, 工具, 服务生命周期或离线 CLI 行为, 并记录证据.
---

# 验证 Rust 应用

在仓库根目录执行命令.
先使用[开发环境](../../../docs/development.zh.md).
本技能驱动 Rust CLI 和 HTTP 应用.

**默认 helper 使用 CPU worker fixture 检查服务和传输.
它不证明 GPU 推理, 数值正确性, 模型容量或性能.**

## Launch

### CPU 服务路径

从被忽略的 `LOCAL.md` 设置 `ARTIFACT_ROOT`.
此目录应位于仓库外部.
服务源码稳定后, 构建服务:

```bash
scripts/with-env.sh cargo build --bin oh-my-vllm-zmq-worker
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature chat --binary target/debug/oh-my-vllm-zmq-worker
```

一次调用依次完成 Launch, Doctor, Drive, Evidence 和 Cleanup.
服务仅在本次调用期间运行.
helper 选择可用的 loopback 端口, 并在新 scratch 目录中使用不同的 IPC 路径.
它为每个 Rust 进程建立其自有的进程组.
它为自身及子进程设置 `CUDA_VISIBLE_DEVICES=''`.

helper 调用[已有 fixture 安装 API](../../../tests/worker_fixture.py).
它不导入 `tests.test_service`, 也不依赖 `OH_MY_VLLM_TEST_BINARY`.
fixture 提供:

- scratch 存储中的 checkpoint 配置和 worker launcher.
- 带有 checkpoint 和 tokenizer 身份的合成 Qwen descriptor.
- Prompt token 准备, 以及固定的文本, 工具和 JSON 回复.
- Worker 就绪消息, 执行结果, cache 消息和 worker 统计.

Rust HTTP server, 请求解析器, 调度器, 响应存储和 ZMQ/msgpack 边界使用各自的服务路径运行.
fixture 不加载 checkpoint 权重, tokenizer, 原生模块或 GPU 状态.
它检查共享服务中的 Qwen 回复解析.
它不检查 DeepSeek 模型执行.
行为依据见[已有服务用例](../../../tests/test_service.py).

Rust 修改期间, `--binary` 可以选择已有的可执行文件.
证据记录一份可执行文件摘要和一份源码快照.
两者的构建对应关系尚未验证.
最终源码冻结后, 重新构建并重复 drive.

### Checkpoint 服务路径

Checkpoint 推理使用[服务指南](../../../docs/serving.zh.md).
从本地配置设置 `MODEL_ROOT`, `CUDA_HOME`, `OH_MY_VLLM_CUTLASS` 和 `OH_MY_VLLM_KERNEL_CACHE`.
GPU 数量, 适配器选项, checkpoint 文件和验收条件见所选模型指南.
按[执行规则](../../../docs/execution.zh.md)取得 GPU 使用权.
使用 CUDA 13.4 及指定的原生构建依赖.

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-env.sh scripts/with-gpu.sh python -m development.native_build
VERIFY_SCRATCH="$(mktemp -d /tmp/omv-native-XXXXXX)"
env -u OH_MY_VLLM_WORKER_PYTHON scripts/with-env.sh scripts/with-gpu.sh target/release/oh-my-vllm-zmq-worker --model "$MODEL_ROOT" --socket "$VERIFY_SCRATCH/worker.ipc" serve --listen 127.0.0.1:8000
```

最后一条命令在前台运行.
示例端口不可用时, 使用本地配置中的其他监听地址.
`env -u` 命令先移除 fixture worker 覆盖值, 再由环境 wrapper 选择推理解释器.
此路径通过 Python inference worker 加载 checkpoint 和原生模块.
CPU helper 不运行此路径.
GPU 数值和性能证据使用[模型验收](../../../docs/testing.zh.md#模型验收).

## Doctor

helper 采集可执行文件的 `--help` 输出及依赖版本.
它轮询 `GET /diagnostics`, 直到取得 worker PID.
它检查报告的 server PID 是否为新启动的 Rust 子进程.
它检查 worker PID 是否属于该服务.

然后, 它读取 `GET /v1/models` 中的公共模型 ID.
`doctor.json` 包含这些记录时, 本步骤完成.

使用 `--launch-timeout` 修改默认 15 秒的启动期限.
失败后读取 `failure.txt`, 进程日志及已记录的 HTTP 错误.
端口选择发生竞争时, 启动会失败. 使用新 scratch 目录重复命令.

对于 checkpoint 路径, 在所选监听地址读取 `/v1/models` 和 `/diagnostics`.
推理前检查所选模型的就绪条件.
fixture descriptor 不是 checkpoint 就绪证据.

## Drive

按所选 API 或 CLI 行为读取[功能图](features/README.zh.md).
通过 `--feature` 选择最小的适用主题.
共享服务行为改变时, 使用 `--feature all`:

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature all --binary target/debug/oh-my-vllm-zmq-worker
```

helper 发送 HTTP 请求并采集完整响应.
operations 主题还运行可执行文件的 `run` 和 `bench` 命令.
drive 失败时, 清理后返回非零退出码.
helper 是带证据的应用驱动工具. 它不增加业务测试.
回归检查使用仓库的[测试范围](../../../docs/testing.zh.md).

## Evidence

helper 打印位于 `ARTIFACT_ROOT` 下的新证据目录.
报告完成前先读取 `result.json`.
其中必须显示 `status=completed`, `cleanup_completed=true` 和 `scratch_removed=true`.
检查所选主题的记录及原始响应.

| 记录 | 内容 |
|---|---|
| `manifest.json` | Git 状态, 源码摘要, 可执行文件摘要, Python 和包版本, 范围限制 |
| `fixture.json` | 安装 API 身份, 替代的准备步骤, 配置, 固定回复 |
| `launch.json` | 所选端口, IPC 路径, scratch 路径 |
| `doctor.json` | 自有 server 和 worker 就绪记录, 公共模型 ID |
| `http/` | 请求方法, URL, headers, JSON body, 响应 headers, 状态, 完整 body 或错误 |
| `features/` | 各个所选主题的观察记录 |
| `processes/` | 命令, 进程身份, 可观察时记录的可执行文件摘要, stdout, stderr, 退出状态 |
| `cleanup.json` | PID 启动身份, 最终观察, 可取得时的接管子进程退出状态, 待停止的进程列表 |

并发修改期间, 源码快照不是原子快照.
可执行文件可能早于该快照.
drive 完成时, manifest 仍会注明此限制.
短命令可能在采集可执行文件身份前退出.
Rust 回收子进程时, 可能无法取得 worker 退出码.
清理记录仅报告观察结果, 不虚构退出码.

证据保存在仓库外部.
在交付回复中给出其路径和范围限制.
主机设置保存在 `LOCAL.md`.

## Cleanup

helper 先请求其 Rust 子进程正常关闭.
它通过自己的进程树监测后代进程.
向已记录的后代进程发送信号前, 它检查 PID 启动身份.
它使用 Linux child subreaper 收集孤儿子进程.

自有进程停止后, 它删除 scratch 存储.
`cleanup.json` 的待停止的进程列表必须为空.
清理后, 外部证据目录仍可读取.

SIGINT 和 SIGTERM 进入清理流程.
SIGKILL 无法执行 Python 清理.
强制停止后, 任何手动进程操作前都应对比已保存的进程身份.
清理失败时保留 scratch 存储供检查.

对于前台 checkpoint 服务, 对自己的服务进程使用 Ctrl-C 或 SIGTERM.
等待服务及其 worker 退出.
删除自己的空 socket 目录:

```bash
rmdir "$VERIFY_SCRATCH"
```

## Helpers

[verify.py](verify.py) 是可执行入口.
使用 `--help` 查看当前选项.
[drive.py](drive.py) 负责 HTTP 证据和主题 drive.
[processes.py](processes.py) 负责进程观察和清理.
[evidence.py](evidence.py) 记录源码和可执行文件身份.

这些 helper 使用标准库及已有 fixture API.
它们不增加生产依赖.

API 或 CLI 改变后, 同步更新功能图及其中文翻译.
用户可通过 `/maintain-verification-skill` 请求维护功能图.

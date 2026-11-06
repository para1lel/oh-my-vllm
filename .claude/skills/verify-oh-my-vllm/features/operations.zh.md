# 诊断和离线命令

## 子功能

- 通过 `GET /v1/models` 发现模型.
- 通过 `GET /diagnostics` 查看服务, worker, cache 和请求状态.
- 不支持的 completion 数量返回 400 校验响应.
- 通过 `run` 离线生成 tokens, 通过 `bench` 检查前缀复用.
- 关闭自有进程并删除 scratch.

参见 [CLI 入口](../../../../crates/zmq-worker/src/main.rs), [诊断](../../../../crates/zmq-worker/src/serving/mod.rs)及 [worker 生命周期](../../../../crates/zmq-worker/src/client/lifecycle.rs).

## 如何进入 (用户视角)

完成 [Launch](../SKILL.zh.md#launch) 后, 在服务监听地址请求模型列表或诊断信息.
离线工作选择 `run` 或 `bench`, 而非 `serve`.
模型, socket 和 cache 选项放在子命令前.
使用 Ctrl-C 或 SIGTERM 正常关闭服务.

## 通过 verify.py 驱动

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature operations --binary target/debug/oh-my-vllm-zmq-worker
```

[Operations drive](../drive.py) 记录被拒绝的 `n=2` Chat 请求及空闲诊断信息.
它使用各自的 IPC 路径和进程组启动每个离线命令.
run 命令使用 token IDs 1 和 2, 输出 8 个 tokens.
bench 命令使用 batch 2, 输入长度 786, 输出长度 8, 前缀复用, 无 warmup, 重复一次.
这些参数来自[已有 fixture 用例](../../../../tests/test_service.py).
读取证据目录中的 `processes/run/`, `processes/bench/`, `features/operations.json` 和 `cleanup.json`.

## 注意事项

离线输出长度固定.
fixture 检查 worker 消息和逻辑前缀复用, 不执行 GPU 计算.
其 bench 耗时不是推理性能.

helper 通过 server PID, worker 父进程和 PID 启动身份核对所有权.
自有进程停止且 scratch 删除后, 清理才完成.
证据保存在 scratch 外部.
失败和强制停止的限制见 [Cleanup](../SKILL.zh.md#cleanup).

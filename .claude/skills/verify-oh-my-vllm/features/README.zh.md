# 验证功能图

## 子功能

| 主题 | 用户行为 |
|---|---|
| [Chat](chat.zh.md) | JSON 回复, SSE chunk, 完成标记, usage |
| [Responses](responses.zh.md) | 创建, 读取, 继续, 删除, 带类型的 SSE |
| [工具和输出格式](tools.zh.md) | Chat 工具, Responses 工具, JSON Schema 输出 |
| [运行操作](operations.zh.md) | 诊断, 校验错误, 离线 run/bench, 进程清理 |

[Rust CLI 入口](../../../../crates/zmq-worker/src/main.rs)和 [HTTP 入口](../../../../crates/zmq-worker/src/serving/mod.rs)提供应用.
[已有服务用例](../../../../tests/test_service.py)定义 CPU 边界示例.

## 如何进入 (用户视角)

先执行 [Launch](../SKILL.zh.md#launch).
应用提供 CLI 命令和 HTTP 路由. 它没有浏览器界面.
调用 API 前, 使用 `GET /v1/models` 取得公共模型 ID.
路由和请求控制字段见所选主题.

## 通过 verify.py 驱动

完成环境设置后, 在仓库根目录执行:

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature all --binary target/debug/oh-my-vllm-zmq-worker
```

使用 `--feature chat`, `--feature responses`, `--feature tools` 或 `--feature operations` 选择单个主题.
一次调用会启动应用, 检查就绪, 驱动主题, 保存证据并停止自有进程.
报告完成前, 阅读[证据契约](../SKILL.zh.md#evidence).

## 注意事项

默认 [worker fixture](../../../../tests/worker_fixture.py) 提供固定回复及合成模型身份.
它只检查服务和传输行为.
它不证明 GPU 推理, 数值正确性, 容量或性能.
使用带模型权重和原生模块的推理时, 采用 [checkpoint 路径](../SKILL.zh.md#checkpoint-服务路径).
原始证据保存在仓库外部.
路由或 CLI 选项变化时, 同步更新此功能图及其中文翻译.

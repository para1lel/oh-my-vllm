# oh-my-vllm

在一张 B200 GPU 上运行 Qwen3.8-27B-FP8, 使用 Chat Completions 和 Responses API.
支持 stream, function tools, 约束输出, 前缀复用, 普通解码, MTP4 和可选 DSpark.
输入与输出总量最多 262144 个 token.

## 启动

安装[环境](docs/development.zh.md), 然后设置本地 checkpoint:

```bash
export OH_MY_VLLM_MODEL="/path/to/Qwen3.8-27B-FP8"
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker serve
```

默认 endpoint 为 `127.0.0.1:8000`.
请求, 工具, 限制和 DSpark 配置见[服务用法](docs/serving.zh.md).

## 请求测量

以下一小时观察使用源码 `3324a7f`, 每种模式有四个并发 OMP 会话.
颜色表示从前缀缓存复用的输入 token 比例.
纵轴是输出 token 数除以请求处理时长, 不是仅 decode 的吞吐率.
每种模式的四次 controller 截止中断在[证据](docs/acceptance.zh.md#源码-3324a7f)中单列.
最终框架性能门槛尚待通过.

![MTP4 请求散点图, 渐变表示缓存命中率](docs/assets/mtp-agentic-soak-3324a7f.svg)

![DSpark 请求散点图, 渐变表示缓存命中率](docs/assets/dspark-agentic-soak-3324a7f.svg)

## 学习与验证

- [交互式代码讲解](code-journey/README.zh.md): 十三章中文教程, 包含当前源码和调度器实验.
- [架构](docs/architecture.zh.md): 所有权与背景概念.
- [测试](docs/testing.zh.md): 正确性, 容量, 服务和性能命令.
- [验收](docs/acceptance.zh.md): 已测源码范围和验证限制.
- [文档索引](docs/README.zh.md): 指南和决策.
- [贡献](CONTRIBUTING.zh.md): 开发检查与提交过程.

[English](README.md)

# oh-my-vllm

本项目提供在单张 NVIDIA B200 上运行 Qwen3.8-27B-FP8 的 Rust 服务.
Python worker 加载模型, 执行文本, 图片和视频推理.
支持普通解码或 MTP4, 并提供约束解码.

安装, 命令和 HTTP 请求见[服务指南](docs/serving.zh.md).

[English](README.md) · [贡献流程](CONTRIBUTING.zh.md)

## 性能测量

2026-10-17 的测量使用单张 NVIDIA B200 和 Qwen3.8-27B-FP8.
运行时源码见版本 [`4356493`](https://github.com/para1lel/oh-my-vllm/commit/43564930ca5d940e8e4ff2643c501c9de1784195).

### 算子

下表为[建模负载](docs/operators/evaluation.zh.md)上的期望 roofline 效率.
每种形状使用冷权重, 预热 2 次, 测量 5 次.

| Serving 算子 | Eager | CUDA Graph |
|---|---:|---:|
| FFN | 48.9% | 54.4% |
| FA | 34.4% | 38.1% |
| GDN | 25.1% | 28.5% |
| Vision | 15.2% | 16.1% |
| 采样 | 65.2% | 68.7% |
| MTP context | 23.6% | 27.8% |
| MTP draft | 63.9% | 69.3% |

全部 7 个算子均通过规定的小形状数值检查, 计时覆盖完整.
仅采样和 MTP draft 在两种执行模式下均达到 60% 性能门槛.
整套算子未通过验收.

### 端到端 HTTP

测试不复用前缀缓存, token 预算为 32768, 最多同时运行 4 个请求.
缓存容量为 1024 个 FA block 和 128 个 GDN slot.
每个请求生成 32 个 token.
下表为完整 batch 延迟, 单位为秒, 从并发提交 HTTP 请求计至收到全部输出.

| 模式 | 每请求输入 token 数 | Batch 1 | Batch 2 | Batch 4 |
|---|---:|---:|---:|---:|
| 普通解码 | 32768 | 3.80 | 7.38 | 14.49 |
| MTP4 | 32768 | 3.69 | 12.33 | 14.64 |
| MTP4 | 65536 | 8.53 | 17.02 | 37.79 |
| MTP4 | 131072 | 22.07 | 47.47 | 86.36 |

全部 12 种配置通过短程功能检查, 共收到 28 份响应.
其 roofline 效率为 8.4% 至 15.8%, 低于 60% 门槛.
这些短测未完成[每请求输出 8192 token 的端到端性能验收](docs/performance.zh.md).

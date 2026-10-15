# oh-my-vllm

本项目提供在单张 NVIDIA B200 上运行 Qwen3.8-27B-FP8 的 Rust 服务.
Python worker 当前为占位实现. 不加载模型权重, 不使用 GPU 计算, 生成随机 token.

安装, 命令和 HTTP 请求见[服务指南](docs/serving.zh.md).

[English](README.md) · [贡献流程](CONTRIBUTING.zh.md)

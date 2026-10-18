# oh-my-vllm

本项目提供在单张 NVIDIA B200 上运行 Qwen3.8-27B-FP8 的 Rust 服务.
Python worker 加载模型, 执行文本, 图片和视频推理.
支持普通解码或 MTP4, 并提供约束解码.

安装, 命令和 HTTP 请求见[服务指南](docs/serving.zh.md).

[English](README.md) · [贡献流程](CONTRIBUTING.zh.md)

## 性能测量

数值检查和完整算子测量见[算子评估](docs/operators/evaluation.zh.md).
完整请求测量见 [HTTP 对比](docs/performance.zh.md).
性能验收需要完成这两项流程.
将测量报告和源码身份保存在仓库外.

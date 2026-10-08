# ADR-004: Rust HTTP 服务

日期: 2026-09-21. 状态: accepted, 已实现.

## 决策

在原离线范围中增加 Chat Completions 和 Responses.
HTTP, 在线接纳, 响应存储, 取消, 调度和逻辑 KV 保留在 Rust.
Template, tokenizer, grammar 和 GPU 计算保留在 Python.
函数工具由客户端执行.
保持 ZMQ/msgpack, 单张 B200 和 block784.

## 影响

提供 stream, 工具结果历史, 思考控制和已保存 Responses 续接.
使用有界, 过期内存, 不跨重启持久化.
保存默认值为 1 小时, 1000 条和 256 MiB 序列化字节.
默认思考为 medium; high 映射为 xhigh.
MTP token 约束在采样前生效.
不支持的 API/schema 功能明确报错.

真实 OMP 验收需要各 API 开启 MTP, 读取文件, 回传工具结果并给出有依据的答案.
不增加 HTTP 吞吐门槛.
[服务合同](../serving.zh.md) 定义支持的子集和资源限制.
[验收](../acceptance.zh.md) 标明实测证据.

# 端到端要求

使用一张 NVIDIA B200, 并遵守[模型契约](operators/model.zh.md).
使用完整 Rust 服务和 Python 推理 worker.
占位输出不能作为推理证据.
[GPU 执行规则](execution.zh.md)规定执行方式和最终验收.

## 配置

每个 batch 同时启动 HTTP 请求.
输入和输出长度适用于每个请求.

| 模式 | Batch 大小 | 每请求输入 token | 每请求输出 token |
|---|---|---|---|
| 普通 decode | 1, 2, 4 | 32768 | 64, 8192 |
| MTP4 | 1, 2, 4 | 32768, 65536, 131072 | 64, 8192 |

共 24 种配置. 最终验收时每种配置必须单独通过.
64-token 输出使测量对 prefill 时间敏感.
保留 8192-token 输出用于长 decode 测量.
使用 32768-token 预算, 最多同时运行 4 个请求.
使用没有缓存前缀的请求.

计时前加载模型.
计时前完成初始化.
计时前构造 prompt 文本.
客户端创建请求线程前开始计时.
客户端将全部响应读取为 JSON 且请求线程停止后结束计时.
包含提交, 请求准备, 排队, prefill, decode, 约束, 采样, 进程通信和输出传输.

不包含模型加载, 初始化和 trace 分析.
不能用首 token 时间代替完整延迟.

## 性能门槛

使用[标称资源及全局公式](execution.zh.md#http-性能).
中间阶段变更可以报告未达标或尚未测量的配置.
最终验收时 24 种配置的效率都必须至少为 50%.
完整 batch 延迟最多为参考时间的 2 倍.
算子计时和公开实现比较没有独立验收门槛.

## 测量命令

将 `MODEL_ROOT` 设为 checkpoint 目录.
通过 `OH_MY_VLLM_GPU_UUID` 选择一个 GPU UUID.
将 `ARTIFACT_ROOT` 设为仓库外的新目录.
按[服务设置](serving.zh.md#启动)构建原生模块和 release 可执行文件.
运行完整比较:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.benchmark --model "$MODEL_ROOT" --artifacts "$ARTIFACT_ROOT"
```

命令为每个模式启动并停止服务.
计时前完成短初始化请求.
它检查 HTTP usage 中的 prompt 长度, 输出长度和无缓存前缀.
默认缓存容量为 1024 个 FA block 和 1024 个 GDN slot.
使用 `--blocks` 和 `--mamba-blocks` 调整这些容量.

使用 `--case ordinary:1:32768` 选择两个输出长度.
使用 `--case ordinary:1:32768:64` 选择一种配置.
使用 `--output-tokens 64` 进行对 prefill 时间敏感的测量.
使用 `--output-tokens 32` 进行短测试.
配置子集或其他输出长度不能通过完整门槛.
重复配置, 缺失配置或任一配置失败都会阻止完整验收.

设置 `OH_MY_VLLM_EXECUTION_TRACE` 后, worker 写入执行 trace.
Trace 路径必须位于仓库外, 且运行前不能存在.
Benchmark 为服务进程设置此变量.

## 下界计算

[参考计算](../development/serving/accounting.py)使用版本 `http-reference-v1`.
它不依赖 runtime 算法选择器或实测校准输入.
[计费假设](accounting.zh.md)规定其算术和省略流量.

[Trace 验证器](../development/serving/trace.py)先检查完整 prefill, 连续接受位置, proposal 依赖和输出数量.
唯一 trace identity 必须匹配服务模式.
Prompt 长度必须匹配 HTTP usage.
Benchmark manifest 将客户端序号映射到唯一响应和 trace ID, 模式及输入/输出长度.
缺失 ID, 重复 ID, 未知 ID 或长度不一致都会阻止计费.

参考调度器按客户端序号接纳请求, 最多同时激活 4 个请求.
它先调度 decode, 再将剩余的 32768-token 预算分配给 prefill 请求.
整数余数分配给较早序号.
未完成的 prefill 停在 784-token 边界.
若还能容纳一个对齐 chunk, 未使用空间分配给较早的待处理请求.
最后的 prompt chunk 可以在边界之间结束.

物理到达顺序和调用边界不控制此调度.
MTP verification 消费上一轮 proposal, 并保留记录的接受结果.
Context 计算停在接受终点; 如果随后有新 proposal, 则再计算一行.
调度器不能创建 proposal, 也不能合并同一请求的不同步骤.

计算先累加所有资源工作量, 再取最大值.
它不累加独立算子时间.
额外 tile, padding, launch, 重复计算或临时复制不能增加计分工作量.

产物包含响应, manifest 及其摘要, 所选 trace 摘要, 资源总量和接受计数.
Manifest 将完整请求计时绑定到产物 identity.
该 identity 绑定源码, 可执行文件, 原生模块, checkpoint, tokenizer, 设备和计费版本.
[产物检查](../development/serving/benchmark.py)拒绝运行期间的变化.
旧测量产物不能作为此版本的验收证据.

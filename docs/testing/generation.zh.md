# 生成与数值检查

## 公共服务采样验收

重复保护测试用独立定义比较有界循环检测, 并检查各草稿前缀的分布.
CPU 用例覆盖多个重叠循环, grammar 必需续接, 提示词隔离, 显式关闭和无效 logits.
GPU 用例检查筛选顺序, grammar 必需续接和各草稿前缀的分布.
HTTP 回归检查公共默认值, null 值和显式布尔值在 Rust, msgpack 和 Python 之间的传递.

Tokenizer 回归检查规范思考结束, 工具起始标记约束, schema, 函数选择, 并行调用和推测回滚.
它们保留标签字面量, 纯文本空白, 禁用工具, 显式非 strict 参数, stop 字符串和总长度.
这些 CPU 用例不能证明模型重复已解决, 也不能证明长程 Agent 验收通过.

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/integration/test_serving_worker.py tests/unit/test_thinking_budget.py -q
```

测试 HTTP 服务使用生产默认值.
两种 API 和各草稿模式共用校验和优先顺序.
默认 presence penalty 为 `0.5`. 独立思考预算默认值为 `-1`.

持续验收必须使用这些默认值. 从原始 diagnostics 和 metric 记录服务默认值及添加服务默认值后的请求采样字段.
最大 effort 和输出 / 上下文限制仍为显式工作负载设置.

离线阶段 / 算子测量和预算边界测试保留各自规定的显式参数.

`tests/integration/test_serving_http.py` 检查真实 Rust HTTP, ZMQ, Python sampler 和 checkpoint 合并, 不执行 GPU 推理.
它检查显式零, null 省略, 服务修改和 worker 启动前的拒绝.
Rust 测试为 CLI 与请求字段共用同一校验函数.

## 原生工具描述兼容性

通过两种 API 使用固定版本 OMP 的原始 Bash 工具定义.
其 `name` 参数具有 `maxLength=48`. 模型 prompt 保留该描述.
请求适配器必须接受该描述, 不添加 string 长度 mask.
OMP 必须拒绝生成的 `49` 字符 name, 在后续模型输入返回该错误, 并完成下一条合法命令.
显式 `strict=true` 必须保留不支持 schema 的拒绝行为.

CPU fixture 使用确定性 API 响应和真实 sandbox OMP 进程.
它不能证明目标模型输出或一小时稳定性.
`OH_MY_VLLM_MODEL` 和 release normalizer 可用时, 它还使用生产 CPU preflight.

```bash
scripts/with-env.sh python -m pytest tests/integration/test_serving_worker.py tests/integration/test_agentic_goal_integration.py -q
```

## 思考预算验收

构建 release binary, 用 `serve --thinking-token-budget 32` 启动各模式服务.
使用 `none`, `mtp` 和 `dspark`, 选择空闲且锁定 UUID 的 B200, 并指定对应启动模式.
脚本通过两种 API 检查零, 小值, 默认, 不限与关闭思考预算.
它检查 stream, JSON, strict tool, 总长度, 有效请求 metric 与源码身份不变.
每次运行后停止所属 service 和 worker. 保留原始请求与响应记录.

```bash
scripts/with-env.sh python scripts/thinking-budget-acceptance.py --mode mtp --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/thinking-budget.json"
```

## 精度

用独立计算的 FP64 参考比较实际舍入输入.
覆盖 prefill, decode, 边界写入, strided view, grouped MTP verification 和 speculative rollback.
BF16 算子输出使用 `atol=rtol=0.03`.
Recurrent state 的 NRMSE 至多 1%, 最大绝对误差至多为参考峰值 2%.
检查每个写入 slot 和未改动的受保护状态.
融合链包括全部舍入, scale, copy 和 reduction 行为.
参见 `tests/probe_worker.py` 和相关内核测试.

`tests/numerical/test_ir_normalized_linear.py` 在 `1/2/4/6/12/24/32` 行检查小行数融合.
Graph 重放读取变化后的输入和权重 scales, 保持紧凑的输出形状, 并保持输入字节不变.
存储偏移用例在 `2/17` 行分别检查三个归一化输入.
之前的精确检查和 FP8 算子容差继续有效.

## 离线执行与数值探针

执行以下命令前构建 release binary. 按 [开发指南](../development.zh.md) 设置环境和模型.

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
```

使用 `--num-speculative-tokens 4` 启用 MTP4.
使用 `--context-repeats 100 --prefix-hit` 跨越 784-token 边界并要求前缀命中.
使用 `--binary` 选择其他构建.
离线固定输出忽略 EOS.

`run --prompt-file PATH` 读取空白分隔的 token ID, 每行一个请求, 最多 32 个请求.
`--arrival-interval N` 按 N 个调度 step 延迟每次接纳.
`--prefix-hit` 先填充 prompt, 随后要求真实缓存命中.
全局 `--scheduler-blocks` 在 worker 报告容量内限制 Rust pool.
`bench` 子命令还提供 `--warmup` 和 `--repetitions`.

以下双请求探针每个请求使用 2048 个输入和 1024 个输出 token:

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-text.ipc --scheduler-blocks 10
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-mtp-text.ipc --scheduler-blocks 18 --num-speculative-tokens 4 --prefix-hit
```

它们要求正抢占计数, 且每个输出的开头和结尾都包含正确城市.
MTP4 必须接受 draft. 前缀模式必须有正初始命中.
检查打印文本, 作为语义 smoke test.
较小 pool 本身不能证明发生抢占.

诊断实际路径数值时, 使用可执行探针作为 worker interpreter:

```bash
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON="$PWD/tests/probe_worker.py" python scripts/smoke-text.py --socket /tmp/probe-text.ipc --max-tokens 32
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON="$PWD/tests/probe_worker.py" OH_MY_VLLM_PROBE_MTP=1 python scripts/smoke-text.py --socket /tmp/probe-mtp.ipc --max-tokens 32 --num-speculative-tokens 4
scripts/with-gpu.sh scripts/with-env.sh python scripts/probe-independent-model.py --max-tokens 32
```

Worker probe 测量真实 GQA, GDN prefill / recurrent verification 和 MTP attention 调用, 保留生产内核.
它基于实际舍入输入, 通过 CPU FP64 计算独立参考并比较结果.
Eager 模式关闭 capture, 让诊断复制观察真实请求.
直接模型探针提供不经过 Rust 调度的短 eager 模型诊断.
性能验收时关闭这些探针和 eager 模式.

# 算子与框架测量

## 验收用 release binary

GPU 采集前, 在 CPU 上将当前 runtime 与已审核的语义契约核对:

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/tooling/test_performance_metadata.py::test_current_runtime_contract_matches_semantic_review -q
```

此检查核对源码摘要, provider, header, compiler flag 和版本 pin, 不初始化 CUDA.
仅在语义审查后修改 `performance/contract.json`. 保持源码与 provider 相等检查启用.

阶段时延采集器 读取仓库 `target/release/oh-my-vllm-zmq-worker`.
显式使用此目录, 防止其他构建位置或旧 binary 改变实测候选:

```bash
export CARGO_TARGET_DIR="$PWD/target"
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
```

框架, 上下文或服务验收前使用此构建.
将 `EVIDENCE_DIR` 设为外部可写目录, 保存完整原始记录和日志.

## 正式算子门槛

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda python benchmarks/kernels.py --numerical-only --output "$EVIDENCE_DIR/operators.json"
```

使用新的外部输出路径. [静态用例](../../development/kernels/cases.py) 定义矩阵.
全部 539 项须通过独立数值, 舍入, 修改与状态契约.
原有 317 个用例 ID 保留. 补充范围增加 200 个合法生产案例与 22 个 attention-gate 案例.

部分选择报告 `coverage.complete=false`, 不能单独证明完整矩阵通过.
诊断可使用 `--operations NAME` 或重复的 `--case-id ID`.

省略 `--numerical-only`, 即在数值验证后执行成对调优测量.
每项包含三轮二十对交错测量, 每个 sample 使用一百次图重放.
分层 bootstrap 的单侧 95% 下界描述相对仓库 CUDA 基线的节省耗时.
算子计时用于诊断. 性能门槛由完整框架的阶段延迟决定.

用例与轮次之间恢复持久状态.
Normalization, Q/K, recurrent 和 convolution 验证须有一次 fast host dispatch, 零次 generic dispatch.
Graph replay 计数不能替代 host dispatch 计数.
完整 GPU 矩阵验证仅数值采集. 独立的选定计时验证候选比较.
来源记录与 profiling 分离见[内核开发](../kernels.zh.md).

## 框架性能

构建当前 release 二进制, 为每次尝试指定新的外部输出路径.
从 [要求](../requirements/performance.zh.md#req-perf-001-阶段时延) 中选择十五行之一.

```bash
scripts/with-gpu.sh scripts/with-env.sh python -m benchmarks.framework --mode ordinary --batch-size 1 --input-len 32768 --output "$EVIDENCE_DIR/ordinary-32768-1.json"
```

使用 `ordinary`, `mtp4`, `prefix` 和 `dspark` 模式, 输入 32768, batch 为 1/2/4.
MTP4 另测输入 131072, batch 相同. `prefix` 模式使用 MTP4.
DSpark 通过 `OH_MY_VLLM_DRAFT_MODEL` 或 `--draft-model` 指定 checkpoint.
输出数量为 4096. 每次尝试执行两次完整预热和五次正式测量.
前缀行必须为每个请求命中 32144 个 token.
其他行在每次重复前重置前缀复用.

采集器记录每个请求的提交, 首 token, 末 token 和 step 边界.
从各请求自己的边界独立计算 prefill 和 decode.
每次重复分别取各阶段最长的请求区间.
清理在末 token 之后执行, 使用独立计时字段.

Worker 缓存有效形状, 状态访问, 草稿运算和编译运算身份.
在正式请求执行后写出 trace.
正式分析使用规范成本模型和官方峰值.
未识别的运算阻止验收.
GPU event 区间是用于诊断的 stream 区间, 可能包含主机提交间隙.

除 prefix-hit prefill 外, 每个阶段的墙钟中位数最多为其理论下界中位数的两倍.

记录 prefix-hit prefill 的时延倍率, 不设两倍门槛.
其波动检查继续有效. Prefix-hit decode 保持两倍门槛.
墙钟时延的相对极差最多为 10%.
保留完整失败集和中断尝试. 调查失败后, 对该负载重新执行两次完整预热和五次测量.

每次尝试绑定源码, 二进制, checkpoint 字节, CUDA module, 硬件, 容量和运行时包.
采集期间保持源码不变.
编译 / cache 审计必须显示稳定执行.
现有日志无法排除全部静默内存编译.
原始日志和 trace 放在外部目录.
审阅使用可移植摘要. 正式验证使用完整原始记录.

未提交源码的实验使用 `--diagnostic`.
诊断记录不提供正式验收结论.
Profiling 与正式计时独立执行.

## DSpark 阈值与草稿统计

`--dspark-confidence-threshold` 选择固定或较短 proposal, 与实际 worker 配置一同记录.
初始设置为 `0.2`. 使用 `0.0` 保留至多 7 个固定数量 proposal.
输出, 上下文和 grammar 限制可减少 proposal 数量.
通过已调度 / 接受草稿数及实测性能比较各设置.
接受率为测量期间接受草稿总数除以已调度草稿总数.

使用 `bench --prompt-file PATH` 对与正式输入不同的文本或工具历史开展实验.
文件每行对应一个请求, token ID 以空白字符分隔.
行数必须等于 `--batch-size`, 每行长度必须等于 `--input-len`.
命令在加载模型或 GPU 前检查这些约束.
实验记录保留文本来源, token 哈希, 模板和采样设置.

将加载的草稿配置和权重哈希与指定 checkpoint 字节比较. 两者必须一致.
实际验证草稿为已调度 candidate. 返回的下一步 proposal 使用不同的诊断计数.
可移植摘要不能替代正式验证所需的原始证据.

源码身份及测量范围见 [验收索引](../acceptance.zh.md).

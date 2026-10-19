# 公开注意力对比

这些工具把 FA 和 GDN 注意力核心与固定版本的公开实现进行对比.
对比包括必要的输入转换和状态导出.
它不测量七个完整服务边界.
这些边界使用[算子评估](../../docs/operators/evaluation.zh.md).
24 个 HTTP 配置使用 [HTTP 流程](../../docs/performance.zh.md).

## 环境

使用与推理分离的环境.
把环境路径和 GPU 设置保存在被忽略的 `LOCAL.md` 中.
[依赖输入](../../requirements/baselines.in) 固定直接依赖版本和源码提交.
[锁文件](../../requirements/baselines.txt) 固定包版本和归档散列.
[头文件清单](headers.json) 固定公开源码头文件的提交和归档散列.

从本地配置设置以下变量:

```bash
export BASELINE_ENV=/external/path/environment
export BASELINE_CACHE=/external/path/cache
export BASELINE_PYTHON=/path/to/python3.12
export CUDA_HOME=/path/to/cuda
export OH_MY_VLLM_CUTLASS=/path/to/project-cutlass
export OH_MY_VLLM_GPU_UUID=GPU-uuid
```

安装环境:

```bash
python3 -m development.baselines.bootstrap \
  --environment "$BASELINE_ENV" --cache "$BASELINE_CACHE" \
  --python "$BASELINE_PYTHON"
```

安装器在解压前检查每个源码归档的散列.
它不修改公开源码.
公开源码归档不包含 Git 子模块.
启动器通过公开的编译器参数提供固定版本的头文件路径.
公开实现的编译器不使用项目的 CUTLASS 头文件.
公开构建禁用无关的分布式专家构建和自动包安装.

GPU 工作使用[启动器](run.sh).
它保留项目 GPU 包装器的锁, 并设置独立的编译器缓存路径.
这些路径包括 cuDNN 编译计划和 CUDA 驱动缓存.

启动器保留 CuTe PTX 和 CUBIN 产物. 报告保存实际使用的 cuDNN GDN 二进制, 文件名包含其内容哈希.
这些文件标识测量使用的核函数. 产物收集不计入计时区间.

报告还使用内容哈希保存编译生成的 CuTe CUBIN 文件. 后续编译不能用不同内容替换这些副本.
编译不使用 CuTe 或 cuDNN 的磁盘缓存. 缓存的可执行文件可能不保留 CUBIN 产物.

## 输入契约

[输入模块](inputs.py) 定义 `CoreInputs` 和 `Prepared`.
本地工厂的签名为 `prepare(case, inputs) -> Prepared`.
通过 `--own MODULE:prepare` 传入模块和函数.
把工厂保存在项目源码树中.

工厂必须使用给定张量.
它不得修改这些张量.
它的 `outputs` 字典包含 `output` 张量.
GDN 还提供私有的 `state_pool` 输出张量.
它的 `reset` 回调在每次调用前恢复可变存储.
该回调在计时区间前执行.
它的 `run` 回调包含所有必要的启动和存储变更.
工厂元数据通过非空路径列表提供 `source_files` 和 `binaries`.
报告记录这些文件的散列.

报告在计时后和运行结束时再次检查这些散列.
它还检查公开包的内容和编译后的公开核函数.

FA 使用以下输入张量:

| 张量 | 形状和类型 |
|---|---|
| `q` | `[total_queries, 24, 256]`, BF16 |
| `cache` | `[pages, 2, 784, 4, 256]`, BF16, NHD |
| `sequence_offsets` | `[batch + 1]`, INT32 查询偏移 |
| `page_offsets` | `[batch + 1]`, INT32 页偏移 |
| `page_indices` | 扁平 INT32 物理页索引 |
| `last_page_lengths` | `[batch]`, INT32 |
| 输出 | `[total_queries, 24, 256]`, BF16 |

缓存已经包含完整上下文.
注意力具有因果掩码, 查询位置位于每个上下文的末尾.
softmax 缩放值为 `1 / sqrt(256)`.
报告记录每个公共实现的中间计算精度.
所有实现都必须通过此边界的相同数值检查.
FA 规划在计时前执行, 因为每个案例的元数据固定.

`inputs.sequence_offsets` 包含主机偏移.
设备张量使用 `inputs.tensors["sequence_offsets"]`.

FA 候选实现使用 FlashInfer 的分页预填充 API 和解码 API.
它们包括自动选择, FA2, FA3, cuDNN, CuTe DSL 和 TensorRT-LLM 后端.
FA3 只能在 SM90a 设备上运行.
CUDA Core 解码使用 FP32 概率, 每个请求支持一个查询.
Tensor Core 解码路径使用 BF16 概率.

当 API 支持查询长度时, 解码 API 可以处理查询长度相同的预填充批次.
适配器在每次使用原子归约的 CuTe 调用前清零输出.
计时调用包含此项工作.
报告把 API, 架构和共享内存限制记录为不支持的路径.

GDN 使用以下输入张量:

| 张量 | 形状和类型 |
|---|---|
| `q`, `k` | `[total_queries, 16, 128]`, BF16, 归一化前 |
| `v` | `[total_queries, 48, 128]`, BF16 |
| `a`, `b` | `[total_queries, 48]`, BF16 门控输入 |
| `A_log`, `dt_bias` | `[48]`, FP32 |
| `state_pool` | `[slots, 48, 128, 128]`, BF16, `[slot, head, value, key]` |
| `read_slots` | `[batch]`, INT32 |
| `checkpoint_rows`, `checkpoint_slots` | INT32 导出行和目标槽 |
| 输出 | `[total_queries, 48, 128]`, BF16 |

初始状态非零.
读取槽和未使用的保护槽必须保持不变.
验证导出每个私有行.
预填充导出每个 784-token 边界和最后一行.
每个序列的位置决定它的首个边界.
实时状态在导出之间保持 FP32.
状态头使用键头 `head // 3`.

公开预填充适配器在要求的 checkpoint 行处分割调用.
它们在调用之间保留 FP32 carry, 并在计时区间内导出 BF16 checkpoint.
当片段起点和长度相同时, 它们合批序列.
它们包含打包, 归一化, 门控, 状态转换和输出复制.
FlashQLA 使用 `auto_cp=False` 保留完整历史.

FlashInfer 候选实现分别启用, 禁用或自动选择上下文并行的预填充路径.
它们还包括 cuDNN 预填充路径和 BF16 状态解码路径.
Cake 路径必须支持给定的头数量和状态类型.
Cake CP 对每个 checkpoint 片段使用一个公开的直接组合计划.
调用者为这些计划提供 graph capture.

cuDNN 计划保留完整历史.
仅当默认计划选择精确的分片状态链或 value 维度切分时, 适配器才使用该默认路径.
报告拒绝允许按衰减阈值截断 warmup 历史的默认计划.
第二个候选设置 `batch_invariant=True`, 为这些形状保留完整历史.

精度策略允许此转换时, chunk API 可以将归一化后的 recurrent Q/K 舍入为 BF16.
每个候选必须通过独立数值检查, 并保留全部私有导出.

## 案例与报告

打印模型关注集合:

```bash
PYTHONPATH=python:. "$BASELINE_ENV/bin/python" -m development.baselines plan
```

为记录的 HTTP 形状同时提供完整执行 trace 和对应的 HTTP 结果文件:

```bash
PYTHONPATH=python:. "$BASELINE_ENV/bin/python" -m development.baselines plan \
  --trace /external/path/ordinary-execution.jsonl \
  --http-result /external/path/ordinary-1-32768-64.json
```

为所有必需文件重复参数.
每个 HTTP 配置提供一个结果文件.
Trace 检查必须表明生成完整, 且符合指定的输出长度.

固定选择包含每个 prefill 形状.
它还包含每个配置和 batch pattern 的首个, 中间和最后一个 decode 形状.
这个选择包含验证调用和草稿调用.
它不声称覆盖每个 decode 上下文.

报告保留完整文件散列和所选源码行.
它在数值检查或计时前记录完整的选中清单.

在不同运行中测量每种模式:

```bash
development/baselines/run.sh "$BASELINE_ENV" "$BASELINE_CACHE" \
  python -m development.baselines run \
  --own development.baselines.gdn_native:prepare \
  --operator gdn_attention --mode eager \
  --output /external/path/gdn-eager.json
```

报告以 `passed: false` 开始.
独立 CPU FP64 参考检查固定的小型数值形状.
限值来自现有命名容差.
大型形状使用相同输入对比公开输出和本地实现.
这些对比包含所有导出状态和不变状态存储.
未检查或检查失败的公开路径不能提供对比时间.

默认 cuDNN 候选还使用一个小型 705-token 数值案例.
这个附加案例使用非零初始状态检查精确分片状态链.
它不改变固定数值案例, 也不提供模型性能结果.

每个公开候选必须在所选 prefill 或循环路径中通过至少一个受支持的独立检查.
所有受支持的检查都必须通过.
不支持的检查保持明确记录, 不计为通过.
错误和检查失败会使候选失去资格.
默认 cuDNN 候选还检查每个计时片段是否选择精确计划.

每次计时调用在状态恢复和 L2 驱逐后开始.
驱逐缓冲区大于所选 GPU L2 容量的两倍.
CUDA event 在调用者 stream 上测量所有必要操作.
流程使用两次预热和五次测量.
报告保留每次时长, 中位数和极差.

每个形状把本地中位数与最快合格公开实现的中位数进行对比.
本地时间必须小于或等于该时间.
每个形状只有通过已检查且适用的公开基线对比才能通过.
为完成总体验收, 提供全部 24 个 HTTP 配置中选定的代表形状.

源码散列, 包身份, GPU 设置, 输入散列和选择规则保留在报告中.
GPU 检查把选中的 UUID 与 Torch 报告的设备 UUID 对比.
把所有报告保存在仓库外.

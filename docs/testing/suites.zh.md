# CPU 与 GPU 测试套件

## 测试分层

- [unit](../../tests/unit/): 关键所有权, 提交/回滚, cache 和故障不变量.
- [numerical](../../tests/numerical/): 独立数学参考, 舍入, schema 和 GPU 状态 witness.
- [integration](../../tests/integration/): 真实进程, HTTP/ZMQ 和 OMP 协议边界.
- [tooling](../../tests/tooling/): 开发检查, 采集器, 静态案例和证据验证.
- [acceptance](../../tests/acceptance/): 真实模型容量和编译单元验收.
- [support](../../tests/support/): 共享构造器和 reference, 无测试模块导入.

目录表示职责. `gpu`, `isolated_gpu` 和 `omp` marker 表示执行条件. 未注册 marker 即报错.
单元测试保留关键不变量. 常规行为由集成测试覆盖, 不再重复.

## CPU 与 GPU 测试套件

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/format_cuda.py --check
scripts/with-env.sh python scripts/check_docs.py
scripts/test.sh cpu
scripts/test.sh full
```

`scripts/test.sh` 构建测试 binary, 提供 `OH_MY_VLLM_TEST_BINARY`.
CPU 模式选择 `not gpu`, GPU visibility 为空.
Full 模式在等待空闲且绑定 UUID 的 B200 前要求模型位置.
模型未配置时, CPU tokenizer / HTTP 集成测试明确跳过.
纯校验测试无需 checkpoint, 仍然执行.

GPU 验收保留全新项目缓存. 不安装预热的项目 JIT cache.

CPU executable 诊断测试检查 executable 生命周期, owner 交集, 等待中的回调, 以及明确的不完整覆盖范围.
测试保留诊断快照, 不保留 executable 的强引用.
HTTP 回归让非空 owner 测量经过 Python msgpack, Rust decoder 和 HTTP diagnostics.
测试检查 tuple 长度 `0/1/2`, owner 交集和零 worker 重启.

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/unit/test_executable_diagnostics.py tests/integration/test_worker_lifecycle.py -q
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/integration/test_serving_http.py::HttpTests::test_executable_diagnostics_cross_msgpack_rust_and_http_boundaries -q
```

Q/K GPU 测试在短输入 dispatch 边界检查 FP64 输出, FP32 边界值, 以及输入改变后的图重放.
保留原数值容差与独立参考.

每个诊断性源码审计任务必须重新成功读取源码.
Grep 输出和之前的历史记录不能满足原有的逐任务源码读取检查.

CPU SDK fixture 保留完整请求体及每个请求体的副本数量.
存在已完成历史时, 两个自动摘要请求可以具有不同的先后顺序.
其他请求须保持相同顺序. CPU fixture 分别测试这两种顺序.

Scripted HTTP fixture 的服务首次就绪 timeout 为 `30` 秒.
这段等待包含 Python import 和 ZMQ 连接. 请求, 恢复与关闭断言保留原有 deadline.

选定 pytest 参数放在 `cpu` 或 `full` 之后.
Full 模式使用两个新的 pytest 进程, 先执行标为 `isolated_gpu` 的 6 个边界 GPU 用例.
边界收集器的 GPU 所有权检查执行时, 没有此前测试的 CUDA context.
Partition plugin 保留所选 node ID 及其重复次数. 空分组不会掩盖失败.

各进程先使用原始收集范围和 pytest filter, 再按 `isolated_gpu` 分组.
这样 `--ignore` 和 `--ignore-glob` 对边界模块继续生效.

使用 `full --evidence-dir "$EVIDENCE_DIR/python-tests"` 指定新的外部证据目录.
未指定时, launcher 打印新的临时目录, 退出后保留文件.
每个已启动的 pytest 进程保留日志和进程出生记录. Pytest 写入 JUnit 结果时保留文件.
缺少分组报告不能视为完整测试证据.
`tests/tooling/test_python_test_partition.py` 用小型 fixture 项目验证 selector, filter 和重复 node 的等价收集, 不执行 GPU 计算.

已有 worker lifecycle 测试在 CPU 上检查模型, callable 和 graph-pool 引用释放.
六项真实模型边界用例检查进程在未改变的 worker shutdown deadline 前正常退出.
移除末尾 `torch.cuda.empty_cache()` 调用使用这些行为检查, 不增加重复源码语句的测试.

测试覆盖调度事务, 缓存所有权, 取消, RPC framing, 服务, 实际路径 FP64 参考和编译模型单元.
GPU 覆盖包括全部 6 个最大上下文用例和 metadata 变化后的 graph replay.

基准所有权检查通过 `nvidia-smi -i` 仅查询选定的 GPU UUID.
查询 timeout 保持 `10` 秒. 查询失败会停止采集.
进程接受条件仍要求精确的所属祖先关系和出生身份.
CPU 回归检查选定设备隔离, 外部进程, 出生身份复用和查询失败.

进程在内核 PID lookup 后, pidfd binding 前被回收时, Linux 可能返回 `EINVAL`.
[Linux 6.8 实现](https://github.com/torvalds/linux/blob/v6.8/kernel/fork.c#L2178) 在 lookup 后检查 thread-group leader.

采集在这个错误后重新检查进程出生身份.
观测器不登记已消失或变化的身份. 相同身份仍导致失败.
其他 `pidfd_open` 错误保留原处理.
CPU 测试检查真实子进程退出, PID 复用及这些失败条件.

`tests/unit/test_graph_capture.py` 在 CPU 上检查实际 allocator 设置与工作区生命周期屏障.
`tests/unit/test_graph_memory.py` 检查活跃共享图及输出保留时的 allocator 释放.
临时分配大于现有各个 default-pool segment 的 `total_size - allocated_size`.
Allocator 能释放新 segment 的全部存储, 同时保留其他 segment 中的活跃 tensor.
新增用例在同一个 default-pool segment 中保留活跃 tensor 和至少 `128` MiB 空闲存储.
Owner, pool, 零驱逐, `32` MiB 释放和精确重放检查继续有效.

`tests/numerical/test_dspark_context_graph_gpu.py` 包含生产使用的 29-by-25600 BF16 上下文投影.
它在捕获 stream 的 pool 外预热, 重置无关 GEMM 图, 施加 allocator 压力, 并重放变化后的输入.
它用 `atol=0.03`, `rtol=0.03` 检查全部五层 cache 输出, 精确检查未修改行的 canary, 并验证工作区预算未变.
完整模型 drift 测试在严格算子容差之外, 使用各自记录的模型级界限.
协议回归使用 scripted worker; GPU / agentic 验收使用真实模型.
记录 warning 和跳过测试及原因.

# 分布式执行契约

## 范围

Rank 控制为模型适配器提供事务所有权, GPU 租约和原生集合通信.
[适配器契约](adapters.zh.md)将这些机制与模型执行分离.
各适配器必须通过自己的数值, 状态, 内存和性能门槛后才能用于服务.

## GPU 租约

使用 [GPU 包装器](../development/serving/gpu.py):

```bash
scripts/with-env.sh scripts/with-gpu.sh COMMAND [ARGS...]
```

将主机设置放在被忽略的 `LOCAL.md` 中.

| 设置 | 契约 |
|---|---|
| `OH_MY_VLLM_GPU_COUNT` | `1` 或 `4`, 默认 `1` |
| `OH_MY_VLLM_GPU_UUID` | 一个 UUID, 仅当数量为 `1` 时有效 |
| `OH_MY_VLLM_GPU_UUIDS` | 长度等于所选数量的有序 UUID 列表 |
| `OH_MY_VLLM_GPU_EXCLUDE_INDICES` | 排除的物理索引, 默认为空 |
| `OH_MY_VLLM_RUN_ID` | 调用方 ID, 为空时生成 UUID |

UUID 必须互不相同, 使用规范小写格式, 并带有 `GPU-` 前缀.
不要同时使用两个 UUID 设置.
列表不能包含空格或空条目.
排除设置优先于显式选择.

无效设置, 缺失设备或非 B200 选择会失败.
显式选择的设备繁忙时等待.

空闲设备最多使用 64 MiB 内存, 利用率为零, 且没有计算进程.
无效或不可用的测量不能证明设备空闲.
自动选择使用物理索引顺序.
显式 UUID 顺序设置逻辑 rank 顺序.
包装器替换 `CUDA_VISIBLE_DEVICES`.

锁按递增 UUID 顺序使用 `/tmp/oh-my-vllm-${uuid}.lock`.
锁文件必须是普通文件.
每次尝试在等待前释放部分锁.
获得全部锁后, 重新检查设备清单和进程.
失败的尝试释放设备组.
重试等待 10 秒, 没有截止时间或公平性保证.

包装器通过 `exec` 启动命令, 并继承锁描述符.
保留描述符, 直到所拥有的全部 GPU 子进程停止.
所有权转移到以 `close_fds=True` 启动的子进程时, 使用 `pass_fds`.
不要移除仍被持有的锁文件.
协作锁不能阻止无关 GPU 使用.

包装器既不改变时钟, 也不停止其他用户的进程.
[CUDA 身份检查](../development/serving/device.py)验证可见设备数量和有序 UUID.
单 GPU 使用方拒绝四设备选择.

## Rank 启动

[组客户端](../crates/zmq-worker/src/rank_client.rs)在启动进程前验证配置.
它使用组协议版本 3, 正 epoch, 预期描述符, 容量限制, 有序 UUID 和正截止时间.
每个所拥有的进程组获得独立的 DEALER 端点.
调用方拥有 IPC 目录.
端点路径必须不存在.
防止其他进程修改 IPC 条目.

每个 worker 仅看见分配给它的 UUID, 本地 CUDA ordinal 为零, 并在就绪前验证该身份.

客户端不拥有 GPU 锁.

回复包含一个 msgpack 帧, 最大为 1 MiB.
连接身份提供来源 rank.
一个启动截止时间覆盖进程创建, 发送, bootstrap 和就绪.
重试不会延长它.
没有消息到达时, 仍检查退出和截止时间.

对于 NCCL, 只有 rank 零提供不透明的 128 字节 unique ID.
全部 bootstrap 回复有效后, Rust 向各 rank 发送该 ID 和新的随机连接 nonce.
原生初始化成功后, 最终就绪必须返回该 nonce.
提前, 重复, 过期或错误的回复会使启动失败.
`none` 传输支持 CPU 对端, 而非 GPU 执行.

[rank 检查器](../crates/zmq-worker/src/group.rs)验证除协商容量之外的全部描述符字段.
容量必须达到请求的最小值, 并为空槽和队列哨兵留出空间.
共享视图必须与其生产者一致.
预算计算所拥有的逻辑槽和空槽, 不重复计算共享视图.
它们不是 GPU 字节预算.
只有全部就绪后, 才提供各 rank 容量和公共最小容量.

## 事务和发布

步骤 ID 递增, 且请求 ID 和事务 ID 唯一.
事务 ID 必须大于此前步骤的事务 ID.
请求数和行数限制在派发前生效.
无效本地计划不会使已就绪的组不可用.

TP/TEP 复制请求工作, 并使用 rank 零作为输出所有者.
DEP 保留各请求的所属 rank.
每个 rank 都收到调度并确认完成.

零行 rank 也遵循此规则.
任何 prefill 都为整个组选择 Eager 执行.
否则, 整个组使用 CUDA Graph 执行.

[类型化接口](../crates/zmq-worker/src/rank_client/model.rs)检查请求顺序, token 切片, 位置, 行数, prefill, 词表和上下文边界.
状态 key 和绑定必须与活动事务, 缓存 schema, rank 放置, 容量和代次一致.
写入绑定包含且仅包含已分配块.
读取绑定来自事务所有者.
两者都不允许重复.

复制的元数据不提供所有权.
模型布局必须使用已声明的绑定.
共享客户端不能检查任意布局引用.

受信任的[策略回调](../crates/scheduler/src/policy.rs)在转移所有权前验证完整身份集合.
客户端在第一次异步等待之前保留分离的事务.
此后调度器阻止后续调度和直接更新.
请求取消不能释放分离的事务所有者.

模型载荷包含已调度分块和状态分配, 而非完整提示历史.
[序列化器](../crates/zmq-worker/src/rank_client/encoding.rs)将载荷限制为 64 MiB, 并为命令元数据另加 1 MiB.
它拒绝未知容器长度, `serde(flatten)` 和 `collect_str`.
回复的限制为 1 MiB.

SHA256 绑定序列化字节, 但不认证 worker 身份.
仅元数据回复和模型回复不能互相替代.

完成信息必须与 epoch, 步骤, rank, 载荷, 本地事务和集合通信计划一致.
worker 仅在设备和传输完成后报告完整操作次数.
仅回传摘要不能证明执行.

客户端拒绝缺失, 重复, 无归属或无效的输出, 并检查推测执行限制.
调度器验证整个批次后, 策略才选择已接受状态.
回复到达顺序不能改变输出顺序.
失败批次的任何子集都不会成为已接受状态.

## 集合通信计划和存储

[Rust 计划](../crates/zmq-worker/src/group/collectives.rs)和 [Python 读取器](../python/oh_my_vllm/operators/runtime/distributed.py)使用相同的规范验证和 SHA256 编码.
版本 1 支持 world size 为 1 或 4, 最多 4096 个操作, 以及空计划.
world size 必须与拓扑一致.
未知字段, 操作和类型无法通过验证.
计数为无符号 64 位整数.
缓冲区总字节数必须能存入有符号 64 位整数.

验证在派发前检查乘积和路由求和.

| 操作 | 每个 rank 的输入元素数 | 每个 rank 的输出元素数 |
|---|---:|---:|
| 使用 sum 的 `all_reduce` | count | count |
| `all_gather` | count | count 乘以 world size |
| 使用 sum 的 `reduce_scatter` | count 乘以 world size | count |
| `route` | 源行之和 | 目标列之和 |

路由包含自身路由和零计数.
输入使用目标顺序, 输出使用来源顺序.
全部 rank 收到相同矩阵.
一个 route 计为一个操作.
worker 独立验证计划, 摘要和操作次数.
Graph key 绑定计划身份和拓扑.

计数和地址在捕获和重放期间保持固定.
适配器提供填充值和有效行元数据.

[原生 API](../python/oh_my_vllm/operators/runtime/collectives.py) 使用一个控制线程和调用方 stream.
在捕获前创建通信器.
在捕获前轮询初始化至 `"ready"`.
外部截止时间和进程终止是必要条件, 因为原生调用可能停滞.
原生检查拒绝错误的计数, 类型, 设备, 步长, 边界和存储重叠.
`cuda:0` 上的连续一维张量使用独立的输入和输出存储.

[私有 staging](../python/oh_my_vllm/operators/runtime/cuda/collective_contracts.cuh) 保护 NCCL LL/LL128 对包围逻辑数据的字节区间的读取.
非空区间仅在地址和字节长度均按 16 字节对齐时直接使用 NCCL.
未对齐的非空区间获得初始化为零且向上取整到 16 字节的私有存储, 并计入 Device 统计.
零元素缓冲区不使用私有存储.
准备好的所有者保留调用方和 staging 张量.
填充不改变传输计数或摘要.

输入复制在 NCCL 前执行, 并在重放时再次执行.
输出复制在主机入队完成后执行.

## 原生完成和关闭

Eager `prepared.run()` 后, 在访问结果或重用前轮询 `comm.poll()` 至 `"ready"`.
该状态证明包含输出复制在内的工作已完成.
在捕获前完成 Eager 预热.
在捕获前为各操作调用一次 `prepared.prepare_capture()`.

在全部 rank 上捕获相同的有序操作.
对每个操作调用 `prepared.run()`.
在截止时间内轮询 `comm.poll_enqueue()` 至 `"ready"`.
然后调用 `prepared.finish_enqueue()`.

在捕获回调返回前完成最后一个操作.

主机入队就绪不能证明设备完成.
捕获期间不要使用 `comm.poll()`.

捕获错误, 取消或超时时, 尝试有界的 CPU/IPC 错误报告.
然后调用 `os._exit(nonzero)`.
不要通过 Device 捕获清理进行异常展开, 也不要在此路径调用 CUDA, abort 或缓冲区释放.

重放后, 在调用方 stream 上调用 `comm.record_completion()`.
在访问结果, 重用或报告 rank 完成前轮询至 `"ready"`.
在整个 graph 生命周期中保留准备好的对象和 workspace 张量.

正常原生关闭时, 销毁源 graph 和可执行实例.
在截止时间内轮询 `comm.abort_ready()` 至 true.
然后调用 `comm.abort()`.
轮询 `comm.poll()` 至 `"aborted"`, 才能释放准备好的对象和张量.
此过程也等待未使用的捕获预留释放.

Graph 回调仅设置标志.
调用线程的清理释放所有者.
强制退出不能证明 Compute Sanitizer 优雅清理.

rank 失败, 无效回复, 传输错误或截止时间到期会永久地使组失败.
任何类型化执行错误都会禁用调度器并关闭设备组.
监督器在释放事务或返回执行错误前停止全部所拥有的进程组和后代.
取消使调度器保持禁用, 并保留事务所有权.
丢弃客户端会启动后台清理.

调用 `close`.
等待其结果后, 才能使用新调度器和 epoch 替换.

取消 close 等待不会取消清理.
IPC 移除检查记录的文件身份.
[正常关闭](../crates/zmq-worker/src/rank_client/shutdown.rs)仅接受绑定的 `closed` 回复, 且全部进程必须在截止时间前以状态零退出.
随后进行进程清理.
取消和截止时间均不能证明终止.

## 命令和门槛

运行 CPU 契约检查:

```bash
scripts/with-env.sh cargo test -p oh-my-vllm-zmq-worker --test group --test rank_client
scripts/with-env.sh python -m unittest tests.test_collective_plan tests.test_collectives
```

构建可选原生传输:

```bash
scripts/with-env.sh python -m development.native_build --module collectives
```

[原生对端](../development/operators/collective_peer.py)提供合成执行, 而非 DeepSeek 适配器.
CPU 检查不能证明 GPU 数值, graph 安全, 内存容量或性能.
模型准备, 物理布局, 已接受状态选择和服务集成是后续工作.

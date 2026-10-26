# KV 卸载

## 范围

Host 缓存保留被中断的请求和可复用的前缀.
Rust 管理逻辑标识, 引用, 准入和淘汰.
Python 管理物理存储和 CUDA 传输.
GPU 缓存为 CUDA Graph 执行保持固定地址.
GPU 容量使用[服务指南](cache.zh.md#cache-内存)中的固定默认值和启动检查.

## 所有权

缓存条目可以处于 GPU 驻留, 换出中, host 驻留或换入中状态.
每次传输在 worker 生命周期内具有唯一标识.
调度器在传输前预留目标容量.
源引用在 worker 报告完成前保持有效.
取消操作不会释放未完成传输仍可访问的内存.
替换 worker 会使该 worker 的全部 host 条目失效.

请求仅在换入完成后才可被调度.
传输期间, 其他已驻留请求仍可被调度.
没有请求可执行时, 服务仍查询传输进度.
计算 stream 不等待无关的换入.

## 存储

Host 内存池在 Ready 前预留可用的 pinned 容量, 并复用空闲存储.
其字节上限包括已分配 slab, 预留空间和传输.
自动上限遵守可用主机内存和进程内存限制.
显式字节上限可以限制此容量.
预留操作等待计算和传输 stream 均空闲.
请求处理不会为每次拷贝分配或释放 pinned 缓存存储.

预留会增加启动时间, 并在首次传输前占用主机内存.
字节上限和前缀保留策略不变.
如果无法分配 pinned 预留空间, worker 启动失败.

容量不足时先删除未使用的 host 前缀.
请求快照在恢复或取消前保持受保护状态.
如果 host 容量不能让处理继续, 服务报告容量错误, 而不是重置请求进度.

独立的 nonblocking stream 执行 H2D 和 D2H 传输.
原生运行时批量提交独立的拷贝.
D2H 拷贝等待源数据的生产者事件.
事件保护源 tensor, 目标 tensor 和 host 切片.
事件查询报告完成状态, 不在 CPU 上等待.
传输可以与计算重叠, 但仍会竞争内存带宽.

## 状态

请求快照包括 target 和 MTP attention block, GDN 状态, convolution 状态和边界 hidden 值.
它包括选中的私有端点, 而不只是逻辑 checkpoint slot.
待验证的随机 draft 保留其 proposal 分布.
CPU 历史, 约束, decoder 状态, 位置, draft 和 RNG offset 保持不变.
卸载不会重新生成 proposal 或重置已计算进度.

可复用前缀包含已验证的 attention block 和匹配的递推 checkpoint.
调度器在 GPU 容量压力要求淘汰时存储可复用前缀, 而不是在每个请求完成后存储.
Host 前缀保留有容量边界, 不保证每个旧前缀均可命中.
部分端点保留精确的 token 数.

Host 前缀不包含其他请求的采样状态.
共享 block 保留引用与写时复制规则.
Host 索引仅在所有必要拷贝完成后发布前缀.
恢复的部分 checkpoint 保持独立读取 slot, 直到第一次续算结果通过验证.

## 验证

所有权检查覆盖延迟完成, 取消, 容量压力和过期结果.
GPU 检查比较恢复后的数值及固定 seed 的后续执行.
时间线检查分别分析传输重叠和分配窗口.

AgentX 测量使用 MTP4 和四个会话树.
其数据集, seed 和正式采样时长保持不变.
[AgentX 指南](agentx.zh.md)给出当前 drain 上限.
[测试范围](../../testing.zh.md)规定其他必要检查.

## Cache 内存

FA cache 容量使用固定默认值: 普通解码为 `1824` 个 block, MTP4 为 `1608` 个 block.
这些默认值针对单张 B200 和默认调度设置, 目标是文本推理时总显存占用达到 85% 至 88%.
Worker 不会根据可用显存计算 block 数量.

使用 `--num-gpu-blocks` 或 `OH_MY_VLLM_NUM_GPU_BLOCKS` 修改容量.
如果提供命令参数, 服务会使用该参数.
否则, 服务会使用已设置的环境变量. 如果环境变量也未设置, 则使用模式默认值.
将参数放在 `serve`, `run` 或 `bench` 之前.

启动时, worker 按 88% 的张量预算检查所选容量.
检查包含已用内存, FA 和 GDN 存储, prefill, 以及保留的 decode 和采样 workspace.
内存快照已包含加载的权重, 检查不会重复计入权重.
容量不足时, worker 会拒绝启动, 不会减小指定或默认的 block 数量.

额外媒体存储会单独报告, 不会减小默认文本 cache 容量.
媒体, CUDA Graph 和 runtime 分配使用剩余显存.
较大的媒体工作负载可能超出可用显存. 此时应减小 block 数量.
启动检查不是进程总显存的严格上限.

[Host 缓存](cache.zh.md)在 pinned 内存中保留被中断的请求和可复用前缀.
请求恢复保留递推状态, 媒体, 待验证 draft 和采样 offset.
其他已驻留请求可以在缓存传输期间执行.

使用 `--host-cache-bytes` 设置以字节为单位的 host 存储上限.
默认上限在启动时计算. Worker 在 Ready 前预留可用的 pinned 容量.
使用零上限关闭 host 卸载以进行比较.

默认逻辑 GDN 容量为 `128` 个槽位.
请求私有状态和临时 checkpoint 使用额外的物理槽位.
使用 `--mamba-blocks` 更改逻辑 GDN 容量.
Worker 在 Ready, 服务日志和执行 trace 的身份记录中报告实际容量和内存计算.
Rust 在 Ready 后创建逻辑池和调度器.
替换 worker 必须提供相同容量, 并满足当前内存预算.


## 前缀 cache

请求可以使用已缓存的完整块, 以及最后一块不完整的前缀.
对于最后一块不完整的提示, 调度器可以在最后一个 token 之前记录 checkpoint.
该 checkpoint 包含同一位置的 FA, GDN, 卷积和 MTP hidden state.
只有对 worker 输出的检查未发现错误时, 调度器才使 checkpoint 对其他请求可用.

对于媒体请求, 调度器只记录完整块 checkpoint.

具有相同提示的请求可以从该 checkpoint 继续计算, 并重新计算最后一个 token.
更长的提示可以从同一个已记录位置继续计算.
Cache 不能从没有 GDN checkpoint 的位置继续计算.

继续计算之前, worker 将共享的 FA 尾部复制到私有块.
复制内容包括目标层和 MTP 层的 K 与 V.
GDN 读取 checkpoint, 并写入另一个 slot.
调度器保留源块, 直到它对 worker 结果的检查未发现错误.

没有足够容量分配这些私有块时, cache 使用完整块.


模型临时缓冲区使用 [CUDA 内存池](../../../python/oh_my_vllm/operators/runtime/cuda/scratch_allocator.cuh#L19)<!-- source: inline cudaMemPool_t scratch_pool(int device) { -->.
Worker 在报告就绪前准备存储.
存储大小根据 token 预算确定, 且最多使用空闲 GPU 显存的一半.
内存池保留这些存储供后续计算使用. `gpu_reserved_bytes` 统计值包含这些存储.

## 参考

设计采用 [vLLM 异步传输模式](https://github.com/vllm-project/vllm/blob/390836a3936629475696a2416378cf2f2aee8827/vllm/v1/kv_offload/cpu/gpu_worker.py).
[SGLang 缓存控制器](https://github.com/sgl-project/sglang/blob/d0a75654368a470d8484e3ea45b06e500fee57af/python/sglang/srt/managers/cache_controller.py)也将传输完成与缓存所有权分离.
这些项目是参考, 不是运行时依赖.

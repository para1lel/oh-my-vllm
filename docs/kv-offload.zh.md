# KV 卸载

## 范围

Host 缓存保留被中断的请求和可复用的前缀.
Rust 管理逻辑标识, 引用, 准入和淘汰.
Python 管理物理存储和 CUDA 传输.
GPU 缓存为 CUDA Graph 执行保持固定地址.
GPU 容量使用[服务指南](serving.zh.md#cache-内存)中的固定默认值和启动检查.

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

Host 内存池按需分配 pinned slab 并复用空闲存储.
其字节上限包括已分配 slab, 预留空间和传输.
自动上限遵守可用主机内存和进程内存限制.
显式字节上限可以限制此容量.
内存池仅在计算和传输 stream 均空闲时扩容.
它不会为每次拷贝分配或释放 pinned 存储.

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
[测试范围](testing.zh.md)规定其他必要检查.

## 参考

设计采用 [vLLM 异步传输模式](https://github.com/vllm-project/vllm/blob/390836a3936629475696a2416378cf2f2aee8827/vllm/v1/kv_offload/cpu/gpu_worker.py).
[SGLang 缓存控制器](https://github.com/sgl-project/sglang/blob/d0a75654368a470d8484e3ea45b06e500fee57af/python/sglang/srt/managers/cache_controller.py)也将传输完成与缓存所有权分离.
这些项目是参考, 不是运行时依赖.

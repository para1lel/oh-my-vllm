# 当前状态与待办

更新: 2026-10-14. 重构仍在推进, 最终交付尚未完成.

## 已确认范围

保持单 B200 Qwen3.8-27B-FP8 推理, Rust 服务/调度器/逻辑缓存, Python GPU 所有权, 以及 ZMQ DEALER/msgpack.
保持 block 大小 784, 16 个 FA 层, 48 个 GDN 层, `mamba_cache_mode="align"`, 以及 262144 个总 token.
保持 API 行为, 数值容差, ordinary/MTP4/DSpark, DSpark 权重不变.

通过仓库 CUDA 使用原生 CUDA 库. 移除外部 Python 算子 provider 和项目比较依赖.
保持独立数值/状态契约. 框架阶段时延须达到下界 2 倍门槛, prefix-hit prefill 保持例外.
保持理论模型, 两次 warmup, 五次重复和 10% spread 门槛.

## 里程碑

1. 文档层次, 源码锚点, 单词限制和定义空行: 已完成并独立审查.
2. 职责集中的代码模块和关键测试分层: 已完成并独立审查.
3. 原生 CUDA provider 迁移, 独立 reference, TileFoundry rebase 和 harness 适配: 已完成并独立审查.
4. Profiler, 请求图, 性能调优和完整验收: 待完成.

每个里程碑 commit 前获得独立 review. 在 `main` 提交, 持续推进到全部工作通过.
不 push, 不开 pull request.

## 验证范围

第一阶段通过完整必需 Rust/style/doc 检查, 55 个选定 Python 测试, 14 个 Node 测试及 9 个浏览器测试.
教程构建成功. 独立审查问题已修正.

第二阶段通过必需 Rust/style/doc 检查, 14 个 Node 测试和 9 个浏览器测试.
完整 CPU 运行通过 1572 个案例及 156 个 subtest; 一个过时环境路径断言失败.
修正后全部 7 个环境测试通过. CPU 没有其他失败.

GPU 分组保留全部 487 个节点, 其中 6 个边界案例隔离运行. 这是收集验证, 并非 GPU 执行.

第三阶段通过必需 Rust/style/doc 检查, 14 个 Node 测试和 9 个浏览器测试.
广泛 CPU 收集通过 1073 个案例及 56 个 subtest; 一个过时证据路径断言失败.
替换为保留的案例 ID 后, 全部 9 个相关 formal-case 测试通过.
最终 harness 检查通过 47 个案例及 31 个 subtest, runtime-contract 检查全部 32 个案例通过.

选定原生 GPU 检查通过 76 个案例, 包括 attention, GDN 状态/replay/非法 offsets, FP8, logits 和 TileFoundry twins.
真实模型 ordinary 冒烟推理完成并释放 worker.
Upstream rebase 后全部 22 个 TileFoundry analysis 及一个实际 CUDA twin CLI 检查通过.

全部 GPU 测试, 框架阶段, 六个容量边界及两次一小时运行留到第四阶段.
下述算子结果仅适用于其记录的源码. 最终源码矩阵仍待运行.
未声明新源码完整 GPU 通过验收. 历史修复结果属于源码 `b98a83a` 及此前记录的提交.

第四阶段的诊断 checkpoint 添加默认关闭的小窗口 Torch profiler 和暗色 cache 命中率请求图.
选定 CPU 检查通过 117 个案例和 11 个 subtest. Rust benchmark 边界检查全部三个案例通过.
Profiler GPU 测试及加载 checkpoint 的小 benchmark 通过. Trace 记录两个 step 和 3930 个 CUDA kernel.

Checkpoint 前通过必需的 Rust, 样式, 文档和教程检查.
独立复核确认 profiler 清理, 默认关闭语义和双语文档范围.

已提交源码 `a17b031` 通过完整 CPU 测试: 1587 个案例和 153 个 subtest.
该版本的正式 517 个算子案例矩阵通过, 覆盖完整且没有干扰.

下一个 CUDA checkpoint 改善 GDN chunk 执行和小形状 FP8 投影, 并保留独立数值与状态契约.
选定的集成 GPU 检查通过 29 个案例. 新增的尾 tile 和非对齐存储检查通过七个案例.
修正前, 一个测试中的 canary 变量遮蔽了输出存储. 修正后全部七个相关检查通过.

独立复核没有发现正确性阻断, 允许刷新源码身份, 无需修改下界模型.

调优源码的诊断矩阵通过全部 517 个算子案例, 覆盖完整且没有干扰.
一次框架诊断完成两次预热和五次测量, 没有 OOM 或抢占.

单个 ordinary 32768-input/4096-output 请求的稳定 prefill 约 1.57 秒, decode 约 42 秒.
两倍门槛尚未达到. 这些诊断结果不属于正式验收.
进一步调优后必须重新采集最终源码矩阵和框架阶段.
外部 CUDA 候选不属于生产变更或验收证据.

源码 `af749d3` 的完整非边界 GPU 分组通过 490 个案例, 失败 20 个案例.
失败原因是永久 stream 槽数量限制以及 Q/K 计时参考选择错误.
修复强制原生调度器使用零 workspace, 并使用每设备一个常驻 sentinel.
Q/K 计时使用仓库 CUDA, 数值观察使用独立 Torch, 并拒绝选择计时基线.

修复后全部 49 个相关 GPU 检查和 56 个选定 CPU 检查通过.
独立复核没有发现阻断. 必需 hooks 通过.
必须在最终源码上重新运行完整 GPU 分组.

在 `10d45c9` 上重复运行的非边界 GPU 分组全部 510 个案例通过.
此前一次运行在检测到其他用户的 GPU 进程后停止. 该次记录单独保留.

下一个 checkpoint 修复相关 key 下 GDN 状态超出容差的问题, 并改善小形状 FP8 投影.
参见 [GDN](../python/oh_my_vllm/kernels/gdn.py) 和[原生 GEMM](../python/oh_my_vllm/kernels/cuda_backend/groupwise.py).
独立复核检查了存储, replay, 源码身份以及保持不变的理论规则.

GDN GPU 检查通过 72 个案例. 修正后的确定性 graph 选路检查单独重复运行并通过.
Split-K 检查通过 26 个 GPU 案例. 七个选定精度与 replay 案例通过 Compute Sanitizer, 错误数为零.
完整 CPU 测试通过 1589 个案例和 153 个 subtest. 必需 hooks 通过.

未提交源码的诊断矩阵全部 517 个算子案例通过, 覆盖完整且没有检测到干扰.
该记录不属于正式验收. 当前源码的完整 GPU, 容量, 阶段以及一小时验收仍待完成.
模型探测仍属于诊断, 没有达到两倍延迟门槛.

已提交源码 `51dea95` 通过正式 517 个算子案例矩阵, 覆盖完整且没有干扰.
其完整非边界 GPU 分组通过 586 个案例, 一个 MTP 元数据变化后的 greedy 等价检查失败.
失败记录已保留. 该结果不属于完整 GPU 验收.

修复保留注意力 sigmoid 的 BF16 舍入, 随后才计算 BF16 乘积.
参见[注意力门控](../python/oh_my_vllm/ir/attention_gate.py).
原有完整模型编译测试及门控精度, replay 检查共五个案例全部通过.
独立原生门控对全部 BF16 门控输入位型均与 Torch 一致.

选定 CPU 检查通过 23 个案例和两个 subtest. 正式案例及覆盖检查通过 16 个案例.
独立复核修正了主机 launch grid 溢出, 没有发现其他源码阻断.

正式 harness 新增 22 个完整和选定门控形状, 不删除此前 517 个 case ID.
其数值门槛为精确一致. 当前源码正式矩阵及完整 GPU 验收仍待完成.
理论方程保持不变. 框架阶段, 容量及一小时验收仍待完成.

带过时比较来源的摘要已从工作树删除; Git 历史保留其原始结果.
当前采集范围见[验收索引](acceptance.zh.md).

## 持续服务例外与清理

用户要求保留的 CPU 教程预览继续运行. 主机地址与进程出生身份记录在被忽略的 `LOCAL.md` 中.
每次运行后停止任务拥有的 GPU worker, 子进程和临时推理 listener.
Agent 测试使用可删除 sandbox 中新建的外部工作目录. 结束后删除工具副本, 生成文件, 容器和自有镜像.
原始测量和临时调优放在外部存储.

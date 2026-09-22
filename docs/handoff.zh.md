# 交接记录 — 2026-09-22

## 当前 CUDA 迁移状态

当前状态以本节为准。`cuda-development.md` 和进度产物中的旧阶段数量属于
历史记录，不是当前结果。

干净提交 `c36d1c9` 上的原生 CUDA 实现通过了全部 **147** 项静态推导的算子
配置对比，对照为冻结的 TileLang 实现。每项均通过三轮独立预热、每轮二十对
交替顺序测量及单侧95% 配对 bootstrap 收益下界大于零的判定。原先173项的
清单早于生产路径的全注意力准备融合，已不再是当前矩阵。冻结 TileLang 只在
`kernels/tilelang_reference` 保留一份实现。

完整 CUDA 正确性测试通过 **173 项测试及28个子测试**，没有放宽参考或容差。
真实模型的普通/MTP4 eager FP64 探针通过。普通/MTP 的真实文本隔离、前缀
复用、强制重算抢占通过。输入258048、输出4096的普通/MTP batch1/2/4共六组
边界测试均完成，输出数量准确，无 OOM、无抢占。

十二组框架矩阵为输入32768的 ordinary/MTP/prefix、各 batch1/2/4，以及仅
ordinary 模式的输入131072、batch1/2/4；所有输出长度均为4096。只与冻结的
vLLM `e9f169d16b9408bb9ae44f75072b91a5521d733c` 比较，没有引入 vLLM 运行时、
源码/环境依赖，也没有重新跑 baseline。十二组均通过吞吐>=95%、TTFT<=110%、spread/median<=10%的门槛。
比率以及完整的失败/排除尝试均记录在当前验收产物中。

MTP4 服务约束、生命周期及长严格 JSON 请求的前缀复用通过。首次 Chat/Responses
的 oh-my-pi 运行已验证真实工具调用、后续请求及接受的草稿 token，但最终回答
混淆了历史与当前数量；Responses 还误称 prefill 排在最后。实际批次规划将
prefill 排在前面。最终收尾前仍需基于更新后的文档重新读回。

完成该读回和默认选择切换之前，默认后端仍为 TileLang。正式采集期间冻结运行时
和 kernel 实现。当前证据见 `acceptance.md`；CUDA/PTX 选择、TileFoundry 估算
与实测 Nsight 观测的区别，以及历史调优记录见 `cuda-development.md`。

## 剩余工作

- 使用 MTP4 和真实工具结果，验证更新文档后的 Chat/Responses 读回。
- 将默认后端切换为 CUDA，并让运行时身份报告同一已选后端；保留显式
  `OH_MY_VLLM_KERNEL_BACKEND=tilelang` 对照。
- 完成相关检查、独立审查，在 main 提交并推送。
- 停止所有本任务启动的 GPU 程序，核实 GPU 和端口资源已释放。

所有采集进程均已退出；nvidia-smi 没有计算进程，18030端口已释放。没有要求
保留运行的服务。全文件 Rust fmt/行宽/clippy/测试及 Ruff 检查通过。

## 操作规则

使用 `scripts/with-env.sh`，通过 `scripts/with-gpu.sh` 选择空闲 GPU UUID。
只在 main 开发，只暂存有意修改的文件，保持 hooks 开启，同步中文 Markdown，
保留规定的提交署名。GPU 服务或子 Worker 完成任务后必须退出。临时算子实验和
原始 GPU trace 留在仓库外；正式算子/框架验收摘要存入 `bench/baseline`。

历史工作保留在 Git 和带日期的产物中。独立运行时和 TileLang 迁移曾分别验收
通过，这些历史结果不能替代当前 CUDA 验收。

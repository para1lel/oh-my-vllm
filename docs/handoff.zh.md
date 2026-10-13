# 当前状态与待办

更新: 2026-10-13. 重构仍在推进, 最终交付尚未完成.

## 已确认范围

保持单 B200 Qwen3.8-27B-FP8 推理, Rust 服务/调度器/逻辑缓存, Python GPU 所有权, 以及 ZMQ DEALER/msgpack.
保持 block 大小 784, 16 个 FA 层, 48 个 GDN 层, `mamba_cache_mode="align"`, 以及 262144 个总 token.
保持 API 行为, 数值容差, ordinary/MTP4/DSpark, DSpark 权重不变.

通过仓库 CUDA 使用原生 CUDA 库. 移除外部 Python 算子 provider 和项目比较依赖.
保持独立数值/状态契约. 框架阶段时延须达到下界 2 倍门槛, prefix-hit prefill 保持例外.
保持理论模型, 两次 warmup, 五次重复和 10% spread 门槛.

## 里程碑

1. 文档层次, 源码锚点, 单词限制和定义空行: 已完成并独立审查.
2. 职责集中的代码模块和关键测试分层: 待完成.
3. 原生 CUDA provider 迁移, 独立 reference, TileFoundry rebase 和 harness 适配: 待完成.
4. Profiler, 请求图, 性能调优和完整验收: 待完成.

每个里程碑 commit 前获得独立 review. 在 `main` 提交, 持续推进到全部工作通过.
不 push, 不开 pull request.

## 验证范围

第一阶段通过完整必需 Rust/style/doc 检查, 55 个选定 Python 测试, 14 个 Node 测试及 9 个浏览器测试.
教程构建成功. 独立审查问题已修正.

GPU 验收与完整 Python 测试将在后续源码稳定后执行, 本阶段未运行.
未声明新源码 GPU 通过验收. 历史修复结果属于源码 `b98a83a` 及此前记录的提交.

带过时比较来源的当前摘要将从工作树删除; Git 历史保留其原始结果.
当前采集范围见[验收索引](acceptance.zh.md).

## 持续服务例外与清理

用户要求保留的 CPU 教程预览继续运行. 主机地址与进程出生身份记录在被忽略的 `LOCAL.md` 中.
每次运行后停止任务拥有的 GPU worker, 子进程和临时推理 listener.
Agent 测试使用可删除 sandbox 中新建的外部工作目录. 结束后删除工具副本, 生成文件, 容器和自有镜像.
原始测量和临时调优放在外部存储.

# 算子实现计划

使用[模型契约](model.zh.md), [融合边界](fusion.zh.md)及[评测流程](evaluation.zh.md).
保持六类请求模板. 优化期间不改变其权重.
在 `main` 上开发. 原始测量产物保存在仓库外.

## 共享接口

运行时模块使用原生 FFI 绑定和执行前分配的张量.
开发 provider 提供 `run()`, `reset()` 和 `check(execute=True)`.
benchmark 包含每个 serving 边界中的全部 launch.
数值检查及每次测量前重置可变状态.

1. 添加共享 BF16 投影和残差归一化.
   使用 `cuda_backend/linear.cuh` 和 `normalization.cuh`.
   在 `native.py` 和构建工具中追踪各原生模块的递归源码依赖.
   检查传递依赖的缓存失效及缺失头文件.

2. 在 `cuda_backend/ffn.cu` 中优化完整 FFN.
   使用相同 CPU 参考及冷权重策略比较小行数和 prefill 投影变体.
   保留有收益的激活和量化融合.

3. 在 `fa.py`, `gdn.py` 及其原生模块中实现 FA 和 GDN.
   检查分页因果注意力, MRoPE, 私有写入, 递归 carry 和 784-token 检查点.
   检查全部持久写入及未改变的 guard 存储.

4. 在 `vision.py` 及其原生模块中实现视觉算子.
   首块包含原始图像准备, 末块包含空间合并.
   检查 patch 顺序, 帧间隔离, 归一化, 激活和输出值.

5. 在 `sample.py` 中实现采样, 在 `mtp.py` 中实现 MTP.
   包含词表投影, 惩罚, mask, 固定 seed 选择, 接受, 拒绝及已接受状态槽位.
   draft 计算复用 FA 和 FFN.
   检查并列值, 遮罩行, 已接受前缀, 拒绝及 bonus token.

6. 将全部 provider 接入评测 CLI.
   每份报告绑定全部原生二进制和 checkpoint shard.
   分别在 eager 与 graph 模式的不同运行中测量全部建模 case.
   数值检查后采集 profile. 优化测出的瓶颈.
   使用[性能计算](evaluation.zh.md#性能计算)测量每个 serving 算子.

## 交付

仅使用少量行为检查. 添加新检查前说明其必要性.
明确数值抽样范围. 不宣称未经检查的元素具备精度证据.
provider 缺失或检查失败时不得验收.

取得独立审查. 修正问题. 更新交接. 提交变更. 推送每个里程碑.

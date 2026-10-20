# 算子结构设计

本文给出算子重构的目标结构.
[交接说明](../handoff.zh.md)给出实现状态.
[运行时契约](../requirements.zh.md), [模型契约](model.zh.md)和 [GPU 执行规则](../execution.zh.md)规定行为.

## 目的

将一个算子的接口, 执行计划, 临时存储和 CUDA 实现放在同一个包中.
模型按依赖顺序调用完整子层.
算子可以修改内部 kernel, 而无需修改模型调用处.
重构可以改变融合边界和执行路径.
反复进行的测量必须表明重构没有造成性能下降.

设计为各算子使用相应的函数和类型.
不引入公共算子基类, backend 注册表, 或通过运算图进行自动融合.

## 包与依赖

目标包是 `python/oh_my_vllm/operators`:

```text
operators/
  runtime/
  primitives/
  fa/
    api.py
    plan.py
    workspace.py
    cuda/
      entry.cu
      input.cuh
      attention/
      output.cuh
  gdn/
  ffn/
  output/
  mtp/
  vision/
```

此目录树表示职责划分.
文件大小或职责使拆分成为必要时, 再拆分文件.
在各算子包内按计算阶段或算法组织 CUDA 文件.

Worker 调用算子接口.
算子包使用 `runtime` 和 `primitives`.
MTP 包还可以使用 FA, FFN 和模型输出提供的显式内部组件接口.
两个共享包均不导入算子包或 worker.

`runtime` 包提供分配, 张量视图, 字节复制, stream, CUDA Graph 资源和 native 模块加载.
`primitives` 包提供多个算子包共用的计算和张量操作.
按索引收集行的操作属于这些原语.
模型专用归一化和 hidden 状态写入属于模型输出.

仅当使用者具有相同的计算或资源契约时, 才将代码移入共享包.
融合专用代码保留在所属算子包中.
算子包的数量不决定 native library 或 kernel launch 的数量.

## 接口与所有权

执行计划和算子执行存储使用[项目术语](../glossary.zh.md)中的定义.
每个算子族有自己的计划类型, 输入类型, 状态视图和输出字段.
以下调用表示接口形式, 不是已实现的函数:

```python
result = fa.run(weights, workspace, inputs, batch_view, cache_view)
```

| 对象 | 职责 |
|---|---|
| 准备后的权重 | 存储一个模型层准备后的张量. 两个或更多算子执行存储可以持有这些张量的引用. |
| 执行计划 | 选择实现, 规定存储, 准备工作和执行条件. |
| 算子执行存储 | 控制其操作使用的临时张量和可变 native 执行 context 的生命周期. |
| 批次视图 | 提供算子使用的请求和 token metadata. |
| 状态视图 | 引用持久 cache 张量以及已分配的读取和写入 slot. |
| 具名输出 | 向调用者提供算子执行存储中输出张量的固定视图. |

每次调用都显式传入层权重.
不要在层间修改算子对象的权重字段.
执行条件允许时, 在顺序执行的模型层之间复用临时存储.
不要仅为绑定权重而给每层分配完整的算子执行存储.

Worker 根据调度数据创建范围明确的批次视图和状态视图.
算子不依赖 worker 的请求, 批次或 cache 类.
每个算子准备其实现专用的 metadata.
视图使用已有张量, 不复制 cache 张量.
保留主机准备工作和设备 metadata 传输.

持久状态存储的生命周期独立于算子临时存储进行管理.
Rust 和 worker 保持现有的逻辑身份管理与状态选择职责.
算子仅对视图提供的状态位置进行读写.

## 规划与执行

计划构建选择实现, 不分配 GPU 存储或执行 GPU 操作.
等价输入产生等价计划.
计划提供复用条件和必要的存储描述.
算子在 Graph 录制前准备 buffer 和 native context.
准备过程可以使用适用的已准备权重.

Worker 为录制的模型操作组合各算子的复用签名.
签名涵盖算法选择, 容量, 布局和影响录制 launch 的标量值.
签名中可以用受支持的规范化边界值替代可变标量值.
不能仅因 buffer 大小不变就从签名中省略某个值.
资源身份检查与计划等价性分别处理.

Worker 在调用者 stream 上更新固定地址的 metadata.
新 metadata 必须满足已录制计划的条件.
超出这些条件的变化会在执行前触发兼容算子执行存储的选择或准备.
Worker 不复制算子分派阈值, 也不通过查看实现字段来选择准备工作.

各算子规定自己的准备工作和 warmup 操作.
不存在执行所有算子的 warmup 流程.
采样和 MTP acceptance 保持现有的录制与 RNG 行为.
[GPU 执行规则](../execution.zh.md)在批次划分前控制完整 Execute 消息.

## 输出与 Graph 生命周期

具名输出视图在录制前即可取得.
调用返回同一组预先准备的输出视图, 不隐式分配或复制.
录制操作使用的输出地址保持固定.
下一次复用算子执行存储或重放 Graph 时, 输出内容可以改变.
持有结果对象不会保留之前的张量值.

在有序的调用者 stream 上, 先排入依赖数据的读取操作, 再排入后续写入.
需要跨复用保留数据时, 显式保存数据.
此顺序不会在模型层之间增加设备同步.

不含可变执行状态的计划可以共享.
可变 native context 属于其算子执行存储及 Graph 生命周期.
不要在独立的 Graph 生命周期之间或与不兼容的 Eager 执行共享该 context.

模型执行资源持有者保留录制操作引用的所有资源.
这些资源包括各层权重, 持久 cache 引用, buffer, native owner 和资源租约.
只保留最后一层的权重绑定并不足够.
[Graph 资源规则](../execution.zh.md#graph-资源)规定录制失败处理, 关闭顺序和释放行为.

## 模型输出与 MTP

FA, GDN, FFN 和 vision 接口提供完整子层.
其内部阶段遵循[模型契约](model.zh.md).
模型输出与 MTP 接口分别承担以下职责:

| 接口 | 计算 |
|---|---|
| `output.hidden` | 最终归一化及 hidden 状态导出, 不进行词表投影. |
| `output.distribution` | 词表投影, 惩罚, 约束和所需概率分布. |
| `output.sample` | 分布计算和 token 选择. |
| `mtp.context` | MTP 上下文准备及其 FA 组件. |
| `mtp.draft` | 使用共享 FA, FFN 和输出组件的完整 MTP draft 计算. |
| `mtp.accept` | 接受判断, 拒绝采样和候选状态 slot 选择. |

各输出接口共享必要的最终归一化计算.
Hidden 导出支持其完整输入行数, 独立于词表投影所选择的较少行数.
分布接口可以在不选择 token 的情况下执行.
这些语义接口可以使用融合的物理执行.

MTP 包包含其组合融合实现, 并控制其存储布局.
组件接口规定兼容的输入, 输出, 存储使用方式和生命周期.
它不根据其他算子的私有 buffer 下标推断兼容性, 也不复制该算子的完整 scratch 布局.

MTP acceptance 返回候选结果, 不返回已发布状态.
Worker 先对返回的 token 应用请求停止条件.
随后由停止条件处理后的接受数量通过现有状态提交过程控制状态选择.
处理停止条件前, GPU 候选 slot 不能替代这一决策.
保持 target, proposal 和 acceptance 各域的 RNG 状态与计数器规则.

## Native 构建

开发构建器和运行时身份检查使用同一份源码路径映射.
目录变化还必须同步修改 include 依赖发现和源码指纹.
保留完整的依赖哈希, 不可变二进制身份和 native library 身份检查.
推理加载器不导入开发工具, 也不调用编译器.

初期只改变源码位置, 保持现有 native 编译单元.
仅当实现有经证实的理由时才改变编译边界.
源码包, 语义接口, native library 和 kernel launch 是不同的边界.

## 迁移与验收

1. 移动源码, 同步修改构建映射和身份检查.
2. 对 FA 及其 MTP 使用者应用接口与所有权设计.
3. 将相同契约应用到 GDN, FFN, output 和 vision.
4. 将新的融合与执行变更和结构迁移分开评估.

随对应实现变更更新[阶段清单](../../python/oh_my_vllm/kernels/specification.py)和 [serving 边界](fusion.zh.md).
迁移各算子时, 移除对私有 buffer 的直接访问及模型层复制的 kernel 选择规则.

使用[测试范围](../testing.zh.md)和[数值流程](evaluation.zh.md).
变更影响 MTP, vision 或长上下文时, 扩展全模型输入.
设备内存或同步行为变化时, 使用 Compute Sanitizer.
新增测试遵循[贡献流程](../../CONTRIBUTING.md).

使用相同负载设置比较全部 24 种 [HTTP 配置](../performance.zh.md).
由抽象引入的性能下降在更多测量中再次出现时, 拒绝该变更.
进行更多测量, 检查性能下降是否由测量噪声造成.
新的融合加速特定负载时, 检查其他配置.
测量产物保存在仓库外, 并更新两个 README 性能表.

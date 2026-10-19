# 数值评测

使用[模型契约](model.zh.md)及[开发命令](development.zh.md).
[Tile 设计](../tile-execution.zh.md)规定执行覆盖范围.
只有 [HTTP 比较](../performance.zh.md)设置性能门槛.

## 数值范围

使用[固定数值用例](../../development/reference/cases.py)进行数值验收.
每个用例都运行 eager 模式和 CUDA Graph.
不对大型建模用例运行 CPU FP64 参考计算.
仅有较小行数不能限制 attention 参考计算量. 上下文长度也决定此计算量.

| 算子 | 数值用例 |
|---|---|
| FFN | 1, 4, 5, 20, 65, 129 和 144 行 |
| GDN | Default 为 1, 4, 5 和 20 行. 一个 embedding 行. 65 行中含两个媒体行 |
| GDN 边界 | Default 为位置 0 的 65 行. 位置 770 的 129 行. 空输入保护检查 |
| FA | 1, 4, 5, 20 和 33 行, 上下文为 256. 五行, 上下文为 786 |
| Sampling | 全部建模用例: 1, 4, 5 和 20 行, 使用完整词表 |
| MTP context | 一个请求共四行. 四个请求共 16 行 |
| MTP draft | 1 和 4 行, 上下文为 256. 包含 `full` 和 `query_only` 两个变体 |
| Vision | 两个帧组, 每组 196 个 patch. 包含 `input`, `default` 和 `merge` 变体 |

偏移位置的 GDN 用例在第 13 和 128 行保存 checkpoint.
它跨越一个 784-token checkpoint, 随后处理一个完整的 64-token 块和一个短尾段.
FA 检查包含 prefill 路径和 cache 页边界.
Vision 检查包含对一个帧组的修改, 并确认另一帧组的输出不变.

65 行 FFN 用例使用 prefill 投影路径.

每个 provider 声明参考检查覆盖的元素.
未检查的元素没有数值证据.
缺少 provider 或检查失败会阻止正确性验收.
不能用参考实现替代缺失的 provider.

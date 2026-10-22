# DeepSeek checkpoint 验证

## 身份命令

使用[开发指南](../development.zh.md)中的开发环境.
将 `MODEL_ROOT` 设为本地 checkpoint 目录.
将 `CHECKPOINT_MANIFEST` 设为仓库和 checkpoint 目录之外的新 JSON 文件.

```bash
scripts/with-env.sh python -m development.model.deepseek_v41.identity \
  --model "$MODEL_ROOT" --output "$CHECKPOINT_MANIFEST" --workers 4
```

该命令检查已发布的 Flash 配置及其预期张量布局.
布局来自模型维度和数学来源, 不来自张量索引.
未知配置字段和不支持的格式会导致错误.

该命令检查张量名称, dtypes, shapes, offsets, 文件长度及索引覆盖.
它拒绝重复 JSON keys, 链接, 路径越界, 重叠存储和未使用的数据字节.
它检测读取期间的文件替换或修改.
该命令不将权重载入 GPU memory.

完整验证为全部已索引权重字节和控制文件计算 SHA256.
所提供的提示编码器和参考源文件存在时, 也绑定它们的身份.
它分别报告 target, vision, DSpark 和 Engram 的数据总量.
这些总量是 checkpoint 存储, 不是运行时内存预算.

使用 `--metadata-only` 仅检查权重头部.
此模式仍计算控制文件的哈希.
其 manifest 标记为 `metadata-only`, 不包含权重 digests.
不得将此模式用于完整权重身份凭据.

该命令不覆盖已有 manifest.
将完整 manifest 及其 digest 与外部数值参考 artifacts 一起保存.
验证或参考生成期间不得修改 checkpoint 文件.
后续参考和运行时集成遵循[阶段计划](plan.zh.md).

## 参考权重读取器

仅供开发使用的 `Weights` 读取器要求完整 manifest 及其匹配的张量布局.
它在构造时检查控制文件, 并在首次使用每个权重 shard 之前计算其哈希.
后续读取拒绝文件替换或修改.
这些检查不能使并发 checkpoint 写入变得安全.

读取器将单个张量或选定行加载到 CPU memory.
Engram 表要求显式行索引, 以防止意外生成完整 FP64 副本.
重复行索引保持请求顺序.
Dense FP8 行选择使用源行对应的块尺度.

独立 NumPy 解码器支持 BF16, F32, E4M3FN, UE8M0 和 packed E2M1 权重.
它先解码 FP4 低半字节, 并乘以对应的 UE8M0 尺度.
它保留有符号零, 并拒绝非有限编码.
输出为 FP64 数学权重数据, 不包含运行时 activation 舍入.

该读取器不执行模型, 也不提供全模型数值凭据.
参考生成必须使用独立模型数学定义, 并限制临时权重存储.
推理代码不得导入这些开发工具.

## Engram 参考

独立 CPU Engram 参考提供 token 归一化, 素数桶, 哈希乘数和行索引.
它使用本地分词器, 不使用 Transformers 分词器适配器.
归一化文本相同的文本 token 共享一个压缩 ID.
不完整的 UTF-8 token 保留原始 token 身份.
空文本和单个空格不会合并.

配置中的压缩词表大小和表行数必须与推导值匹配.
素数桶在各层之间互不重叠.
乘数生成器使用源实现中逐层不同的 PCG64 seeds.
哈希乘积必须在有符号 64-bit 整数范围内.

`hash_rows` 接收压缩 token IDs, 并返回行索引和续算历史.
dead-token 标记 `-1` 阻止向前查找跨越媒体区间.
缺失或被阻断的历史使用压缩 padding ID.
历史最多保留 `max_ngram_size-1` 个 IDs.
dead-token 标记是该历史的一部分.
该参考不分配 pinned storage, 也不执行 Engram projections.

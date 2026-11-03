# 计划: 重建 DeepSeek V4.1 数值 fixture, 去掉外部参考目录

状态: 只是计划, 本文档不记录已实现的行为.
写于 2026-11-03, 在维护者删除外部参考目录之后; 该路径记在被忽略的 `LOCAL.md` 文件中.

## 为什么要写这个计划

冻结参考目录占 492 GiB.
转换后的 checkpoint 分片占 487 GiB: 四份 MP4 模型分片和一个 tokenizer.
Capture 目录占 3.1 GiB: 413 个目录, 每个对应一类算子.
其他 capture 占 1.9 GiB: 各族的 order, mean 和 endpoint capture.

checkpoint 分片占 492 GiB 中的 487 GiB.
它们不是 fixture, 而是转换成 MP4 布局的模型.

`development/model/deepseek_candidate.py` 只读取该目录中的 56 MiB.
这部分是一个含七个文件的目录, 加一个 checkpoint 摘要文件.
656 项冻结检查用该目录中的 endpoint 向量与新跑的四 rank candidate 结果比较.

`--model` 选项本来就可以指向原始 checkpoint 根目录.
除了比较分片字节的运行, 没有检查会读转换后的副本.

## 目标设计

1. **fixture 小而带版本.** 只保留检查真正读取的文件.
   打包成一个不超过几百 MiB 的 bundle.
   加一个列出每个文件 SHA256 的 manifest.
2. **不复制 checkpoint.** candidate 运行接收模型根目录和期望分片摘要的 manifest.
   转换步骤先校验原始分片再读取.
   运行结束后不保留转换副本.
3. **每个捕获族都有生成器.** 每个 fixture 目录记录确切命令, seed 和输入用例集合.
   因此 bundle 可以从 checkpoint 重建.
   每个生成器是仓库内 seed 固定的脚本.
4. **gate 的含义不变.** 656 项检查, 0.01 容差和冻结 manifest 都不变.
   改变的只有存储位置和重建路径.

## 工作项

1. 新增 `development/model/fixture_bundle.py`.
   它提供 `pack`, `verify`, `unpack` 三个子命令.
   `pack` 写出 bundle 和摘要 manifest.
   `verify` 在 gate 运行前校验 bundle.
   `unpack` 把 bundle 展开到临时目录.
2. 把小 fixture 文件移入 bundle.
   目录名保持不变, candidate runner 的路径不变.
3. 用原始分片的摘要校验替代对转换 checkpoint 的依赖.
   把校验结果写入运行输出.
4. 为每个目前在 GPU 上运行的捕获族增加生成器.
   记录 seed, 用例列表, Torch 版本和 CUDA 版本.
5. 在 `docs/models/deepseek_v41/evaluation.md` 及其中文翻译中记录 bundle 位置,
   重建步骤和预期大小.
   原始 capture 放在外部目录.
6. 重建 bundle 后跑完整 656 项检查.
   把结果与已交付报告比较.
   只有比较通过后, 新 bundle 才能替代已记录的证据.

## 改动后的大小

| 项目 | 大小 |
|---|---:|
| fixture bundle | 预期约 0.4 GiB |
| 原始 checkpoint | 已在本机, 不复制 |
| 运行输出 | 外部目录, 运行后删除 |

## 风险

- 重建的 bundle 可能与已删除版本差一位, 这个差别会改变 gate.
  用已交付报告中的数值比较重建结果.
  不能只看 bundle 摘要.
- 部分 capture 来自参考驱动的 release 构建, 本仓库无法重建.
   这些族保留已记录的 endpoint 向量.
- bundle 必须小到能在一台工作站上运行.
   不要加入检查并不读取的 capture 目录.

# 交互代码教程

静态 Twine/SugarCube2.37.3 教程用 13 个中文章节讲解当前推理代码.
每章包含逻辑, 决策, 实现及展示的字段 / 变量.
读者通过兴趣选项和内部前置知识选择路线.
默认熟悉基础 Rust 和 Python 语法.

## 构建与预览

使用 Node24, curl, unzip 和 [开发指南](../docs/development.zh.md) 中配置的 Rust 环境.
在仓库根目录执行:

```bash
npm --prefix code-journey ci --ignore-scripts
npm --prefix code-journey run build
npm --prefix code-journey run serve
```

静态 listener 默认 `0.0.0.0:18084`, 不使用 GPU 或模型.
实际预览地址和进程信息记入被忽略的 `LOCAL.md`.
首次构建下载固定 Tweego, SugarCube 和 Fira Code Nerd Font 产物.
解压或执行前检查 SHA-256.
其余依赖由 npm lock 固定.
字体, KaTeX 和许可证本地提供, 阅读时不依赖 CDN.

构建通过 `scripts/with-env.sh` 使用真实 scheduler/KV crate.
CPU 实验使用 token42 的合成 worker 反馈.
演示调度和提交, 不作为模型输出或 GPU 性能结论.

## 章节与源码

| 章节 | 主题 |
|---|---|
| 1 | HTTP 准备, token ID, 位置, prefill 和 decode. |
| 2 | Rust/Python 进程, 环境和初始化. |
| 3 | 请求历史, batching, 回复校验和 commit. |
| 4 | Token 预算与对齐 prefill. |
| 5 | FA/GDN layout, 分配和 CPU planning. |
| 6 | 前缀哈希, 引用, 队列和抢占. |
| 7 | ZMQ, 注册, 准备, 取消和失败. |
| 8 | 加载, FP8 / 普通 projection, residual, FA, GDN 和 logits. |
| 9 | 采样, grammar, 停止, 解析和 stream. |
| 10 | MTP proposal, grouped verification 和状态所有权. |
| 11 | Semantic IR, lowering, graph cache 和 CUDA 内核. |
| 12 | 正确性, 最大 shape, TTFT, 吞吐和测量审计. |
| 13 | DSpark 目标特征, context KV, 七候选提议, confidence, 概率验证和测量. |

`src/curriculum.mjs` 将每个有实质内容的运行路径 Rust/Python/CUDA 文件映射到章节和入口符号.
构建拒绝缺失覆盖, anchor 和记录字段解释.
记录当前源码行和完整文件哈希.
冻结比较 provider, 测试和仅含文档的 initializer 单独限定范围.
独立库在项目调用点讲解.

条件源码列表在相关行打开本地高亮完整文件.
源码页链接回章节, 支持两种主题.
引用源码变化后重新构建.

## 导航与实验

每个页面跳转都是包含条件和目标的无序列表项.
未读目标先定位到最早缺失的内部前置章节.
继续阅读保留选择的路线.
只有显式完成选项会将章节标记为完成.
复习 / 地图访问保留阅读进度.

经过校验的 localStoragev2 schema 恢复上一章节和实验状态.
旧短 passage 迁移为访问记录, 不算扩展章节完成.
重新开始清除阅读 schema 和 Twine history; 主题选择独立保留.
5 个 prompt 长度为 784, 785, 1568, 1569, 32768.
最后一个 trace 在首个合成输出前调度 32144 和 624 token.
控件使用发布的 Rust scheduler trace.

## 字体与片段

栏宽 820px, 桌面 / 移动正文字号 17/16px, 标题 28/24px, 代码 14/13px.
正文使用 LXGW WenKai; 代码使用 Fira Code Nerd Font Mono, 中文 glyph 回退 LXGW WenKai.
KaTeX 提供公式和可访问 MathML.
亮色和暗色主题都可持久保存.
中文使用半角标点和指定空格, 采用 Humanizer-zh 编辑.
源码标识符和语法保持不变.

章节正文在 `src/chapters/`; 入口 / 地图在 `src/story.twee`.
行内代码使用 `icode` macro 和 `textContent`, 使 Python `//` 保持原样.
`src/source-specs.mjs` 选择片段; `src/source-notes.mjs` 解释字段, 参数, 状态和局部变量.
移动端含义 / 用途表保留明确标签.

`scripts/excerpt.mjs` 删除公共 literal whitespace prefix, 规范空行.
保留相对缩进, 文档, attribute, decorator, body docstring, 源码行号和文件哈希.
展示与复制使用相同规范片段; 完整源码页保留全部原始文件.
Shiki4.5.0 在构建时 tokenize Rust/Python/C++.
可聚焦代码区支持键盘滚动.
Clipboard API 和 HTTP selection fallback 复制展示代码.

## 验证

静态 listener 运行时执行:

```bash
npm --prefix code-journey test
scripts/with-env.sh cargo test --locked --manifest-path code-journey/trace/Cargo.toml
scripts/with-env.sh cargo fmt --manifest-path code-journey/trace/Cargo.toml --check
scripts/with-env.sh cargo clippy --manifest-path code-journey/trace/Cargo.toml --all-targets -- -D warnings
```

Node 测试覆盖文档 / metadata 边界, 去缩进, tab, 空行和相邻元素.
Playwright Chromium 覆盖全部 13 条路线, 选项, 前置知识, 条件列表, 完成 / 复习, 迁移, 历史, 重启和 5 个调度用例.
DSpark 检查包含配置字段, 概率公式, 源码复制和桌面 / 移动端主题.

还检查文档复制, 字段, 字体, 公式, 主题, 源码 hash/anchor/URL, 键盘滚动和至少 4.5:1 的代码对比度.
分别检查桌面 1536x1024 和移动端 390x844 截图.
Browser plugin 可用时优先使用; 否则记录采用 Playwright 的原因.

固定 Chromium 不存在时, 在 `code-journey/` 安装:

```bash
npx playwright install chromium
```

`JOURNEY_URL` 选择测试 URL.
`JOURNEY_QA_DIR` 选择仓库外截图目录; 默认使用临时目录.
生成的 `dist`, `.tools`, `node_modules`, `data` 和浏览器临时文件被忽略.
仓库级检查仍然必须执行.

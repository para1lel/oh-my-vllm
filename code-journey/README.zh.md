# oh-my-vllm 代码之旅

使用真正的 Twine / SugarCube 2.37.3 与 Tweego 2.1.1 构建的中文教程.
12 个完整章节沿当前推理流程讲解 HTTP 输入, 调度, 缓存, 模型计算和输出,
再展开编译与验证. 每章连接问题背景, 执行步骤, 设计取舍, 字段变量说明和当前源码.
默认读者熟悉 Rust 和 Python 的基础语法. 推理概念通过教程内部的前置章节逐步介绍.

## 阅读与运行

用户要求的固定内网预览为 http://10.30.64.14:18084/.
静态监听器绑定 0.0.0.0:18084, 无需 GPU 或模型.

从仓库根目录运行:

    npm --prefix code-journey ci --ignore-scripts
    npm --prefix code-journey run build
    npm --prefix code-journey run serve

需要 Node 24, curl, unzip 和已有 Rust 环境. 首次构建下载固定版本的 Tweego,
SugarCube 与 Fira Code Nerd Font, 解压或执行前检查 SHA-256.
npm 依赖在 lockfile 中固定. 字体, KaTeX 和许可证均本地提供, 阅读无需 CDN.

构建通过 scripts/with-env.sh 调用真实 scheduler / KV crate, 使用必需的环境变量.
CPU 实验用合成 token 42 反馈检查调度与提交契约, 展示真实分块数量,
不产生模型输出或 GPU 性能结果.

## 完整章节

| 章节 | 主题 | 主要实现 |
|---|---|---|
| 1 | HTTP 输入, token ID, 位置, prefill 与 decode | serving/request.rs; worker/serving.py; scheduler/request.rs |
| 2 | Rust / Python 进程分工与初始化 | zmq-worker/main.rs; worker/model_runner.py; runtime.py; logging_utils.py |
| 3 | 请求状态, 连续批处理, 校验与提交 | scheduler/request.rs; lib.rs; output.rs |
| 4 | token 预算与分块 prefill | Scheduler::schedule; Scheduler::aligned_prefill |
| 5 | FA / GDN 布局, 分配与 CPU 计划 | kv-cache/group.rs; coordinator.rs; worker/batch_plan.py; tensors.py |
| 6 | 前缀哈希, 引用, 队列与抢占 | kv-cache/hash.rs; pool.rs; block.rs; scheduler/lib.rs |
| 7 | ZMQ 消息, 注册, 准备, 取消与错误 | zmq-worker/client.rs; protocol.rs; error.rs; worker/zmq_bridge.py; protocol.py |
| 8 | 权重加载, FP8, 残差, FA, GDN 与 logits | models/qwen.py; kernels/fp8.py; attention.py; attention_prepare.py; convolution.py; gdn.py; normalization.py; elementwise.py |
| 9 | 采样, 语法, 停止, 解析与流式输出 | worker/sampling.py; sampler.py; serving.py; serving/mod.rs; parser.rs; events.rs |
| 10 | MTP 提议, 分组验证与状态保留 | worker/mtp.py; kernels/mtp_attention.py; decode_attention.py; batch_plan.py |
| 11 | Semantic IR, lowering, 图捕获与缓存, CUDA 算子 | ir/; worker/decode_graph.py; graph_cache.py; kernels/backend.py; cuda_backend/ |
| 12 | 正确性, 最大形状, TTFT, 吞吐与测量审计 | ir/coverage.py; tests/; benchmarks/ttft.py; measurement.py; development/kernels/cases.py |

src/curriculum.mjs 将实际 Rust / Python / CUDA 运行文件映射到章节职责与实现取舍.
构建逐文件检查默认推理主线中的自有代码覆盖, 核对入口锚点, 记录真实行号和
完整文件摘要. 测试模块, 只有文档的初始化文件和冻结 TileLang 比较实现不属于
默认主线清单; 第 11 章解释比较后端的共同契约. 独立库通过项目调用入口讲解.

各章的条件源码列表可打开本地高亮的完整文件并定位入口行.
Rust, Python 与 C++ 使用统一亮暗配色. 源码页可返回对应章节, 仍检查未读前置内容.
引用源码修改后需要重新构建.

## 阅读与跳转

所有页面跳转使用无序列表, 每项说明条件和目标, 参考用户提供的 Matrix67 示例.
首页记录对基础, 请求管理或模型计算的兴趣. 未读目标先解析到缺少的最早前置章节,
继续链接保留已选路线.

只有明确已理解并继续的选项才确认当前章已读. 复习, 地图和顶部入口保留阅读中.
已读标记在返回历史后仍保留. 校验后的 localStorage v2 恢复最后章节与实验步骤.
旧短文的访问记录迁移为阅读中, 不视为已完成新扩写章节.
重新开始清空两个阅读 schema 与 Twine 历史, 主题偏好独立保存.

实验长度为 784, 785, 1568, 1569 和 32768. 最后一组分成 32144 + 624,
尾段完成时产生第一个合成输出. 控件按实际发布的调度轨迹更新.

## 字体与源码编写

Matrix67 阅读风格采用白底, 蓝灰文字和绿色下划线链接. 栏宽 820px,
桌面 / 手机正文 17/16px, 标题 28/24px, 代码 14/13px.
正文使用 LXGW WenKai, 代码使用 Fira Code Nerd Font Mono, 中文字符回退到 LXGW WenKai,
公式使用本地 KaTeX 并带可访问 MathML. 亮暗主题持久化.

章节位于 src/chapters/. 行内代码使用 icode 宏与 textContent,
让 Python // 等符号保持代码文本, 避免 Twine 排版改变后续 DOM. 入口与地图位于 src/story.twee,
标题, 前置关系与源码清单位于 src/curriculum.mjs.
正文使用半角标点与用户要求的空格, 代码与标识符保留原有语法.
Humanizer-zh 已安装在用户 Codex skills 中, 用于检查重复和空泛表述,
同时保留事实与有意义的限定.

src/source-specs.mjs 选择节选, src/source-notes.mjs 解释所有显示字段,
函数参数, 状态和局部变量. 构建拒绝缺少说明的公开 Rust 字段与 Python dataclass 字段.
手机字段表按含义和用途纵向排列, 保留标签.

scripts/excerpt.mjs 去掉非空行公共空白前缀, 规范空白行.
节选保留元素所附的文档注释, Rust 属性与 Python 装饰器,
以及选定代码内的 Python docstring.
相对缩进, 原始行号与完整文件摘要保留. 显示与复制使用去公共缩进后的节选,
完整源码页保留完整文件. Shiki 4.5.0 在构建时高亮, 阅读无需运行时高亮器.
代码区域可键盘横向滚动, Clipboard API 与 HTTP 选区备用路径都复制显示代码.

## 验证

静态监听器运行时执行:

    npm --prefix code-journey test
    scripts/with-env.sh cargo test --locked --manifest-path code-journey/trace/Cargo.toml
    scripts/with-env.sh cargo fmt --manifest-path code-journey/trace/Cargo.toml --check
    scripts/with-env.sh cargo clippy --manifest-path code-journey/trace/Cargo.toml --all-targets -- -D warnings

npm test 先运行 Node 的文档与元数据边界, Rust / Python, tab, 相对缩进和空白行测试,
再执行固定版本 Chromium 工作流. Playwright 检查文档注释的显示与复制,
确认基础语法说明已移除, 并检查全部 12 章路线, 兴趣和前置关系,
条件无序跳转, 完成与复习, 进度迁移, 五组调度轨迹, 亮暗 / 刷新 / 重置 / 返回,
所有显示字段, 公式, 字体和手机溢出. 同时核对节选与源码摘要和行号,
两条复制路径的去缩进文本, 代码对比度 >=4.5:1, 键盘滚动,
全部完整源码 URL 及 Rust / Python / CUDA 源码页.

缺少 Chromium 时, 从 code-journey 安装固定浏览器:

    npx playwright install chromium

JOURNEY_URL 调整测试地址, JOURNEY_QA_DIR 将截图保存在仓库外,
默认 /tmp/oh-my-vllm-journey-qa. 功能检查之外单独查看实际截图,
桌面 1536x1024, 手机 390x844. 本轮 Browser 插件不可用, 使用 Playwright Chromium.
仓库整体 Rust / Ruff 检查仍需通过.

dist, .tools, node_modules, 生成数据和浏览器临时文件均忽略.
推理运行时, kernel 实现与基线证据未改动.

# oh-my-vllm 代码之旅

这是讲解当前仓库的中文交互文章, 使用真正的 Twine / SugarCube 2.37.3
和 Tweego 2.1.1. 首个切片跟踪一条请求从 token ID 到第一个确认输出 token
的过程. 已实现 11 篇阅读文章和阅读地图. 下面的 12 章大纲覆盖其他核心代码,
后续章节按该大纲继续制作.

## 阅读与启动

固定内网预览地址是 http://10.30.64.14:18084/. 监听器只提供生成的静态站点,
绑定 0.0.0.0:18084, 运行时无需 GPU 或模型.

在仓库根目录执行:

    npm --prefix code-journey ci --ignore-scripts
    npm --prefix code-journey run build
    npm --prefix code-journey run serve

构建需要 Node 24, curl, unzip 和已有的 Rust 环境. 首次构建下载固定版本的
Tweego, SugarCube 和 Fira Code Nerd Font, 在解压或执行前核对 SHA-256,
并将下载缓存保存在 .tools. KaTeX 0.19.0, LXGW WenKai 和 Playwright
已固定在 package-lock.json. 运行资源和许可证由本站提供, 阅读无需 CDN.

Rust 构建通过 scripts/with-env.sh 执行, 设置两个必需的环境变量.
trace crate 直接依赖当前 scheduler/KV crate, 用合成 worker 返回值执行 CPU
输出契约. 页面上的输出数量用于说明这个契约; 测试中的 token 42 是合成值.
页面不运行模型或测量 GPU 性能.

## 阅读行为

入口选项记录读者对基础概念, 请求流程或 GPU 职责的兴趣. 导航先展开未读的
前置文章, 再回到选定目标. 核心文章只假设读者已读过列出的前置内容.

读者选择继续时, 文章标为已读; 入口文章在打开时标记. 访问记录单独区分
已打开和已完成的文章. 返回操作保留已读标记. 刷新从经过检查的 localStorage
结构恢复上篇文章和实验步骤. 重新开始清空文章状态和 Twine 历史.
主题偏好单独保存; 首次跟随系统, 手动选择后保留该选择.

实验包含 784, 785, 1568, 1569 和 32768 五种长度. 源码节选, 行号和完整
文件哈希在构建时提取. 源码锚点不匹配时构建停止. 修改 Rust/Python 后应重新构建.

## 完整大纲

| 章 | 问题 | 当前源码入口 |
|---|---|---|
| 1 | 文字怎样成为输入与位置? | crates/scheduler/src/request.rs; python/oh_my_vllm/worker/serving.py |
| 2 | 谁决定, 谁计算? | crates/zmq-worker/src/client.rs; python/oh_my_vllm/worker/zmq_bridge.py; docs/architecture.md |
| 3 | 请求怎样进入连续批处理? | crates/scheduler/src/lib.rs; request.rs; output.rs |
| 4 | 怎样选择 token 预算与 prefill 分块? | Scheduler::schedule; Scheduler::aligned_prefill |
| 5 | FA 页和 GDN 状态为什么不同? | crates/kv-cache/src/group.rs; coordinator.rs; python/oh_my_vllm/worker/batch_plan.py |
| 6 | 前缀复用, 引用计数与抢占怎样工作? | kv-cache/src/hash.rs; pool.rs; block.rs; Scheduler::preempt_request |
| 7 | 传输, 取消和错误怎样协作? | zmq-worker/src/protocol.rs; client.rs; error.rs; worker/protocol.py; zmq_bridge.py |
| 8 | Qwen 怎样执行 FP8, FA 和 GDN 层? | models/qwen.py; kernels/fp8.py; attention.py; decode_attention.py; attention_prepare.py; gdn.py; convolution.py; normalization.py; elementwise.py; mtp_attention.py |
| 9 | 采样, 语法约束与流式输出怎样工作? | worker/sampler.py; sampling.py; serving.py; zmq-worker/src/serving/request.rs; parser.rs; events.rs; mod.rs |
| 10 | MTP 提议怎样验证与提交? | worker/mtp.py; batch_plan.py; model_runner.py; Scheduler::validate_output; Scheduler::update |
| 11 | Semantic IR, 编译与 CUDA Graph 怎样配合? | ir/core.py; ir/coverage.py; ir/attention.py; ir/recurrent.py; ir/fp8.py; ir/pointwise.py; ir/logits.py; ir/state.py; ir/gdn_prefill.py; worker/graph_cache.py; decode_graph.py; kernels/cuda_backend/kernels.cu |
| 12 | 怎样验证正确性与性能验收? | tests/; worker/runtime.py; logging_utils.py; tensors.py; benchmarks/ttft.py; measurement.py; development/kernels/cases.py; bench/baseline/; docs/acceptance.md |

后续每章连接一个具体问题, 一个可操作的预测, 当前源码, 一项设计选择和检查.
历史 vLLM adapter/V2 ADR 放在注明日期的支线文章. 当前 runtime, 默认 CUDA
和 semantic IR 构成主线. 性能证据沿用验收记录标明的源码版本.

## 设计与写作

Matrix67 参考站提供简洁的分支阅读方式: 真白背景, 蓝灰正文和绿色下划线选项.
Image Gen 分别生成亮色阅读页, 实验页和暗色阅读页. 设计使用开放的 950px 单栏,
明确设置正文与控件字号, 不使用卡片网格. 手机两侧留 24px.
暗色配色为 #161D23, #D5DFE6 和 #8FC99F.

正文使用 LXGW WenKai, 代码使用 Fira Code Nerd Font Mono. 公式宏调用本地
KaTeX, 同时提供 MathML. 内容与控件均由 HTML 和 SugarCube passage 实现.
短代码预览与可展开节选从同一份源码提取.

Shiki 4.5.0 在构建时解析完整的 Rust / Python 节选, 保留跨行语法与空白.
GitHub 亮暗主题的配色按代码背景调整, 随页面主题通过 CSS 切换. 浏览器通过
textContent 创建 token span, 阅读时无需高亮引擎或 CDN. 完整节选保留源码行号;
所有代码块带语言标记和复制按钮, 复制内容为原始源码. HTTP 预览在 Clipboard API 不可用时
使用选区复制. 长代码在可获得焦点的区域内横向滚动, 手机端也可操作.
语言与配色在 scripts/highlight.mjs 配置, 背景与行号样式在 src/style.css 配置.

中文正文使用半角标点, 左括号前留空格, 标点后以及中文与英文或公式之间留空格.
源码, 标识符, 命令和公式保留自身语法. Humanizer-zh 已安装在用户的 Codex
skills 中, 用于检查重复表达和缺少依据的陈述.

## 验证

静态预览运行时执行:

    npm --prefix code-journey test
    scripts/with-env.sh cargo test --locked --manifest-path code-journey/trace/Cargo.toml
    scripts/with-env.sh cargo fmt --manifest-path code-journey/trace/Cargo.toml --check
    scripts/with-env.sh cargo clippy --manifest-path code-journey/trace/Cargo.toml --all-targets -- -D warnings

如缺少 Chromium, 在 code-journey 中调用固定版本的 Playwright 安装器:

    npx playwright install chromium

JOURNEY_URL 可指定测试地址. JOURNEY_QA_DIR 指定仓库外的截图目录,
默认 /tmp/oh-my-vllm-journey-qa. Playwright 检查真实 Twine 标识, 前置关系,
五组边界轨迹, 源码显示, 主题与刷新和重置, 手机溢出以及控制台健康.
语法测试还检查所有节选文本, Rust / Python 主题颜色, token / 工具栏 / 行号至少 4.5:1 的对比度,
两种复制路径和键盘横向滚动.
截图覆盖 1536x1024 桌面和 390x844 手机的两种主题.
单独的 CPU 测试将公布的分块计数与真实 Rust 调度器核对.
仓库全局 Rust/Ruff 检查仍然必需.

当前没有 Browser 插件, 因此使用 Playwright Chromium 验证.
生成的设计图和浏览器截图均通过 view_image 检查.
功能测试和视觉对照分别执行.

## 文件

- src/story.twee: passage 与前置文章内容.
- src/setup.js: SugarCube 宏, 经检查的持久化结构, 主题与轨迹播放.
- src/style.css 和 src/head.html: 设计参数, 字体与本地 KaTeX.
- scripts/build.mjs: 校验工具下载, 提取当前源码与轨迹.
- scripts/highlight.mjs: 固定版本的 Shiki 语法规则与代码主题配置.
- scripts/serve.mjs: 固定端口的静态文件监听器.
- trace/: CPU 轨迹程序和直接使用调度器的边界测试.
- tests/: 文档规定的浏览器验收检查.

生成的 dist, .tools, node_modules, generated 数据和浏览器临时文件均不纳入版本控制.
这个切片未修改推理 runtime, 内核或基线记录.

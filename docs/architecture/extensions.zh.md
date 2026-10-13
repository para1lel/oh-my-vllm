# 测量与扩展边界

## 阶段模型

`performance/` 将观测到的语义工作映射到官方计算与 HBM 上限.
Rust 记录请求提交, 首 token 和末 token 的时刻及其对应 step.
二进制包含 Cargo 输入与 Rust 源码的构建标记.
采集器验证该标记, 源码身份, checkpoint, 已加载库及各自独立的重复区间.
未知 GPU 工作会阻止验收.
参见 [时延模型](../latency-model.zh.md).

## 扩展边界

新模型需要经过验证的加载, 语义合同和缓存 layout.
多 GPU 需要新进程所有权和 collective 合同.
其他 NVIDIA 后端需要设备专属 provider 和实测验收.
当前接口没有这些支持声明.

DSpark 运行时支持仅覆盖已校验的本地 checkpoint 架构.
不同草稿 checkpoint 需要新增加载, 语义, 数值和验收测试.
源码身份及测量范围见 [验收索引](../acceptance.zh.md).

## Agentic 测试所有权

推理服务和 GPU worker 与 agent sandbox 分开.
主机 collector 管理记录代理, SDK trace endpoint, context 检查和资源测量.
各 sandbox 管理新建 CPU 项目, OMP 状态, 工具和临时文件.
Collector 在完成, 失败或取消后释放这些 owner, 并在独立证据目录中保留日志.
[ADR-017](../decisions/ADR-017-disposable-native-goal-soaks.zh.md) 定义原生 goal 和清理流程.

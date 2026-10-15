# ADR-017: 可删除的原生 goal 持续测试

状态: 实现在验收中. 日期: 2026-10-13.

## 问题

此前四会话负载针对项目副本发送源码审计 prompt.
之后的 fixture 让每个会话从空目录开始数据库项目.
改为翻译同一本完整英文书籍的已接受负载.
Goal 可能在一小时之前完成, 暂停或停止自动续行.
这些正常 goal 转换不能导致服务测试提前结束.

## 决策

固定 OMP `18.10.3`, 该版本提供原生 RPC goal 和 `rpc` 自动续行.
四个持久会话分别在独立 rootless Docker sandbox 中翻译同一本已校验的英文书籍.
Corpus 从外部目录连同其原始 manifest 暂存, 各卷均为只读输入.
预先准备的后续 goal 继续下一段落, 下一卷, 最后进行校验.
Goal 提前结束后, 在同一进程, 历史和工作目录中提交下一项.

只有主机可以创建 goal 和重新启动续行. Agent 可以读取, 完成或放弃当前 goal.
逐条核对已记录 goal 身份与主机命令. 每个新 goal 开始前清空旧 todo phase.
保留原始转换原因, 时间戳, 原生 journal 和完整工具输出日志.
采集窗口包含正常转换. 最后的暂停尾部耗时单独报告.

Sandbox 具有独立 PID 和文件系统 namespace, 无网络接口, 不访问 GPU 设备, 不挂载推理项目或 Docker socket.
每个容器只挂载对应 agent 的外部目录和一个自有 socket 目录.
每条 lane 有一个本任务自有的 AF_UNIX bridge 承载模型调用, 另一个承载 telemetry.
Guest loopback endpoint 将原始请求和响应 body 转发到这些挂载 socket.
每个上游都是本地明文 HTTP collector origin, lane 因此无法回退到网络路径.
Body, 连接, listener 和关闭过程都有上限, 所有进程退出后移除 socket.
Lane guard 保持 corpus 与 specification 只读, 拒绝固定的 hashline `edit` 工具, 并保持译文和笔记可写.

采集结束后记录 corpus 和 specification 哈希以及翻译进展, 不宣称翻译质量或书籍已译完.
删除 agent 容器, 修改文件, 安装工具, 缓存和自有镜像.
失败和取消使用相同清理流程. 原始日志和哈希 manifest 保存在独立证据目录中.
不得 prune 其他任务的 Docker 资源.

这一小时检查推理服务稳定性. Goal 状态, 翻译进展, 工具错误和最终答案属于负载观察.
它们不决定框架验收. 独立检查原始模型请求, 资源上限和本任务 sandbox 清理.
期限到达后暂停续行, 发送原生 RPC `abort`, 再执行最后的模型请求排空.

## 验证范围

CPU fixture 检查原生续行, goal 提前完成, 两种 API, 自动压缩, 隔离, 清理失败, corpus 校验和挂载 socket framing.
传输 fixture 提供 fake 模型服务和 fake OTLP receiver. 它证明 SSE 顺序与字节一致, protobuf export framing 及 loopback-only listener.
Docker fixture 在密封容器内走通同样的 hop, 因此挂载 socket 路径和只读输入 guard 也已覆盖.

截止 fixture 让请求保持到原生 goal 截止时间.
它绑定已记录的 pause 与 abort 以及被中止的原生结果.
它还绑定未改变的请求字节, 下游断开, 以及每个挂载 hop 自身失败的那一次尝试及其方向与采集窗口.
该 fixture 走通已接受的 collector 路径: 从框架审计到传输审计, 请求探针与 metric 使用固定数据.
这些 fixture 使用固定数据, 不能证明目标模型推理, 保留书籍的实际内容或一小时服务稳定性.
正式 MTP4 和 DSpark 测试仍需完整原始记录及独立资源审查.
十五行框架矩阵全部通过后才启动真实一小时 soak.
命令和证据规则见 [测试](../testing/soak.zh.md).

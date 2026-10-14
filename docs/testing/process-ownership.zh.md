# 专用服务进程归属

将服务进程树与采集器的 OMP, Docker 和 CPU 请求归一化进程分开.
[服务监督进程](../../benchmarks/engine_supervisor.py#L201) <!-- source: def supervise( --> 在新的 Linux 进程中启动一个 engine.
它的初始子进程清单必须为空. 它不启动其他子进程根.

内核将失去父进程的后代交给这个专用 subreaper.
即使中间父进程在观测前退出, 这个范围仍能证明后代来源.
监督进程使用 pidfd, 并记录 PID, 进程出生时间和清理动作.
它不使用共享用户身份判断进程归属.

[关键测试](../../tests/tooling/test_engine_supervisor.py) 覆盖 session leader 消失, 根进程提前退出, 控制器失败和根绑定失败.
这些测试还检查无关的采集器子进程继续运行.

## 控制器契约

向模块提供外部日志和完整服务命令:

```bash
scripts/with-env.sh python -m benchmarks.engine_supervisor --log "$EVIDENCE_DIR/server.log" --audit "$EVIDENCE_DIR/processes.json" -- "${SERVICE_COMMAND[@]}"
```

采集期间保持监督进程的标准输入打开. Engine 使用独立的标准输入.
采集结束后发送 `stop` 和换行符. 等待监督进程退出, 并检查进程审计记录.
输入 EOF, 服务意外退出, `SIGINT` 或 `SIGTERM` 都会触发清理, 并使监督结果失败.

不完整控制输入不会阻塞进程观测或信号处理.
监督进程记录实际观测间隔. 标称间隔本身不能证明进程覆盖完整.

有限的协作者任务使用 `--natural-completion`, 无须由控制器停止.
只有任务退出码为零且后代清理得到验证时, 此模式才通过.
默认服务模式仍要求控制器发送 `stop`.

将监督进程的意外退出连接到[采集器取消事件](../../benchmarks/agentic_soak.py#L73) <!-- source: def run(args, -->.
等待采集器清理, 监督进程退出和控制器监控线程退出.
监控线程仍使用归属句柄时, 不要关闭这些句柄.

## 结果范围

分别保留采集, 进程清理和资源审查结果.
采集器返回字典. 检查 `operational_passed`, 而不是将字典与退出码比较.

运行结果不代表正式验收. 不要改写原始采集器的 `passed=false`.
将独立资源审查和正式验收保存在其他记录中.

同时保留主要错误及后续清理, 快照或句柄关闭错误.
清理重试成功不能删除原始错误.
检查任务进程, 临时监听端口, 容器, 文件和镜像均已释放.
用户授权的服务例外保存在忽略的本机记录中.

这些 CPU 测试不能证明推理稳定运行一小时.
最终源码仍须完成 MTP4 和 DSpark 采集.

# ADR-018: 客户端持有的工具描述

状态: 已接受.

## 背景

OMP `18.10.3` 的 Bash 工具在可选 `name` 参数上具有 `maxLength=48`.
现有 XML schema validator 在模型执行前拒绝该描述.
客户端持有此参数检查, 模型生成更长 name 时向模型返回失败信息.

工具描述可以包含指导模型的 keyword, 而不提供解码约束.
项目必须接受这些描述, 同时保留显式 strict 请求和现有推理路径.

## 决策

模型 prompt 保留每条原始工具描述和参数 schema.
工具未显式设置 `strict=true` 时, 为请求准备生成独立 schema 副本.
副本保留现有支持的 XML 类型, property, required 字段, 引用, 有限候选以及 numeric/array 约束.
String 长度, pattern, format, default, examples 及其他省略 keyword 保留在原始描述中.

模型必须遵守这些说明. OMP 校验返回参数, 并在后续模型输入中提供错误信息.

对独立副本使用现有 validator 和 XGrammar compiler.
不修改 GPU 计算, sampling, speculative mask 或 matcher 实现.
不为兼容性修改 OMP 原生工具定义, 执行或 provider 请求.

显式 `strict=true` 使用原始 schema 和现有拒绝规则.
显式 `strict=false` 保持现有不应用参数 schema 约束的行为.
XML 仍要求直接 closed root object 及已知, 无歧义的参数类型.
JSON answer schema 保持现有行为.

## 验证

CPU 测试必须在 prompt 中保留原始描述.
它们必须接受字符数多于原始长度限制的字符串.
它们必须保留 strict 模式下不支持 schema 的拒绝行为和现有 typed tool 约束.
OMP 可执行程序 fixture 必须依据原始 `48` 字符限制拒绝 `49` 字符 Bash name.
它必须转发该错误, 并通过每种 API 执行下一条合法命令.

确定性 fixture 响应不能证明目标模型推理或一小时稳定性.
提交源码后重测正式服务和长程门槛.

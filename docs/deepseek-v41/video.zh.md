# DeepSeek V4.1 视频 codec reference

## 状态与来源

此 Phase 0 reference 提供 CPU 帧选择和有序内容块.
它不解码视频, 不执行模型, 也不增加服务视频支持.
它不提供模型数值输出或性能结果.

[模型契约](model.zh.md#图像和视频边界) 标识 `LOCAL-V41:encoding/encoding.py` 和独立的图像编码器.
该本地源代码接受有序图像块.
它不提供视频帧采样器或时间戳协议.
下述规则是计划中的 DeepSeek 扩展的一项明确服务设计决策.
它们不声称本地模型源代码具有视频契约.

选择规则沿用 [Qwen 媒体源代码](../../python/oh_my_vllm/models/qwen38/media.py) 中的 `video_media`.
DeepSeek 将每个选中帧作为一张图像.
它不配对帧, 不重复奇数帧视频的最后一帧, 也不使用时序视觉编码器.
它不使用 Qwen 视觉标记或三轴语言模型位置.

## 元数据与选择

[纯 reference](../../development/model/deepseek_v41/reference_video.py) 有三个公共 API 项:

- `VideoMetadata(frame_count, fps)` 存储不可变且已验证的元数据.
- `sample_plan(metadata)` 返回由选中源帧索引组成的不可变元组.
- `video_records(metadata, frames)` 返回一个视频的有序时间戳文本块和图像块.

`frame_count` 必须为不大于 `2**53` 的正 Python 整数.
最后一个源索引不大于 `2**53-1`.
此范围内的每个整数端点都有精确的 float64 值.
`fps` 必须为可转换为有限浮点数的正有限 Python 整数或浮点数.
构造函数拒绝布尔值, 字符串, 零, 负值, NaN 和无穷大.
选择函数只接受已验证的 `VideoMetadata`.

目标采样率约为每秒两帧.
采样数量为:

```python
count = min(frame_count, max(4, min(768, int(frame_count / fps * 2))))
```

实现先应用上限再转换为整数, 以避免正采样率很小时发生溢出.
对于有限的计算结果, 此顺序得到相同的数量.

选中索引遵循此 NumPy 规则:

```python
np.linspace(0, frame_count - 1, count).round()
```

索引从零开始.
NumPy 将中点值舍入到最近的偶数整数.
采样计划保留第一个和最后一个源帧.
帧数不足四帧的视频使用全部可用帧.
计划最多包含 768 帧.

## 有序内容块

`frames` 是由 `(source_index, image_block)` 对组成的有序序列.
记录数量和完整索引序列必须与 `sample_plan(metadata)` 一致.
每个索引必须为 Python 整数.
缺失, 多余, 重复, 逆序, 布尔值和浮点数索引会导致错误.
函数不排序记录, 也不丢弃记录.

每个图像块使用本地编码器的内部格式:

```python
{"type": "image", "data": b"caller-supplied-frame-image"}
```

本地源代码还接受此格式中的非空 `source` 或 `url` 字段.
这些字段是调用方提供的图像引用.
函数不打开路径, 不获取 URL, 不解码字节, 也不导入推理运行时.
它不根据不透明记录计算字节大小.
它复制图像字典并保留其字段值.
它不修改调用方的记录.

对于每个选中帧, 函数在其图像块之前立即返回此文本块:

```python
{"type": "text", "text": f"<{source_index / fps:.1f} seconds>"}
```

时间戳是根据帧索引和源帧率计算的估计值.
它不是显示时间戳或可变帧率时间.
展开函数拒绝非有限的时间戳.

不同帧舍入后可能具有相同的时间戳文本.
这些帧保留源顺序.
时间戳字符串是供本地提示词编码器使用的普通文本.
此函数不对它们执行 tokenization.

调用方将返回的元组插入视频的原始内容位置:

```python
content = [*before_blocks, *video_records(metadata, frames), *after_blocks]
message = {"role": "user", "content": content}
```

此插入方式保留前后文本和静态图像的顺序.
随后本地编码器按提示词顺序收集图像记录.
图像处理器应用本地缩放, 填充, 归一化, patch 顺序和图像片段规则.
语言模型位置遵循连续 token 顺序.

## 资源上限与后续工作

在调用此函数之前, 解码器边界必须拒绝超过 64 MiB 的编码视频.
原始已解码采样 RGB 字节总量不得大于 512 MiB.
解码字节检查发生在堆叠, 缩放, 填充或图像准备之前.
这些检查是未来解码器的前置条件.
此纯 reference 无法通过不透明图像记录实施这些检查.
解码器集成还必须限制临时存储和解码工作量.

每帧保留本地图像片段的 1024 tokens 上限.
在时间戳文本和其他提示词内容之前, 视频图像片段的最大预算为 `768*1024 = 786432` tokens.
此有限上限不保证请求准入.
服务随后必须依据 1048576-token 上下文上限检查完整编码提示词与请求输出的总量.
它必须拒绝超限输入, 不得静默截断.
独立图像和多个视频计入同一个请求预算.

[固定媒体夹具](fixtures.zh.md#媒体端点) 提供一张静态图片和两张合成已解码视频帧的完整模型端点.
[视频提示词夹具](video-fixtures.zh.md) 绑定混合媒体顺序, 格式化后相同的时间戳和上下文续接.
视频解码, 运行时分配边界和服务集成属于后续工作.
[行为测试](../../development/model/deepseek_v41/test_reference_video.py) 仅检查选择, 内容顺序和无效输入.

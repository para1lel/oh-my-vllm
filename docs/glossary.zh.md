# 项目技术术语

[terms.json](terms.json) 是技术术语, 含义, 翻译和注册词形的权威来源.
由该来源生成以下参考页:

- [术语参考 A](glossary/model.zh.md).
- [术语参考 B](glossary/runtime.zh.md).
- [术语参考 C](glossary/gpu-service.zh.md).

ASD-STE100 管理语法和通用词. 技术条目只许可其定义的项目含义.
代码标识符, 产品专名和协议文本保持源码词形.

## 维护

仅在已批准词典无法准确表达技术含义时添加术语.
提供名词或动词, 定义, 中文翻译和明确词形.
依据源码及 Rule 1.5 或 Rule 1.12 review.
通用的已批准措辞可以表达相同行为时使用通用措辞.

```bash
scripts/with-env.sh python scripts/build_glossary.py
scripts/with-env.sh python scripts/build_glossary.py --check
```

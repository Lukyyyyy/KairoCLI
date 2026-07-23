# Kairo CLI Agent Guide

- Python：3.11–3.14
- 包：`src/kairocli`
- 测试：`pytest`
- 质量门：`ruff check .`、`mypy src/kairocli`
- 用户目录：`~/.kairocli`
- 项目目录：`.kairocli`
- 项目记忆：`KAIRO.md`

修改命令时同步命令解析测试和 README；修改工具时同步 schema、安全策略和 Agent 提示词。
应用日志只能记录脱敏后的生命周期元数据，不得记录 prompt、工具参数、回答正文、图片或 reasoning；
reasoning 仍只允许通过用户明确开启的私有 trace 保存。
不要提交 `.env`、真实密钥、缓存、日志或运行时数据库。

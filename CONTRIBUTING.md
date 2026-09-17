# 参与贡献

感谢你愿意改进 Kairo CLI。欢迎提交缺陷报告、功能建议、文档修正和代码贡献。

参与项目前，请遵守[行为准则](CODE_OF_CONDUCT.md)。安全漏洞请按[安全策略](SECURITY.md)
私下报告，不要创建公开 Issue。

## 开始之前

提交较大的功能或行为变更前，请先创建 Issue 说明问题、使用场景和建议方案，避免重复工作。
小型缺陷修复、测试和文档修正可以直接提交 Pull Request。

## 本地开发

项目需要 Python `3.11`–`3.14`：

```bash
git clone https://github.com/Lukyyyyy/KairoCLI.git
cd KairoCLI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Windows PowerShell：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
```

## 提交改动

1. 从最新的 `main` 创建功能分支。
2. 保持改动聚焦，并为非平凡逻辑补充最小有效测试。
3. 修改命令时同步命令解析测试和 README。
4. 修改工具时同步 schema、安全策略和 Agent 提示词。
5. 使用 Conventional Commits，例如 `fix: 修复会话恢复失败`。

不要提交 `.env`、真实密钥、缓存、日志、trace 或运行时数据库。应用日志不得记录 prompt、
工具参数、回答正文、图片或 reasoning；reasoning 只能由用户明确开启的私有 trace 保存。

## 质量检查

提交 Pull Request 前运行：

```bash
ruff check .
ruff format --check .
mypy src/kairocli
pytest
kairocli --version
```

Pull Request 应说明改动原因、实现范围和验证方式；涉及终端或 Web 界面时，请附上截图或录屏。

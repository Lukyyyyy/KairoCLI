<p align="center">
  <img src="src/kairocli/web_static/favicon.svg" width="128" alt="Kairo CLI logo">
</p>

<h1 align="center">Kairo CLI</h1>

<p align="center">从意图，到可审计的执行。</p>

<p align="center">
  <strong>简体中文</strong> · <a href="README.en.md">English</a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11--3.14-3776AB?logo=python&amp;logoColor=white" alt="Python 3.11–3.14"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&amp;logoColor=white" alt="FastAPI 0.115+"></a>
  <a href="https://docs.pydantic.dev/"><img src="https://img.shields.io/badge/Pydantic-2.9+-E92063?logo=pydantic&amp;logoColor=white" alt="Pydantic 2.9+"></a>
  <a href="https://modelcontextprotocol.io/"><img src="https://img.shields.io/badge/MCP-1.0+-5A67D8" alt="MCP 1.0+"></a>
  <a href="https://textual.textualize.io/"><img src="https://img.shields.io/badge/Textual-1.0+-111111" alt="Textual 1.0+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Lukyyyyy/KairoCLI" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#使用方式">使用方式</a> ·
  <a href="docs/architecture.md">架构说明</a> ·
  <a href="CONTRIBUTING.md">参与贡献</a> ·
  <a href="SECURITY.md">安全策略</a> ·
  <a href="landing/index.html">产品页</a>
</p>

Kairo CLI 是一个 Python 原生的开源 Agent 运行时。它把代码理解、任务规划、工具调用、
多人协作和人工审批收束到统一的安全边界中，可作为终端助手、自动化命令、本地 Runtime
API 或多用户 Web 服务使用。

**当前状态：** Kairo CLI 当前版本为 `0.1.0`，仍在积极开发中。接口、配置和数据格式可能在后续版本中调整。

## 为什么选择 Kairo CLI

- **面向真实工程**：从文件编辑、受控 Shell、LSP 诊断到代码索引与调用关系查询，覆盖完整开发链路。
- **执行过程可控**：支持 ReAct、Plan-and-Execute 和 Team 三种模式，计划、审批、执行与结果均可追踪。
- **默认安全**：限制工作区路径，拦截灾难性命令，对写入与高风险工具分级审批，并记录脱敏审计事件。
- **上下文可延续**：支持项目指令、长期记忆、自动压缩、Todo、工作区快照和可恢复会话。
- **入口不受限**：同一运行时可通过交互式 CLI、结构化脚本输出、HTTP API、Web 控制台或微信通道访问。
- **集成可扩展**：原生支持 MCP、Skill、Chrome DevTools 和七种模型服务配置。

执行链路保持简单且明确：

```text
Inspect → Plan → Approve → Execute → Record
```

## 核心能力

| 领域 | 能力 |
| --- | --- |
| Agent | ReAct、可审阅 Plan、依赖感知 Team 协作、并行只读工具调用 |
| 代码理解 | AST / Tree-sitter 分块、混合检索、定义与调用关系、LSP 诊断 |
| 工程工具 | 文件读写、安全补丁、受控命令、持久 Shell、网页检索、图片输入 |
| 上下文 | `KAIRO.md` 指令、长期记忆、自动压缩、Todo、快照、会话恢复 |
| 扩展 | MCP tools/resources/prompts、分层 Skill、Chrome DevTools 浏览器会话 |
| 自动化 | `text` / `json` / `jsonl` 输出、本地 Runtime API、可恢复 SSE |
| 交互界面 | Inline、Plain、Textual TUI、多用户 Web 控制台、微信通道 |
| 模型服务 | GLM、DeepSeek、Step、Kimi、FreeLLMAPI、讯飞星火 MaaS、Agnes AI |

完整实现状态见[功能验收矩阵](docs/feature-matrix.md)。

## 环境要求

- Python `3.11`–`3.14`
- macOS、Linux 或 Windows
- 至少一个受支持模型服务的 API Key
- 可选：Chrome 与 `npx`，用于默认的 Chrome DevTools MCP 集成

## 快速开始

### 1. 从源码安装

```bash
git clone https://github.com/Lukyyyyy/KairoCLI.git
cd KairoCLI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell 使用：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### 2. 配置模型

复制配置模板，并填写至少一个 Provider 的 API Key：

```bash
cp .env.example .env
```

例如：

```dotenv
GLM_API_KEY=your-api-key
KAIROCLI_PROVIDER=glm
```

**密钥安全：** 不要提交 `.env` 或任何真实密钥。Kairo CLI 依次读取 `~/.kairocli/.env`、
项目 `.env` 和进程环境变量，后读取的配置优先。

### 3. 启动

在需要处理的项目目录中运行：

```bash
kairocli
```

首次进入项目后，可执行 `/init` 创建项目指令文件，再直接描述任务：

```text
/init
分析这个项目的测试结构，并给出三个最值得优先修复的问题。
```

输入 `/help` 查看完整命令索引，输入 `@` 可补全并引用工作区文件或目录。

## 使用方式

### 交互式 CLI

常用命令：

```text
/plan TASK                 生成计划，审阅后执行
/team TASK                 使用多 Agent 团队执行
/index [PATH]              建立工作区代码索引
/search QUERY              搜索已索引代码
/graph SYMBOL              查询定义、包含和调用关系
/memory list               查看长期记忆
/session list              查看可恢复会话
/todo list                 查看当前任务清单
/mcp list                  查看 MCP 服务
/skill list                查看可用 Skill
/policy                    查看当前安全策略
/snapshot list             查看工作区快照
/cancel                    取消当前任务
/exit                      退出 Kairo CLI
```

使用 `@relative/path` 或 `@<path with spaces>` 将本地内容加入上下文：

```text
请审查 @src/kairocli/policy.py
总结 @docs/feature-matrix.md
```

消息输入支持 `Shift+Enter` 换行、`Enter` 提交；默认 Inline 渲染器支持 Markdown、代码高亮
和工具结果折叠，也可切换为 Plain 或 Textual TUI。

### Plan 与 Team 模式

```bash
kairocli -p "规划并实现缓存层" --mode plan
kairocli -p "并行审查安全性、性能和测试覆盖" --mode team
```

交互式 Plan 会在执行前等待审阅；非交互模式会直接执行生成的计划。Team 模式按依赖关系
调度任务，并在完成后汇总审查结果。

### 非交互式执行

`-p` / `--print` 可用于脚本和 CI：

```bash
kairocli -p "检查项目并给出风险"
kairocli -p "运行测试并解释失败" --output-format json
printf '%s' '总结当前改动' | kairocli --print --output-format jsonl
```

输出格式：

- `text`：仅输出最终回答；
- `json`：输出一条 schema v1 结果；
- `jsonl`：依次输出 start/result 事件。

退出码为 `0`（成功）、`1`（执行失败）、`2`（参数或启动配置错误）和 `130`（取消）。
无人值守模式不会等待人工审批，危险工具默认拒绝；需要时按工具名精确授权：

```bash
kairocli -p "修复格式问题" \
  --allow-tool write_file \
  --allow-tool apply_patch
```

仅当外部运行环境已经提供等价隔离时，才应使用 `--dangerously-skip-approvals`。

### 会话恢复

```bash
kairocli --continue
kairocli --resume session_xxxxxxxxxxxx
kairocli -p "记录本次检查结果" --save-session
```

`--continue` 恢复当前工作区最近的非空会话；`--resume` 按 ID 恢复。会话始终绑定原工作区，
不能从当前工作区删除其他项目的会话。

### Runtime API 与 Web 控制台

启动仅监听 `127.0.0.1` 的 Runtime API：

```bash
KAIROCLI_RUNTIME_API_KEY=local-secret kairocli serve --http --port 8080
```

客户端可使用 `X-Kairo-CLI-API-Key` 或 `Authorization: Bearer <key>` 鉴权。API 支持 thread
创建、turn 执行与取消，以及通过游标恢复的 SSE 事件流。

启动带账号、会话、模型配置、计划审批和工具审批的多用户 Web 控制台：

```bash
kairocli serve --http --web --port 8080
```

Web 默认只监听本机；可信局域网部署可增加 `--lan`。首次交互式启动会创建管理员，
非交互式部署需设置 `KAIROCLI_WEB_ADMIN_EMAIL`，并妥善保存终端输出的一次性密码。
公开注册与密码重置依赖配置完整的腾讯云 SES 邮件服务。

### 微信通道

多用户部署可在 Web 控制台的“设置 → 通道 → 微信”中扫码绑定。微信消息按账号串行处理；
读取工具自动允许，写入工具需要一次性审批码，Shell 与 MCP 默认拒绝。

旧版单用户 CLI 通道仍可独立运行：

```bash
kairocli wechat setup
kairocli wechat start
kairocli wechat status
kairocli wechat daemon start
```

## 配置

配置按以下优先级合并，后者覆盖前者：

```text
内置默认值
  → ~/.kairocli/config.json
  → ~/.kairocli/.env
  → <workspace>/.env
  → 进程环境变量
  → CLI 参数
```

### 模型服务

| Provider | API Key 环境变量 | 默认模型 |
| --- | --- | --- |
| GLM | `GLM_API_KEY` | `glm-5.1` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` |
| Step | `STEP_API_KEY` | `step-3.5-flash` |
| Kimi | `KIMI_API_KEY` | `kimi-k2.6` |
| FreeLLMAPI | `FREELLMAPI_API_KEY` | `auto` |
| 讯飞星火 MaaS | `XFYUN_MAAS_API_KEY` | `Qwen3.6-35B-A3B` |
| Agnes AI | `AGNES_API_KEY` | `agnes-2.0-flash` |

交互式 CLI 可查看或修改 Provider 配置：

```text
/model
/model deepseek
/config
/config provider deepseek api-key YOUR_KEY
/config provider deepseek model deepseek-v4-flash
```

完整环境变量及可选服务配置见 [.env.example](.env.example)。

### 长期记忆与代码索引

长期记忆默认使用本地关键词检索。明确允许 embedding 服务处理已保存内容后，可启用混合检索：

```dotenv
KAIROCLI_MEMORY_SEMANTIC_SEARCH=true
KAIROCLI_EMBEDDING_PROVIDER=ollama
KAIROCLI_EMBEDDING_MODEL=nomic-embed-text:latest
KAIROCLI_EMBEDDING_BASE_URL=http://localhost:11434
```

也支持 `alicloud`、`openai` 和 `zhipu` embedding provider。远程服务会接收长期记忆文本
和被索引的代码片段；更换模型或向量维度后需要重新执行完整 `/index`。

代码索引使用 Python AST 和常见语言的 Tree-sitter 语法树按声明分块，解析器不可用时自动
回退。`/graph <符号>` 与只读工具 `query_code_graph` 共用持久化的定义、包含和语法级调用
关系；关键词重排支持英文标识符、camelCase 与 Jieba 中文分词。

### MCP、Skill 与浏览器

- 用户级与项目级 MCP 配置分别位于 `~/.kairocli/mcp.json` 和 `.kairocli/mcp.json`；
- Skill 按用户、项目和内置层级加载，可从本地目录、精选目录或 Git 仓库安装；
- 首次运行会生成隔离模式的 Chrome DevTools MCP 默认配置；敏感页面始终逐次审批；
- 单个 MCP 服务启动失败不会阻止主程序启动。

## 安全模型

Kairo CLI 默认实施以下边界：

- 文件工具只访问工作区允许的真实路径，并拒绝符号链接逃逸；
- 审批框按低危、中危、高危展示工具能力和风险原因；
- 工作区写入可按会话授权，`execute_command`、`shell_exec` 等高风险 Shell 能力始终逐次审批；
- 灾难性 Shell 模式和路径越界由策略直接拦截，并返回明确原因；
- 非交互模式默认拒绝所有需要审批的工具；
- 写入型工具串行执行，只读工具受控并发；
- 取消或超时会清理命令和 MCP 服务的完整子进程树；
- Provider、MCP、工具和终端错误在输出前执行凭据脱敏和大小限制；
- 应用日志只记录脱敏后的生命周期元数据，不记录 prompt、工具参数、回答正文、图片或 reasoning；
- 私有 trace 仅在用户明确开启后记录，reasoning 需要再次单独开启。

这些边界用于降低误操作风险，不能替代代码审查、最小权限凭据、隔离环境和可靠备份。
执行来自不可信来源的 Skill、MCP 服务或 Shell 命令前，请先审查其内容。

## 数据与项目指令

Kairo CLI 不读取其他 Agent 产品的数据目录：

| 范围 | 位置 | 内容 |
| --- | --- | --- |
| 用户级 | `~/.kairocli/` | 配置、会话、记忆、任务、审计、日志和 Runtime 数据 |
| 项目级 | `<workspace>/.kairocli/` | 项目 MCP、Skill、索引和运行状态 |
| 项目指令 | `KAIRO.md`、`KAIRO.local.md` | 长期规则与本地覆盖规则 |

项目指令按用户级、工作区、`.kairocli` 和 local 的顺序加载，越靠后的规则优先级越高；
子目录中的指令文件只作用于对应目录树。

## 架构

```text
CLI / Web / Runtime API / WeChat
                 │
          Agent orchestration
       ┌─────────┼─────────┐
     ReAct      Plan      Team
       └─────────┼─────────┘
                 │
       Policy · Tools · Context
       ┌─────────┼─────────┐
     Built-in   MCP      Skills
```

更完整的组件边界、生命周期、并发模型和持久化设计见[架构说明](docs/architecture.md)。

## 开发与贡献

欢迎通过 Issue 和 Pull Request 参与。提交改动前请先创建分支并安装开发依赖：

```bash
python -m pip install -e '.[dev]'
```

运行与 CI 一致的质量门：

```bash
ruff check .
ruff format --check .
mypy src/kairocli
pytest
kairocli --version
```

CI 在 Linux 与 macOS 上运行测试，并在 Windows 上执行 Python `3.11`–`3.14` 安装与导入
冒烟检查。贡献时请同时遵守以下约定：

- 修改命令时同步命令解析测试和 README；
- 修改工具时同步 schema、安全策略和 Agent 提示词；
- 应用日志不得记录 prompt、工具参数、回答正文、图片或 reasoning；
- 不提交 `.env`、真实密钥、缓存、日志或运行时数据库。

## 文档

- [架构说明](docs/architecture.md)
- [功能验收矩阵](docs/feature-matrix.md)
- [贡献指南](CONTRIBUTING.md)
- [行为准则](CODE_OF_CONDUCT.md)
- [安全策略](SECURITY.md)
- [响应式产品页](landing/index.html)
- [产品海报](assets/kairocli-poster-1200x1600.png)

## 许可

本项目基于 [MIT License](LICENSE) 开源。

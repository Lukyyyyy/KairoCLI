# Kairo CLI

> 从意图，到可审计的执行。

Kairo CLI 是用 Python 开发的 Agent CLI，将规划、协作、工具与上下文收束进一个安全可控的工程运行时。它既可以作为交互式终端助手使用，也可以通过结构化输出接入脚本，或以本地 Runtime API 的形式嵌入其他应用。

<p align="center">
  <a href="landing/index.html">
    <img src="assets/kairocli-poster-1200x1600.png" width="520" alt="Kairo CLI：从意图，到可审计的执行">
  </a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a>
  ·
  <a href="landing/index.html">产品页</a>
  ·
  <a href="docs/architecture.md">架构说明</a>
</p>

> [!IMPORTANT]
> 当前版本为 `0.1.0`，仍处于积极开发阶段。

## 目录

- [核心能力](#核心能力)
- [产品预览](#产品预览)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用方式](#使用方式)
- [配置](#配置)
- [安全模型](#安全模型)
- [项目数据与指令](#项目数据与指令)
- [开发](#开发)
- [文档与演示](#文档与演示)
- [许可](#许可)

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 多种执行模式 | 支持 ReAct、带人工审阅的 Plan-and-Execute，以及依赖感知的 Team 协作 |
| 工程工具 | 文件读写、安全补丁、受控 Shell、代码索引、语义搜索、符号关系、LSP 诊断和图片输入 |
| 上下文管理 | 项目指令、长期记忆、自动压缩、结构化 Todo、工作区快照和可恢复会话 |
| 可扩展集成 | 支持 MCP tools/resources/prompts、Skill 分层加载和 Chrome DevTools 浏览器会话 |
| 自动化入口 | 提供非交互式 `text` / `json` / `jsonl` 输出和本地 Runtime API |
| 多模型接入 | 内置 GLM、DeepSeek、Step、Kimi、FreeLLMAPI、讯飞星火 MaaS 和 Agnes AI 配置 |
| 安全与审计 | 工作区路径围栏、危险操作审批、默认拒绝的无人值守策略、凭据脱敏和审计记录 |
| 多终端体验 | 支持 inline、plain 和 Textual 全屏 TUI，并可通过微信通道远程交互 |

更细的实现状态见[功能验收矩阵](docs/feature-matrix.md)，系统边界见[架构说明](docs/architecture.md)。

## 产品预览

新版响应式产品页围绕 Kairo CLI 的完整执行链展开：

```text
Inspect → Plan → Approve → Execute → Record
```

- 三种执行模式：ReAct、Plan、Team
- 扩展入口：MCP、Skill、Browser、Runtime API、WeChat
- 控制边界：工作区路径围栏、危险操作审批、默认拒绝策略、凭据脱敏与审计

查看[响应式产品页](landing/index.html)、[海报源文件](poster/index.html)或[导出的 1200×1600 产品海报](assets/kairocli-poster-1200x1600.png)。

## 环境要求

- Python `3.11`–`3.14`
- macOS、Linux 或 Windows
- 至少一个受支持模型服务的 API Key
- 可选：Chrome 和可用的 `npx`，用于默认的 Chrome DevTools MCP 集成

## 快速开始

### 1. 安装

项目目前从源码安装。获取源码后进入项目目录：

```bash
cd /path/to/KairoCLI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell 的虚拟环境激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

### 2. 配置模型

复制配置模板，并填写至少一个 Provider 的 API Key：

```bash
cp .env.example .env
```

例如使用 GLM：

```dotenv
GLM_API_KEY=your-api-key
KAIROCLI_PROVIDER=glm
```

也可以使用对应环境变量：

| Provider | API Key 环境变量 | 默认模型 |
| --- | --- | --- |
| GLM | `GLM_API_KEY` | `glm-5.1` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` |
| Step | `STEP_API_KEY` | `step-3.5-flash` |
| Kimi | `KIMI_API_KEY` | `kimi-k2.6` |
| FreeLLMAPI | `FREELLMAPI_API_KEY` | `auto` |
| 讯飞星火 MaaS | `XFYUN_MAAS_API_KEY` | `Qwen3.6-35B-A3B` |
| Agnes AI | `AGNES_API_KEY` | `agnes-2.0-flash` |

> [!CAUTION]
> `.env` 和真实密钥不得提交到版本库。Kairo CLI 会读取全局的
> `~/.kairocli/.env` 和当前项目的 `.env`，项目配置覆盖全局配置，进程环境变量优先于两者。

若希望所有工作区共享同一套 Provider 配置，可创建 KairoCLI 专用的全局配置：

```bash
mkdir -p ~/.kairocli
cp .env.example ~/.kairocli/.env
chmod 600 ~/.kairocli/.env
```

### 3. 启动

在需要处理的项目目录中运行：

```bash
kairocli
```

首次进入项目后，可初始化项目级指令：

```text
/init
```

然后直接描述任务，例如：

```text
分析这个项目的测试结构，并给出三个最值得优先修复的问题。
```

## 使用方式

### 交互式 CLI

Kairo CLI 默认使用 inline 渲染器。输入 `/help` 查看完整命令索引，常用命令包括：

```text
/plan TASK                 生成计划，审阅后执行
/team TASK                 使用多 Agent 团队执行
/index [PATH]              建立工作区代码索引
/search QUERY              搜索已索引代码
/session                   管理可恢复会话
/todo                      管理当前任务清单
/mcp                       管理 MCP 服务和资源
/skill                     安装和管理 Skill
/policy                    查看当前安全策略
/snapshot                  查看工作区快照
/cancel                    取消当前任务
/exit                      退出 Kairo CLI
```

使用 `@relative/path` 或 `@<path with spaces>` 将工作区文件或目录加入上下文：

```text
请审查 @src/kairocli/policy.py
总结 @docs/feature-matrix.md 中仍需推进的能力
```

输入 `@` 时会像 `/` 命令一样显示最多 8 行可滚动的纵向路径菜单；路径菜单不显示说明文字，可使用方向键选择，并用 `Tab` 或 `Enter` 补全。

消息输入框支持多行编辑：按 `Shift+Enter` 插入换行，按 `Enter` 提交；粘贴多行文本时会保留并显示原有换行。

### Plan 与 Team 模式

交互模式下使用 `/plan` 或 `/team`；脚本中使用 `--mode`：

```bash
kairocli -p "规划并实现缓存层" --mode plan
kairocli -p "并行审查安全性、性能和测试覆盖" --mode team
```

Plan 模式会在交互终端中等待审阅；非交互模式会直接执行生成的计划。

### 非交互式执行

使用 `-p` / `--print` 执行单轮任务：

```bash
kairocli -p "检查项目并给出风险"
kairocli -p "运行测试并解释失败" --output-format json
printf '%s' '总结当前改动' | kairocli --print --output-format jsonl
```

`--output-format` 支持：

- `text`：只向 stdout 写入最终回答；
- `json`：输出一条 schema v1 结果；
- `jsonl`：依次输出 start/result 事件。

成功、执行失败和取消的退出码分别为 `0`、`1` 和 `130`；参数或启动配置错误使用 `2`。

无人值守模式不会等待人工审批，危险工具默认拒绝。按工具名精确授权：

```bash
kairocli -p "修复格式问题" \
  --allow-tool write_file \
  --allow-tool apply_patch
```

只有调用环境已经提供等价安全边界时，才应使用 `--dangerously-skip-approvals`。

### 恢复会话

```bash
kairocli --continue
kairocli --resume session_xxxxxxxxxxxx
kairocli -p "记录本次检查结果" --save-session
```

`--continue` 恢复当前工作区最近的非空会话；`--resume` 按 ID 恢复。交互式 CLI
可使用 `/session list` 查看按最近更新时间排序的会话，当前会话以 `●` 标记。列表默认
隐藏非当前的空会话；使用 `/session list --all` 查看全部会话。

会话删除始终限制在当前工作区，且不能删除当前活动会话：

```text
/session delete --empty
/session delete session_xxxxxxxxxxxx session_yyyyyyyyyyyy
```

`--empty` 删除全部非当前空会话；指定多个 ID 时会先校验全部会话，再执行原子批量删除。

### Runtime API

设置独立 API Key 后启动仅监听 `127.0.0.1` 的服务：

```bash
KAIROCLI_RUNTIME_API_KEY=local-secret \
  kairocli serve --http --port 8080
```

客户端可通过 `X-Kairo-CLI-API-Key` 或 `Authorization: Bearer <key>` 鉴权。Runtime API 提供 thread 创建、turn 执行与取消，以及支持游标恢复的 SSE 事件流；持久化数据库位于 `~/.kairocli/runtime/runtime.db`。

### 微信通道

```bash
kairocli wechat setup
kairocli wechat start
kairocli wechat status
kairocli wechat daemon start
kairocli wechat daemon logs
kairocli wechat daemon stop
```

运行 `wechat setup` 时，Kairo CLI 会先显示微信 Agent 可访问的目录；直接按 Enter 使用当前目录，输入另一个已存在的目录，或按 Esc 取消，然后按提示扫码绑定。

交互模式的命令列表只显示一级入口。对于包含子命令的命令，输入一级命令和空格后会按需展开，例如 `/wechat ` 会显示 `setup`、`start`、`status` 和 `stop`；直接执行 `/wechat` 则会显示当前状态与子命令提示。

微信通道采用独立的默认拒绝策略，不会发送 reasoning、工具进度或 diff。当前媒体消息仅保留安全的元数据，文件和图片下载/解密尚未启用。

## 配置

配置按以下优先级合并，后者覆盖前者：

1. 内置默认值；
2. `~/.kairocli/config.json`；
3. `~/.kairocli/.env`；
4. 当前项目的 `.env`；
5. 进程环境变量；
6. CLI 参数。

常用环境变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `KAIROCLI_PROVIDER` | `glm` | 默认 Provider |
| `KAIROCLI_RENDERER` | `inline` | `inline`、`plain` 或 `tui` |
| `KAIROCLI_TASK_WORKERS` | `2` | 持久后台任务 worker 数量，范围 `1`–`32` |
| `KAIROCLI_RUNTIME_API_KEY` | 空 | Runtime API 鉴权密钥 |
| `KAIROCLI_NO_TUI` | `false` | 禁止使用全屏 TUI |
| `KAIROCLI_LOG_ENABLED` | `true` | 启用脱敏后的应用日志 |
| `KAIROCLI_TRACE_ENABLED` | `false` | 启用私有模型诊断 trace，记录每次模型请求的消息、工具 schema 与响应 |
| `KAIROCLI_TRACE_REASONING` | `false` | 在 trace 中额外记录 reasoning，需显式开启 |

在交互式 CLI 中可查看或修改 Provider 配置：

```text
/model
/model deepseek
/config
/config provider deepseek api-key YOUR_KEY
/config provider deepseek model deepseek-v4-flash
```

Provider 的 `base-url`、`model`、`lora-id`、`context-window`、`temperature` 和 `max-tokens` 也可通过 `/config` 修改。详细模板见 [.env.example](.env.example)。

### MCP

用户级和项目级 MCP 配置分别位于：

```text
~/.kairocli/mcp.json
<workspace>/.kairocli/mcp.json
```

项目级同名 server 覆盖用户级配置。Kairo CLI 支持 stdio 和 Streamable HTTP MCP，并在首次运行时创建隔离模式的 Chrome DevTools MCP 默认配置；单个 MCP 启动失败不会阻止主程序启动。

## 安全模型

Kairo CLI 的默认安全边界包括：

- 文件工具只能访问当前工作区允许的真实路径，并拒绝符号链接逃逸；
- 危险操作在交互模式中需要审批，在非交互模式中默认拒绝；
- 写入型工具串行执行，只读工具可受控并发；
- Shell 命令运行在独立进程组中，取消或超时会清理子进程树；
- Provider、MCP、工具和终端错误在输出前执行凭据脱敏和大小限制；
- 应用日志只记录脱敏后的生命周期元数据，不记录 prompt、工具参数、回答正文、图片或 reasoning；
- 私有模型 trace 仅在用户明确开启时记录实际请求上下文与响应，并持续脱敏凭据、以占位符替代图片数据；
- reasoning 仅在用户同时明确开启 trace 与 reasoning trace 时保存。

安全边界降低误操作风险，但不能代替代码审查、最小权限凭据、隔离环境或可靠备份。执行来自不可信来源的 Skill、MCP 服务或 Shell 命令前，请先审查其内容。

## 项目数据与指令

Kairo CLI 不读取其他 Agent 产品的数据目录：

| 范围 | 位置 | 内容 |
| --- | --- | --- |
| 用户级 | `~/.kairocli/` | 配置、会话、记忆、任务、审计、日志和 Runtime 数据 |
| 项目级 | `<workspace>/.kairocli/` | 项目 MCP、Skill、索引和运行状态 |
| 项目指令 | `KAIRO.md`、`KAIRO.local.md` | 长期规则与仅本地覆盖规则 |

项目指令按 user → workspace → `.kairocli` → local 的顺序加载，越靠后的规则优先级越高。子目录中的 `KAIRO.md` 和 `KAIRO.local.md` 仅作用于对应目录树。

## 开发

安装开发依赖：

```bash
python -m pip install -e '.[dev]'
```

提交前运行与 CI 一致的质量门：

```bash
ruff check .
ruff format --check .
mypy src/kairocli
pytest
kairocli --version
```

CI 在 Ubuntu、macOS 和 Windows 上覆盖 Python `3.11`、`3.12`、`3.13` 和 `3.14`。

开发时请遵循以下约定：

- 代码实际行为优先于文档；用户可见的名称、路径和环境变量统一使用 Kairo CLI；
- 修改命令时同步命令解析测试和 README；
- 修改工具时同步 schema、安全策略和 Agent 提示词；
- 不提交 `.env`、真实密钥、缓存、日志或运行时数据库。

## 文档与演示

- [架构说明](docs/architecture.md)
- [功能验收矩阵](docs/feature-matrix.md)
- [响应式产品页](landing/index.html)
- [海报源文件](poster/index.html)
- [1200×1600 产品海报](assets/kairocli-poster-1200x1600.png)

## 许可

本项目基于 [MIT License](LICENSE) 开源。

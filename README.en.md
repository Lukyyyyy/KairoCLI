<p align="center">
  <img src="src/kairocli/web_static/favicon.svg" width="128" alt="Kairo CLI logo">
</p>

<h1 align="center">Kairo CLI</h1>

<p align="center">From intent to auditable execution.</p>

<p align="center">
  <a href="README.md">简体中文</a> · <strong>English</strong>
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
  <a href="#quick-start">Quick start</a> ·
  <a href="#usage">Usage</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="SECURITY.md">Security</a> ·
  <a href="landing/index.html">Product page</a>
</p>

Kairo CLI is a Python-native, open-source agent runtime. It brings code understanding, planning,
tool use, multi-agent collaboration, and human approval into one controlled security boundary. Use
it as an interactive terminal assistant, a structured automation command, a local Runtime API, or a
multi-user web service.

**Project status:** Kairo CLI is currently at version `0.1.0` and under active development.
Interfaces, configuration, and data formats may change in future releases.

## Why Kairo CLI

- **Built for real engineering work**: file editing, controlled shell access, LSP diagnostics,
  code indexing, and call-graph queries cover the complete development loop.
- **Controlled execution**: ReAct, Plan-and-Execute, and Team modes keep planning, approval,
  execution, and results traceable.
- **Secure by default**: workspace path boundaries, catastrophic-command blocking, risk-based
  approval, and redacted audit events reduce accidental damage.
- **Persistent context**: project instructions, long-term memory, automatic compaction, todos,
  workspace snapshots, and resumable sessions preserve continuity.
- **Multiple entry points**: use the same runtime through the CLI, structured output, HTTP API,
  web console, or WeChat channel.
- **Extensible integrations**: native support for MCP, Skills, Chrome DevTools, and seven model
  provider configurations.

The execution path stays explicit:

```text
Inspect → Plan → Approve → Execute → Record
```

## Core capabilities

| Area | Capabilities |
| --- | --- |
| Agent | ReAct, reviewable plans, dependency-aware Team mode, parallel read-only tools |
| Code intelligence | AST / Tree-sitter chunking, hybrid retrieval, definitions, calls, LSP diagnostics |
| Engineering tools | File I/O, safe patches, controlled commands, persistent shells, web research, images |
| Context | `KAIRO.md`, long-term memory, compaction, todos, snapshots, session recovery |
| Extensions | MCP tools/resources/prompts, layered Skills, Chrome DevTools sessions |
| Automation | `text` / `json` / `jsonl`, local Runtime API, resumable SSE |
| Interfaces | Inline, Plain, Textual TUI, multi-user web console, WeChat |
| Model providers | GLM, DeepSeek, Step, Kimi, FreeLLMAPI, iFlytek Spark MaaS, Agnes AI |

See the [feature acceptance matrix](docs/feature-matrix.md) for detailed implementation status.

## Requirements

- Python `3.11`–`3.14`
- macOS, Linux, or Windows
- An API key for at least one supported model provider
- Optional: Chrome and `npx` for the default Chrome DevTools MCP integration

## Quick start

### 1. Install from source

```bash
git clone https://github.com/Lukyyyyy/KairoCLI.git
cd KairoCLI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### 2. Configure a model

Copy the configuration template and set at least one provider API key:

```bash
cp .env.example .env
```

For example:

```dotenv
GLM_API_KEY=your-api-key
KAIROCLI_PROVIDER=glm
```

**Credential safety:** Never commit `.env` or real credentials. Kairo CLI reads
`~/.kairocli/.env`, the workspace `.env`, and process environment variables in that order; later
sources take precedence.

### 3. Start

Run Kairo CLI inside the project you want it to work on:

```bash
kairocli
```

On the first run, use `/init` to create project instructions, then describe the task:

```text
/init
Analyze this project's test structure and identify the three highest-priority issues.
```

Use `/help` for the complete command index. Type `@` to complete and reference workspace files or
directories.

## Usage

### Interactive CLI

Common commands:

```text
/plan TASK                 Generate, review, and execute a plan
/team TASK                 Run a task with multiple agents
/index [PATH]              Index workspace code
/search QUERY              Search indexed code
/graph SYMBOL              Query definitions, containment, and calls
/memory list               List long-term memory
/session list              List resumable sessions
/todo list                 List current todos
/mcp list                  List MCP servers
/skill list                List available Skills
/policy                    Show the active security policy
/snapshot list             List workspace snapshots
/cancel                    Cancel the active task
/exit                      Exit Kairo CLI
```

Use `@relative/path` or `@<path with spaces>` to add local content to the context:

```text
Review @src/kairocli/policy.py
Summarize @docs/feature-matrix.md
```

Press `Shift+Enter` for a newline and `Enter` to submit. The default Inline renderer supports
Markdown, syntax highlighting, and collapsible tool output. Plain and Textual TUI modes are also
available.

### Plan and Team modes

```bash
kairocli -p "Plan and implement a cache layer" --mode plan
kairocli -p "Review security, performance, and test coverage in parallel" --mode team
```

Interactive Plan mode waits for review before execution. Non-interactive mode executes the generated
plan directly. Team mode schedules tasks by dependency and performs a final review.

### Non-interactive execution

Use `-p` / `--print` in scripts and CI:

```bash
kairocli -p "Inspect the project and report risks"
kairocli -p "Run the tests and explain failures" --output-format json
printf '%s' 'Summarize the current changes' | kairocli --print --output-format jsonl
```

Output formats:

- `text`: print only the final answer;
- `json`: print one schema v1 result;
- `jsonl`: emit start/result events in order.

Exit codes are `0` for success, `1` for execution failure, `2` for argument or startup errors, and
`130` for cancellation. Unattended execution never waits for approval and denies dangerous tools by
default. Allow only the tools a task needs:

```bash
kairocli -p "Fix formatting issues" \
  --allow-tool write_file \
  --allow-tool apply_patch
```

Use `--dangerously-skip-approvals` only when the surrounding runtime provides an equivalent security
boundary.

### Resume sessions

```bash
kairocli --continue
kairocli --resume session_xxxxxxxxxxxx
kairocli -p "Record the inspection result" --save-session
```

`--continue` resumes the latest non-empty session in the current workspace. `--resume` selects a
session by ID. Sessions remain bound to their original workspace.

### Runtime API and web console

Start the Runtime API on `127.0.0.1`:

```bash
KAIROCLI_RUNTIME_API_KEY=local-secret kairocli serve --http --port 8080
```

Clients authenticate with `X-Kairo-CLI-API-Key` or `Authorization: Bearer <key>`. The API supports
thread creation, turn execution and cancellation, plus cursor-based SSE recovery.

Start the multi-user web console with accounts, sessions, model settings, plan review, and tool
approval:

```bash
kairocli serve --http --web --port 8080
```

The web server listens locally by default; add `--lan` for a trusted LAN. Interactive first launch
creates an administrator. Non-interactive deployment requires `KAIROCLI_WEB_ADMIN_EMAIL` and prints
a one-time password. Public registration and password recovery require Tencent Cloud SES.

### WeChat channel

Multi-user deployments can bind WeChat under **Settings → Channels → WeChat** in the web console.
Messages are processed serially per account. Read-only tools are allowed automatically, writes need
a one-time approval code, and Shell and MCP are denied by default.

The legacy single-user CLI channel remains available:

```bash
kairocli wechat setup
kairocli wechat start
kairocli wechat status
kairocli wechat daemon start
```

## Configuration

Configuration sources are merged in this order, with later values taking precedence:

```text
built-in defaults
  → ~/.kairocli/config.json
  → ~/.kairocli/.env
  → <workspace>/.env
  → process environment
  → CLI arguments
```

### Model providers

| Provider | API key environment variable | Default model |
| --- | --- | --- |
| GLM | `GLM_API_KEY` | `glm-5.1` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` |
| Step | `STEP_API_KEY` | `step-3.5-flash` |
| Kimi | `KIMI_API_KEY` | `kimi-k2.6` |
| FreeLLMAPI | `FREELLMAPI_API_KEY` | `auto` |
| iFlytek Spark MaaS | `XFYUN_MAAS_API_KEY` | `Qwen3.6-35B-A3B` |
| Agnes AI | `AGNES_API_KEY` | `agnes-2.0-flash` |

Inspect or update provider settings from the interactive CLI:

```text
/model
/model deepseek
/config
/config provider deepseek api-key YOUR_KEY
/config provider deepseek model deepseek-v4-flash
```

See [.env.example](.env.example) for every environment variable and optional service.

### Long-term memory and code indexing

Long-term memory uses local keyword retrieval by default. If you explicitly allow an embedding
service to process saved content, enable hybrid retrieval:

```dotenv
KAIROCLI_MEMORY_SEMANTIC_SEARCH=true
KAIROCLI_EMBEDDING_PROVIDER=ollama
KAIROCLI_EMBEDDING_MODEL=nomic-embed-text:latest
KAIROCLI_EMBEDDING_BASE_URL=http://localhost:11434
```

The `alicloud`, `openai`, and `zhipu` embedding providers are also supported. Remote services receive
long-term memory text and indexed code chunks. Rebuild the full `/index` after changing an embedding
model or vector dimension.

Code indexing uses Python AST and Tree-sitter declarations for common languages, with automatic
fallbacks when a parser is unavailable. `/graph <symbol>` and the read-only `query_code_graph` tool
share persisted definition, containment, and syntax-level call relationships. Keyword reranking
supports identifiers, camelCase, and Jieba tokenization for Chinese.

### MCP, Skills, and browser

- User and project MCP configuration lives in `~/.kairocli/mcp.json` and `.kairocli/mcp.json`;
- Skills are loaded from built-in, user, and project layers and can be installed from local folders,
  a curated catalog, or Git repositories;
- First launch creates a default isolated Chrome DevTools MCP configuration; sensitive pages always
  require per-call approval;
- One MCP server failing to start does not prevent the main application from starting.

## Security model

Kairo CLI enforces these defaults:

- file tools only access permitted real paths inside the workspace and reject symlink escapes;
- approval prompts explain low, medium, and high-risk capabilities and reasons;
- workspace writes may be approved for a session, while high-risk Shell tools such as
  `execute_command` and `shell_exec` always require per-call approval;
- catastrophic shell patterns and path escapes are blocked directly by policy;
- non-interactive mode denies every approval-gated tool by default;
- mutating tools run serially while read-only tools use bounded concurrency;
- cancellation and timeouts clean up complete command and MCP subprocess trees;
- provider, MCP, tool, and terminal errors are redacted and size-limited before display;
- application logs contain only redacted lifecycle metadata—never prompts, tool arguments, answers,
  images, or reasoning;
- private traces are recorded only when explicitly enabled, and reasoning requires a separate opt-in.

These boundaries reduce accidental damage but do not replace code review, least-privilege credentials,
runtime isolation, or reliable backups. Review untrusted Skills, MCP servers, and shell commands before
running them.

## Data and project instructions

Kairo CLI does not read data directories belonging to other agent products:

| Scope | Location | Contents |
| --- | --- | --- |
| User | `~/.kairocli/` | Configuration, sessions, memory, tasks, audits, logs, runtime data |
| Project | `<workspace>/.kairocli/` | Project MCP, Skills, index, and runtime state |
| Instructions | `KAIRO.md`, `KAIRO.local.md` | Durable rules and local overrides |

Instructions load from user, workspace, `.kairocli`, and local scopes in order; later rules take
precedence. Instruction files in subdirectories apply only to their corresponding subtree.

## Architecture

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

See the [architecture guide](docs/architecture.md) for component boundaries, lifecycle management,
concurrency, and persistence design.

## Development and contributing

Issues and pull requests are welcome. Install the development dependencies before making changes:

```bash
python -m pip install -e '.[dev]'
```

Run the same quality gates as CI:

```bash
ruff check .
ruff format --check .
mypy src/kairocli
pytest
kairocli --version
```

CI runs tests on Linux and macOS and installation/import smoke tests on Windows with Python
`3.11`–`3.14`. When contributing:

- update command parsing tests and the README when commands change;
- update schemas, security policy, and agent prompts when tools change;
- never log prompts, tool arguments, answer content, images, or reasoning in application logs;
- never commit `.env`, real credentials, caches, logs, or runtime databases.

See the [contribution guide](CONTRIBUTING.md) before opening a pull request.

## Documentation

- [Architecture](docs/architecture.md)
- [Feature acceptance matrix](docs/feature-matrix.md)
- [Contribution guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Responsive product page](landing/index.html)
- [Product poster](assets/kairocli-poster-1200x1600.png)

## License

Kairo CLI is released under the [MIT License](LICENSE).

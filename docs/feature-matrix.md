# Kairo CLI Feature Acceptance Matrix

| Phase | Capability | Python acceptance |
|---:|---|---|
| 1 | ReAct Agent CLI | Serialized generation-fenced model/tool loop, paired cancellation evidence, iteration and token budgets |
| 2 | Plan and DAG | Strict finite validated DAG, review/replan, parallel ready nodes, failure propagation |
| 3 | Memory and context | Strict finite locked scoped memory, compact, clear and export |
| 4 | RAG | Atomic hash-incremental symbol chunks, strict embedding transport/storage recovery and hybrid retrieval |
| 5 | Team collaboration | Dependency batches, isolated workers, per-step review/retry and final reviewer |
| 6 | HITL | Globally serialized cancelable prompts, atomic tool/server approve-all caches, strict edited arguments, repeated policy preflight and locked corruption-tolerant audit records |
| 7 | Parallel tools | Four-way bounded read concurrency, fair serialized workspace mutations, per-tool timeout, ordered results, bounded command input/output and full process-group cleanup across pipe ownership |
| 8 | Multiple models | seven provider adapters with strict finite response, capability, reasoning and error contracts |
| 9 | Web | Strict typed search providers, fetch extraction, redirect and SSRF policy |
| 10 | MCP core | strict finite stdio and session-bound Streamable HTTP JSON-RPC, nonblocking bounded server requests, fair per-server call/lifecycle quiescence, idempotent manager lifecycle, rollback-safe restart and dynamic tools |
| 11 | MCP advanced | generation-safe bounded single-flight resources, lifecycle-leased prompts/logs, exact double-phase dynamic-tool unregistration, enable, disable, argument-switch restart and approval invalidation |
| 12 | Long context | usage accounting, conservative estimation, bounded CAS compaction and budgets |
| 13 | Chrome DevTools | strict finite version probe, atomic validated tab discovery, connection status and ownership |
| 14 | Browser session reuse | transactional autoConnect/browser-url/isolated MCP switching, atomic serialized guard/approval/operation/state commits, uncertain-state cancellation recovery and sensitive-page per-call approval |
| 15 | Skill system | deterministic three-layer overrides, strict finite locked state, bounded bodies and bounded safe references |
| 16 | Product terminal | inline REPL, incremental width-aware Markdown, low-noise tool-call events, terminal-control isolation, bounded idempotent shutdown and full-screen Textual interface |
| 17 | Diagnostics | post-edit parser fallback plus strict restartable stdio LSP lifecycle and merged diagnostics |
| 18 | Side history | isolated sync-pre/async-post snapshots, pre-turn restore, undo points and automatic failed-checkout rollback |
| 19 | Prompt layering | base, mode, project, memory and Skill contexts |
| 20 | Runtime API | authenticated persistent multi-turn context, transactional state/event transitions, serialized/idempotent turns, shutdown-terminalized active tasks, strict corruption-tolerant bounded/coalesced SSE replay and private retention |
| 21 | Image input | robust refs/clipboard, bounded processing, MCP parity and provider fallback |
| 22 | Line editing | private locked bounded corruption-tolerant history, completion, highlighting and status toolbar |
| 23 | WeChat channel | QR binding, cancellable bounded queue, mobile rendering, deny-by-default policy, strict finite typed API/account transport and generation-fenced daemon |
| 24 | Incremental editing | checked workspace-fenced unified patches with cross-validated paths, bounded process/presentation output and non-authoritative failure-isolated post-edit diagnostics |
| 25 | Resumable sessions | workspace-scoped revision-CAS, finite-JSON, counter-bounded, protocol-safe canceled-tool-aware CLI and TUI recovery |
| 26 | Structured todos | session-scoped atomic list, one-active invariant, CLI/TUI and Agent tools |
| 27 | Persistent shell | bounded stateful cwd/env, guarded commands, timeout/cancel process-tree cleanup |
| 28 | Scoped instructions | safe imports, deterministic layers, subtree index and on-demand resolution |
| 29 | Noninteractive CLI | stdin/argument input, finite redacted Unicode-safe text/JSON/JSONL terminal contracts, deny-by-default/exact-name approvals, streamed tool round trips, exit codes and installed-wheel subprocess E2E |
| 30 | Private model traces | opt-in scoped finite JSONL, double-gated multi-form credential-redacted reasoning and locked bounded retention |
| 31 | Secret-bearing config | strict finite-JSON, bounded private cross-process merge and atomic storage without copying environment keys |
| 32 | Durable background tasks | private bounded queue, cross-process claims, generation-fenced restart, isolated terminal notifications and stale-result CAS |

Every phase must pass unit or contract coverage plus the global Ruff, Mypy and Pytest gates.
Every persistent or user-facing failure boundary must also use total Unicode-safe exception
terminalization, shared credential redaction and bounded UTF-8 output; diagnostic rendering may not
replace the original result or leave a running state behind.

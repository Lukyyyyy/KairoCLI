# Kairo CLI Architecture

Kairo CLI uses one async runtime shared by the interactive CLI, Runtime API, durable workers and the
WeChat channel. `Agent` owns conversation state; `ToolRegistry` owns schemas and bounded execution;
policy checks happen before optional approval and tool dispatch.

The package is organized around shallow domain packages. `agent/` separates the core runtime,
planning, orchestration, context accounting and compaction. `cli/` separates argument parsing,
bootstrap, interactive, noninteractive, completion and private history responsibilities. `tools/`
owns the registry, schemas, validation, filesystem, process, shell and web implementations. `mcp/` separates configuration,
protocol framing, client transport, lifecycle management, resources and commands. WeChat account,
client, channel, daemon and formatting implementations live under `channels/wechat/`. Package
`__init__.py` files preserve established imports such as `from kairocli.agent import Agent` and the
`kairocli.cli:main` entry point without restoring the former monolithic modules. Compatibility files
for older `kairocli.context`, `kairocli.compaction`, `kairocli.shell`, `kairocli.web` and
`kairocli.wechat` imports contain imports only; they do not own runtime behavior.

The WeChat channel separates polling from one active Agent task. Ordinary messages enter a bounded
FIFO, while help/status/pause/resume/stop bypass it; active polling uses a three-second ceiling so
cancellation remains responsive. Typing refresh is best-effort and failures never block an answer.
The remote policy defaults shell and MCP to deny, supports exact command plus exact/server MCP
allowlists, permits workspace-fenced file creation, and always denies remote restore/browser-session
switches. Final content alone passes through a mobile Markdown/ANSI/table/code formatter and
3,800-character chunker.

The secret-bearing WeChat account uses bounded, symlink-rejecting, fsynced atomic storage with private
Unix modes and validated workspace/HTTPS endpoint fields. Its strict finite JSON rejects duplicate
keys, non-standard constants and trees above 16 levels/10,000 nodes. Full replacement, clear and sync
cursor updates share a private no-follow cross-process lock. Cursor writes compare-and-swap the complete
loaded account generation, so a daemon with stale credentials exits instead of overwriting a new setup.
Account reads use a limit-plus-one 128 KiB boundary, while daemon PID reads stop at 65 bytes; post-check growth cannot reach JSON parsing or
process ownership probes. iLink calls share public-network DNS and
rate policy, reject redirects and embedded credentials, stream at most 2 MiB, and strictly reject
duplicate/non-finite/invalid-UTF-8 responses above 32 levels or 200,000 nodes. Request dictionaries
are serialized once with standard finite JSON before rate acquisition and those exact validated bytes
are sent. Token, identity, cursor and media fields use explicit string/integer contracts instead of
generic coercion; update/message/item counts, sync state and polling timeout are bounded before channel
state changes.
Daemon mode owns a separate process group and closes parent log descriptors immediately after spawn.
Its atomic private PID file rejects symlinks; POSIX status/stop also verifies the live command belongs
to a Kairo CLI WeChat process before signaling the group. Private stdout/stderr rotate at 5 MiB with
three generations, and log inspection seeks only the final 64 KiB before returning 100 lines.

Inbound WeChat file/image items retain bounded encrypted-media metadata rather than being flattened
into text. The channel passes only media count/capability context to the Agent; AES keys and encrypted
query parameters never enter prompts or output. A private symlink-rejecting media directory establishes
the future CDN download/decryption boundary, which remains disabled until its protocol is implemented.

Each top-level run owns a shared cancellation event. ReAct, Plan workers, team workers, model
requests and tool batches inherit that event. Shell commands run in a dedicated process group so a
timeout or cancellation terminates descendants instead of leaving orphan processes behind. A single
Agent instance admits only one active run, preventing concurrent history mutation. Every admitted run
also receives a monotonically increasing generation; streaming observers require both that generation
and a clear cancellation event, so a detached provider request cannot emit stale deltas after the next
turn resets cancellation.
Cancellation during a tool batch preserves the assistant call and appends one synthetic canceled tool
result per call. Workspace mutations and MCP calls explicitly disclose that side effects may already
have completed; dangerous tools also receive a credential-redacted canceled audit entry. This keeps
provider history structurally valid and recoverable without erasing evidence of an ambiguous commit.
ReAct's optional token budget is charged per top-level run. A terminal model answer is retained even
when its usage reaches the limit; only another tool batch or model iteration is blocked. Budget
exhaustion is checked before appending an assistant tool-call message or dispatching side effects, so
the in-memory protocol remains resumable. Direct budget construction validates all limits, provider
usage counters cannot reduce totals with negative values, and limit errors report their threshold.

The Textual TUI gates submissions around one active turn. Only explicit `/cancel` sets the shared
cancellation event; other input is rejected as busy instead of replacing the Textual worker. ReAct,
Plan and Team share image preparation, workspace-fenced local mentions and MCP resource expansion.
MCP initialization runs as a separate UI worker and registers against the Agent's existing tool
registry. Textual `Unmount` cancels and awaits both turn and startup workers before session save,
MCP shutdown and tool shutdown, preventing the previous non-event `on_shutdown` hook from leaking
resources.

Terminal rendering is a separate trust boundary. Inline streaming, normal CLI output and text-mode
noninteractive output remove ANSI CSI, OSC, DCS and related string controls, unsafe C0/C1 bytes and
bidirectional override/isolate characters before writing to a terminal. The Textual transcript uses
the same sanitizer and converts every plain string to a Rich `Text` object with markup disabled, so
model responses, prompts, tool output, indexed source and exception text cannot inject styles,
terminal titles or clipboard operations. JSON and JSONL remain lossless structured transports and
rely on JSON escaping rather than display sanitization.

Streaming surfaces keep a constant-memory terminal-protocol state machine across model deltas. CSI
and OSC/DCS/SOS/PM/APC sequences remain suppressed when their introducer, payload and terminator land
in different chunks; CRLF normalization is also stateful. An unfinished protocol is discarded at a
turn or tool boundary, and the final writer still applies complete-string sanitization as defense in
depth.

User input has a separate pre-history boundary. CLI, Textual and Runtime normalize CRLF/bare CR to LF
before command parsing, transcript or turn reservation, and `Agent.run` repeats the check for every
surface and internal worker. The shared 1 MiB limit is measured in UTF-8 bytes; invalid surrogates and
oversized prompts fail before image processing, memory context refresh, history mutation or model
dispatch.

Interactive submissions containing an explicit inline image token are the sole ordering exception:
the bounded 50 MiB image processor validates and removes that payload first, then the remaining text
must pass the normal 1 MiB Agent boundary. Raw noninteractive and Runtime requests retain their 1 MiB
transport cap. Merely including an image marker cannot bypass the final post-preparation text check.

The interactive CLI then feeds sanitized deltas into a stateful lightweight Markdown renderer. It
buffers only the unfinished line and pending table, so chunks split inside Markdown or Unicode text
do not duplicate output. Heading, nested-list, quote and fenced-code layouts mirror the reference
terminal behavior. Tables use Unicode display widths and the detected terminal columns; wide cells
wrap inside the border, while long two-column tables become key/value blocks. Noninteractive and
Runtime transports deliberately bypass presentation rendering to preserve machine-facing content.

Prompt completion does not maintain a second integration catalog. Every completion request reads
canonical providers from the live config, server names from the MCP manager and effective names from
the three-layer Skill registry. Static slash/subcommand, local/image path and MCP resource candidates
share the same 50-item cap; dynamic state changes are visible on the next completion without rebuilding
the prompt session.

Renderer selection is presentation-only. `inline` owns the stateful Markdown renderer, while
`plain` streams sanitized model text without interpreting Markdown. Both paths keep identical Agent
history and tool behavior, and neither changes Runtime or JSON/JSONL payloads. `tui` remains the
separate Textual surface.

Before surface startup, terminal capability resolution verifies both input/output TTY state and
rejects `TERM=dumb`. TUI additionally requires an 80x24 terminal and honors `KAIROCLI_NO_TUI`;
fallback selects inline only when an interactive ANSI terminal remains, otherwise plain. Resolution
does not mutate durable configuration. `NO_COLOR` affects styling but not cursor/layout capability.

Tool-call visibility is an observer boundary on `Agent`, emitted after budget and stagnation checks
but before dispatch. A renderer failure is isolated from protocol history and side effects. The
shared formatter groups calls, maps built-in and MCP names to compact labels, extracts only one key
argument, redacts secrets, strips terminal controls and caps details at 80 characters; tool result
bodies are never copied into the progress line. Inline and Textual surfaces use compact summaries,
while plain mode expands grouped details. Recursive, non-finite, non-serializable or invalid-Unicode
arguments render a fixed safe placeholder instead of throwing from this observer boundary.

After a tool batch completes and cancellation is rechecked, a second failure-isolated observer emits
paired calls and `ToolOutput` metadata. Its formatter reports aggregate success/failure, maximum
elapsed time, timeout, truncation and image counts without rendering result text or image URLs. Tool
execution policy, Browser state commits and this formatter share one strict finite failure classifier:
duplicate fields, non-standard constants, over-deep JSON, `isError` and approval/policy denials cannot
be displayed or committed as success. Tool protocol messages are appended only after this
informational event returns.

Plan and Team children inherit both tool observers from the parent, keeping worker side effects
visible on every user surface. They intentionally do not inherit content-delta rendering because
parallel token streams cannot be composed into one valid Markdown transcript; child text remains
dependency/reviewer input and only the parent emits the final answer.

Successful `write_file` and `apply_patch` outputs may carry presentation-only `FileDiff` metadata.
Before/after capture rejects non-regular, symlinked, non-UTF-8 and oversized files and is capped by
characters, lines and changed-file count. The CLI/TUI formatter uses bounded unified diffs with two
context lines and terminal sanitization. `Agent` appends only `ToolOutput.text` to protocol history,
so diffs do not consume model context or become durable session/trace/Runtime payloads.

TUI approval returns the same structured `ApprovalResult` used by the inline CLI. The modal supports
allow-once, tool-scoped approval, MCP-server-scoped approval, skip, reject and edited JSON arguments.
Edited arguments share a 1 MiB/32-level/100,000-node strict decoder that rejects duplicate keys and
non-standard constants. Invalid JSON leaves the modal open with an inline error. Edited arguments return to `ToolRegistry`,
which repeats path/command/browser preflight before dispatch; UI approval therefore cannot turn an
initially safe request into a workspace escape.
All governed tool calls share one cancelable approval lock per registry. Cache recheck, prompt,
edited-argument validation and approve-all cache publication are one critical section, matching the
single-consumer terminal/TUI surface: parallel calls cannot overlap prompts or miss a just-published
tool/server approval. Invalid handler return types fail closed. Sensitive Browser writes still enter
the serialized prompt on every call and never publish or consume an approve-all cache entry.

Durable task command formatting is shared by inline and full-screen modes, so IDs, lifecycle fields,
validation and retention semantics cannot drift. The TUI starts the durable manager only after its
event loop is mounted. A terminal callback writes bounded completion/failure/cancellation previews
above the input; callback failures are isolated from task state and worker survival. Unmount closes
the manager, which cancels owned Agents and requeues genuinely interrupted rows before tool shutdown.
Claims persist a unique manager owner and PID. Store startup and every subsequent claim requeue only
rows whose owner process is no longer alive; a second live Kairo CLI instance cannot duplicate active
work. Completion/failure updates include the original owner predicate, so a detached or stale worker
cannot publish over a task that has been recovered and claimed by another manager. Every manager
start creates a fresh generation with its own stop event, owner token and current PID. Detached cleanup
from a previous generation conditionally removes only its own Agent mapping and remains fenced by its
old token, so rapid close/restart cannot publish stale output or lose cancellation ownership.

Code index, search and symbol graph commands operate on the same `CodeIndex` in inline and Textual
modes. Index file reads yield to the event loop, while a throttled first/every-25th/final progress
callback updates the TUI status without producing unbounded transcript output. Callback failures are
non-fatal, but task cancellation still propagates. Browser command parsing and status output are also
shared; both modes validate optional CDP ports and report empty tabs consistently.
CodeIndex selects its embedding implementation from Kairo CLI-prefixed environment variables: an
offline deterministic hash is the no-network default, while explicit Ollama/OpenAI/Zhipu settings
construct a bounded HTTP client. Remote responses stream into a 10 MiB cap and must return exactly one
finite, numeric, consistently-sized vector per input; dimensions cannot drift during a client session.

Memory/save and Skill command handlers are shared between inline and Textual modes. Skill state
changes rebuild the Agent's available-skill index immediately. Policy, bounded audit-tail and export
commands stay local; an unrecognized slash command is rejected before image/mention preparation or
any model call, preventing local administrative intent from leaking upstream. Session exports are
capped at 20 MiB and use a private fsynced temporary file plus atomic replace. Audit inspection reads
only the final 1 MiB/100 records and rejects every directory/daily-log symlink component. Export rendering retains the
system prompt and reasoning metadata, bounds tool results, normalizes line endings and selects a
backtick fence longer than any run in embedded tool JSON/output, so nested Markdown cannot terminate
the export block early.
Skill discovery preserves builtin < user < project precedence but refuses symlinked roots,
directories, manifests and references. The always-injected metadata index XML-escapes descriptions;
full bodies remain on-demand. Disabled state is size/count/name bounded and committed under a private
cross-process lock using fsync plus atomic replacement, with in-memory enablement changed only after
durable publication succeeds. Manifest and state reads use limit-plus-one byte boundaries after
metadata checks; state, lock and publication paths reject every symlink component. State JSON rejects
duplicate keys, non-standard constants and trees beyond 16 levels/10,000 nodes while degrading invalid
state to a warning. Reference discovery scans at most 10,000 entries/16 levels and returns at most 1,000
files instead of materializing an unbounded recursive glob before applying the result limit.

Provider/model and config commands also share one transactional handler. Aliases normalize before
validation, failed saves restore every mutated in-memory field (including private API-key tracking),
and successful changes explicitly require session restart to rebuild the active client. Config status
only reports `configured`/`missing`; Textual transcript rendering masks API-key flag values before
writing the submitted command to the RichLog.

Long-term memory accepts at most 1,000 validated entries, 10,000 characters per fact and a 10 MiB
file. It scans newest-to-oldest until it has 1,000 valid records, so malformed tail rows cannot evict
valid history. Its JSON is strict and finite: duplicate keys, non-standard numeric constants and trees
beyond 16 levels/10,000 nodes degrade the entire damaged file to empty, after which the next explicit
save can recover a canonical store. Reads are byte-bounded, and writes reject every symlink component before using a
private random `O_EXCL` temporary file, fsync and atomic replacement. Every save/delete/clear
transaction also owns a private lock file; POSIX uses `flock` and Windows uses an `msvcrt` byte lock,
preventing cross-process read-modify-write loss between CLI, TUI, Runtime and channel processes. Lock
nodes must be regular files, preventing FIFOs or device-like paths from blocking state operations.
Audit append is independently failure-isolated, recursively secret-redacted and bounded to 1 MiB per
finite standard-JSON event. Directory, daily-file, lock and rotation paths reject every symlink
component; a private regular-file cross-process lock serializes append, tail reads and five-generation
10 MiB rotation. Tail inspection strictly decodes and canonicalizes valid records while skipping
corrupt/partial rows. Failures never change the owning tool result.

Independent commands use a fresh process. Commands that genuinely need shared cwd or environment
may use a persistent `/bin/sh` session (or `cmd.exe` on Windows). The session manager admits at most
four shells, confines startup cwd to a real workspace directory, serializes commands within each
session, and frames results with an unguessable completion marker so exit status remains observable.
Commands are limited to 100 KiB and 300 seconds, output uses the same bounded head/tail evidence
shape, and idle shells expire after 30 minutes. Timeout, cancellation, explicit stop and registry
shutdown terminate the full process group. `shell_exec` passes the same command deny rules, HITL and
redacted audit path as one-shot execution. Timeout teardown drains remaining pipe data into the same
bounded collector, preserving diagnostic evidence. POSIX framing temporarily disables and then
restores xtrace/verbose around the marker; Windows uses a non-echoed `@echo` frame, so user-controlled
shell tracing cannot impersonate a completion frame or shift output into the next command.

Model calls share an OpenAI-compatible transport but retain provider contracts. Step requests
negotiate DeepSeek-style reasoning (and 2603 high effort), Kimi and DeepSeek alone replay assistant
reasoning history, Xfyun omits unsupported tools and sends its LoRA header, and GLM-5V selects the
multimodal endpoint/raw-base64 form. Streaming accepts three reasoning field shapes, preserves cache
usage aliases, surfaces in-band error events and rejects truly empty responses instead of silently
ending a turn. Both response modes use strict finite JSON, rejecting duplicate keys, non-standard
constants and trees above 32 levels/200,000 nodes. Streaming consumes raw bytes through the total/event
budgets, decodes strict UTF-8 and aggregates standard multi-line SSE data fields before parsing.
Tool-call normalization is shared by streaming and non-streaming responses: names and
object arguments are validated and bounded before dispatch, duplicate/non-finite JSON is rejected, calls are
limited to 100, and missing/invalid/duplicate IDs receive deterministic unique protocol IDs. Canonical
provider names and documented aliases resolve before configuration lookup. Non-streaming JSON is read
through a 20 MiB chunk budget rather than eager `post()` buffering; SSE applies the same total plus a
2 MiB event ceiling, and upstream HTTP error evidence is read only through a 10,000-byte budget. Both
HTTP and in-band SSE errors pass the shared credential redactor before reaching any caller or trace.
The Agent retries only transport failures and explicitly transient HTTP/in-band errors. Two retries
with bounded exponential backoff are the default; numeric Retry-After is honored within the wait cap.
Backoff uses the shared cancellation event. An internal delta observer permits retries before any
visible content while permanently disabling them after the first emitted content chunk, so output and
tool side effects cannot be duplicated. Auxiliary planner, compaction and reviewer calls share this policy.

MCP stdio connections use one reader loop with request-ID-to-future dispatch. Writes are serialized,
but requests remain concurrent and may complete out of order. Server list-changed notifications
refresh dynamic tools and resource metadata without restarting the Kairo CLI process.
Explicit resource mentions are expanded forward with a shared character/target budget. Resource
text, MIME metadata and errors are XML-escaped, oversized content carries partial metadata, and a
repeated server/URI becomes a compact duplicate reference instead of a second remote read.
MCP configuration and enablement state use strict finite JSON: duplicate keys, non-standard numeric
constants, trees deeper than 16 levels and trees above 10,000 nodes are rejected before semantic
validation. Configuration is also size-, count-, shape- and field-bounded. Process environment values override
workspace and user dotenv files; unresolved substitutions and invalid transports remain isolated as
that server's error instead of aborting the manager. HTTP URLs reject embedded credentials and
non-HTTP schemes. Config, enablement state and dotenv files use limit-plus-one reads after metadata
checks, and every config/state path component rejects symlinks; user-level secret-bearing
configuration is private on Unix. Runtime
enablement overrides are atomically stored in the private user directory instead of rewriting source
configuration. A private no-follow cross-process lock serializes read-modify-write updates, and each
operation merges only its target server into the latest state instead of publishing a stale snapshot.
CLI and TUI use the same management handler; lifecycle changes unregister all
namespaced tools/helpers and replace the Agent resource index rather than accumulating stale blocks.
Both stdio and Streamable HTTP enforce a 2 MiB JSON-RPC message boundary. The subprocess reader's
actual stream limit matches that contract, while a chunked line assembler retains at most one message
budget and keeps draining after an oversized stdout/stderr line. One malformed line therefore cannot
permanently stop response dispatch or fill the child pipe. Outgoing requests are checked before writing,
and HTTP responses are consumed incrementally and decoded as strict UTF-8 before JSON/SSE parsing.
All three protocol surfaces reject duplicate object keys, non-standard numeric constants and trees over
32 levels/200,000 nodes; outgoing payloads must also serialize as finite standard JSON before transport.
Invalid stdio lines and SSE events are isolated while later correlated responses remain usable. Unknown notifications are
diagnostic-only; actionable refresh tasks are capped at 100, and a dropped resource notification
invalidates the cache conservatively. Direct resource-tool rendering is
escaped and sized including markup, so entity expansion cannot make the generic tool cap cut a
forged or unterminated context block.
Server-initiated JSON-RPC requests are answered by a separate bounded task set, so a blocked stdin
write cannot stop the stdout reader from resolving unrelated client requests. Streamable HTTP binds
the first validated session identifier for the client lifetime and rejects later response headers that
attempt to rebind it.
Stdio servers own a dedicated POSIX session or Windows process group. Initialization is transactional:
any timeout, cancellation or protocol failure closes transport tasks and the full process tree before
the error escapes. Normal close/restart/disable uses the same TERM-then-KILL group cleanup (Windows
uses `taskkill /T /F`), preventing server-spawned workers from surviving the Agent that owned them.
POSIX cleanup addresses the original PGID even after its leader has exited and escalates to SIGKILL
when descendants ignore SIGTERM. Manager startup is lifecycle-locked and idempotent; cancellation
closes both fully registered and partially starting clients before the start operation unwinds.
Restart and argument-switch restart are rollback transactions: after the old client is closed, any
candidate start/registration failure or cancellation closes the candidate and reconstructs the old
client before reporting failure. Only a ready candidate replaces the in-memory config. Successful
restart/disable/enable invalidates both server-scoped and individual-tool approval caches.
Each MCP server has a fair shared/exclusive gate in `ToolRegistry`. Calls hold shared ownership through
approval, handler execution and result processing, retaining concurrency with sibling calls. Lifecycle
restart/disable/close takes exclusive ownership, drains existing calls and blocks later readers once a
writer queues. A queued call resolves its definition only after acquiring shared ownership, so it uses
the replacement schema/client rather than a stale handler captured before restart. Cancellation-safe
acquisition releases any ownership won concurrently with the cancel signal.
Dynamic registration is tracked by exact public names on each client. Manager cleanup detaches the
client registry before close and repeats exact unregistration after all notification tasks drain, so a
late list-changed refresh cannot leave a ghost tool. Resource mention expansion and MCP logs/resource/
prompt inspection also hold shared server ownership. Unknown tool/server inputs do not allocate gates,
keeping the per-registry gate table bounded by real configured or registered servers.
Resource content reads use a 128-item, URI-keyed single-flight table. Waiter cancellation is shielded
from the shared request, while client close owns and cancels all remaining read tasks. Per-URI versions
and a list-wide epoch are sampled around the remote read/subscribe operation; an update racing the first
read forces one fresh read, and repeated churn fails explicitly instead of caching a stale response.
URI input is schema/runtime bounded to 8,192 control-free characters.

Tool execution has two interfaces: the public compatibility API returns text, while the Agent path
uses structured `ToolOutput` values containing text plus validated image data URLs. MCP image
content is processed under the same 5 MiB provider limit as local images and appended only after
all tool-role messages. Historical image payloads are stripped at the next top-level turn.

Tool evidence is bounded before entering model context. `read_file` is line- and character-paged,
rejects binary content and reports UTF-8 replacement. Directory, glob and grep results expose
`partial` plus continuation metadata. Shell stdout/stderr are drained concurrently with bounded
head/tail previews while total byte counts remain observable. Arbitrary text is capped at 200,000
characters; oversized structured results are wrapped in valid JSON rather than sliced into invalid
syntax. JSON-looking tool results are strictly decoded for failure classification; ambiguous,
non-finite or over-complex payloads fail conservatively and cannot commit BrowserGuard success state.
Parallel calls retain input order, have a per-tool timeout and preserve completed siblings.
Web fetch streams at most 5 MiB and independently caps extracted Markdown.

Web search configuration is loaded from process environment plus user/workspace dotenv files.
Both dotenv sources reject symlinks and use a limit-plus-one 1 MiB read after metadata checks;
oversized or post-check growth is ignored as a whole, with process > workspace > user precedence.
Automatic selection tries configured Zhipu, SerpAPI and SearXNG providers in that order, falling back
on bounded failures or empty results; an explicit provider disables fallback. Provider JSON is
limited to 2 MiB and strict finite 32-level/100,000-node trees without duplicate fields. Result text
must remain text, and URLs reject credentials, controls, oversized input and unsafe schemes. Transient
429/5xx responses retry once, and errors redact credentials. Step 3.7 Flash can preferentially route through a registered Step search
MCP while preserving its schema aliases and approval path. Fetch and search share a sliding request
limit. Every fetch redirect repeats URL/DNS policy, embedded credentials are rejected, and the
connected peer is checked when exposed by the transport. Main-content extraction removes semantic
noise and scores text against link density; empty JavaScript/login shells return an explicit browser
fallback hint rather than plausible-looking empty evidence.

Focused edits use a bounded unified-diff tool. Patch headers are parsed before approval and fenced to
real workspace paths; binary data, rename/copy, `.git`, symlink/submodule and mode operations are
rejected. The `---/+++` markers must agree with each `diff --git` header. A full `git apply --check`
prevents partially applying a multi-file patch, and the actual apply runs in a cancellable process
group with bounded stdout/stderr capture. Changed files immediately pass through the same LSP/parser
diagnostics as `write_file`. Patch source is redacted from the audit log. Full-file writes use an
fsynced same-directory temporary file and atomic replacement while preserving an existing mode.
Write-specific path resolution rejects every existing symlink component; POSIX publication is
anchored to a verified no-follow directory descriptor, so a failed replace leaves the old inode intact.
The Python grep fallback treats a leading `**/` as zero-or-more directories, reads at most 5 MiB plus
one boundary byte per file and reports skipped oversized files as partial. Diff before-images use the
same limit-plus-one pattern under the existing display budget.
`read_file` uses sized `readline` fragments under its output character budget; oversized logical lines
are truncated without first allocating the complete line, and draining skipped lines retains exact
line-based continuation offsets.

Interactive conversation recovery uses a separate WAL-mode SQLite database under the user directory.
Sessions are keyed by an unguessable ID and exact resolved workspace; cross-workspace load, save and
delete operations return no state. Structured messages preserve assistant tool calls, tool result IDs,
reasoning needed by compatible providers and usage counters, but image bytes are replaced by text.
Before saving/loading, the history validator drops orphan tool messages and truncates an interrupted
assistant tool-call suffix, while a fully paired canceled batch is retained verbatim. State is capped at 1,000 messages/10 MiB and 100 sessions per workspace;
serialization uses standard finite JSON with explicit 32-level/200,000-node structural budgets, and
usage counters are normalized into the non-negative signed SQLite integer range. Unix storage is
private, and both the database and its direct parent reject symlinks. Raw terminal history independently
rejects likely secrets and image payloads.
Message saves use a monotonically increasing, automatically migrated revision and an immediate SQLite
CAS transaction. A Store remembers the revision it loaded; a stale CLI/TUI/noninteractive process
receives an explicit conflict instead of overwriting newer history. Todo replacement updates session
recency but deliberately does not advance the message revision, keeping its independent transaction
from producing false message conflicts.

Structured todos share the session database and exact workspace/session scope. Updates replace the
entire list and read back the returned snapshot inside one write transaction, preserve stable item
IDs and creation timestamps, and enforce a 50-item limit plus a single `in_progress` item. The
interactive CLI and TUI attach one mutable
controller to the active session, so switching sessions also switches the Agent's `read_todos` and
`update_todos` view. Session deletion cascades to its todos; noninteractive agents do not expose
session-only tools.

MCP resources are available through both namespaced tools and explicit
`@server:protocol://resource` mentions. Mention expansion is asynchronous, XML-escapes metadata,
caps inline text at 200,000 characters, caches reads, and invalidates cached content on resource
notifications.

Context capacity is derived from the active model rather than a fixed global limit. Before every
model call, Kairo CLI estimates system prompt, tool schemas, text, tool arguments and image cost. At
the model-specific threshold it summarizes older turns at a user-message boundary, then keeps the
three most recent user rounds intact. Manual `/compact` keeps the most recent round. Failed or
non-shrinking summaries never mutate history. Transcript construction is capped at 60,000 characters
and degrades recursive/non-finite tool arguments without failing token estimation. Empty or over-20,000
character summaries are rejected. Compaction snapshots message identities and commits only if history
is unchanged while awaiting the model; manual compaction, clear and system-context mutation share the
Agent active-run exclusion boundary.

All model call sites share scoped diagnostics: ReAct, planner, plan/team workers, step reviewers and
compaction. Tracing is off by default. The first opt-in records only request/response shape, latency
and usage; a second independent opt-in permits bounded reasoning text. Prompt bodies, images and
answer content are never directly serialized. Reasoning and upstream error strings redact credential
assignments including quoted JSON/CLI flags/provider prefixes, credential URLs/query parameters,
Bearer/Basic headers, private keys, JWTs, common vendor token prefixes, data images and long base64.
JSONL files live in a private user directory,
rotate at 10 MiB with ten-file retention, and trace I/O failures are swallowed so observability cannot
change execution semantics. Trace directories, daily files and pruning candidates reject every
symlink component. A private regular-file cross-process lock serializes target selection, no-follow
append/fsync and pruning, preventing concurrent runtimes from deleting or overfilling each other's
active trace. Compaction now records its actual
model usage instead of undercounting
calls and tokens.

Long-term memory is queried per user turn instead of being injected wholesale at startup. Search
uses CJK phrase fragments plus complete Latin words, relevance coverage and time decay. Only global
and current-project facts can enter the prompt. Exact normalized facts are deduplicated, writes are
atomic, and malformed legacy storage degrades to an empty store rather than breaking startup. The
10 MiB store uses a limit-plus-one read after metadata checks, while reads degrade safely and writes
reject every symlink component in the memory and lock paths.

The three execution paths are:

- ReAct: direct iterative model and tool execution.
- Plan: a bounded structured JSON DAG, interactive review, dependency batches, one early-failure
  replan and final synthesis.
- Team: dependency-aware specialist batches with isolated histories, bounded concurrency, strict
  per-step review with corrective retries, failure blocking and a final reviewer.

Planner and reviewer payloads use strict finite 1 MiB/16-level/10,000-node JSON. Duplicate keys or
non-standard constants cannot change a DAG or approval decision. Reviewer issue arrays are bounded
typed strings, and invalid JSON-shaped output is never re-approved by the natural-language fallback.

Planner, worker and reviewer calls contribute to the parent Agent's usage counters. Durable task
rows record start/finish times and duration; interrupted running rows return to the queue during
startup or graceful shutdown, so stopping the CLI does not strand work in a permanent running state.
Task completion, failure and cancellation are conditional SQLite state transitions, preventing a
late answer from overwriting cancellation. Agent construction and tool cleanup failures are isolated
to one row rather than terminating the worker. The private WAL database rejects symlinks; prompts and
outputs are each bounded to 1 MiB, errors are redacted and capped, and only the newest 1,000 terminal
rows are retained while queued/running work is never pruned.

The Runtime API requires a product API key, returns stable `thread_` and `turn_` identifiers, and
accepts `Idempotency-Key` on turn creation. Reusing a key with the same input returns the original
turn; reusing it with different input is rejected. SSE events carry monotonic IDs and can resume
with `after` or `Last-Event-ID`; replay responses disable proxy buffering and caching.
The development extra installs Starlette 1.x's `httpx2` TestClient backend explicitly, allowing the
Runtime contract suite to run with Python development mode and warnings promoted to errors without
changing the production HTTP/provider transport.
Turn cancellation is an authenticated idempotent endpoint. The store permits terminal transitions
only from `running`, so cancellation wins races against late completion/failure. Active Agents receive
the shared cancellation signal, shutdown signals every remaining Agent, and tool-cleanup exceptions
cannot rewrite a committed terminal state. Runtime database files and their direct parent reject
symlink redirection. Each running turn persists a unique Runtime owner and PID. Store startup and each
reservation recover only dead owners, so another live Runtime process cannot mark active work failed;
completion/failure writes also require the original owner token. A post-fork worker refreshes both
owner fields before accepting turns instead of inheriting its parent's identity.

The one-shot CLI is a separate presentation boundary over the same Agent, MCP, image, snapshot and
session components. A prompt comes from `--print [PROMPT]` or bounded stdin. Text mode reserves
stdout for the final answer; JSON emits one schema-v1 result and JSONL emits a start/result sequence,
with stable status, provider/model/mode/session, usage, duration, warnings and typed errors. Ephemeral
execution is the default; explicit save/resume flags opt into the session database. Noninteractive
approval never reads stdin: dangerous and MCP tools are denied unless their exact name is repeated
with `--allow-tool`, while the intentionally conspicuous bypass flag disables all approval gates.
Input is capped at 1 MiB, JSONL flushes each event, and success/error/cancellation map to deterministic
process exit codes. Every terminal payload passes one schema normalizer after cleanup: answers are
UTF-8-safe and capped at the provider's 20 MiB response boundary; errors are credential-redacted and
limited to 4,000 bytes; at most 100 redacted 2,000-byte warnings survive. Usage and duration accept
only bounded non-negative integers, and `allow_nan=False` makes JSON/JSONL serialization total for
all internal result states. An exception whose own string conversion fails receives a stable typed
fallback instead of preventing the terminal event.

The local Runtime API stores each completed prompt/final-answer pair and restores a bounded recent
history before the next turn on that thread. A database transaction permits only one new running
turn per thread while preserving idempotent retries of the same request. Streaming callbacks append
individual delta events; non-streaming providers append one fallback delta. Completion status is
committed atomically with its terminal event; thread creation and running-turn reservation use the same
state/event transaction boundary. An event insertion failure therefore rolls back the related state
change instead of leaving an unobservable terminal result or an orphaned running turn. Startup and
subsequent reservations mark only dead-owner running turns failed, so a crashed process cannot
permanently lock the thread while overlapping live instances remain isolated. Legacy databases gain
response/error/owner columns in place; the runtime directory and database are private on Unix.
Streaming callbacks accumulate into 4,096-character event chunks, avoiding one SQLite transaction
per model token. Event JSON is capped at 256 KiB with a valid partial envelope; each thread retains
10,000 events and 100 turns, while creation keeps the latest 1,000 inactive threads and never evicts
a running one. Pruning events, turns and thread-owned rows occurs inside the writer transaction.
Event writes require a line-safe bounded type and finite JSON with depth/node budgets. Replay treats
stored rows as untrusted: duplicate keys, non-standard constants, invalid shapes and SSE-injecting type
names become a same-ID `runtime.event.invalid` envelope, preserving cursor progress and later events.
Terminal errors share the trace redactor before either database or SSE persistence. Replay queries
read one look-ahead row, so `X-Kairo-CLI-Has-More` reflects an actual next event rather than merely a
full current page.

Resumable session loading treats its private SQLite payload as potentially corrupt. The database
path cannot be a symlink, and SQL returns the message JSON only when its byte length is within the
10 MiB contract. Deserialization accepts user/assistant/tool roles only, begins at a user boundary,
normalizes text-only content parts, replaces historical images, bounds messages/reasoning/tool calls
and strictly finite JSON arguments, then repairs complete assistant-tool protocol groups. Explicit
node/depth budgets are checked after parsing rather than relying on a Python-version-specific recursion
limit. Stored system roles, invalid identifiers, orphan tool results, non-standard constants and
oversized payloads never enter an Agent prompt; corrupt counters degrade to bounded non-negative values.

Skills load in deterministic builtin, user and project order, with later layers overriding the
same declared name. The registry parses a documented YAML subset, retains source/version/author/
tags, records non-fatal warnings and persists only disabled names using atomic replacement. The
system prompt contains a bounded sorted index; `load_skill` returns at most 5 KiB of relevant body.
References are listed by the skill and loaded one at a time through `load_skill_reference`, which
confines real paths to `references/`, rejects binary files and caps text at 100,000 characters.

Project instructions use a separate `InstructionResolver`. Global, workspace, project-state and
local `KAIRO.md` layers are deterministic, while a one-token `@relative/path` line expands imports
only within its user or workspace root. Imports reject traversal and symlink escape, detect cycles,
stop beyond depth three, reject binary sources, bound each read and share a 24,000-character render
budget. Nested instruction files are discovered without following symlinks or entering dependency,
build, VCS, virtual-environment and state directories. Their 8,000-character location index is in
the system prompt; `load_project_instructions` resolves only the root-to-target ancestor chain, with
more-specific and local layers last. Target paths are length-bounded and fenced to the workspace.

Local-path mentions use the same realpath fence. Files are read with a byte-limited binary handle,
directories stop scanning after the configured entry cap, and all mentions share a character and
target-count budget. Repository text and names are XML-escaped before insertion; truncation,
binary omission, duplicate references and exhausted budgets remain visible to the model. Colon
mentions are reserved for MCP/image inputs and are never partially interpreted as local paths.

Browser reuse starts isolated. Connecting validates port range, probes `/json/version`, then reads
`/json/list` before switching to shared mode. Both responses use bounded strict finite JSON; version
fields and tab metadata have explicit types/lengths, malformed tabs are isolated and duplicate IDs
are rejected. Port, mode, URL and tabs commit together only after full validation, so a failed reconnect
preserves the prior live session. The session tracks the last successful navigation and
IDs of tabs created by Kairo CLI; shared mode refuses to close any other Chrome tab. Chrome DevTools
MCP calls pass through `BrowserGuard`: writes on configured bank/payment/settings/admin pages always
require a fresh approval even after tool/server approve-all. The approval UI receives the matching
rule, and every Chrome MCP result is audited with mode, sensitivity and a query-free URL. Additional
glob rules can be placed in `~/.kairocli/sensitive_patterns.txt`.
One per-registry Browser operation lock spans guard evaluation, HITL, the remote MCP call and successful
session mutation, so parallel navigation cannot change the page between approval and a write. A call
canceled after its remote handler starts marks mutable Browser state uncertain; later writes require
fresh per-call approval until a successful navigation commits a known URL. Cancellation while merely
waiting for the operation lock leaves the prior known state intact.
Interactive Browser mode changes use that same operation lock. Bare `/browser connect` restarts the
configured Chrome MCP with `--autoConnect`; an explicit port first probes version/list without mutation
and then restarts with `--browser-url`; disconnect restarts with `--isolated=true`. Browser session mode,
tabs and approval invalidation commit only after the candidate client is ready. Failed transitions use
the manager rollback and leave the previous Browser session untouched.
The optional rule file rejects every symlink component, uses a limit-plus-one 128 KiB read and admits
at most 1,000 non-comment patterns of 1,024 characters each. Invalid user storage never removes the
built-in bank, payment, settings, admin and cloud protections.

The code index stores a SHA-256 manifest per path. Unchanged files reuse existing vectors; changed
paths replace chunks and manifest state in one SQLite transaction, while deletion is reconciled for
the exact full or subdirectory indexing scope. Embedding provider/model/dimension signatures
invalidate incompatible vectors. Remote embedding JSON rejects duplicate fields, non-finite values,
deep/complex trees and ambiguous indices. Stored vectors are revalidated before cosine scoring;
one corrupt, oversized, dimension-mismatched or malformed chunk is skipped and its path manifest is
invalidated so the next index pass rebuilds it without taking down retrieval. Python AST and lightweight multi-language declarations produce
class/method chunks with symbol names, falling back to overlapping line windows for unstructured or
invalid files. Retrieval blends cosine and query-token coverage with a small symbol-kind boost.
Binary files, files over 2 MiB and symlinks escaping the workspace are never indexed.
Indexing and symbol-graph scans share a limit-plus-one 2 MiB source reader. Oversized, binary or
post-check growth atomically clears that path's stale chunks, while I/O failure preserves the last
valid path state; partial bytes never reach chunking, tree-sitter or embedding calls.

Every successful `write_file` runs bounded post-edit diagnostics. Python, JSON and TOML use native
parsers, while common brace-based languages retain a lightweight delimiter fallback. When a matching
language server is installed, Kairo CLI starts it over framed stdio JSON-RPC, negotiates `initialize`,
sends versioned `didOpen`/`didChange` notifications and merges published diagnostics into the tool
result. Language servers are shared per Agent and shut down with the CLI, TUI, channel, Runtime turn
or durable task that owns them; failed initialization terminates the complete child process group.
Requests for one document are serialized, dead-process restart clears open/version state, duplicate
responses are ignored, and diagnostics are accepted only for opened workspace-owned URIs. Framing uses
a 16 KiB header/10 MiB body contract; incoming JSON rejects duplicate keys, non-standard constants and
trees above 32 levels/200,000 nodes, while outgoing JSON is finite and size-checked before pipe writes.
Responses require JSON-RPC 2.0 and exactly one valid result/error. A malformed body with a known frame is
isolated, while a dead reader attached to a still-live process is detected on the next start and forces
full process-group cleanup plus reinitialization. Discovery
supports explicit and automatic Python, TypeScript, Go, Rust, Java and Clang language servers; the
local delimiter fallback also covers common C/C++ suffixes.
The read-only `lsp_inspect` tool can refresh diagnostics and request `textDocument/codeAction` for a
bounded line range. It returns at most 50 metadata summaries and deliberately strips WorkspaceEdit
text plus command arguments; applying a suggestion must still use audited workspace write tools.
Unsupported/failed code-action requests degrade to an empty action list without discarding diagnostics.
The companion `lsp_workspace_diagnostics` tool merges LSP 3.17 `workspace/diagnostic` reports with a
bounded local parser sweep. It scans at most 500 supported files/20,000 entries, skips dependency,
build, VCS and Kairo CLI state directories, rejects symlinks/external reports, deduplicates evidence,
and shares a limit-plus-one 1 MiB source reader with single-file inspect/diagnose entrypoints. Binary,
post-check growth and unreadable files never reach a parser or language server. The sweep returns at
most 1,000 sorted diagnostics with explicit partial metadata.

All user state is under `~/.kairocli`; workspace state is under `.kairocli`. Path tools resolve real
paths against the workspace before approval. Runtime events, tasks and RAG vectors use separate
SQLite databases. Snapshots use an isolated bare Git repository outside the workspace. Each CLI
turn commits a synchronous `pre-turn` baseline and queues an asynchronous `post-turn`; restore only
counts pre-turn commits and first records a `pre-restore` undo point. Side history has its own ignore
rules and never mutates the workspace's `.git` repository or refs. Snapshot repository operations
reject symlinks in every path component. Git pipes are drained concurrently into bounded buffers;
oversized tree output fails explicitly, while NUL-delimited tree entries preserve filenames that
contain newlines. Commit messages are bounded and sanitize embedded NUL before process launch. A
restore checkout failure automatically checks the just-captured pre-restore revision back out; if that
rollback also fails, both bounded causes are surfaced instead of claiming a successful restore.

Starter-project creation is staged beside the final target and published with a same-directory atomic
rename anchored to a verified no-follow directory descriptor on POSIX. Python, Node and Java templates
include the reference implementation's entry/manifest structures plus bounded modern metadata. A
non-empty destination is a policy denial before approval, and a publish failure restores a pre-existing
empty directory and removes the private staging tree.

Provider configuration is treated as secret-bearing state. Loading rejects symlinks, oversized or
malformed/non-object JSON, duplicate keys, non-finite constants, trees above 16 levels/10,000 nodes,
and invalid provider, renderer, URL, worker, token, temperature and context values. Config and dotenv
reads use both a metadata fast rejection and a limit-plus-one bounded read,
so growth after the metadata check cannot cause an unbounded allocation; existing Unix modes are
hardened to private. Saving validates again, writes and fsyncs a
0600 temporary file, then atomically replaces the destination under a 0700 directory. A failed save
leaves both disk and live CLI values unchanged. Publication is serialized by a private no-follow
cross-process lock. Each writer reloads the latest persisted payload while holding that lock and
overlays only fields changed relative to its own loaded runtime snapshot; disjoint updates from stale
CLI/TUI processes therefore merge instead of being lost. Temperature and max-token fields participate
in the same tracking. Effective API keys remember their origin: process or
dotenv overrides are usable in memory but are not copied to disk by unrelated configuration changes,
and a temporary environment override does not overwrite an already persisted key. Publication also
uses finite standard JSON, keeping load/save acceptance symmetric.

Image references are normalized at the Agent boundary so CLI, TUI, Runtime, durable tasks, WeChat,
Plan and Team share one path. `@image:path`, angle-wrapped/file URI paths and boundary-safe
`@clipboard` tokens become processed data URLs plus direct-inspection instructions. Images are capped
at 50 MiB source and 5 MiB base64 API payload, alpha is flattened, and only oversized payloads are
resized or converted to JPEG. Local image files are size-checked before opening, then read with a
source-limit-plus-one bounded read; non-regular filesystem entries are rejected before decoding, so
the source limit is also an allocation boundary. Decoding has an independent 40-million-pixel hard
limit, and Pillow decompression-bomb failures are never passed through as undecoded raw images. User
images and MCP image content use the same processor, and resized image metadata includes coordinate
mapping, while historical binary payloads are removed on the next turn.

Interactive completion never passes user text to a glob engine. It resolves a traversal-free parent
inside the workspace, scans at most 1,000 entries, rejects escaping/broken symlinks and returns at
most 50 sorted candidates. Space-containing local/image paths use angle syntax; MCP resource and
slash/subcommand candidates are also bounded. Input history drops empty, secret-bearing, image/base64,
and over-8,000-character lines before prompt-toolkit can persist them. Its directory and file are
private, every path component is checked for symlinks, and appends use a no-follow descriptor anchored
to a verified directory inode. Append, load and clear share a private cross-process lock; append trims
only at complete entry boundaries and keeps the file within 5 MiB. Unsafe or unavailable storage
degrades to in-memory history. Startup loads only the newest 5 MiB and at most 2,000 parsed entries;
malformed UTF-8 and legacy secret-bearing entries are isolated rather than restored.
The lexer overlays command, mention, image, secret, dangerous-shell and unclosed-delimiter styles
without mutating submitted text. Cancellation raised while a turn is executing is converted into the
shared Agent cancel signal and returns control to the REPL; input-level EOF/interrupt still performs
the normal session/MCP/task/tool shutdown path.

`/help` is a first-class parsed command rather than an inline-loop exception. Inline completion derives
from the command enum, while inline and Textual surfaces render one shared exhaustive reference. Plain
`help` and `?` remain inline compatibility aliases; unknown slash input stays local and is never sent
to the model.

## Total exception and text boundary

`text_safety.safe_text` is the lowest-level conversion boundary for provider, Agent, tool, MCP,
Runtime, Task and UI values. It catches failures raised by object string conversion, repairs invalid
Unicode and feeds `bound_utf8`, whose limit is expressed in encoded bytes rather than Python
characters. Secret-bearing errors additionally pass `trace.safe_redacted_text` before persistence,
logging or display. Plan/Team orchestration, background workers and persistent turns must commit a
terminal status even when the originating exception is unprintable; diagnostic cleanup is never
allowed to replace an already completed answer.

## Resource ownership and shutdown

Every long-lived process, client and task has one owning component. MCP owns its HTTP client, stdio
process, protocol readers, notification tasks and pending futures; ToolRegistry owns Shell, Snapshot
and LSP. Close operations transfer and clear ownership before awaiting cleanup, are idempotent, and
attempt every independent resource even when one close fails. Background done callbacks always
retrieve exceptions before removing tasks from tracking sets.

Runtime API additionally tracks the asyncio task behind every active turn. Lifespan shutdown first
commits `canceled` and its event through the turn CAS, then sends Agent cancellation and performs a
bounded task join. Late work cannot overwrite that terminal state. CLI, Textual and WeChat close
independent managers concurrently so an unavailable MCP server cannot leak Shell/LSP/Task resources.
Interactive CLI installs a single idempotent session-save/close path immediately after component
startup, including prompt initialization, input backend, external cancellation and broken-pipe exits.
Textual worker joins and post-turn Snapshot flushes have short grace periods; overdue auxiliary waits
are canceled and exception-consumed so UI shutdown remains bounded without abandoning process-tree
termination.

The release contract includes a black-box CLI harness backed by a loopback OpenAI-compatible SSE
server. It launches separate Python processes for every output format and for session save/resume,
plus a two-request streamed tool-call round trip. The tool scenarios verify a real `read_file`, default
denial of approval-gated writes, exact `--allow-tool write_file` authorization and the resulting file
side effect. Command scenarios additionally verify deny-by-default/exact-name authorization and that
the timeout covers both shell exit and inherited output-pipe closure, terminating a background POSIX
process group even when its original leader has exited. The harness then repeats the same suite with
imports forced to an isolated wheel
installation. This verifies the actual environment/configuration, HTTP framing, streaming aggregation,
persistence, approval policy, stdout purity and exit codes rather than inferring executable behavior
from in-process mocks.

Workspace-mutating built-ins share a fair per-registry serialization lock. Serialized approval still
occurs before the mutation lock, but policy is rechecked at the execution boundary; write, patch, project, revert and shell
side effects therefore cannot overlap inside one parallel tool batch. Post-edit diagnostics are an
auxiliary result: unexpected LSP failures become bounded credential-redacted warnings and never replace
a successfully committed write or patch with an error result.

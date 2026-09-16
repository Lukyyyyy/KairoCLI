from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ..agent import Agent
from ..browser import (
    BrowserGuard,
    BrowserSession,
    SensitivePagePolicy,
    handle_browser_command,
)
from ..config import (
    AppConfig,
)
from ..instructions import InstructionResolver
from ..llm import create_llm_client
from ..mcp import (
    McpServerManager,
    refresh_agent_resource_index,
)
from ..memory import MemoryStore
from ..paths import KairoPaths
from ..policy import ApprovalPolicy, AuditLog
from ..pricing import PricingConfig
from ..prompts import PromptAssembler
from ..rag import HashEmbeddingClient, embedding_client_from_environment
from ..skill_installer import SkillInstallRequest, install_skill
from ..skills import SkillRegistry, refresh_agent_skill_index
from ..snapshot import SnapshotService
from ..todos import SessionTodoController
from ..tools import ToolDefinition, ToolRegistry
from ..trace import LlmTraceLogger

log = logging.getLogger(__name__)
MAX_INTERACTIVE_ERROR_BYTES = 4_000


def make_agent(
    paths: KairoPaths,
    config: AppConfig,
    provider: str | None = None,
    approval_policy: ApprovalPolicy | None = None,
    approver: Any = None,
    skill_registry: SkillRegistry | None = None,
    browser_session: BrowserSession | None = None,
    todo_controller: SessionTodoController | None = None,
) -> Agent:
    llm = create_llm_client(config, provider)
    embedding = None
    if os.getenv("KAIROCLI_MEMORY_SEMANTIC_SEARCH", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        candidate = embedding_client_from_environment()
        if isinstance(candidate, HashEmbeddingClient):
            log.warning("memory_semantic_search_disabled reason=real_embedding_required")
        else:
            embedding = candidate
    try:
        semantic_min_score = float(os.getenv("KAIROCLI_MEMORY_SEMANTIC_MIN_SCORE", "0.45"))
    except ValueError as exc:
        raise ValueError("KAIROCLI_MEMORY_SEMANTIC_MIN_SCORE must be a number") from exc
    memory = MemoryStore(paths, embedding, semantic_min_score)
    skills = skill_registry or SkillRegistry(paths)
    if not skills.skills:
        skills.reload()
    extras: list[str] = []
    skill_index = skills.index()
    if skill_index:
        extras.append(f"<available_skills>\n{skill_index}\n</available_skills>")
    extra = "\n\n".join(extras)
    instruction_resolver = InstructionResolver(paths)
    prompt = PromptAssembler(paths, instruction_resolver=instruction_resolver).assemble(extra=extra)
    browser = browser_session or BrowserSession()
    browser_guard = BrowserGuard(
        browser, SensitivePagePolicy(paths.user_dir / "sensitive_patterns.txt")
    )
    tools = ToolRegistry(
        paths.workspace,
        AuditLog(paths),
        approval_policy,
        approver,
        browser_guard=browser_guard,
    )
    tools.snapshot_service = SnapshotService(paths)
    tools.current_provider = llm.provider
    tools.current_model = llm.model
    if todo_controller is not None:
        todo_controller.register(tools)
    agent_ref: Agent | None = None

    async def load_project_instructions(arguments: dict[str, Any]) -> str:
        target = str(arguments.get("path", ".")).strip() or "."
        content = await asyncio.to_thread(instruction_resolver.for_target, target)
        return content or "No KAIRO.md instructions apply to this target."

    tools.register(
        ToolDefinition(
            "load_project_instructions",
            "Load the ordered global, project, and nested KAIRO.md instructions that apply "
            "to one workspace path. Call before editing under a subtree listed in the "
            "instruction scope index.",
            {
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 4096}},
                "required": ["path"],
                "additionalProperties": False,
            },
            load_project_instructions,
        )
    )

    async def install_skill_tool(arguments: dict[str, Any]) -> str:
        request = SkillInstallRequest(
            source=str(arguments.get("source", "")).strip(),
            scope=str(arguments.get("scope", "user")).strip().casefold(),
            ref=(str(arguments["ref"]).strip() if arguments.get("ref") is not None else None),
            subdirectory=(
                str(arguments["path"]).strip() if arguments.get("path") is not None else None
            ),
            name=(str(arguments["name"]).strip() if arguments.get("name") is not None else None),
            force=bool(arguments.get("force", False)),
        )
        result = await asyncio.to_thread(install_skill, skills, request)
        refresh_agent_skill_index(agent_ref, skills)
        action = "Replaced" if result.replaced else "Installed"
        active_note = "" if result.active else " A higher-priority Skill remains active."
        return (
            f"{action} Skill {result.name} [{result.scope}] from {result.source} at "
            f"{result.target}.{active_note}"
        )

    tools.register(
        ToolDefinition(
            "install_skill",
            "Install a Kairo CLI Skill only when the user explicitly asks to install or "
            "replace one. Sources may be a curated name, local directory, GitHub owner/repo, "
            "GitHub tree URL, or HTTPS Git URL. Installation writes outside ordinary workspace "
            "files for user scope and always participates in HITL approval.",
            {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "minLength": 1, "maxLength": 4_096},
                    "scope": {"type": "string", "enum": ["user", "project"]},
                    "ref": {"type": "string", "minLength": 1, "maxLength": 255},
                    "path": {"type": "string", "minLength": 1, "maxLength": 1_024},
                    "name": {"type": "string", "minLength": 1, "maxLength": 128},
                    "force": {"type": "boolean"},
                },
                "required": ["source"],
                "additionalProperties": False,
            },
            install_skill_tool,
        )
    )

    async def load_skill(arguments: dict[str, Any]) -> str:
        name = str(arguments.get("name", "")).strip()
        skill = skills.skills.get(name)
        if skill is None or not skill.enabled:
            raise ValueError(f"Skill is unavailable: {name}")
        references = skill.references()
        reference_note = (
            "\n\nAvailable references (load only if needed):\n- " + "\n- ".join(references)
            if references
            else ""
        )
        return f"## Loaded Skill: {name}\n\n{skill.load_for_agent()}{reference_note}"

    tools.register(
        ToolDefinition(
            "load_skill",
            "Load the full instructions for an available Kairo CLI skill.",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            load_skill,
        )
    )

    async def load_skill_reference(arguments: dict[str, Any]) -> str:
        name = str(arguments.get("name", "")).strip()
        relative_path = str(arguments.get("path", "")).strip()
        skill = skills.skills.get(name)
        if skill is None or not skill.enabled:
            raise ValueError(f"Skill is unavailable: {name}")
        return skill.load_reference(relative_path, int(arguments.get("max_chars", 100_000)))

    tools.register(
        ToolDefinition(
            "load_skill_reference",
            "Load one text file from an already relevant skill's references directory. "
            "Use only paths listed by load_skill; traversal and binary files are rejected.",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "path": {"type": "string"},
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1_000,
                        "maximum": 100_000,
                    },
                },
                "required": ["name", "path"],
            },
            load_skill_reference,
        )
    )

    async def save_memory(arguments: dict[str, Any]) -> str:
        fact = str(arguments.get("fact", "")).strip()
        scope = str(arguments.get("scope", "project")).lower()
        key = str(arguments["key"]).strip() if arguments.get("key") is not None else None
        replaces = [str(value) for value in arguments.get("replaces", [])]
        entry, indexed = await memory.save_with_embedding(fact, scope, key=key, replaces=replaces)
        suffix = "" if memory.embedding is None or indexed else " (semantic indexing deferred)"
        return f"Saved long-term memory ({entry.scope}): {entry.fact}{suffix}"

    tools.register(
        ToolDefinition(
            "save_memory",
            "Only when the user explicitly asks to remember a durable fact or preference, save a "
            "concise reusable fact. Default to project scope; use global only across projects. "
            "Never save temporary task steps, one-off filenames, or model speculation. Before "
            "saving a correction, search memory, reuse a stable semantic-slot key, and replace "
            "legacy IDs. Store only the canonical current fact, without the superseded value as "
            "a negation, comparison, parenthetical, note, or history.",
            {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": (
                            "Canonical current fact only; omit superseded values "
                            "and change history."
                        ),
                    },
                    "scope": {"type": "string", "enum": ["project", "global"]},
                    "key": {
                        "type": "string",
                        "pattern": "^[a-z0-9][a-z0-9._:-]{0,127}$",
                        "description": "Stable semantic slot, never a current or previous value.",
                    },
                    "replaces": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[0-9a-f]{12}$"},
                        "maxItems": 20,
                        "uniqueItems": True,
                    },
                },
                "required": ["fact"],
                "additionalProperties": False,
            },
            save_memory,
        )
    )

    async def search_memory(arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"]).strip()
        limit = int(arguments.get("limit", 10))
        entries = await memory.search_hybrid(query, limit)
        return {
            "query": query,
            "results": [
                {
                    "id": entry.id,
                    "key": entry.key,
                    "fact": entry.fact,
                    "scope": entry.scope,
                    "created_at": entry.created_at,
                }
                for entry in entries
            ],
        }

    tools.register(
        ToolDefinition(
            "search_memory",
            "Search durable facts and preferences that the user explicitly asked Kairo to "
            "remember. Use when the user asks what Kairo remembers, asks how Kairo knows a "
            "preference or fact, "
            "or when a follow-up may depend on saved memory. Results are stored user-approved "
            "memories, not model guesses. Current user statements override stored memory. Do not "
            "present results as verbatim quotes or claim a specific prior conversation.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 2_000},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            search_memory,
        )
    )
    agent_ref = Agent(
        llm,
        tools,
        prompt,
        memory_store=memory,
        image_cache_dir=paths.user_dir / "cache" / "clipboard",
        trace_logger=LlmTraceLogger.from_environment(paths),
        pricing=PricingConfig.load(paths),
    )
    return agent_ref


def _inject_mcp_resource_index(agent: Agent, manager: McpServerManager) -> None:
    refresh_agent_resource_index(agent, manager)


def _register_browser_agent_tools(
    agent: Agent,
    browser: BrowserSession,
    manager: McpServerManager,
) -> None:
    """Expose the same transactional browser session controls to the Agent and CLI."""

    async def run_browser(operation: str) -> str:
        result = await handle_browser_command(
            operation,
            browser,
            mcp_manager=manager,
            approval_policy=agent.tools.approval_policy,
            tools=agent.tools,
        )
        failure_prefixes = (
            "Browser connection failed:",
            "Browser disconnect failed:",
        )
        if result.startswith(failure_prefixes):
            _, _, detail = result.partition(":")
            raise RuntimeError(detail.strip() or result)
        return result

    empty_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    definitions = (
        (
            "browser_connect",
            "Connect chrome-devtools to the user's shared local Chrome only when a page "
            "requires an existing login or the user explicitly requests the shared session. "
            "Do not use it preemptively for public pages.",
            "connect",
        ),
        (
            "browser_disconnect",
            "Return chrome-devtools to its isolated browser after shared-login work is complete.",
            "disconnect",
        ),
        (
            "browser_status",
            "Inspect the current isolated/shared browser mode and chrome-devtools MCP status.",
            "status",
        ),
    )
    for name, description, operation in definitions:

        async def handler(_arguments: dict[str, Any], selected: str = operation) -> str:
            return await run_browser(selected)

        agent.tools.register(
            ToolDefinition(
                name,
                description,
                empty_schema,
                handler,
            )
        )

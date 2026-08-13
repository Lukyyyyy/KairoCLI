# FastAPI intentionally declares dependency providers in callable defaults.
# ruff: noqa: B008

from __future__ import annotations

import asyncio
import copy
import json
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import OAuth2PasswordRequestForm

from .agent import AgentCanceled
from .config import (
    PROVIDER_DEFAULTS,
    AppConfig,
    normalize_provider_base_url,
    normalize_provider_name,
    validate_provider_protocol_fields,
)
from .runtime_api import (
    _IDEMPOTENCY_KEY,
    _THREAD_ID,
    _TURN_ID,
    MAX_RUNTIME_EVENT_RESPONSE_BYTES,
    MAX_SQLITE_INTEGER,
    RUNTIME_DELTA_CHARS,
    RUNTIME_SHUTDOWN_GRACE_SECONDS,
    RuntimeState,
    RuntimeThreadStore,
    _await_runtime_shutdown,
    _finish_detached_runtime_task,
    _valid_identifier,
)
from .trace import safe_redacted_text
from .user_input import UserInputError, normalize_user_input
from .web_auth import (
    JwtSecretStore,
    LoginRateLimiter,
    WebUser,
    WebUserStore,
    create_access_token,
    make_get_current_user,
    make_require_admin,
    verify_password,
)

_DETACHED_WEB_TASKS: set[asyncio.Task[Any]] = set()
MAX_CONFIG_PRESET_NAME_CHARS = 128


class WebApprover:
    """Per-turn approver: pauses the agent and waits for the web user's decision."""

    TIMEOUT_SECONDS = 300.0

    def __init__(self, thread_id: str, turn_id: str, state: RuntimeState) -> None:
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.state = state
        self._pending: asyncio.Event | None = None
        self._result: bool = False

    async def __call__(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        self.state.event(
            self.thread_id,
            "tool.approval_request",
            {
                "turn_id": self.turn_id,
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )
        self._pending = asyncio.Event()
        self._result = False
        try:
            await asyncio.wait_for(self._pending.wait(), timeout=self.TIMEOUT_SECONDS)
        except TimeoutError:
            self.state.event(
                self.thread_id,
                "tool.approval_timeout",
                {"turn_id": self.turn_id, "tool_name": tool_name},
            )
        return self._result

    def respond(self, approved: bool) -> None:
        self._result = approved
        if self._pending is not None:
            self._pending.set()


class WebPlanReviewer:
    """Per-turn plan reviewer: pauses the plan agent and waits for user decision."""

    TIMEOUT_SECONDS = 600.0

    def __init__(self, thread_id: str, turn_id: str, state: RuntimeState) -> None:
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.state = state
        self._pending: asyncio.Event | None = None
        self._result: str | bool = ""

    async def __call__(self, plan: Any) -> str | bool:
        tasks = [
            {
                "id": t.id,
                "description": t.description,
                "dependencies": list(t.dependencies),
            }
            for t in plan.tasks.values()
        ]
        self.state.event(
            self.thread_id,
            "plan.review_request",
            {"turn_id": self.turn_id, "tasks": tasks},
        )
        self._pending = asyncio.Event()
        self._result = ""
        try:
            await asyncio.wait_for(self._pending.wait(), timeout=self.TIMEOUT_SECONDS)
        except TimeoutError:
            self.state.event(
                self.thread_id,
                "plan.review_timeout",
                {"turn_id": self.turn_id},
            )
        return self._result

    def respond(self, action: str, feedback: str = "") -> None:
        if action == "cancel":
            self._result = False
        elif action == "supplement" and feedback.strip():
            self._result = feedback.strip()
        else:
            self._result = ""
        if self._pending is not None:
            self._pending.set()


class WebRuntimeState(RuntimeState):
    """Extends RuntimeState with per-turn approver and plan reviewer tracking."""

    def __init__(self, agent_factory: Any, store: RuntimeThreadStore) -> None:
        super().__init__(agent_factory, store)
        self.active_approvers: dict[str, WebApprover] = {}
        self.active_plan_reviewers: dict[str, WebPlanReviewer] = {}

    def refresh_owner_identity(self) -> None:
        super().refresh_owner_identity()
        self.active_approvers.clear()
        self.active_plan_reviewers.clear()


def _effective_user_config(base: AppConfig, stored: dict[str, Any]) -> AppConfig:
    config = copy.deepcopy(base)
    default_provider = normalize_provider_name(str(stored.get("default_provider", "")))
    if default_provider in config.providers:
        config.default_provider = default_provider
    stored_providers = stored.get("providers")
    if not isinstance(stored_providers, dict):
        return config
    for name, values in stored_providers.items():
        if name not in config.providers or not isinstance(values, dict):
            continue
        provider = config.providers[name]
        for field_name in ("api_key", "model", "base_url", "lora_id"):
            value = values.get(field_name)
            if isinstance(value, str):
                setattr(provider, field_name, value)
                if field_name == "api_key":
                    provider._persisted_api_key = value
                    provider._loaded_api_key = value
        for field_name in ("temperature", "max_tokens", "context_window"):
            if field_name in values:
                setattr(provider, field_name, values[field_name])
    return config


def _stored_user_config(
    config: AppConfig,
    existing: dict[str, Any],
    *,
    updated_api_key_provider: str | None = None,
) -> dict[str, Any]:
    existing_providers = existing.get("providers", {})
    if not isinstance(existing_providers, dict):
        existing_providers = {}
    providers: dict[str, Any] = {}
    for name, provider in config.providers.items():
        values: dict[str, Any] = {
            "model": provider.model,
            "base_url": provider.base_url,
            "lora_id": provider.lora_id,
            "temperature": provider.temperature,
            "max_tokens": provider.max_tokens,
            "context_window": provider.context_window,
        }
        previous = existing_providers.get(name, {})
        if updated_api_key_provider == name or (
            isinstance(previous, dict) and "api_key" in previous
        ):
            values["api_key"] = provider.api_key
        providers[name] = values
    return {"default_provider": config.default_provider, "providers": providers}


def _public_config(config: AppConfig) -> dict[str, Any]:
    return {
        "default_provider": config.default_provider,
        "providers": {
            name: {
                "model": provider.model,
                "base_url": provider.base_url,
                "has_key": bool(provider.api_key),
                "lora_id": provider.lora_id,
                "temperature": provider.temperature,
                "max_tokens": provider.max_tokens,
                "context_window": provider.context_window,
            }
            for name, provider in config.providers.items()
        },
        "provider_names": list(PROVIDER_DEFAULTS),
    }


def _preset_config(config: AppConfig) -> dict[str, Any]:
    public = _public_config(config)
    for provider in public["providers"].values():
        provider.pop("has_key", None)
    public.pop("provider_names", None)
    return public


def create_web_app(
    agent_factory: Any,
    *,
    runtime_database: Path,
    users_database: Path,
    jwt_secret_path: Path,
    allow_origins: list[str] | None = None,
    model_info: dict[str, str] | None = None,
    app_config: AppConfig | None = None,
) -> FastAPI:
    user_store = WebUserStore(users_database)
    jwt_secret = JwtSecretStore(jwt_secret_path).load_or_generate()
    store = RuntimeThreadStore(runtime_database)
    state = WebRuntimeState(agent_factory, store)
    get_current_user = make_get_current_user(user_store, jwt_secret)
    require_admin = make_require_admin(get_current_user)
    rate_limiter = LoginRateLimiter()

    if user_store.count() == 0:
        temp_password = secrets.token_urlsafe(16)
        user_store.create_user("admin", temp_password, is_admin=True)
        print(
            f"\n[KairoCLI Web] First run — admin account created.\n"
            f"  Username : admin\n"
            f"  Password : {temp_password}\n"
            f"  Please change this password after first login.\n",
            flush=True,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        state.refresh_owner_identity()
        try:
            yield
        finally:
            async def finish_shutdown() -> None:
                for turn_id in tuple(state.active_turn_ids):
                    active_agent = state.active_agents.get(turn_id)
                    if active_agent is not None:
                        try:
                            active_agent.cancel()
                        except Exception:
                            pass
                    try:
                        state.store.update_turn_status(
                            turn_id,
                            "canceled",
                            owner_token=state.owner_token,
                            event_type="turn.canceled",
                            event_data={"turn_id": turn_id},
                        )
                    except Exception:
                        pass
                    approver = state.active_approvers.pop(turn_id, None)
                    if approver is not None:
                        approver.respond(False)
                    reviewer = state.active_plan_reviewers.pop(turn_id, None)
                    if reviewer is not None:
                        reviewer.respond("cancel")
                active_tasks = tuple(state.active_turn_tasks)
                for task in active_tasks:
                    task.cancel()
                if active_tasks:
                    done, pending = await asyncio.wait(
                        active_tasks, timeout=RUNTIME_SHUTDOWN_GRACE_SECONDS
                    )
                    if done:
                        await asyncio.gather(*done, return_exceptions=True)
                    for task in pending:
                        _DETACHED_WEB_TASKS.add(task)
                        task.add_done_callback(_finish_detached_runtime_task)
                state.active_agents.clear()
                state.active_turn_ids.clear()
                state.active_turn_tasks.clear()

            shutdown_task = asyncio.create_task(finish_shutdown(), name="kairo-web-shutdown")
            await _await_runtime_shutdown(shutdown_task)

    app = FastAPI(title="Kairo CLI Web", version="1", lifespan=lifespan)
    app.state.runtime = state

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or [],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "PUT"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "Last-Event-ID"],
    )

    # ── Auth endpoints ───────────────────────────────────────────────────────

    @app.post("/auth/register", status_code=201)
    async def register(payload: dict[str, Any]) -> dict[str, str]:
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        if not username or not password:
            raise HTTPException(status_code=422, detail="用户名和密码不能为空")
        try:
            user_store.create_user(username, password, is_admin=False)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return {"status": "ok"}

    @app.post("/auth/login")
    async def login(
        request: Request,
        form: OAuth2PasswordRequestForm = Depends(),
    ) -> dict[str, str]:
        ip = request.client.host if request.client else "unknown"
        rate_limiter.check_and_record(ip)
        user = user_store.get_by_username(form.username)
        if user is None or not verify_password(form.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        token = create_access_token(user.id, user.username, user.is_admin, jwt_secret)
        return {"access_token": token, "token_type": "bearer"}

    @app.get("/auth/me")
    async def me(user: WebUser = Depends(get_current_user)) -> dict[str, Any]:
        return {"id": user.id, "username": user.username, "is_admin": user.is_admin}

    @app.post("/auth/logout")
    async def logout(_user: WebUser = Depends(get_current_user)) -> dict[str, str]:
        return {"status": "ok"}

    @app.put("/auth/me/password")
    async def change_own_password(
        payload: dict[str, Any],
        user: WebUser = Depends(get_current_user),
    ) -> dict[str, str]:
        old_password = str(payload.get("old_password", ""))
        new_password = str(payload.get("new_password", ""))
        if not old_password or not new_password:
            raise HTTPException(status_code=422, detail="旧密码和新密码不能为空")
        stored = user_store.get_by_username(user.username)
        if stored is None or not verify_password(old_password, stored.hashed_password):
            raise HTTPException(status_code=400, detail="旧密码不正确")
        try:
            ok = user_store.update_password(user.id, new_password)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        if not ok:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"status": "ok"}

    # ── Admin endpoints ──────────────────────────────────────────────────────

    @app.get("/admin/users")
    async def list_users(
        _admin: WebUser = Depends(require_admin),
    ) -> dict[str, Any]:
        users = user_store.list_users()
        return {
            "object": "list",
            "data": [
                {
                    "id": u.id,
                    "username": u.username,
                    "is_admin": u.is_admin,
                    "created_at": u.created_at,
                }
                for u in users
            ],
        }

    @app.post("/admin/users", status_code=201)
    async def create_user(
        payload: dict[str, Any],
        _admin: WebUser = Depends(require_admin),
    ) -> dict[str, Any]:
        username = payload.get("username", "")
        password = payload.get("password", "")
        is_admin = bool(payload.get("is_admin", False))
        if not username or not password:
            raise HTTPException(status_code=422, detail="用户名和密码不能为空")
        try:
            user = user_store.create_user(str(username), str(password), is_admin=is_admin)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return {"id": user.id, "username": user.username, "is_admin": user.is_admin}

    @app.put("/admin/users/{user_id}/password")
    async def reset_password(
        user_id: str,
        payload: dict[str, Any],
        admin: WebUser = Depends(require_admin),
    ) -> dict[str, str]:
        new_password = payload.get("password", "")
        if not new_password:
            raise HTTPException(status_code=422, detail="密码不能为空")
        try:
            ok = user_store.update_password(user_id, str(new_password))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        if not ok:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"status": "ok"}

    @app.delete("/admin/users/{user_id}")
    async def delete_user(
        user_id: str,
        admin: WebUser = Depends(require_admin),
    ) -> dict[str, str]:
        if user_id == admin.id:
            raise HTTPException(status_code=400, detail="不能删除自己的账号")
        ok = user_store.delete_user(user_id)
        if not ok:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"status": "ok"}

    # ── User-scoped model config endpoints ──────────────────────────────────

    @app.get("/v1/config")
    async def get_config(user: WebUser = Depends(get_current_user)) -> dict[str, Any]:
        if app_config is None:
            raise HTTPException(status_code=503, detail="Config not available")
        return _public_config(_effective_user_config(app_config, user_store.get_config(user.id)))

    @app.put("/v1/config")
    async def update_config(
        payload: dict[str, Any],
        user: WebUser = Depends(get_current_user),
    ) -> dict[str, str]:
        if app_config is None:
            raise HTTPException(status_code=503, detail="Config not available")
        existing = user_store.get_config(user.id)
        config = _effective_user_config(app_config, existing)
        provider_name = normalize_provider_name(str(payload.get("provider", "")))
        if provider_name and provider_name not in config.providers:
            raise HTTPException(status_code=422, detail=f"Unknown provider: {provider_name}")
        if "default_provider" in payload:
            dp = normalize_provider_name(str(payload["default_provider"]))
            if dp not in config.providers:
                raise HTTPException(status_code=422, detail=f"Unknown provider: {dp}")
            config.default_provider = dp
        if provider_name:
            provider = config.providers[provider_name]
            for field_name in ("model", "base_url", "lora_id"):
                if field_name in payload and payload[field_name] != "":
                    value = str(payload[field_name]).strip()
                    if field_name == "base_url":
                        try:
                            value = normalize_provider_base_url(value, provider_name)
                        except ValueError as exc:
                            raise HTTPException(status_code=422, detail=str(exc)) from None
                    if field_name == "model" and not value:
                        raise HTTPException(status_code=422, detail="model cannot be empty")
                    setattr(provider, field_name, value)
            if payload.get("api_key"):
                key = str(payload["api_key"]).strip()
                provider.api_key = key
                provider._persisted_api_key = key
                provider._loaded_api_key = key
            for num_field, lo, hi in (
                ("temperature", 0.0, 2.0),
                ("max_tokens", 1, 1_000_000),
                ("context_window", 0, 10_000_000),
            ):
                if num_field in payload and payload[num_field] != "":
                    try:
                        if num_field == "temperature":
                            v = float(payload[num_field])
                            if not lo <= v <= hi:
                                raise ValueError
                            provider.temperature = v
                        else:
                            v_int = int(payload[num_field])
                            if not lo <= v_int <= hi:
                                raise ValueError
                            if num_field == "context_window" and v_int != 0 and v_int < 8_000:
                                raise HTTPException(
                                    status_code=422,
                                    detail="context_window must be 0 or at least 8000",
                                )
                            setattr(provider, num_field, v_int)
                    except (ValueError, TypeError):
                        raise HTTPException(
                            status_code=422,
                            detail=f"Invalid value for {num_field}",
                        ) from None
            try:
                validate_provider_protocol_fields(provider, provider_name)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
        try:
            user_store.save_config(
                user.id,
                _stored_user_config(
                    config,
                    existing,
                    updated_api_key_provider=(
                        provider_name if provider_name and payload.get("api_key") else None
                    ),
                ),
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from None
        return {"status": "ok"}

    @app.get("/v1/config/presets")
    async def list_presets(user: WebUser = Depends(get_current_user)) -> dict[str, Any]:
        return {"object": "list", "data": user_store.list_config_presets(user.id)}

    @app.post("/v1/config/presets", status_code=201)
    async def save_preset(
        payload: dict[str, Any],
        user: WebUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        if app_config is None:
            raise HTTPException(status_code=503, detail="Config not available")
        name = str(payload.get("name", "")).strip()
        if not name or len(name) > MAX_CONFIG_PRESET_NAME_CHARS:
            raise HTTPException(
                status_code=422,
                detail=f"name must be 1–{MAX_CONFIG_PRESET_NAME_CHARS} characters",
            )
        config = _effective_user_config(app_config, user_store.get_config(user.id))
        return user_store.save_config_preset(user.id, name, _preset_config(config))

    @app.delete("/v1/config/presets/{preset_name}")
    async def delete_preset(
        preset_name: str,
        user: WebUser = Depends(get_current_user),
    ) -> dict[str, str]:
        if not user_store.delete_config_preset(user.id, preset_name):
            raise HTTPException(status_code=404, detail="Preset not found")
        return {"status": "ok"}

    @app.post("/v1/config/presets/{preset_name}/apply")
    async def apply_preset(
        preset_name: str,
        user: WebUser = Depends(get_current_user),
    ) -> dict[str, str]:
        if app_config is None:
            raise HTTPException(status_code=503, detail="Config not available")
        preset = user_store.get_config_preset(user.id, preset_name)
        if preset is None:
            raise HTTPException(status_code=404, detail="Preset not found")
        existing = user_store.get_config(user.id)
        config = _effective_user_config(app_config, existing)
        preset_config = _effective_user_config(config, preset)
        try:
            user_store.save_config(user.id, _stored_user_config(preset_config, existing))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from None
        return {"status": "ok"}

    # ── User-scoped v1 endpoints ─────────────────────────────────────────────

    @app.get("/v1/info")
    async def get_info(user: WebUser = Depends(get_current_user)) -> dict[str, Any]:
        if app_config is not None:
            config = _effective_user_config(app_config, user_store.get_config(user.id))
            provider = config.providers[config.default_provider]
            return {"provider": config.default_provider, "model": provider.model}
        return model_info or {"provider": "unknown", "model": "unknown"}

    @app.delete("/v1/threads/{thread_id}")
    async def delete_thread(
        thread_id: str,
        user: WebUser = Depends(get_current_user),
    ) -> dict[str, str]:
        if not _valid_identifier(thread_id, _THREAD_ID):
            raise HTTPException(status_code=404, detail="Thread not found")
        if state.active_turn_ids & {
            t for t in state.active_turn_ids
            if state.store.exists_for_user(thread_id, user.id)
        }:
            pass  # allow deletion even with running turns; cancel is separate
        ok = state.store.delete_thread(thread_id, user.id)
        if not ok:
            raise HTTPException(status_code=404, detail="Thread not found")
        return {"status": "ok"}

    @app.post("/v1/threads")
    async def create_thread(user: WebUser = Depends(get_current_user)) -> dict[str, str]:
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        state.store.create(
            thread_id,
            owner_user_id=user.id,
            event_type="thread.created",
            event_data={"thread_id": thread_id},
        )
        return {"id": thread_id, "object": "thread"}

    @app.get("/v1/threads")
    async def list_threads(user: WebUser = Depends(get_current_user)) -> dict[str, Any]:
        threads = state.store.list_threads(user.id)
        return {"object": "list", "data": threads}

    async def execute_turn(
        thread_id: str,
        prompt: str,
        turn_id: str,
        approver: WebApprover,
        user_id: str,
        mode: str = "agent",
    ) -> None:
        current_task = asyncio.current_task()
        if current_task is not None:
            state.active_turn_tasks.add(current_task)
        agent = None
        delta_buffer = ""
        emitted_delta = False

        def flush_delta(*, final: bool = False) -> None:
            nonlocal delta_buffer
            while len(delta_buffer) >= RUNTIME_DELTA_CHARS or (final and delta_buffer):
                chunk = delta_buffer[:RUNTIME_DELTA_CHARS]
                delta_buffer = delta_buffer[len(chunk):]
                state.event(thread_id, "message.delta", {"turn_id": turn_id, "delta": chunk})

        def emit_plan_events(plan: Any) -> None:
            tasks = [
                {"id": t.id, "description": t.description, "dependencies": list(t.dependencies)}
                for t in plan.tasks.values()
            ]
            state.event(
                thread_id,
                "plan.created",
                {"turn_id": turn_id, "tasks": tasks, "mode": mode},
            )

        def emit_task_started(task: Any) -> None:
            state.event(
                thread_id,
                "plan.task.started",
                {"turn_id": turn_id, "task_id": task.id, "description": task.description},
            )

        def emit_task_completed(task: Any, success: bool) -> None:
            state.event(
                thread_id,
                "plan.task.completed",
                {
                    "turn_id": turn_id,
                    "task_id": task.id,
                    "success": success,
                    "status": task.status.value,
                },
            )

        try:
            if state.store.turn_status(thread_id, turn_id) != "running":
                return
            if app_config is None:
                agent = agent_factory(approver=approver)
            else:
                user_config = _effective_user_config(
                    app_config, user_store.get_config(user_id)
                )
                agent = agent_factory(approver=approver, config=user_config)
            state.active_agents[turn_id] = agent
            agent.history = state.store.completed_messages(thread_id, before_turn_id=turn_id)

            def emit_delta(delta: str) -> None:
                nonlocal delta_buffer, emitted_delta
                if not delta:
                    return
                emitted_delta = True
                delta_buffer += delta
                flush_delta()

            agent.on_content_delta = emit_delta

            def emit_reasoning_delta(delta: str) -> None:
                if delta:
                    state.event(thread_id, "reasoning.delta", {"turn_id": turn_id, "delta": delta})

            def emit_tool_calls(calls: list[Any]) -> None:
                state.event(
                    thread_id,
                    "tool.calls",
                    {
                        "turn_id": turn_id,
                        "calls": [
                            {"id": c.id, "name": c.name, "arguments": c.arguments}
                            for c in calls
                        ],
                    },
                )

            def emit_tool_results(calls: list[Any], results: list[Any]) -> None:
                state.event(thread_id, "tool.results", {
                    "turn_id": turn_id,
                    "results": [
                        {"id": c.id, "name": c.name, "text": r.text[:2000]}
                        for c, r in zip(calls, results, strict=True)
                    ],
                })

            agent.on_reasoning_delta = emit_reasoning_delta
            agent.on_tool_calls = emit_tool_calls
            agent.on_tool_results = emit_tool_results

            if mode == "plan":
                from .agent import PlanExecuteAgent
                reviewer = WebPlanReviewer(thread_id, turn_id, state)
                state.active_plan_reviewers[turn_id] = reviewer
                plan_agent = PlanExecuteAgent(agent, review_handler=reviewer)
                plan_agent.on_plan_created = emit_plan_events
                plan_agent.on_task_started = emit_task_started
                plan_agent.on_task_completed = emit_task_completed
                answer = await plan_agent.run(prompt)
            elif mode == "team":
                from .agent import AgentOrchestrator
                team_agent = AgentOrchestrator(agent)
                team_agent.on_plan_created = emit_plan_events
                team_agent.on_task_started = emit_task_started
                team_agent.on_task_completed = emit_task_completed
                answer = await team_agent.run(prompt)
            else:
                answer = await agent.run(prompt)

            if answer and not emitted_delta:
                emit_delta(answer)
            flush_delta(final=True)
            state.store.update_turn_status(
                turn_id,
                "completed",
                response=answer,
                owner_token=state.owner_token,
                event_type="turn.completed",
                event_data={"turn_id": turn_id},
            )
        except AgentCanceled:
            state.store.update_turn_status(
                turn_id,
                "canceled",
                owner_token=state.owner_token,
                event_type="turn.canceled",
                event_data={"turn_id": turn_id},
            )
        except asyncio.CancelledError:
            state.store.update_turn_status(
                turn_id,
                "canceled",
                owner_token=state.owner_token,
                event_type="turn.canceled",
                event_data={"turn_id": turn_id},
            )
            raise
        except Exception as exc:
            flush_delta(final=True)
            error = safe_redacted_text(exc, 2_000, "...[runtime error truncated]")
            state.store.update_turn_status(
                turn_id,
                "failed",
                error=error,
                owner_token=state.owner_token,
                event_type="turn.failed",
                event_data={"turn_id": turn_id, "error": error},
            )
        finally:
            state.active_agents.pop(turn_id, None)
            state.active_approvers.pop(turn_id, None)
            state.active_plan_reviewers.pop(turn_id, None)
            state.active_turn_ids.discard(turn_id)
            if current_task is not None:
                state.active_turn_tasks.discard(current_task)
            if agent is not None:
                try:
                    await agent.tools.close()
                except Exception:
                    pass

    @app.post("/v1/threads/{thread_id}/turns", status_code=202)
    async def create_turn(
        thread_id: str,
        payload: dict[str, Any],
        background: BackgroundTasks,
        user: WebUser = Depends(get_current_user),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, str]:
        if not _valid_identifier(thread_id, _THREAD_ID):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not state.store.exists_for_user(thread_id, user.id):
            raise HTTPException(status_code=404, detail="Thread not found")
        raw_prompt = payload.get("input") if "input" in payload else payload.get("prompt")
        if raw_prompt is None:
            raise HTTPException(status_code=422, detail="input is required")
        if not isinstance(raw_prompt, str):
            raise HTTPException(status_code=422, detail="input must be a string")
        if not raw_prompt:
            raise HTTPException(status_code=422, detail="input is required")
        raw_mode = str(payload.get("mode", "agent")).lower().strip()
        turn_mode = raw_mode if raw_mode in {"agent", "plan", "team"} else "agent"
        try:
            prompt = normalize_user_input(raw_prompt)
        except UserInputError as exc:
            raise HTTPException(
                status_code=413,
                detail=safe_redacted_text(exc, 4_000, "...[runtime error truncated]"),
            ) from None
        if not prompt.strip():
            raise HTTPException(status_code=422, detail="input is required")
        if idempotency_key is not None:
            if not _IDEMPOTENCY_KEY.fullmatch(idempotency_key):
                raise HTTPException(status_code=422, detail="Invalid Idempotency-Key")
        state.refresh_owner_identity()
        try:
            turn_id, status_str, created = state.store.reserve_turn(
                thread_id,
                prompt,
                idempotency_key,
                owner_token=state.owner_token,
                owner_pid=state.owner_pid,
                event_type="turn.started",
                event_data={"input": prompt, "mode": turn_mode},
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail=safe_redacted_text(exc, 4_000, "...[runtime error truncated]"),
            ) from None
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail=safe_redacted_text(exc, 4_000, "...[runtime error truncated]"),
            ) from None
        if not created:
            return {"id": turn_id, "object": "turn", "status": status_str}
        approver = WebApprover(thread_id, turn_id, state)
        state.active_approvers[turn_id] = approver
        state.active_turn_ids.add(turn_id)
        background.add_task(
            execute_turn, thread_id, prompt, turn_id, approver, user.id, turn_mode
        )
        return {"id": turn_id, "object": "turn", "status": "running"}

    @app.post("/v1/threads/{thread_id}/turns/{turn_id}/cancel")
    async def cancel_turn(
        thread_id: str,
        turn_id: str,
        user: WebUser = Depends(get_current_user),
    ) -> dict[str, str]:
        if not _valid_identifier(thread_id, _THREAD_ID):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not state.store.exists_for_user(thread_id, user.id):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not _valid_identifier(turn_id, _TURN_ID):
            raise HTTPException(status_code=404, detail="Turn not found")
        turn_status = state.store.turn_status(thread_id, turn_id)
        if turn_status is None:
            raise HTTPException(status_code=404, detail="Turn not found")
        if turn_status == "running" and state.store.update_turn_status(
            turn_id,
            "canceled",
            event_type="turn.canceled",
            event_data={"turn_id": turn_id},
        ):
            active = state.active_agents.get(turn_id)
            if active is not None:
                active.cancel()
            approver = state.active_approvers.pop(turn_id, None)
            if approver is not None:
                approver.respond(False)
            turn_status = "canceled"
        return {"id": turn_id, "object": "turn", "status": turn_status}

    @app.post("/v1/threads/{thread_id}/turns/{turn_id}/approval")
    async def respond_to_approval(
        thread_id: str,
        turn_id: str,
        payload: dict[str, Any],
        user: WebUser = Depends(get_current_user),
    ) -> dict[str, str]:
        if not _valid_identifier(thread_id, _THREAD_ID):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not state.store.exists_for_user(thread_id, user.id):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not _valid_identifier(turn_id, _TURN_ID):
            raise HTTPException(status_code=404, detail="Turn not found")
        approver = state.active_approvers.get(turn_id)
        if approver is None:
            raise HTTPException(status_code=404, detail="No pending approval for this turn")
        approver.respond(bool(payload.get("approved", False)))
        return {"status": "ok"}

    @app.post("/v1/threads/{thread_id}/turns/{turn_id}/plan_review")
    async def respond_to_plan_review(
        thread_id: str,
        turn_id: str,
        payload: dict[str, Any],
        user: WebUser = Depends(get_current_user),
    ) -> dict[str, str]:
        if not _valid_identifier(thread_id, _THREAD_ID):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not state.store.exists_for_user(thread_id, user.id):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not _valid_identifier(turn_id, _TURN_ID):
            raise HTTPException(status_code=404, detail="Turn not found")
        reviewer = state.active_plan_reviewers.get(turn_id)
        if reviewer is None:
            raise HTTPException(status_code=404, detail="No pending plan review for this turn")
        action = str(payload.get("action", "approve"))
        feedback = str(payload.get("feedback", ""))
        reviewer.respond(action, feedback)
        return {"status": "ok"}

    @app.get("/v1/threads/{thread_id}/events", response_class=PlainTextResponse)
    async def events(
        thread_id: str,
        user: WebUser = Depends(get_current_user),
        after: int = Query(default=0, ge=0, le=MAX_SQLITE_INTEGER),
        limit: int = Query(default=100, ge=1, le=1_000),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> PlainTextResponse:
        if not _valid_identifier(thread_id, _THREAD_ID):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not state.store.exists_for_user(thread_id, user.id):
            raise HTTPException(status_code=404, detail="Thread not found")
        cursor = after
        if after == 0 and last_event_id:
            try:
                from .runtime_api import _last_event_cursor
                cursor = _last_event_cursor(last_event_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid Last-Event-ID") from None
        chunks: list[str] = []
        response_bytes = 0
        next_event_id = cursor
        selected = state.store.events(thread_id, cursor, limit + 1)
        has_more = len(selected) > limit
        for event in selected[:limit]:
            chunk = (
                f"id: {event.id}\nevent: {event.type}\n"
                f"data: {json.dumps(event.data, ensure_ascii=False)}\n\n"
            )
            chunk_bytes = len(chunk.encode("utf-8"))
            if chunks and response_bytes + chunk_bytes > MAX_RUNTIME_EVENT_RESPONSE_BYTES:
                has_more = True
                break
            chunks.append(chunk)
            response_bytes += chunk_bytes
            next_event_id = event.id
        body = "".join(chunks)
        return PlainTextResponse(
            body,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Kairo-CLI-Next-Event-ID": str(next_event_id),
                "X-Kairo-CLI-Has-More": str(has_more).lower(),
            },
        )

    # ── SPA fallback (must be last) ──────────────────────────────────────────

    _static_index = Path(__file__).parent / "web_static" / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_catchall(full_path: str) -> FileResponse:
        return FileResponse(_static_index)

    return app

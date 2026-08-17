import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kairocli.agent as agent_module
from kairocli.agent import Agent
from kairocli.config import AppConfig, ProviderConfig
from kairocli.llm import LlmClient
from kairocli.models import LlmResponse, Message
from kairocli.plan import ExecutionPlan, PlanTask, TaskStatus
from kairocli.tools import ToolRegistry
from kairocli.web_app import WebPlanReviewer, create_web_app
from kairocli.web_auth import JwtSecretStore, WebUserStore, create_access_token


class WebClient(LlmClient):
    provider = "test"
    model = "web-model"

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return LlmResponse(content="web answer")


def _web_app(tmp_path: Path) -> tuple[Any, dict[str, str]]:
    users_database = tmp_path / "web" / "users.db"
    secret_path = tmp_path / "web" / "jwt_secret.bin"
    user = WebUserStore(users_database).create_user("tester", "password-123")
    secret = JwtSecretStore(secret_path).load_or_generate()
    token = create_access_token(user.id, user.username, user.is_admin, secret)

    def factory(approver: Any = None, workspace: Path | None = None) -> Agent:
        return Agent(
            WebClient(),
            ToolRegistry(workspace or tmp_path, approver=approver),
            "system",
        )

    app = create_web_app(
        factory,
        runtime_database=tmp_path / "runtime" / "runtime.db",
        users_database=users_database,
        jwt_secret_path=secret_path,
        model_info={"provider": "test", "model": "web-model"},
        default_workspace=tmp_path,
        workspace_roots=[tmp_path],
    )
    return app, {"Authorization": f"Bearer {token}"}


def _sse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip().split("\n\n"):
        event_type = ""
        data: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if event_type:
            events.append((event_type, data))
    return events


def test_web_info_and_invalid_mode_fallback(tmp_path: Path) -> None:
    app, headers = _web_app(tmp_path)

    with TestClient(app) as client:
        assert client.get("/v1/info").status_code == 401
        assert client.get("/v1/info", headers=headers).json() == {
            "provider": "test",
            "model": "web-model",
        }
        thread_id = client.post("/v1/threads", headers=headers).json()["id"]
        response = client.post(
            f"/v1/threads/{thread_id}/turns",
            headers=headers,
            json={"input": "hello", "mode": "unsupported"},
        )
        assert response.status_code == 202
        events = _sse_events(
            client.get(f"/v1/threads/{thread_id}/events", headers=headers).text
        )

    started = [data for event, data in events if event == "turn.started"]
    assert len(started) == 1
    assert started[0]["input"] == "hello"
    assert started[0]["mode"] == "agent"
    assert started[0]["turn_id"].startswith("turn_")
    assert any(event == "turn.completed" for event, _data in events)


def test_web_favicon_uses_the_kairo_brand_icon(tmp_path: Path) -> None:
    app, _headers = _web_app(tmp_path)

    with TestClient(app) as client:
        response = client.get("/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert b'<linearGradient id="brand"' in response.content


def test_web_workspace_browser_and_thread_binding(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    used_workspaces: list[Path] = []

    users_database = tmp_path / "web" / "users.db"
    secret_path = tmp_path / "web" / "jwt_secret.bin"
    user = WebUserStore(users_database).create_user("tester", "password-123")
    secret = JwtSecretStore(secret_path).load_or_generate()
    headers = {
        "Authorization": f"Bearer {create_access_token(user.id, user.username, False, secret)}"
    }

    def factory(approver: Any = None, workspace: Path | None = None) -> Agent:
        assert workspace is not None
        used_workspaces.append(workspace)
        return Agent(WebClient(), ToolRegistry(workspace, approver=approver), "system")

    app = create_web_app(
        factory,
        runtime_database=tmp_path / "runtime" / "runtime.db",
        users_database=users_database,
        jwt_secret_path=secret_path,
        default_workspace=first,
        workspace_roots=[tmp_path],
    )

    with TestClient(app) as client:
        listing = client.get("/v1/workspaces", headers=headers).json()
        assert listing["default"] == str(first)
        assert {item["name"] for item in listing["directories"]} == set()

        root_listing = client.get(
            "/v1/workspaces", headers=headers, params={"path": str(tmp_path)}
        ).json()
        assert {item["name"] for item in root_listing["directories"]} >= {
            "first",
            "second",
        }
        assert [item["path"] for item in root_listing["projects"]] == [str(first)]

        selected = client.post(
            "/v1/workspaces", headers=headers, json={"path": str(second)}
        )
        assert selected.status_code == 200
        assert selected.json()["path"] == str(second)
        projects = client.get("/v1/workspaces", headers=headers).json()["projects"]
        assert [item["path"] for item in projects] == [str(first), str(second)]

        created = client.post(
            "/v1/threads", headers=headers, json={"workspace": str(second)}
        )
        assert created.status_code == 200
        thread = created.json()
        assert thread["workspace"] == str(second)
        assert client.post(
            f"/v1/threads/{thread['id']}/turns",
            headers=headers,
            json={"input": "hello"},
        ).status_code == 202
        assert client.get("/v1/threads", headers=headers).json()["data"][0][
            "workspace"
        ] == str(second)

        denied = client.post(
            "/v1/threads", headers=headers, json={"workspace": str(outside)}
        )
        assert denied.status_code == 422

        removed = client.request(
            "DELETE", "/v1/workspaces", headers=headers, json={"path": str(second)}
        )
        assert removed.status_code == 200
        assert removed.json()["deleted_threads"] == 1
        assert client.get("/v1/threads", headers=headers).json()["data"] == []
        assert [
            item["path"]
            for item in client.get("/v1/workspaces", headers=headers).json()["projects"]
        ] == [str(first)]

        assert client.request(
            "DELETE", "/v1/workspaces", headers=headers, json={"path": str(first)}
        ).status_code == 200
        assert client.get("/v1/workspaces", headers=headers).json()["projects"] == []

        assert client.post(
            "/v1/workspaces", headers=headers, json={"path": str(second)}
        ).status_code == 200
        assert [
            item["path"]
            for item in client.get("/v1/workspaces", headers=headers).json()["projects"]
        ] == [str(second)]

    assert used_workspaces == [second]


@pytest.mark.parametrize(
    ("mode", "class_name", "answer"),
    [("plan", "PlanExecuteAgent", "planned"), ("team", "AgentOrchestrator", "teamed")],
)
def test_web_execution_modes_emit_plan_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    class_name: str,
    answer: str,
) -> None:
    class FakeModeAgent:
        def __init__(self, agent: Agent, **_kwargs: Any) -> None:
            self.agent = agent
            self.on_plan_created: Any = None
            self.on_task_started: Any = None
            self.on_task_completed: Any = None

        async def run(self, _prompt: str) -> str:
            task = PlanTask("step-1", "执行测试步骤")
            plan = ExecutionPlan([task])
            self.on_plan_created(plan)
            task.status = TaskStatus.RUNNING
            self.on_task_started(task)
            task.status = TaskStatus.COMPLETED
            self.on_task_completed(task, True)
            return answer

    monkeypatch.setattr(agent_module, class_name, FakeModeAgent)
    app, headers = _web_app(tmp_path)

    with TestClient(app) as client:
        thread_id = client.post("/v1/threads", headers=headers).json()["id"]
        response = client.post(
            f"/v1/threads/{thread_id}/turns",
            headers=headers,
            json={"input": "run", "mode": mode},
        )
        assert response.status_code == 202
        events = _sse_events(
            client.get(f"/v1/threads/{thread_id}/events", headers=headers).text
        )

    assert any(
        event == "plan.created" and data["mode"] == mode for event, data in events
    )
    assert any(event == "plan.task.started" for event, _data in events)
    assert any(
        event == "plan.task.completed" and data["success"] is True
        for event, data in events
    )


def test_plan_review_endpoint_is_user_scoped(tmp_path: Path) -> None:
    app, headers = _web_app(tmp_path)

    with TestClient(app) as client:
        thread_id = client.post("/v1/threads", headers=headers).json()["id"]
        reviewer = WebPlanReviewer(thread_id, "turn_review", app.state.runtime)
        app.state.runtime.active_plan_reviewers["turn_review"] = reviewer

        assert (
            client.post(
                f"/v1/threads/{thread_id}/turns/turn_review/plan_review",
                headers=headers,
                json={"action": "supplement", "feedback": "增加测试"},
            ).status_code
            == 200
        )
        assert reviewer._result == "增加测试"
        assert (
            client.post(
                "/v1/threads/thread_missing/turns/turn_review/plan_review",
                headers=headers,
                json={"action": "approve"},
            ).status_code
            == 404
        )


def test_model_config_and_presets_are_isolated_per_user(tmp_path: Path) -> None:
    users_database = tmp_path / "web" / "users.db"
    secret_path = tmp_path / "web" / "jwt_secret.bin"
    user_store = WebUserStore(users_database)
    alice = user_store.create_user("alice", "password-123")
    bob = user_store.create_user("bob", "password-456")
    secret = JwtSecretStore(secret_path).load_or_generate()
    alice_headers = {
        "Authorization": f"Bearer {create_access_token(alice.id, alice.username, False, secret)}"
    }
    bob_headers = {
        "Authorization": f"Bearer {create_access_token(bob.id, bob.username, False, secret)}"
    }
    base_config = AppConfig(
        default_provider="glm",
        providers={
            "glm": ProviderConfig(
                base_url="https://open.bigmodel.cn/api/paas/v4",
                model="shared-model",
            ),
            "deepseek": ProviderConfig(
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
            ),
        },
    )
    used_configs: list[AppConfig] = []

    def factory(
        approver: Any = None,
        config: AppConfig | None = None,
        workspace: Path | None = None,
    ) -> Agent:
        assert config is not None
        used_configs.append(config)
        return Agent(
            WebClient(), ToolRegistry(workspace or tmp_path, approver=approver), "system"
        )

    app = create_web_app(
        factory,
        runtime_database=tmp_path / "runtime" / "runtime.db",
        users_database=users_database,
        jwt_secret_path=secret_path,
        app_config=base_config,
        default_workspace=tmp_path,
        workspace_roots=[tmp_path],
    )

    with TestClient(app) as client:
        assert client.get("/v1/config").status_code == 401
        assert client.get("/v1/config", headers=alice_headers).status_code == 200
        response = client.put(
            "/v1/config",
            headers=alice_headers,
            json={
                "provider": "glm",
                "default_provider": "glm",
                "model": "alice-model",
                "api_key": "alice-secret",
            },
        )
        assert response.status_code == 200
        assert client.post(
            "/v1/config/presets", headers=alice_headers, json={"name": "Alice preset"}
        ).status_code == 201

        alice_config = client.get("/v1/config", headers=alice_headers).json()
        bob_config = client.get("/v1/config", headers=bob_headers).json()
        assert alice_config["providers"]["glm"]["model"] == "alice-model"
        assert alice_config["providers"]["glm"]["has_key"] is True
        assert bob_config["providers"]["glm"]["model"] == "shared-model"
        assert bob_config["providers"]["glm"]["has_key"] is False
        assert client.get("/v1/config/presets", headers=bob_headers).json()["data"] == []
        assert client.post(
            "/v1/config/presets/Alice%20preset/apply", headers=bob_headers
        ).status_code == 404
        assert client.get("/v1/info", headers=alice_headers).json() == {
            "provider": "glm",
            "model": "alice-model",
        }

        thread_id = client.post("/v1/threads", headers=alice_headers).json()["id"]
        assert client.post(
            f"/v1/threads/{thread_id}/turns",
            headers=alice_headers,
            json={"input": "hello"},
        ).status_code == 202

    assert used_configs[-1].providers["glm"].model == "alice-model"
    assert used_configs[-1].providers["glm"].api_key == "alice-secret"
    assert base_config.providers["glm"].model == "shared-model"
    assert base_config.providers["glm"].api_key == ""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kairocli.agent as agent_module
from kairocli.agent import Agent
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

    def factory(approver: Any = None) -> Agent:
        return Agent(
            WebClient(),
            ToolRegistry(tmp_path, approver=approver),
            "system",
        )

    app = create_web_app(
        factory,
        runtime_database=tmp_path / "runtime" / "runtime.db",
        users_database=users_database,
        jwt_secret_path=secret_path,
        model_info={"provider": "test", "model": "web-model"},
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

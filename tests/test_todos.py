import json
from pathlib import Path

from kairocli.sessions import SessionStore, TodoStatus
from kairocli.todos import SessionTodoController
from kairocli.tools import ToolRegistry


async def test_todo_tools_follow_attached_session(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    store = SessionStore(tmp_path / "sessions.db")
    first = store.create(workspace, "glm", "model")
    second = store.create(workspace, "glm", "model")
    controller = SessionTodoController(store, workspace)
    tools = ToolRegistry(workspace)
    controller.register(tools)
    controller.attach(first.meta.id)

    created = json.loads(
        await tools.execute(
            "update_todos",
            {
                "items": [
                    {"content": "first task", "status": "in_progress"},
                    {"content": "verify", "status": "pending"},
                ]
            },
        )
    )
    assert created["total"] == 2
    controller.attach(second.meta.id)
    assert json.loads(await tools.execute("read_todos", {}))["todos"] == []
    controller.attach(first.meta.id)
    assert json.loads(await tools.execute("read_todos", {}))["total"] == 2
    await tools.close()


async def test_todo_cli_transitions_keep_one_active_item(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    store = SessionStore(tmp_path / "sessions.db")
    session = store.create(workspace, "glm", "model")
    controller = SessionTodoController(store, workspace)
    controller.attach(session.meta.id)

    await controller.command("add inspect behavior")
    await controller.command("add implement feature")
    items = await controller.list()
    await controller.command(f"start {items[0].id}")
    await controller.command(f"start {items[1].id}")
    updated = await controller.list()

    assert [item.status for item in updated] == [
        TodoStatus.PENDING,
        TodoStatus.IN_PROGRESS,
    ]
    await controller.command(f"done {items[1].id}")
    assert (await controller.list())[1].status == TodoStatus.COMPLETED
    assert "1/2 completed" in await controller.command("list")

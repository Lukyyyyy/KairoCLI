from datetime import datetime

from kairocli.rendering.session_display import format_session_list
from kairocli.sessions import SessionMeta


def _session(
    session_id: str,
    *,
    title: str,
    updated_at: str,
    message_count: int,
) -> SessionMeta:
    return SessionMeta(
        session_id,
        "/workspace",
        "glm",
        "model",
        title,
        updated_at,
        updated_at,
        message_count,
    )


def test_session_list_is_readable_and_hides_noncurrent_empty_sessions() -> None:
    current_id = "session_111111111111"
    rendered = format_session_list(
        [
            _session(
                current_id,
                title="配置 GLM",
                updated_at="2026-07-24T02:12:06+00:00",
                message_count=35,
            ),
            _session(
                "session_222222222222",
                title="New session",
                updated_at="2026-07-23T12:00:00+00:00",
                message_count=0,
            ),
            _session(
                "session_333333333333",
                title="你好",
                updated_at="2026-07-23T02:03:00+00:00",
                message_count=1,
            ),
        ],
        current_id,
        width=100,
        now=datetime.fromisoformat("2026-07-24T10:30:00+08:00"),
    )

    assert rendered.startswith("Sessions (2)\n\n● 配置 GLM")
    assert "Current · 35 messages · Today 10:12" in rendered
    assert "○ 你好\n  1 message · Yesterday 10:03" in rendered
    assert "session_222222222222" not in rendered
    assert "1 empty session hidden · /session list --all" in rendered


def test_session_list_all_includes_empty_sessions_and_narrow_layout() -> None:
    current_id = "session_111111111111"
    rendered = format_session_list(
        [
            _session(
                current_id,
                title="这是一个需要按终端显示宽度截断的很长中文会话标题",
                updated_at="2025-12-30T12:00:00+00:00",
                message_count=0,
            )
        ],
        current_id,
        show_all=True,
        width=40,
        now=datetime.fromisoformat("2026-07-24T10:30:00+08:00"),
    )

    lines = rendered.splitlines()
    assert lines[2].endswith("…")
    assert "Current · 0 messages · 2025-12-30 20:00" in rendered
    assert "  session_111111111111" in rendered
    assert "hidden" not in rendered


def test_session_list_preserves_malformed_timestamp_instead_of_failing() -> None:
    rendered = format_session_list(
        [
            _session(
                "session_111111111111",
                title="Title",
                updated_at="unknown",
                message_count=2,
            )
        ],
        "session_other",
    )

    assert "2 messages · unknown · session_111111111111" in rendered

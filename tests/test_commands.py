import pytest

from kairocli.commands import (
    SLASH_COMMAND_DESCRIPTIONS,
    SLASH_HELP,
    CommandType,
    parse_command,
)


def test_regular_input_is_not_command() -> None:
    assert parse_command("explain /plan").type == CommandType.NONE


def test_alias_and_payload() -> None:
    parsed = parse_command("/mem search architecture")
    assert parsed.type == CommandType.MEMORY
    assert parsed.payload == "search architecture"


def test_unknown_slash_command_is_rejected() -> None:
    parsed = parse_command("/not-real")
    assert parsed.type == CommandType.UNKNOWN
    assert parsed.payload == "/not-real"


def test_plain_exit_compatibility() -> None:
    assert parse_command("quit").type == CommandType.EXIT


@pytest.mark.parametrize("value", ["/help", "help", "?"])
def test_help_is_a_real_command(value: str) -> None:
    assert parse_command(value).type == CommandType.HELP


def test_help_covers_every_user_command() -> None:
    documented_groups = {
        token.removeprefix("/")
        for token in SLASH_HELP.replace("|", " ").split()
        if token.startswith("/")
    }
    expected = {
        item.value
        for item in CommandType
        if item
        not in {
            CommandType.NONE,
            CommandType.UNKNOWN,
            CommandType.HISTORY_CLEAR,
        }
    }
    assert expected <= documented_groups
    assert "/history clear" in SLASH_HELP


def test_every_slash_command_has_a_short_completion_description() -> None:
    expected = {
        f"/{item.value}"
        for item in CommandType
        if item not in {CommandType.NONE, CommandType.UNKNOWN, CommandType.HISTORY_CLEAR}
    }
    assert set(SLASH_COMMAND_DESCRIPTIONS) == expected
    assert all(description.strip() for description in SLASH_COMMAND_DESCRIPTIONS.values())


def test_snapshot_subcommands_preserve_payload() -> None:
    snapshot = parse_command("/snapshot status")
    restore = parse_command("/restore 2")
    assert snapshot.type == CommandType.SNAPSHOT
    assert snapshot.payload == "status"
    assert restore.type == CommandType.RESTORE
    assert restore.payload == "2"


def test_session_command_preserves_operation() -> None:
    parsed = parse_command("/session resume session_123456789abc")
    assert parsed.type == CommandType.SESSION
    assert parsed.payload == "resume session_123456789abc"


def test_task_command_preserves_full_identifier_and_prompt() -> None:
    task = parse_command("/task add inspect the worker")
    cancel = parse_command("/task cancel task_123456789abc")
    assert task.type == CommandType.TASK
    assert task.payload == "add inspect the worker"
    assert cancel.type == CommandType.TASK
    assert cancel.payload == "cancel task_123456789abc"

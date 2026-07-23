from kairocli.cli import _try_capture_snapshot
from kairocli.snapshot import SnapshotError


class FailingSnapshots:
    async def capture(self, message: str) -> str:
        raise SnapshotError(f"failed to capture {message}")


class RecordingConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str) -> None:
        self.messages.append(message)


async def test_snapshot_failure_becomes_non_fatal_warning() -> None:
    console = RecordingConsole()

    captured = await _try_capture_snapshot(
        FailingSnapshots(),  # type: ignore[arg-type]
        "pre-turn",
        console,
    )

    assert captured is False
    assert "Snapshot warning:" in console.messages[0]
    assert "failed to capture pre-turn" in console.messages[0]

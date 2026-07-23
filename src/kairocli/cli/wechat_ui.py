from __future__ import annotations

from pathlib import Path


def workspace_prompt(default: Path) -> str:
    return (
        "\nConnect WeChat\n"
        f"Workspace: {default}\n"
        "Press Enter to connect, enter another directory, or Ctrl+C to cancel:\n"
        "› "
    )


QR_SCAN_PROMPT = "Next: Scan this QR code with the WeChat account you want to connect:"

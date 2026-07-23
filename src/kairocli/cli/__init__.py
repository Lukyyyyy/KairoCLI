"""Command-line interface for Kairo CLI."""

from .bootstrap import _inject_mcp_resource_index as _inject_mcp_resource_index
from .bootstrap import _register_browser_agent_tools as _register_browser_agent_tools
from .bootstrap import make_agent as make_agent
from .completion import (
    _completion_word as _completion_word,
)
from .completion import (
    _highlight_input_line as _highlight_input_line,
)
from .completion import (
    _local_path_completion_candidates as _local_path_completion_candidates,
)
from .completion import (
    _slash_completion_candidates as _slash_completion_candidates,
)
from .history import (
    _append_input_history as _append_input_history,
)
from .history import (
    _clear_input_history as _clear_input_history,
)
from .history import (
    _is_sensitive_history_input as _is_sensitive_history_input,
)
from .history import (
    _load_input_history as _load_input_history,
)
from .interactive import _await_shutdown as _await_shutdown
from .interactive import _close_components as _close_components
from .interactive import _CommandOutputConsole as _CommandOutputConsole
from .interactive import _console as _console
from .interactive import _end_answer_block as _end_answer_block
from .interactive import _erase_with_default_background as _erase_with_default_background
from .interactive import _EscapeInterrupt as _EscapeInterrupt
from .interactive import _handle_browser as _handle_browser
from .interactive import _handle_command as _handle_command
from .interactive import _handle_config as _handle_config
from .interactive import _handle_mcp as _handle_mcp
from .interactive import _handle_memory as _handle_memory
from .interactive import _handle_session_command as _handle_session_command
from .interactive import _handle_shell as _handle_shell
from .interactive import _handle_skill as _handle_skill
from .interactive import _handle_task as _handle_task
from .interactive import _handle_trace as _handle_trace
from .interactive import _has_rich as _has_rich
from .interactive import _InteractiveWechatRuntime as _InteractiveWechatRuntime
from .interactive import _print_answer_prefix as _print_answer_prefix
from .interactive import _print_command_output as _print_command_output
from .interactive import _print_snapshot_warning as _print_snapshot_warning
from .interactive import _print_status as _print_status
from .interactive import _print_untrusted as _print_untrusted
from .interactive import _print_welcome as _print_welcome
from .interactive import _prompt_session as _prompt_session
from .interactive import _read_input as _read_input
from .interactive import _render_interactive_answer as _render_interactive_answer
from .interactive import _resume_session_hint as _resume_session_hint
from .interactive import _safe_cli_error as _safe_cli_error
from .interactive import _save_session as _save_session
from .interactive import _session_startup_notice as _session_startup_notice
from .interactive import _StreamingAnswerDisplay as _StreamingAnswerDisplay
from .interactive import _terminal_background_block as _terminal_background_block
from .interactive import _terminal_columns as _terminal_columns
from .interactive import _try_capture_snapshot as _try_capture_snapshot
from .interactive import _TurnLifecycleState as _TurnLifecycleState
from .interactive import _welcome_lines as _welcome_lines
from .interactive import _working_status_text as _working_status_text
from .interactive import _WorkingIndicator as _WorkingIndicator
from .interactive import _write_stream as _write_stream
from .interactive import expand_local_mentions as expand_local_mentions
from .interactive import interactive as interactive
from .main import _mask_secret as _mask_secret
from .main import _silence_broken_pipe as _silence_broken_pipe
from .main import handle_wechat as handle_wechat
from .main import main as main
from .main import run_server as run_server
from .noninteractive import _bounded_noninteractive_text as _bounded_noninteractive_text
from .noninteractive import _emit_noninteractive as _emit_noninteractive
from .noninteractive import _emit_noninteractive_error as _emit_noninteractive_error
from .noninteractive import _noninteractive_counter as _noninteractive_counter
from .noninteractive import _noninteractive_exception_text as _noninteractive_exception_text
from .noninteractive import _normalize_noninteractive_payload as _normalize_noninteractive_payload
from .noninteractive import _read_noninteractive_stdin as _read_noninteractive_stdin
from .noninteractive import _write_json as _write_json
from .noninteractive import noninteractive as noninteractive
from .parser import build_parser

__all__ = [
    "build_parser",
    "handle_wechat",
    "interactive",
    "main",
    "make_agent",
    "noninteractive",
    "run_server",
]

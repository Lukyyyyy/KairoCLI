from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .paths import KairoPaths, reject_symlink_components
from .text_safety import safe_text

MAX_CONFIG_BYTES = 1024 * 1024
MAX_DOTENV_BYTES = 1024 * 1024
MAX_PROVIDER_API_KEY_CHARS = 16_384
MAX_PROVIDER_MODEL_CHARS = 1_024
MAX_PROVIDER_LORA_ID_CHARS = 1_024
MAX_CONFIG_JSON_DEPTH = 16
MAX_CONFIG_JSON_NODES = 10_000
_CONFIG_THREAD_LOCK = threading.Lock()

PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "glm": ("https://open.bigmodel.cn/api/coding/paas/v4", "glm-5.1"),
    "deepseek": ("https://api.deepseek.com", "deepseek-v4-flash"),
    "step": ("https://api.stepfun.com/v1", "step-3.5-flash"),
    "kimi": ("https://api.moonshot.ai/v1", "kimi-k2.6"),
    "freellmapi": ("http://localhost:5173/v1", "auto"),
    "xfyun": ("https://maas-api.cn-huabei-1.xf-yun.com/v2", "Qwen3.6-35B-A3B"),
    "agnes": ("https://apihub.agnes-ai.com/v1", "agnes-2.0-flash"),
}
PROVIDER_ALIASES = {
    "stepfun": "step",
    "step-fun": "step",
    "moonshot": "kimi",
    "moonshotai": "kimi",
    "moonshot-ai": "kimi",
    "free-llm-api": "freellmapi",
    "free_llm_api": "freellmapi",
    "freellm": "freellmapi",
    "free-llm": "freellmapi",
    "xfyun-maas": "xfyun",
    "xfyun_maas": "xfyun",
    "iflytek": "xfyun",
    "iflytek-maas": "xfyun",
    "iflytek_maas": "xfyun",
    "maas": "xfyun",
    "agnes-ai": "agnes",
    "agnes_ai": "agnes",
    "sapiens": "agnes",
    "sapiens-ai": "agnes",
    "sapiens_ai": "agnes",
}


def normalize_provider_name(value: str) -> str:
    normalized = value.strip().casefold()
    return PROVIDER_ALIASES.get(normalized, normalized)


def handle_model_command(payload: str | None, config: AppConfig, paths: KairoPaths) -> str:
    requested = normalize_provider_name(payload or "")
    if not requested:
        return f"Current provider: {config.default_provider}; available: " + ", ".join(
            config.providers
        )
    if requested not in config.providers:
        return f"Unsupported provider: {requested}"
    previous = config.default_provider
    previous_dirty = set(config._dirty_values)
    config.default_provider = requested
    config._dirty_values.add("default_provider")
    try:
        config.save(paths)
    except (OSError, ValueError) as exc:
        config.default_provider = previous
        config._dirty_values = previous_dirty
        return "Provider was not saved: " + safe_text(exc)
    return "Provider saved. Restart the session to rebuild the active model client."


def handle_config_command(payload: str | None, config: AppConfig, paths: KairoPaths) -> str:
    parts = (payload or "").split()
    if not parts:
        lines = [
            f"default_provider={config.default_provider}",
            f"renderer={config.renderer}",
            f"task_workers={config.task_workers}",
        ]
        lines.extend(
            f"{name}: model={provider.model} base_url={provider.base_url} "
            f"key={'configured' if provider.api_key else 'missing'} "
            f"context_window={provider.context_window or 'default'}"
            for name, provider in config.providers.items()
        )
        return "\n".join(lines)
    rollback: list[tuple[Any, str, Any]] = []
    if len(parts) == 2 and parts[0].casefold() == "provider":
        name = normalize_provider_name(parts[1])
        if name not in config.providers:
            return f"Unsupported provider: {name}"
        rollback.append((config, "default_provider", config.default_provider))
        rollback.append((config, "_dirty_values", set(config._dirty_values)))
        config.default_provider = name
        config._dirty_values.add("default_provider")
    elif len(parts) >= 4 and parts[0].casefold() == "provider":
        name = normalize_provider_name(parts[1])
        if name not in config.providers:
            return f"Unsupported provider: {name}"
        provider = config.providers[name]
        field = parts[2].casefold().replace("-", "_")
        allowed = {
            "api_key",
            "base_url",
            "model",
            "lora_id",
            "context_window",
            "temperature",
            "max_tokens",
        }
        if field not in allowed:
            return (
                "Config field must be api-key, base-url, model, lora-id, "
                "context-window, temperature, or max-tokens."
            )
        value = " ".join(parts[3:])
        rollback.append((provider, field, getattr(provider, field)))
        rollback.append((config, "_dirty_values", set(config._dirty_values)))
        if field == "context_window":
            if not value.isdigit() or (int(value) != 0 and int(value) < 8_000):
                return "context-window must be 0 or an integer of at least 8000."
            provider.context_window = int(value)
        elif field == "temperature":
            try:
                provider.temperature = _number(value, f"{name}.temperature", 0, 2)
            except ValueError as exc:
                return safe_text(exc)
        elif field == "max_tokens":
            try:
                provider.max_tokens = _integer(value, f"{name}.max_tokens", 1, 1_000_000)
            except ValueError as exc:
                return safe_text(exc)
        else:
            if field == "base_url":
                try:
                    value = normalize_provider_base_url(value)
                except ValueError as exc:
                    return safe_text(exc)
            if field == "model" and not value.strip():
                return "model cannot be empty."
            setattr(provider, field, value)
            if field == "api_key":
                rollback.extend(
                    [
                        (provider, "_persisted_api_key", provider._persisted_api_key),
                        (provider, "_loaded_api_key", provider._loaded_api_key),
                    ]
                )
                provider._persisted_api_key = value
                provider._loaded_api_key = value
        config._dirty_values.add(f"providers.{name}.{field}")
    else:
        return (
            "Usage: /config provider NAME "
            "[api-key|base-url|model|lora-id|context-window|temperature|max-tokens VALUE]"
        )
    try:
        config.save(paths)
    except (OSError, ValueError) as exc:
        for target, field, old_value in reversed(rollback):
            setattr(target, field, old_value)
        return "Configuration was not saved: " + safe_text(exc)
    return "Configuration saved; provider changes apply fully after restart."


@dataclass(slots=True)
class ProviderConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    lora_id: str = ""
    temperature: float = 0.7
    max_tokens: int = 8192
    context_window: int = 0
    _loaded_api_key: str = field(default="", repr=False)
    _persisted_api_key: str = field(default="", repr=False)


@dataclass(slots=True)
class AppConfig:
    default_provider: str = "glm"
    renderer: str = "inline"
    task_workers: int = 2
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    _loaded_values: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _persisted_values: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _dirty_values: set[str] = field(default_factory=set, repr=False, compare=False)

    @classmethod
    def load(cls, paths: KairoPaths, overrides: dict[str, Any] | None = None) -> AppConfig:
        raw: dict[str, Any] = {}
        _check_config_paths(paths)
        if paths.config_file.is_file():
            try:
                config_bytes = _read_bounded_file(
                    paths.config_file,
                    MAX_CONFIG_BYTES,
                    "Kairo CLI config file exceeds the 1 MiB limit",
                )
            except OSError as exc:
                raise ValueError(f"Cannot read Kairo CLI config: {type(exc).__name__}") from exc
            try:
                parsed = json.loads(
                    config_bytes.decode("utf-8"),
                    object_pairs_hook=_config_object_without_duplicates,
                    parse_constant=_reject_config_json_constant,
                )
                _validate_config_json_shape(parsed)
            except (
                OSError,
                OverflowError,
                RecursionError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError(f"Cannot read Kairo CLI config: {type(exc).__name__}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("Kairo CLI config root must be a JSON object")
            raw = parsed
            if os.name == "posix":
                paths.config_file.chmod(0o600)
        dotenv = _read_dotenv(paths.user_dotenv) | _read_dotenv(paths.workspace / ".env")
        env = dotenv | dict(os.environ)
        providers: dict[str, ProviderConfig] = {}
        persisted_values: dict[str, Any] = {}
        raw_providers = raw.get("providers", {})
        if not isinstance(raw_providers, dict):
            raise ValueError("Kairo CLI config providers must be a JSON object")
        for name, (default_url, default_model) in PROVIDER_DEFAULTS.items():
            source = raw_providers.get(name, {})
            if not isinstance(source, dict):
                raise ValueError(f"Provider config must be an object: {name}")
            prefix = _provider_prefix(name)
            persisted_api_key = _text(source.get("api_key", ""))
            persisted_base_url = normalize_provider_base_url(
                _text(source.get("base_url", default_url)), name
            )
            persisted_model = _text(source.get("model", default_model))
            persisted_lora_id = _text(source.get("lora_id", ""))
            persisted_context_window = _integer(
                source.get("context_window", 0),
                f"{name}.context_window",
                0,
                10_000_000,
            )
            _validate_provider_text_values(
                name, persisted_api_key, persisted_model, persisted_lora_id
            )
            if persisted_context_window and persisted_context_window < 8_000:
                raise ValueError(f"Provider context window must be 0 or at least 8000: {name}")
            effective_api_key = _text(env.get(f"{prefix}_API_KEY", persisted_api_key))
            provider = ProviderConfig(
                api_key=effective_api_key,
                base_url=_text(env.get(f"{prefix}_BASE_URL", persisted_base_url)),
                model=_text(env.get(f"{prefix}_MODEL", persisted_model)),
                lora_id=_text(env.get(f"{prefix}_LORA_ID", persisted_lora_id)),
                temperature=_number(source.get("temperature", 0.7), f"{name}.temperature", 0, 2),
                max_tokens=_integer(
                    source.get("max_tokens", 8192), f"{name}.max_tokens", 1, 1_000_000
                ),
                context_window=_integer(
                    env.get(f"{prefix}_CONTEXT_WINDOW", persisted_context_window),
                    f"{name}.context_window",
                    0,
                    10_000_000,
                ),
                _loaded_api_key=effective_api_key,
                _persisted_api_key=persisted_api_key,
            )
            provider.base_url = normalize_provider_base_url(provider.base_url, name)
            validate_provider_protocol_fields(provider, name)
            if provider.context_window and provider.context_window < 8_000:
                raise ValueError(f"Provider context window must be 0 or at least 8000: {name}")
            providers[name] = provider
            persisted_values.update(
                {
                    f"providers.{name}.base_url": persisted_base_url,
                    f"providers.{name}.model": persisted_model,
                    f"providers.{name}.lora_id": persisted_lora_id,
                    f"providers.{name}.temperature": provider.temperature,
                    f"providers.{name}.max_tokens": provider.max_tokens,
                    f"providers.{name}.context_window": persisted_context_window,
                }
            )
        persisted_default_provider = normalize_provider_name(
            _text(raw.get("default_provider", "glm"))
        )
        if persisted_default_provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"Unsupported default provider: {persisted_default_provider}")
        default_provider = normalize_provider_name(
            _text(env.get("KAIROCLI_PROVIDER", persisted_default_provider))
        )
        if default_provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"Unsupported default provider: {default_provider}")
        persisted_renderer = _text(raw.get("renderer", "inline"))
        if persisted_renderer not in {"inline", "plain", "tui"}:
            raise ValueError("KAIROCLI_RENDERER must be inline, plain, or tui")
        renderer = _text(env.get("KAIROCLI_RENDERER", persisted_renderer))
        if renderer not in {"inline", "plain", "tui"}:
            raise ValueError("KAIROCLI_RENDERER must be inline, plain, or tui")
        persisted_task_workers = _integer(raw.get("task_workers", 2), "task_workers", 1, 32)
        config = cls(
            default_provider=default_provider,
            renderer=renderer,
            task_workers=_integer(
                env.get("KAIROCLI_TASK_WORKERS", persisted_task_workers),
                "task_workers",
                1,
                32,
            ),
            providers=providers,
        )
        if env.get("KAIROCLI_TUI", "").lower() in {"1", "true", "yes", "on"}:
            config.renderer = "tui"
        for key, value in (overrides or {}).items():
            if value is not None:
                setattr(
                    config,
                    key,
                    normalize_provider_name(str(value)) if key == "default_provider" else value,
                )
        persisted_values.update(
            {
                "default_provider": persisted_default_provider,
                "renderer": persisted_renderer,
                "task_workers": persisted_task_workers,
            }
        )
        config._persisted_values = persisted_values
        config._loaded_values = _config_runtime_values(config)
        return config

    def save(self, paths: KairoPaths) -> None:
        _check_config_paths(paths)
        _validate_config_for_save(self)
        paths.config_file.parent.mkdir(parents=True, exist_ok=True)
        _check_config_paths(paths)
        if os.name == "posix":
            paths.user_dir.chmod(0o700)
            paths.config_file.parent.chmod(0o700)
        desired = _config_payload(self)
        with _config_file_lock(paths):
            latest = _config_payload(AppConfig.load(paths))
            payload = _merge_config_payload(self, desired, latest)
            _publish_config(paths, payload)
            raw_providers = payload["providers"]
            for name, provider in self.providers.items():
                provider._persisted_api_key = str(raw_providers[name]["api_key"])
                provider._loaded_api_key = provider.api_key
            self._persisted_values = _config_payload_values(payload)
            self._loaded_values = _config_runtime_values(self)
            self._dirty_values.clear()


def _config_payload(config: AppConfig) -> dict[str, Any]:
    return {
        "default_provider": _persisted_or_changed(
            config, "default_provider", config.default_provider
        ),
        "renderer": _persisted_or_changed(config, "renderer", config.renderer),
        "task_workers": _persisted_or_changed(config, "task_workers", config.task_workers),
        "providers": {
            name: {
                "api_key": (
                    provider.api_key
                    if provider.api_key != provider._loaded_api_key
                    else provider._persisted_api_key
                ),
                "base_url": _persisted_or_changed(
                    config, f"providers.{name}.base_url", provider.base_url
                ),
                "model": _persisted_or_changed(config, f"providers.{name}.model", provider.model),
                "lora_id": _persisted_or_changed(
                    config, f"providers.{name}.lora_id", provider.lora_id
                ),
                "temperature": _persisted_or_changed(
                    config, f"providers.{name}.temperature", provider.temperature
                ),
                "max_tokens": _persisted_or_changed(
                    config, f"providers.{name}.max_tokens", provider.max_tokens
                ),
                "context_window": _persisted_or_changed(
                    config,
                    f"providers.{name}.context_window",
                    provider.context_window,
                ),
            }
            for name, provider in config.providers.items()
        },
    }


def _merge_config_payload(
    config: AppConfig,
    desired: dict[str, Any],
    latest: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        key: latest[key] for key in ("default_provider", "renderer", "task_workers")
    }
    for key in ("default_provider", "renderer", "task_workers"):
        if _config_value_changed(config, key, getattr(config, key)):
            merged[key] = desired[key]
    desired_providers = desired["providers"]
    latest_providers = latest["providers"]
    merged_providers: dict[str, dict[str, Any]] = {}
    for name, provider in config.providers.items():
        desired_provider = desired_providers[name]
        merged_provider = dict(latest_providers[name])
        for field_name in (
            "api_key",
            "base_url",
            "model",
            "lora_id",
            "temperature",
            "max_tokens",
            "context_window",
        ):
            key = f"providers.{name}.{field_name}"
            changed = (
                key in config._dirty_values
                or (field_name == "api_key" and provider.api_key != provider._loaded_api_key)
                or (
                    field_name != "api_key"
                    and _config_value_changed(config, key, getattr(provider, field_name))
                )
            )
            if changed:
                merged_provider[field_name] = desired_provider[field_name]
        merged_providers[name] = merged_provider
    merged["providers"] = merged_providers
    return merged


def _config_value_changed(config: AppConfig, key: str, current: Any) -> bool:
    return (
        key in config._dirty_values
        or key not in config._loaded_values
        or current != config._loaded_values[key]
    )


def _publish_config(paths: KairoPaths, payload: dict[str, Any]) -> None:
    serialized = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    if len(serialized.encode()) > MAX_CONFIG_BYTES:
        raise ValueError("Kairo CLI config exceeds the 1 MiB limit")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".config-", suffix=".tmp", dir=paths.config_file.parent
    )
    temporary = Path(temporary_name)
    try:
        if os.name == "posix":
            os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, paths.config_file)
        if os.name == "posix":
            paths.config_file.chmod(0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _config_file_lock(paths: KairoPaths) -> Iterator[None]:
    directory = paths.config_file.parent
    reject_symlink_components(directory, "Kairo CLI config directory")
    directory.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(directory, "Kairo CLI config directory")
    lock_file = directory / ".config.lock"
    reject_symlink_components(lock_file, "Kairo CLI config lock")
    descriptor = os.open(
        lock_file,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    locked = False
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with _CONFIG_THREAD_LOCK:
            if os.name == "posix":
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
            elif os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
                locked = True
            yield
    finally:
        if locked and os.name == "posix":
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        elif locked and os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        os.close(descriptor)


def _provider_prefix(name: str) -> str:
    return {"xfyun": "XFYUN_MAAS", "freellmapi": "FREELLMAPI"}.get(name, name.upper())


def _check_config_paths(paths: KairoPaths) -> None:
    if paths.user_dir.is_symlink() or paths.config_file.is_symlink():
        raise ValueError("Kairo CLI config cannot use a symbolic link")


def _config_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate Kairo CLI config key: {key}")
        result[key] = value
    return result


def _reject_config_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _validate_config_json_shape(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_CONFIG_JSON_NODES:
            raise ValueError("Kairo CLI config JSON exceeds the node limit")
        if depth > MAX_CONFIG_JSON_DEPTH:
            raise ValueError("Kairo CLI config JSON exceeds the nesting limit")
        children: Iterable[Any]
        if isinstance(current, dict):
            child_count = len(current)
            children = current.values()
        elif isinstance(current, list):
            child_count = len(current)
            children = current
        else:
            continue
        if visited + len(stack) + child_count > MAX_CONFIG_JSON_NODES:
            raise ValueError("Kairo CLI config JSON exceeds the node limit")
        stack.extend((child, depth + 1) for child in children)


def _persisted_or_changed(config: AppConfig, key: str, current: Any) -> Any:
    if key in config._dirty_values or key not in config._loaded_values:
        return current
    if current != config._loaded_values[key]:
        return current
    return config._persisted_values.get(key, current)


def _config_runtime_values(config: AppConfig) -> dict[str, Any]:
    values: dict[str, Any] = {
        "default_provider": config.default_provider,
        "renderer": config.renderer,
        "task_workers": config.task_workers,
    }
    for name, provider in config.providers.items():
        values.update(
            {
                f"providers.{name}.base_url": provider.base_url,
                f"providers.{name}.model": provider.model,
                f"providers.{name}.lora_id": provider.lora_id,
                f"providers.{name}.temperature": provider.temperature,
                f"providers.{name}.max_tokens": provider.max_tokens,
                f"providers.{name}.context_window": provider.context_window,
            }
        )
    return values


def _config_payload_values(payload: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {
        "default_provider": payload["default_provider"],
        "renderer": payload["renderer"],
        "task_workers": payload["task_workers"],
    }
    raw_providers = payload["providers"]
    if isinstance(raw_providers, dict):
        for name, raw in raw_providers.items():
            if not isinstance(raw, dict):
                continue
            for field_name in (
                "base_url",
                "model",
                "lora_id",
                "temperature",
                "max_tokens",
                "context_window",
            ):
                values[f"providers.{name}.{field_name}"] = raw[field_name]
    return values


def normalize_provider_base_url(value: str, provider: str = "provider") -> str:
    if any(character.isspace() for character in value) or "\\" in value:
        raise ValueError(f"Provider base URL contains invalid whitespace: {provider}")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Provider base URL is invalid: {provider}") from exc
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError(f"Provider base URL must use HTTP(S): {provider}")
    if not parsed.hostname:
        raise ValueError(f"Provider base URL requires a host: {provider}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"Provider base URL cannot contain credentials: {provider}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"Provider base URL cannot contain query or fragment: {provider}")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    netloc = host
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme.casefold(), netloc, parsed.path.rstrip("/"), "", ""))


def _read_dotenv(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        content = _read_bounded_file(path, MAX_DOTENV_BYTES, "dotenv exceeds size limit")
        lines = content.decode("utf-8", errors="replace").splitlines()
    except (OSError, ValueError):
        return {}
    result: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip("\"'")
    return result


def _read_bounded_file(path: Path, max_bytes: int, oversized_message: str) -> bytes:
    if path.stat().st_size > max_bytes:
        raise ValueError(oversized_message)
    with path.open("rb") as source:
        content = source.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(oversized_message)
    return content


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Kairo CLI config text values must be strings")
    return value.strip()


def _integer(value: Any, field_name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Kairo CLI config {field_name} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        parsed = int(value)
    else:
        raise ValueError(f"Kairo CLI config {field_name} must be an integer")
    if not minimum <= parsed <= maximum:
        raise ValueError(f"Kairo CLI config {field_name} must be between {minimum} and {maximum}")
    return parsed


def _number(value: Any, field_name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Kairo CLI config {field_name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Kairo CLI config {field_name} must be numeric") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(
            f"Kairo CLI config {field_name} must be between {minimum:g} and {maximum:g}"
        )
    return parsed


def _validate_config_for_save(config: AppConfig) -> None:
    if config.default_provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unsupported default provider: {config.default_provider}")
    if config.renderer not in {"inline", "plain", "tui"}:
        raise ValueError("KAIROCLI_RENDERER must be inline, plain, or tui")
    _integer(config.task_workers, "task_workers", 1, 32)
    for name, provider in config.providers.items():
        if name not in PROVIDER_DEFAULTS:
            raise ValueError(f"Unsupported provider config: {name}")
        for value in (
            provider.api_key,
            provider.base_url,
            provider.model,
            provider.lora_id,
            provider._loaded_api_key,
            provider._persisted_api_key,
        ):
            _text(value)
        normalize_provider_base_url(provider.base_url, name)
        validate_provider_protocol_fields(provider, name)
        _number(provider.temperature, f"{name}.temperature", 0, 2)
        _integer(provider.max_tokens, f"{name}.max_tokens", 1, 1_000_000)
        context_window = _integer(provider.context_window, f"{name}.context_window", 0, 10_000_000)
        if context_window and context_window < 8_000:
            raise ValueError(f"Provider context window must be 0 or at least 8000: {name}")


def validate_provider_protocol_fields(provider: ProviderConfig, name: str = "provider") -> None:
    _validate_provider_text_values(name, provider.api_key, provider.model, provider.lora_id)


def _validate_provider_text_values(name: str, api_key: str, model: str, lora_id: str) -> None:
    _validate_header_value(api_key, f"{name}.api_key", MAX_PROVIDER_API_KEY_CHARS, allow_empty=True)
    _validate_header_value(lora_id, f"{name}.lora_id", MAX_PROVIDER_LORA_ID_CHARS, allow_empty=True)
    if (
        not isinstance(model, str)
        or not model.strip()
        or len(model) > MAX_PROVIDER_MODEL_CHARS
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in model)
    ):
        raise ValueError(
            f"Provider model must be 1 to {MAX_PROVIDER_MODEL_CHARS} characters "
            f"without control characters: {name}"
        )


def _validate_header_value(
    value: str, field_name: str, max_chars: int, *, allow_empty: bool
) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"Provider {field_name} must be a string")
    if len(value) > max_chars or any(not 0x21 <= ord(character) <= 0x7E for character in value):
        raise ValueError(
            f"Provider {field_name} must contain at most {max_chars} visible ASCII characters"
        )

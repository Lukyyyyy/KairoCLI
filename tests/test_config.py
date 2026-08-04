import io
import json
import os
from pathlib import Path

import pytest

import kairocli.config as config_module
from kairocli.cli import _handle_config
from kairocli.config import (
    MAX_CONFIG_BYTES,
    MAX_DOTENV_BYTES,
    AppConfig,
    handle_config_command,
    handle_model_command,
    normalize_provider_name,
)
from kairocli.paths import KairoPaths


class RecordingConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str) -> None:
        self.messages.append(message)


def test_config_precedence(monkeypatch: object, tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.config_file.parent.mkdir(parents=True)
    paths.config_file.write_text(
        json.dumps({"default_provider": "glm", "providers": {"glm": {"model": "file-model"}}}),
        encoding="utf-8",
    )
    (paths.workspace).mkdir(parents=True)
    (paths.workspace / ".env").write_text("GLM_MODEL=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv("KAIROCLI_PROVIDER", "deepseek")  # type: ignore[attr-defined]
    monkeypatch.setenv("GLM_MODEL", "env-model")  # type: ignore[attr-defined]
    config = AppConfig.load(paths)
    assert config.default_provider == "deepseek"
    assert config.providers["glm"].model == "env-model"


def test_config_save_roundtrip(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    config = AppConfig.load(paths)
    config.default_provider = "agnes"
    config.providers["agnes"].context_window = 777_000
    config.save(paths)
    loaded = AppConfig.load(paths)
    assert loaded.default_provider == "agnes"
    assert loaded.providers["agnes"].context_window == 777_000


def test_config_concurrent_saves_merge_distinct_changed_fields(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    seed = AppConfig.load(paths)
    seed.save(paths)
    first = AppConfig.load(paths)
    stale = AppConfig.load(paths)

    first.default_provider = "agnes"
    first.save(paths)
    stale.providers["glm"].temperature = 1.25
    stale.providers["glm"].max_tokens = 16_000
    stale.save(paths)

    stored = json.loads(paths.config_file.read_text(encoding="utf-8"))
    assert stored["default_provider"] == "agnes"
    assert stored["providers"]["glm"]["temperature"] == 1.25
    assert stored["providers"]["glm"]["max_tokens"] == 16_000


def test_stale_config_with_environment_key_preserves_newer_stored_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    seed = AppConfig.load(paths)
    seed.providers["glm"].api_key = "original-stored"
    seed.save(paths)
    monkeypatch.setenv("GLM_API_KEY", "environment-only")
    first = AppConfig.load(paths)
    stale = AppConfig.load(paths)

    first.providers["glm"].api_key = "newer-stored"
    first.save(paths)
    stale.providers["glm"].model = "stale-instance-model-change"
    stale.save(paths)

    stored = json.loads(paths.config_file.read_text(encoding="utf-8"))
    assert stored["providers"]["glm"]["api_key"] == "newer-stored"
    assert stored["providers"]["glm"]["model"] == "stale-instance-model-change"


def test_provider_aliases_and_reference_defaults() -> None:
    assert normalize_provider_name("StepFun") == "step"
    assert normalize_provider_name("iflytek-maas") == "xfyun"
    assert normalize_provider_name("sapiens-ai") == "agnes"


def test_config_storage_is_atomic_private_and_hardens_existing_mode(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    config = AppConfig.load(paths)
    config.providers["glm"].api_key = "private-key"
    config.save(paths)

    assert not list(paths.config_file.parent.glob(".config-*.tmp"))
    if os.name == "posix":
        assert (paths.user_dir.stat().st_mode & 0o777) == 0o700
        assert (paths.config_file.stat().st_mode & 0o777) == 0o600
        paths.config_file.chmod(0o644)
        AppConfig.load(paths)
        assert (paths.config_file.stat().st_mode & 0o777) == 0o600


def test_config_rejects_symlinks_malformed_shapes_and_oversized_files(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.config_file.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        paths.config_file.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="symbolic link"):
        AppConfig.load(paths)
    paths.config_file.unlink()

    paths.config_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        AppConfig.load(paths)
    paths.config_file.write_text('{"providers": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="providers"):
        AppConfig.load(paths)
    paths.config_file.write_bytes(b"{" + b"x" * MAX_CONFIG_BYTES)
    with pytest.raises(ValueError, match="1 MiB"):
        AppConfig.load(paths)


@pytest.mark.parametrize(
    "content",
    [
        '{"default_provider":"glm","default_provider":"agnes"}',
        '{"unused":NaN}',
        '{"unused":' + "[" * 32 + "0" + "]" * 32 + "}",
    ],
)
def test_config_rejects_duplicate_nonstandard_and_overdeep_json(
    tmp_path: Path, content: str
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.config_file.parent.mkdir(parents=True)
    paths.config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot read Kairo CLI config"):
        AppConfig.load(paths)


def test_config_save_rejects_symlink_lock(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    config = AppConfig.load(paths)
    paths.config_file.parent.mkdir(parents=True)
    sentinel = tmp_path / "outside-lock"
    sentinel.write_text("unchanged", encoding="utf-8")
    (paths.config_file.parent / ".config.lock").symlink_to(sentinel)

    with pytest.raises(ValueError, match="symlink"):
        config.save(paths)

    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_config_bounded_read_detects_growth_after_metadata_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.config_file.parent.mkdir(parents=True)
    paths.config_file.write_text("{}", encoding="utf-8")
    requested: list[int] = []

    class GrowingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    def growing_open(
        _path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> GrowingReader:
        assert mode == "rb"
        return GrowingReader(b"x" * (MAX_CONFIG_BYTES + 1))

    monkeypatch.setattr(Path, "open", growing_open)
    with pytest.raises(ValueError, match="1 MiB"):
        AppConfig.load(paths)

    assert requested == [MAX_CONFIG_BYTES + 1]


def test_dotenv_bounded_read_ignores_growth_after_metadata_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("x", encoding="utf-8")
    requested: list[int] = []

    class GrowingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    def growing_open(
        _path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> GrowingReader:
        assert mode == "rb"
        return GrowingReader(b"x" * (MAX_DOTENV_BYTES + 1))

    monkeypatch.setattr(Path, "open", growing_open)

    assert config_module._read_dotenv(dotenv) == {}
    assert requested == [MAX_DOTENV_BYTES + 1]


def test_config_rejects_symlinked_user_directory_without_external_writes(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    config = AppConfig.load(paths)
    paths.home.mkdir(parents=True)
    outside = tmp_path / "outside-config"
    outside.mkdir()
    sentinel = outside / "config.json"
    sentinel.write_text('{"default_provider":"agnes"}', encoding="utf-8")
    paths.user_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        AppConfig.load(paths)
    config.default_provider = "kimi"
    with pytest.raises(ValueError, match="symbolic link"):
        config.save(paths)

    assert sentinel.read_text(encoding="utf-8") == '{"default_provider":"agnes"}'
    assert list(outside.iterdir()) == [sentinel]


def test_config_ignores_symlinked_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.workspace.mkdir(parents=True)
    outside = tmp_path / "outside.env"
    outside.write_text("GLM_API_KEY=external-secret\n", encoding="utf-8")
    (paths.workspace / ".env").symlink_to(outside)
    monkeypatch.delenv("GLM_API_KEY", raising=False)

    config = AppConfig.load(paths)

    assert config.providers["glm"].api_key == ""


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"task_workers": 0}, "task_workers"),
        ({"renderer": "unknown"}, "RENDERER"),
        ({"default_provider": "unknown"}, "Unsupported"),
        ({"providers": {"glm": {"temperature": 3}}}, "temperature"),
        ({"providers": {"glm": {"context_window": 100}}}, "context window"),
        ({"providers": {"glm": {"base_url": "file:///tmp/model"}}}, "HTTP"),
        (
            {"providers": {"glm": {"base_url": "https://user:pass@model.test/v1"}}},
            "credentials",
        ),
        (
            {"providers": {"glm": {"base_url": "https://model.test/v1?token=x"}}},
            "query or fragment",
        ),
        ({"providers": {"glm": {"base_url": "https:///missing"}}}, "host"),
        ({"providers": {"glm": {"base_url": "https://model .test/v1"}}}, "whitespace"),
        ({"providers": {"glm": {"api_key": "bad key"}}}, "visible ASCII"),
        ({"providers": {"glm": {"api_key": "密钥"}}}, "visible ASCII"),
        ({"providers": {"glm": {"api_key": "x" * 16_385}}}, "visible ASCII"),
        ({"providers": {"glm": {"lora_id": "bad value"}}}, "visible ASCII"),
        ({"providers": {"glm": {"model": "good\nbad"}}}, "control characters"),
        ({"providers": {"glm": {"model": "m" * 1_025}}}, "1024"),
    ],
)
def test_config_validates_untrusted_values(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.config_file.parent.mkdir(parents=True)
    paths.config_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        AppConfig.load(paths)


def test_failed_atomic_replace_preserves_previous_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    config = AppConfig.load(paths)
    config.save(paths)
    original = paths.config_file.read_bytes()
    config.default_provider = "agnes"

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("kairocli.config.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        config.save(paths)
    assert paths.config_file.read_bytes() == original
    assert not list(paths.config_file.parent.glob(".config-*.tmp"))


def test_workspace_dotenv_precedes_user_dotenv_and_environment_secrets_are_not_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.workspace.mkdir(parents=True)
    paths.user_dir.mkdir(parents=True)
    paths.user_dotenv.write_text(
        "GLM_MODEL=user-model\nGLM_API_KEY=user-secret\n", encoding="utf-8"
    )
    (paths.workspace / ".env").write_text(
        "export GLM_MODEL=workspace-model\nGLM_API_KEY=workspace-secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GLM_MODEL", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)

    config = AppConfig.load(paths)
    assert config.providers["glm"].model == "workspace-model"
    assert config.providers["glm"].api_key == "workspace-secret"
    config.default_provider = "agnes"
    config.save(paths)
    stored = json.loads(paths.config_file.read_text(encoding="utf-8"))
    assert stored["providers"]["glm"]["api_key"] == ""


def test_environment_override_does_not_replace_an_existing_stored_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.config_file.parent.mkdir(parents=True)
    paths.config_file.write_text(
        json.dumps({"providers": {"glm": {"api_key": "stored-key"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GLM_API_KEY", "environment-key")
    config = AppConfig.load(paths)
    assert config.providers["glm"].api_key == "environment-key"
    config.save(paths)
    stored = json.loads(paths.config_file.read_text(encoding="utf-8"))
    assert stored["providers"]["glm"]["api_key"] == "stored-key"


def test_all_environment_overrides_remain_ephemeral_until_explicitly_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    monkeypatch.setenv("KAIROCLI_PROVIDER", "agnes")
    monkeypatch.setenv("KAIROCLI_RENDERER", "plain")
    monkeypatch.setenv("KAIROCLI_TASK_WORKERS", "7")
    monkeypatch.setenv("GLM_BASE_URL", "https://runtime.example/v1/")
    monkeypatch.setenv("GLM_MODEL", "runtime-model")
    monkeypatch.setenv("GLM_LORA_ID", "runtime-lora")
    monkeypatch.setenv("GLM_CONTEXT_WINDOW", "16000")
    config = AppConfig.load(paths)

    assert config.default_provider == "agnes"
    assert config.renderer == "plain"
    assert config.task_workers == 7
    assert config.providers["glm"].model == "runtime-model"
    assert (
        "saved" in handle_config_command("provider glm temperature 1.1", config, paths).casefold()
    )

    stored = json.loads(paths.config_file.read_text(encoding="utf-8"))
    assert stored["default_provider"] == "glm"
    assert stored["renderer"] == "inline"
    assert stored["task_workers"] == 2
    assert stored["providers"]["glm"]["base_url"] == ("https://open.bigmodel.cn/api/coding/paas/v4")
    assert stored["providers"]["glm"]["model"] == "glm-5.1"
    assert stored["providers"]["glm"]["lora_id"] == ""
    assert stored["providers"]["glm"]["context_window"] == 0
    assert stored["providers"]["glm"]["temperature"] == 1.1

    assert (
        "saved"
        in handle_config_command("provider glm model runtime-model", config, paths).casefold()
    )
    assert "saved" in handle_model_command("agnes", config, paths).casefold()
    stored = json.loads(paths.config_file.read_text(encoding="utf-8"))
    assert stored["providers"]["glm"]["model"] == "runtime-model"
    assert stored["default_provider"] == "agnes"


def test_config_save_validation_and_cli_failure_restore_in_memory_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    config = AppConfig.load(paths)
    config.save(paths)
    original = paths.config_file.read_bytes()
    config.providers["glm"].base_url = "file:///unsafe"
    with pytest.raises(ValueError, match="HTTP"):
        config.save(paths)
    assert paths.config_file.read_bytes() == original
    config.providers["glm"].base_url = "https://safe.example/v1"

    old_key = config.providers["glm"].api_key
    old_loaded = config.providers["glm"]._loaded_api_key
    old_persisted = config.providers["glm"]._persisted_api_key

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("no replace")

    monkeypatch.setattr("kairocli.config.os.replace", fail_replace)
    console = RecordingConsole()
    _handle_config("provider glm api-key new-secret", config, paths, console)

    assert config.providers["glm"].api_key == old_key
    assert config.providers["glm"]._loaded_api_key == old_loaded
    assert config.providers["glm"]._persisted_api_key == old_persisted
    assert "not saved" in console.messages[-1]


def test_shared_model_and_config_commands_validate_and_mask(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    config = AppConfig.load(paths)
    status = handle_config_command("", config, paths)
    assert "key=missing" in status
    assert "api_key" not in status
    assert "context-window must" in handle_config_command(
        "provider glm context-window 7999", config, paths
    )
    assert (
        "saved" in handle_config_command("provider glm context-window 0", config, paths).casefold()
    )
    assert (
        "saved" in handle_config_command("provider glm temperature 1.25", config, paths).casefold()
    )
    assert config.providers["glm"].temperature == 1.25
    original_url = config.providers["glm"].base_url
    assert "credentials" in handle_config_command(
        "provider glm base-url https://user:secret@model.test/v1", config, paths
    )
    assert config.providers["glm"].base_url == original_url
    original_key = config.providers["glm"].api_key
    assert (
        "not saved"
        in handle_config_command("provider glm api-key bad key", config, paths).casefold()
    )
    assert config.providers["glm"].api_key == original_key
    assert "saved" in handle_model_command("moonshot", config, paths).casefold()
    assert config.default_provider == "kimi"
    assert "Unsupported" in handle_model_command("missing", config, paths)

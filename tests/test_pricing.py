import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from kairocli.paths import KairoPaths
from kairocli.pricing import PricingConfig, ensure_default_pricing_config


def test_pricing_config_bootstrap_is_private_and_idempotent(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")

    assert ensure_default_pricing_config(paths)
    assert not ensure_default_pricing_config(paths)
    assert paths.pricing_file.stat().st_mode & 0o777 == 0o600
    assert paths.user_dir.stat().st_mode & 0o777 == 0o700
    assert not list(paths.user_dir.glob(".pricing-bootstrap.*.tmp"))


def test_user_pricing_config_overrides_rates_on_next_load(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    ensure_default_pricing_config(paths)
    payload = json.loads(paths.pricing_file.read_text(encoding="utf-8"))
    payload["providers"]["deepseek"]["models"][1]["peak"] = {
        "input": 30,
        "cached": 2,
        "output": 90,
    }
    paths.pricing_file.write_text(json.dumps(payload), encoding="utf-8")

    pricing = PricingConfig.load(paths)
    peak = datetime(2026, 8, 17, 10, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert (
        pricing.estimated_cost_cny(
            "deepseek",
            2_000_000,
            1_000_000,
            1_000_000,
            model="deepseek-v4-flash",
            at=peak,
        )
        == 122.0
    )


def test_invalid_user_pricing_falls_back_without_failing_startup(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.user_dir.mkdir(parents=True)
    paths.pricing_file.write_text("{broken", encoding="utf-8")

    pricing = PricingConfig.load(paths)

    assert pricing.warning
    assert pricing.estimated_cost_cny("glm", 1_000_000, 0, 0) == 5.0


def test_unknown_provider_uses_configured_default_rates() -> None:
    pricing = PricingConfig.default()

    assert pricing.estimated_cost_cny("custom", 1_000_000, 1_000_000, 0) == 20.0

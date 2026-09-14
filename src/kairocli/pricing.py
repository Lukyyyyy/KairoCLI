from __future__ import annotations

import json
import os
import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from importlib.resources import files
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .paths import KairoPaths, reject_symlink_components

MAX_PRICING_CONFIG_BYTES = 1024 * 1024
MAX_PRICING_JSON_DEPTH = 16
MAX_PRICING_JSON_NODES = 10_000
MAX_PRICING_PROVIDERS = 256
MAX_PRICING_MODEL_RULES = 1_000
MAX_PRICING_PEAK_RANGES = 100


@dataclass(frozen=True, slots=True)
class TokenRates:
    input: float
    cached: float
    output: float


@dataclass(frozen=True, slots=True)
class ModelPricing:
    contains: str
    peak: TokenRates
    off_peak: TokenRates


@dataclass(frozen=True, slots=True)
class ProviderPricing:
    default: TokenRates
    timezone: ZoneInfo | None = None
    effective_at: datetime | None = None
    peak_hours: tuple[tuple[int, int], ...] = ()
    models: tuple[ModelPricing, ...] = ()

    def rates(self, model: str | None, at: datetime | None) -> TokenRates:
        if self.effective_at is None or self.timezone is None or not self.models:
            return self.default
        current = at or datetime.now(tz=self.timezone)
        if current.tzinfo is None:
            current = current.replace(tzinfo=self.timezone)
        else:
            current = current.astimezone(self.timezone)
        if current < self.effective_at:
            return self.default
        normalized_model = (model or "").casefold()
        rule = next(
            (item for item in self.models if item.contains.casefold() in normalized_model),
            None,
        )
        if rule is None:
            return self.default
        is_peak = any(start <= current.hour < end for start, end in self.peak_hours)
        return rule.peak if is_peak else rule.off_peak


@dataclass(frozen=True, slots=True)
class PricingConfig:
    providers: dict[str, ProviderPricing]
    warning: str = ""

    @classmethod
    def default(cls) -> PricingConfig:
        return cls.from_bytes(_default_pricing_bytes())

    @classmethod
    def load(cls, paths: KairoPaths) -> PricingConfig:
        default = cls.default()
        try:
            ensure_default_pricing_config(paths)
            return cls.from_bytes(_read_pricing_file(paths))
        except (OSError, ValueError) as exc:
            return cls(
                default.providers,
                f"价格配置无效（{type(exc).__name__}），已使用内置默认单价。",
            )

    @classmethod
    def from_bytes(cls, encoded: bytes) -> PricingConfig:
        if len(encoded) > MAX_PRICING_CONFIG_BYTES:
            raise ValueError("Pricing config exceeds the 1 MiB limit")
        try:
            payload = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (OverflowError, RecursionError, UnicodeError, ValueError) as exc:
            raise ValueError(f"Cannot parse pricing config: {type(exc).__name__}") from exc
        _validate_json_shape(payload)
        if not isinstance(payload, dict) or not isinstance(payload.get("providers"), dict):
            raise ValueError("Pricing config must contain a providers object")
        if len(payload["providers"]) > MAX_PRICING_PROVIDERS:
            raise ValueError("Pricing config exceeds the provider limit")
        providers = {
            str(name).casefold(): _provider_pricing(str(name), value)
            for name, value in payload["providers"].items()
        }
        if "default" not in providers:
            raise ValueError("Pricing config must contain the default provider")
        return cls(providers)

    def estimated_cost_cny(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        *,
        model: str | None = None,
        at: datetime | None = None,
    ) -> float:
        pricing = self.providers.get(provider.casefold(), self.providers["default"])
        rates = pricing.rates(model, at)
        cached = min(max(0, cached_tokens), max(0, input_tokens))
        uncached = max(0, input_tokens - cached)
        return (
            uncached / 1_000_000 * rates.input
            + cached / 1_000_000 * rates.cached
            + max(0, output_tokens) / 1_000_000 * rates.output
        )

    def cost_units_and_rates(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        *,
        model: str | None = None,
        at: datetime | None = None,
    ) -> tuple[int, tuple[str, str, str]]:
        """Return exact 1e-8 CNY units plus the rate snapshot used."""

        pricing = self.providers.get(provider.casefold(), self.providers["default"])
        rates = pricing.rates(model, at)
        cached = min(max(0, cached_tokens), max(0, input_tokens))
        uncached = max(0, input_tokens - cached)
        input_rate = Decimal(str(rates.input))
        cached_rate = Decimal(str(rates.cached))
        output_rate = Decimal(str(rates.output))
        # Prices are CNY per million tokens; one stored unit is 1e-8 CNY.
        units = (
            Decimal(uncached) * input_rate * 100
            + Decimal(cached) * cached_rate * 100
            + Decimal(max(0, output_tokens)) * output_rate * 100
        ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return int(units), (str(input_rate), str(cached_rate), str(output_rate))


def ensure_default_pricing_config(paths: KairoPaths) -> bool:
    target = paths.pricing_file
    reject_symlink_components(target, "Pricing config")
    if target.exists():
        if not target.is_file():
            raise ValueError(f"Pricing config is not a regular file: {target}")
        return False
    paths.user_dir.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(target, "Pricing config")
    if os.name != "nt":
        paths.user_dir.chmod(0o700)
    temporary = paths.user_dir / f".pricing-bootstrap.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_default_pricing_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            return False
        if os.name != "nt":
            target.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _default_pricing_bytes() -> bytes:
    return files("kairocli").joinpath("pricing.default.json").read_bytes()


def _read_pricing_file(paths: KairoPaths) -> bytes:
    path = paths.pricing_file
    reject_symlink_components(path, "Pricing config")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        encoded = os.read(descriptor, MAX_PRICING_CONFIG_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(encoded) > MAX_PRICING_CONFIG_BYTES:
        raise ValueError("Pricing config exceeds the 1 MiB limit")
    return encoded


def _provider_pricing(name: str, value: Any) -> ProviderPricing:
    if not name or len(name) > 128:
        raise ValueError("Pricing provider name must contain 1 to 128 characters")
    if not isinstance(value, dict):
        raise ValueError(f"Pricing provider must be an object: {name}")
    default = _rates(value.get("default"), f"{name}.default")
    raw_models = value.get("models", [])
    raw_hours = value.get("peak_hours", [])
    if not isinstance(raw_models, list) or not isinstance(raw_hours, list):
        raise ValueError(f"Pricing models and peak_hours must be arrays: {name}")
    if len(raw_models) > MAX_PRICING_MODEL_RULES:
        raise ValueError(f"Pricing config exceeds the model rule limit: {name}")
    if len(raw_hours) > MAX_PRICING_PEAK_RANGES:
        raise ValueError(f"Pricing config exceeds the peak range limit: {name}")
    timezone: ZoneInfo | None = None
    effective_at: datetime | None = None
    if raw_models:
        timezone_name = value.get("timezone")
        effective_value = value.get("effective_at")
        if not isinstance(timezone_name, str) or not isinstance(effective_value, str):
            raise ValueError(f"Dynamic pricing requires timezone and effective_at: {name}")
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown pricing timezone: {timezone_name}") from exc
        try:
            effective_at = datetime.fromisoformat(effective_value)
        except ValueError as exc:
            raise ValueError(f"Invalid pricing effective_at: {name}") from exc
        if effective_at.tzinfo is None:
            effective_at = effective_at.replace(tzinfo=timezone)
        else:
            effective_at = effective_at.astimezone(timezone)
    peak_hours: list[tuple[int, int]] = []
    for item in raw_hours:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or isinstance(item[0], bool)
            or isinstance(item[1], bool)
            or not isinstance(item[0], int)
            or not isinstance(item[1], int)
            or not 0 <= item[0] < item[1] <= 24
        ):
            raise ValueError(f"Invalid pricing peak hour range: {name}")
        peak_hours.append((item[0], item[1]))
    models: list[ModelPricing] = []
    for index, item in enumerate(raw_models):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("contains"), str)
            or len(item["contains"]) > 1_024
        ):
            raise ValueError(f"Invalid pricing model rule: {name}[{index}]")
        models.append(
            ModelPricing(
                item["contains"],
                _rates(item.get("peak"), f"{name}.models[{index}].peak"),
                _rates(item.get("off_peak"), f"{name}.models[{index}].off_peak"),
            )
        )
    return ProviderPricing(default, timezone, effective_at, tuple(peak_hours), tuple(models))


def _rates(value: Any, label: str) -> TokenRates:
    if not isinstance(value, dict):
        raise ValueError(f"Pricing rates must be an object: {label}")
    numbers: list[float] = []
    for field in ("input", "cached", "output"):
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Pricing rate must be a number: {label}.{field}")
        number = float(raw)
        if not 0 <= number <= 1_000_000:
            raise ValueError(f"Pricing rate is out of range: {label}.{field}")
        numbers.append(number)
    return TokenRates(*numbers)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate pricing config key: {key}")
        result[key] = value
    return result


def _validate_json_shape(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_PRICING_JSON_NODES:
            raise ValueError("Pricing config exceeds the JSON node limit")
        if depth > MAX_PRICING_JSON_DEPTH:
            raise ValueError("Pricing config exceeds the JSON nesting limit")
        children: Iterable[Any]
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, list):
            children = current
        else:
            continue
        stack.extend((child, depth + 1) for child in children)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")

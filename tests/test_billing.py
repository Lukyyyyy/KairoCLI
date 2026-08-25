from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from kairocli.billing import BillingStore, cny_to_units
from kairocli.pricing import PricingConfig
from kairocli.web_auth import WebUserStore


def test_default_quota_and_idempotent_usage_charge(tmp_path: Path) -> None:
    database = tmp_path / "web" / "users.db"
    user = WebUserStore(database).create_user("user", "password-123")
    billing = BillingStore(database)
    pricing = PricingConfig.default()
    at = datetime(2026, 8, 17, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
    cost, rates = pricing.cost_units_and_rates(
        "deepseek",
        1_000_000,
        100_000,
        200_000,
        model="deepseek-v4-flash",
        at=at,
    )

    assert billing.quota(user.id).balance_cny == "1.00"
    first = billing.charge(
        user.id,
        provider="deepseek",
        model="deepseek-v4-flash",
        input_tokens=1_000_000,
        cached_tokens=200_000,
        output_tokens=100_000,
        cost_units=cost,
        rates=rates,
        charged_at=at,
        call_id="call-1",
    )
    second = billing.charge(
        user.id,
        provider="deepseek",
        model="deepseek-v4-flash",
        input_tokens=1_000_000,
        cached_tokens=200_000,
        output_tokens=100_000,
        cost_units=cost,
        rates=rates,
        charged_at=at,
        call_id="call-1",
    )

    assert cost == cny_to_units("3.32")
    assert first.balance_units == -cny_to_units("2.32")
    assert second == first
    assert len(billing.list_usage(user.id)) == 1


def test_monthly_reset_is_opt_in_and_replaces_balance(tmp_path: Path) -> None:
    database = tmp_path / "web" / "users.db"
    user = WebUserStore(database).create_user("user", "password-123")
    billing = BillingStore(database)
    billing.set_quota(user.id, cny_to_units("0.25"), cny_to_units("2.00"))

    assert billing.apply_monthly_reset(datetime(2026, 9, 1)) is False
    billing.set_monthly_reset_enabled(True)
    assert billing.apply_monthly_reset(datetime(2026, 9, 1)) is True
    assert billing.quota(user.id).balance_cny == "2.00"
    assert billing.apply_monthly_reset(datetime(2026, 9, 2)) is False

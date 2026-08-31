"""邮件服务（腾讯云 SES）单元测试。"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest

from kairocli.mail import (
    MailCodeService,
    MailSender,
    MailSendError,
    MailSettings,
    build_mail_service,
    mask_email,
    tc3_headers,
    validate_email,
)

FULL_ENV: dict[str, str] = {
    "KAIROCLI_MAIL_ENABLED": "true",
    "KAIROCLI_MAIL_PROVIDER": "tencent-ses",
    "KAIROCLI_TENCENT_SECRET_ID": "AKIDexample",
    "KAIROCLI_TENCENT_SECRET_KEY": "secret-key",
    "KAIROCLI_MAIL_FROM_ADDRESS": "noreply@example.com",
    "KAIROCLI_MAIL_TEMPLATE_TEST": "213821",
    "KAIROCLI_MAIL_TEMPLATE_VERIFICATION": "213820",
}


class FakeSender:
    def __init__(self, error: Exception | None = None) -> None:
        self.sent: list[tuple[str, str, int]] = []
        self.error = error

    async def send_verification_code(self, to: str, code: str, minutes: int) -> None:
        if self.error is not None:
            raise self.error
        self.sent.append((to, code, minutes))


def test_settings_defaults_disable_mail() -> None:
    settings = MailSettings.from_environ({})
    assert settings.enabled is False
    assert settings.provider == "tencent-ses"
    assert settings.is_configured is False
    assert settings.code_ttl_minutes == 5
    assert settings.code_length == 6
    assert settings.resend_seconds == 60
    assert build_mail_service({}) is None


def test_settings_parse_full_env() -> None:
    settings = MailSettings.from_environ(FULL_ENV)
    assert settings.enabled is True
    assert settings.provider == "tencent-ses"
    assert settings.is_configured is True
    assert settings.from_alias == "Kairo CLI"
    assert settings.region == "ap-guangzhou"
    assert settings.test_template_id == 213821
    assert settings.verification_template_id == 213820


def test_settings_enabled_but_incomplete_stays_disabled() -> None:
    env = {**FULL_ENV, "KAIROCLI_TENCENT_SECRET_KEY": ""}
    settings = MailSettings.from_environ(env)
    assert settings.is_configured is False
    with pytest.raises(ValueError, match="配置不完整"):
        build_mail_service(env)


def test_settings_reject_unsupported_provider() -> None:
    env = {**FULL_ENV, "KAIROCLI_MAIL_PROVIDER": "smtp"}
    assert MailSettings.from_environ(env).is_configured is False
    with pytest.raises(ValueError, match="配置不完整"):
        build_mail_service(env)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("User@Example.COM ", "user@example.com"),
        ("first.last+tag@sub.example.io", "first.last+tag@sub.example.io"),
    ],
)
def test_validate_email_normalizes(raw: str, expected: str) -> None:
    assert validate_email(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "no-at-sign",
        "user@",
        "@example.com",
        "user@nodot",
        "user name@example.com",
        "a@" + "b" * 260,
    ],
)
def test_validate_email_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        validate_email(raw)


def test_mask_email_keeps_metadata_only() -> None:
    assert mask_email("alice@example.com") == "a***@example.com"
    assert mask_email("not-an-email") == "***"


def test_tc3_headers_shape_and_determinism() -> None:
    kwargs: dict[str, Any] = dict(
        secret_id="AKIDexample",
        secret_key="secret-key",
        service="ses",
        host="ses.tencentcloudapi.com",
        action="SendEmail",
        version="2020-10-02",
        timestamp=1_551_113_065,
        payload='{"k":"v"}',
        region="ap-guangzhou",
    )
    headers = tc3_headers(**kwargs)
    again = tc3_headers(**kwargs)
    assert headers == again
    assert headers["X-TC-Action"] == "SendEmail"
    assert headers["X-TC-Version"] == "2020-10-02"
    assert headers["X-TC-Timestamp"] == "1551113065"
    assert headers["X-TC-Region"] == "ap-guangzhou"
    match = re.fullmatch(
        r"TC3-HMAC-SHA256 Credential=AKIDexample/2019-02-25/ses/tc3_request, "
        r"SignedHeaders=content-type;host, Signature=([0-9a-f]{64})",
        headers["Authorization"],
    )
    assert match is not None


def _capture_transport(
    captured: list[httpx.Request],
    status_code: int = 200,
    body: dict[str, Any] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status_code, json=body or {"Response": {"RequestId": "req-1"}})

    return httpx.MockTransport(handler)


async def test_send_builds_template_payload() -> None:
    captured: list[httpx.Request] = []
    settings = MailSettings.from_environ(FULL_ENV)
    sender = MailSender(settings, transport=_capture_transport(captured))
    await sender.send_verification_code("User@Example.com", "013456", 5)
    request = captured[0]
    assert str(request.url) == "https://ses.tencentcloudapi.com/"
    assert request.headers["X-TC-Action"] == "SendEmail"
    payload = json.loads(request.content.decode("utf-8"))
    assert payload["FromEmailAddress"] == "Kairo CLI <noreply@example.com>"
    assert payload["Destination"] == ["user@example.com"]
    assert payload["Subject"] == "Kairo CLI 邮箱验证码"
    assert payload["Template"]["TemplateID"] == 213820
    assert payload["Template"]["TemplateData"] == '{"code": "013456", "minutes": "5"}'


async def test_send_test_mail_uses_test_template() -> None:
    captured: list[httpx.Request] = []
    settings = MailSettings.from_environ(FULL_ENV)
    sender = MailSender(settings, transport=_capture_transport(captured))
    await sender.send_test_mail("user@example.com")
    payload = json.loads(captured[0].content.decode("utf-8"))
    assert payload["Subject"] == "Kairo CLI 邮件服务测试"
    assert payload["Template"]["TemplateID"] == 213821
    assert payload["Template"]["TemplateData"] == "{}"


async def test_send_raises_on_provider_error() -> None:
    captured: list[httpx.Request] = []
    settings = MailSettings.from_environ(FULL_ENV)
    body = {"Response": {"Error": {"Code": "InvalidTemplateID", "Message": "模板不可用"}}}
    sender = MailSender(
        settings,
        transport=_capture_transport(captured, status_code=200, body=body),
    )
    with pytest.raises(MailSendError) as excinfo:
        await sender.send_test_mail("user@example.com")
    assert excinfo.value.code == "InvalidTemplateID"


async def test_send_raises_on_http_error_status() -> None:
    captured: list[httpx.Request] = []
    settings = MailSettings.from_environ(FULL_ENV)
    sender = MailSender(
        settings, transport=_capture_transport(captured, status_code=403)
    )
    with pytest.raises(MailSendError) as excinfo:
        await sender.send_test_mail("user@example.com")
    assert excinfo.value.code == "HTTP403"


async def test_send_requires_configuration_and_template() -> None:
    captured: list[httpx.Request] = []
    incomplete = MailSettings.from_environ(
        {**FULL_ENV, "KAIROCLI_TENCENT_SECRET_KEY": ""}
    )
    sender = MailSender(incomplete, transport=_capture_transport(captured))
    with pytest.raises(MailSendError) as excinfo:
        await sender.send_test_mail("user@example.com")
    assert excinfo.value.code == "NotConfigured"
    missing_template = MailSettings.from_environ(
        {**FULL_ENV, "KAIROCLI_MAIL_TEMPLATE_VERIFICATION": ""}
    )
    sender = MailSender(missing_template, transport=_capture_transport(captured))
    with pytest.raises(MailSendError) as excinfo:
        await sender.send_verification_code("user@example.com", "123456", 5)
    assert excinfo.value.code == "MissingTemplateId"
    assert captured == []


def _code_service(
    sender: FakeSender | None = None,
    env: dict[str, str] | None = None,
    clock_state: dict[str, float] | None = None,
) -> tuple[MailCodeService, FakeSender, dict[str, float]]:
    sent_sender = sender or FakeSender()
    state = clock_state or {"now": 1_000.0}
    settings = MailSettings.from_environ(env or FULL_ENV)
    service = MailCodeService(sent_sender, settings, clock=lambda: state["now"])
    return service, sent_sender, state


async def test_code_service_issue_and_verify() -> None:
    service, sender, _ = _code_service()
    await service.issue("User@Example.com")
    to, code, minutes = sender.sent[0]
    assert to == "user@example.com"
    assert re.fullmatch(r"\d{6}", code)
    assert minutes == 5
    assert service.verify("user@example.com", f" {code} ") is True
    # 成功后验证码立即作废。
    assert service.verify("user@example.com", code) is False


async def test_code_service_keeps_registration_and_reset_codes_separate() -> None:
    service, sender, _ = _code_service()
    await service.issue("user@example.com", "reset")
    code = sender.sent[0][1]
    assert service.verify("user@example.com", code, "register") is False
    assert service.verify("user@example.com", code, "reset") is True


async def test_code_service_rejects_wrong_code_and_expires() -> None:
    service, _, state = _code_service()
    await service.issue("user@example.com")
    assert service.verify("user@example.com", "000000") is False
    state["now"] += 6 * 60
    assert service.verify("user@example.com", "000000") is False


async def test_code_service_enforces_attempt_limit() -> None:
    service, sender, _ = _code_service()
    await service.issue("user@example.com")
    code = sender.sent[0][1]
    for _ in range(4):
        assert service.verify("user@example.com", "000000") is False
    # 第 5 次失败后整条作废，即使随后输入正确验证码也失效。
    assert service.verify("user@example.com", "000000") is False
    assert service.verify("user@example.com", code) is False


async def test_code_service_resend_throttle() -> None:
    service, sender, state = _code_service()
    await service.issue("user@example.com")
    with pytest.raises(ValueError, match="秒"):
        await service.issue("user@example.com")
    state["now"] += 61
    await service.issue("user@example.com")
    assert len(sender.sent) == 2


async def test_code_service_send_failure_does_not_throttle() -> None:
    sender = FakeSender(error=MailSendError("InvalidTemplateID", "模板不可用"))
    service, sender, _ = _code_service(sender=sender)
    with pytest.raises(MailSendError):
        await service.issue("user@example.com")
    assert service.verify("user@example.com", "000000") is False
    sender.error = None
    await service.issue("user@example.com")
    assert len(sender.sent) == 1


async def test_code_service_builds_from_environ() -> None:
    service = build_mail_service(FULL_ENV)
    assert service is not None
    assert service.ttl_minutes == 5

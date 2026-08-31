"""腾讯云 SES 事务邮件：测试邮件与邮箱验证码。

仅用于系统事务邮件（注册验证码、配置自检测试邮件）。日志只记录脱敏后的
生命周期元数据（收件人脱敏、模板 ID、结果码），不得记录验证码等敏感内容。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

_SES_HOST = "ses.tencentcloudapi.com"
_SES_SERVICE = "ses"
_SES_VERSION = "2020-10-02"
_SES_ACTION_SEND_EMAIL = "SendEmail"

_EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_MAX_EMAIL_CHARS = 254
_MAX_PENDING_CODES = 10_000

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class MailSendError(RuntimeError):
    """邮件发送失败：服务商返回错误、网络异常或本地配置缺失。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class MailSettings:
    """邮件服务配置，来自 KAIROCLI_* 环境变量。"""

    enabled: bool = False
    provider: str = "tencent-ses"
    secret_id: str = ""
    secret_key: str = ""
    from_address: str = ""
    from_alias: str = "Kairo CLI"
    region: str = "ap-guangzhou"
    test_template_id: int = 0
    verification_template_id: int = 0
    code_ttl_minutes: int = 5
    code_length: int = 6
    resend_seconds: int = 60
    max_verify_attempts: int = 5

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> MailSettings:
        env = os.environ if environ is None else environ

        def integer(name: str, default: int, minimum: int, maximum: int) -> int:
            raw = env.get(name, "").strip()
            if not raw:
                return default
            try:
                value = int(raw, 10)
            except ValueError:
                return default
            return min(max(value, minimum), maximum)

        return cls(
            enabled=env.get("KAIROCLI_MAIL_ENABLED", "").strip().lower() in _TRUE_VALUES,
            provider=env.get("KAIROCLI_MAIL_PROVIDER", "").strip().lower() or "tencent-ses",
            secret_id=env.get("KAIROCLI_TENCENT_SECRET_ID", "").strip(),
            secret_key=env.get("KAIROCLI_TENCENT_SECRET_KEY", "").strip(),
            from_address=env.get("KAIROCLI_MAIL_FROM_ADDRESS", "").strip(),
            from_alias=env.get("KAIROCLI_MAIL_FROM_ALIAS", "").strip() or "Kairo CLI",
            region=env.get("KAIROCLI_MAIL_REGION", "").strip() or "ap-guangzhou",
            test_template_id=integer("KAIROCLI_MAIL_TEMPLATE_TEST", 0, 0, 999_999_999),
            verification_template_id=integer(
                "KAIROCLI_MAIL_TEMPLATE_VERIFICATION", 0, 0, 999_999_999
            ),
            code_ttl_minutes=integer("KAIROCLI_MAIL_CODE_TTL_MINUTES", 5, 1, 30),
            code_length=integer("KAIROCLI_MAIL_CODE_LENGTH", 6, 4, 8),
            resend_seconds=integer("KAIROCLI_MAIL_RESEND_SECONDS", 60, 10, 600),
            max_verify_attempts=integer("KAIROCLI_MAIL_MAX_VERIFY_ATTEMPTS", 5, 1, 20),
        )

    @property
    def is_configured(self) -> bool:
        return (
            self.enabled
            and self.provider == "tencent-ses"
            and bool(self.secret_id)
            and bool(self.secret_key)
            and bool(self.from_address)
        )


def mask_email(address: str) -> str:
    """脱敏邮箱地址，用于日志等生命周期元数据输出。"""
    local, separator, domain = address.partition("@")
    if not separator:
        return "***"
    prefix = local[:1] if local else ""
    return f"{prefix}***@{domain}"


def validate_email(address: str) -> str:
    """校验并归一化邮箱地址（小写）；不合法时抛出 ValueError。"""
    normalized = address.strip().lower()
    if (
        not normalized
        or len(normalized) > _MAX_EMAIL_CHARS
        or not _EMAIL_PATTERN.match(normalized)
    ):
        raise ValueError("邮箱地址格式不正确")
    return normalized


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def tc3_headers(
    *,
    secret_id: str,
    secret_key: str,
    service: str,
    host: str,
    action: str,
    version: str,
    timestamp: int,
    payload: str,
    region: str = "",
) -> dict[str, str]:
    """按腾讯云 API 3.0 TC3-HMAC-SHA256 规范构造请求头。"""
    date = datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d")
    canonical_request = "\n".join(
        [
            "POST",
            "/",
            "",
            f"content-type:application/json; charset=utf-8\nhost:{host}\n",
            "content-type;host",
            _sha256_hex(payload),
        ]
    )
    string_to_sign = "\n".join(
        [
            "TC3-HMAC-SHA256",
            str(timestamp),
            f"{date}/{service}/tc3_request",
            _sha256_hex(canonical_request),
        ]
    )
    secret_date = _hmac_sha256(f"TC3{secret_key}".encode(), date)
    secret_service = _hmac_sha256(secret_date, service)
    secret_signing = _hmac_sha256(secret_service, "tc3_request")
    signature = hmac.new(
        secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    headers = {
        "Authorization": (
            f"TC3-HMAC-SHA256 Credential={secret_id}/{date}/{service}/tc3_request, "
            f"SignedHeaders=content-type;host, Signature={signature}"
        ),
        "Content-Type": "application/json; charset=utf-8",
        "X-TC-Action": action,
        "X-TC-Version": version,
        "X-TC-Timestamp": str(timestamp),
    }
    if region:
        headers["X-TC-Region"] = region
    return headers


class MailSender:
    """通过腾讯云 SES SendEmail 接口按模板发送邮件。"""

    def __init__(
        self,
        settings: MailSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], int] | None = None,
        endpoint: str = f"https://{_SES_HOST}/",
    ) -> None:
        self.settings = settings
        self.endpoint = endpoint
        self._transport = transport
        self._clock = clock or (lambda: int(time.time()))

    def _from_email_address(self) -> str:
        # 腾讯云 SES 要求别名与地址之间恰好一个空格，别名内不能有冒号。
        alias = self.settings.from_alias.replace(":", " ")
        if alias:
            return f"{alias} <{self.settings.from_address}>"
        return self.settings.from_address

    async def send(
        self, to: str, subject: str, template_id: int, params: dict[str, str]
    ) -> None:
        if not self.settings.is_configured:
            raise MailSendError("NotConfigured", "邮件服务未配置完整")
        if template_id <= 0:
            raise MailSendError("MissingTemplateId", "未配置邮件模板 ID")
        address = validate_email(to)
        payload = json.dumps(
            {
                "FromEmailAddress": self._from_email_address(),
                "Destination": [address],
                "Subject": subject,
                "Template": {
                    "TemplateID": template_id,
                    "TemplateData": json.dumps(params, ensure_ascii=False),
                },
            },
            ensure_ascii=False,
        )
        headers = tc3_headers(
            secret_id=self.settings.secret_id,
            secret_key=self.settings.secret_key,
            service=_SES_SERVICE,
            host=_SES_HOST,
            action=_SES_ACTION_SEND_EMAIL,
            version=_SES_VERSION,
            timestamp=self._clock(),
            payload=payload,
            region=self.settings.region,
        )
        try:
            async with httpx.AsyncClient(timeout=10.0, transport=self._transport) as client:
                response = await client.post(
                    self.endpoint, content=payload.encode("utf-8"), headers=headers
                )
        except httpx.HTTPError:
            logger.warning("邮件发送失败 收件人=%s result=network_error", mask_email(address))
            raise MailSendError("NetworkError", "邮件服务网络异常，请稍后重试") from None
        body: object = {}
        try:
            body = response.json()
        except ValueError:
            pass
        error: object = None
        if isinstance(body, dict):
            response_body = body.get("Response")
            error = response_body.get("Error") if isinstance(response_body, dict) else None
        if response.status_code != 200 or error:
            if isinstance(error, dict) and error.get("Code"):
                code = str(error["Code"])
                message = str(error.get("Message", ""))
            else:
                code = f"HTTP{response.status_code}"
                message = response.text[:200]
            logger.warning("邮件发送失败 收件人=%s result=%s", mask_email(address), code)
            raise MailSendError(code, message or "邮件发送失败")
        logger.info("邮件已提交发送 收件人=%s 模板=%d", mask_email(address), template_id)

    async def send_test_mail(self, to: str) -> None:
        await self.send(to, "Kairo CLI 邮件服务测试", self.settings.test_template_id, {})

    async def send_verification_code(self, to: str, code: str, minutes: int) -> None:
        await self.send(
            to,
            "Kairo CLI 邮箱验证码",
            self.settings.verification_template_id,
            {"code": code, "minutes": str(minutes)},
        )


class VerificationSender(Protocol):
    """MailCodeService 所需的最小发送接口。"""

    async def send_verification_code(self, to: str, code: str, minutes: int) -> None: ...


@dataclass(slots=True)
class _PendingCode:
    code: str
    issued_at: float
    expires_at: float
    attempts: int = 0


class MailCodeService:
    """生成、下发并校验邮箱验证码；验证码只在进程内存活。"""

    def __init__(
        self,
        sender: VerificationSender,
        settings: MailSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sender = sender
        self.settings = settings
        self._clock = clock
        self._pending: dict[tuple[str, str], _PendingCode] = {}
        self._last_sent: dict[tuple[str, str], float] = {}

    @property
    def ttl_minutes(self) -> int:
        return self.settings.code_ttl_minutes

    def generate_code(self) -> str:
        return "".join(
            str(secrets.randbelow(10)) for _ in range(self.settings.code_length)
        )

    async def issue(self, email: str, purpose: str = "register") -> str:
        """生成并下发验证码；触发重发限频时抛出 ValueError。"""
        address = validate_email(email)
        key = (address, purpose)
        now = self._clock()
        last = self._last_sent.get(key)
        if last is not None:
            remaining = self.settings.resend_seconds - (now - last)
            if remaining > 0:
                raise ValueError(f"发送过于频繁，请 {int(remaining) + 1} 秒后再试")
        code = self.generate_code()
        await self.sender.send_verification_code(
            address, code, self.settings.code_ttl_minutes
        )
        self._last_sent[key] = now
        self._pending[key] = _PendingCode(
            code=code,
            issued_at=now,
            expires_at=now + self.settings.code_ttl_minutes * 60,
        )
        self._evict(now)
        return address

    def verify(self, email: str, code: str, purpose: str = "register") -> bool:
        """校验验证码；成功或失效后立即作废，超过尝试上限后整条作废。"""
        address = validate_email(email)
        key = (address, purpose)
        entry = self._pending.get(key)
        if entry is None:
            return False
        now = self._clock()
        if now >= entry.expires_at:
            self._pending.pop(key, None)
            return False
        if not hmac.compare_digest(entry.code, code.strip()):
            entry.attempts += 1
            if entry.attempts >= self.settings.max_verify_attempts:
                self._pending.pop(key, None)
            return False
        self._pending.pop(key, None)
        return True

    def _evict(self, now: float) -> None:
        expired_pending = [key for key, item in self._pending.items() if item.expires_at <= now]
        for key in expired_pending:
            self._pending.pop(key, None)
        expired_sent = [
            key
            for key, issued_at in self._last_sent.items()
            if now - issued_at > self.settings.resend_seconds
        ]
        for key in expired_sent:
            self._last_sent.pop(key, None)
        overflow = len(self._pending) - _MAX_PENDING_CODES
        if overflow > 0:
            oldest = sorted(self._pending.items(), key=lambda item: item[1].issued_at)
            for key, _ in oldest[:overflow]:
                self._pending.pop(key, None)


def build_mail_service(environ: Mapping[str, str] | None = None) -> MailCodeService | None:
    """按环境变量构建验证码服务；未启用或配置不完整时返回 None。"""
    settings = MailSettings.from_environ(environ)
    if not settings.enabled:
        return None
    if not settings.is_configured or settings.verification_template_id <= 0:
        raise ValueError("邮件服务已启用但配置不完整")
    return MailCodeService(MailSender(settings), settings)

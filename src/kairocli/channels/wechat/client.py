from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
from typing import Any
from urllib.parse import quote

from ...channels.wechat.formatting import (
    format_wechat_text as format_wechat_text,
)
from ...channels.wechat.formatting import (
    split_message as split_message,
)
from ...json_boundary import decode_strict_json
from ...web import NetworkPolicy
from .accounts import (
    LoginResult,
    QrLogin,
    WechatAccount,
    WechatMediaItem,
    WechatMessage,
    WechatUpdate,
    _normalize_wechat_base_url,
    _safe_int,
    _validate_wechat_request_url,
    _wechat_string,
)

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
MAX_WECHAT_ACCOUNT_BYTES = 128 * 1024
MAX_WECHAT_REQUEST_BYTES = 1024 * 1024
MAX_WECHAT_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_WECHAT_API_JSON_DEPTH = 32
MAX_WECHAT_API_JSON_NODES = 200_000
MAX_WECHAT_MESSAGE_CHARS = 100_000
MAX_WECHAT_MESSAGES_PER_UPDATE = 1_000
MAX_WECHAT_DAEMON_LOG_BYTES = 5 * 1024 * 1024
MAX_WECHAT_ACCOUNT_JSON_DEPTH = 16
MAX_WECHAT_ACCOUNT_JSON_NODES = 10_000
_WECHAT_ACCOUNT_THREAD_LOCK = threading.RLock()


class IlinkClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        network_policy: NetworkPolicy | None = None,
        transport: Any = None,
    ) -> None:
        self.base_url = _normalize_wechat_base_url(base_url)
        self.network_policy = network_policy or NetworkPolicy(max_requests=120)
        self.transport = transport

    async def start_qr_login(self, bot_type: str = "3") -> QrLogin:
        payload = await self._request(
            "POST", f"ilink/bot/get_bot_qrcode?bot_type={quote(bot_type)}", {}
        )
        qrcode_id = _wechat_string(payload.get("qrcode") or payload.get("qrcode_id"), 4_096)
        qrcode_url = _wechat_string(
            payload.get("qrcode_img_content") or payload.get("qrcode_url"), 16_384
        )
        if not qrcode_id or not qrcode_url:
            raise RuntimeError("Failed to obtain a valid WeChat QR code response")
        return QrLogin(qrcode_id, qrcode_url)

    async def poll_qr_status(self, qrcode_id: str) -> LoginResult:
        if not qrcode_id or len(qrcode_id) > 4_096:
            raise ValueError("WeChat QR code ID is empty or too long")
        payload = await self._request(
            "GET", f"ilink/bot/get_qrcode_status?qrcode={quote(qrcode_id)}"
        )
        status = _wechat_string(payload.get("status"), 64)
        if status == "confirmed":
            token = _wechat_string(payload.get("bot_token"), 16_384)
            account_id = _wechat_string(payload.get("ilink_bot_id"), 1_024)
            base_url = _wechat_string(payload.get("baseurl") or DEFAULT_BASE_URL, 8_192)
            user_id = _wechat_string(payload.get("ilink_user_id"), 1_024)
            if not token or not account_id or not base_url or not user_id:
                raise RuntimeError("Confirmed WeChat login response is incomplete")
            try:
                base_url = _normalize_wechat_base_url(base_url)
            except ValueError as exc:
                raise RuntimeError("Confirmed WeChat login returned an invalid base URL") from exc
            return LoginResult(
                True,
                False,
                status,
                token,
                account_id,
                base_url,
                user_id,
                "connected",
            )
        return LoginResult(
            False,
            status == "expired",
            status,
            message=_wechat_string(payload.get("retmsg") or status, 4_096),
        )

    async def get_updates(self, account: WechatAccount, timeout_ms: int = 35_000) -> WechatUpdate:
        timeout_ms = max(1_000, min(timeout_ms, 60_000))
        payload = await self._request(
            "POST",
            "ilink/bot/getupdates",
            {"get_updates_buf": account.sync_buf},
            account.token,
            account.base_url,
            request_timeout=max(5, timeout_ms / 1000 + 5),
        )
        raw_messages = payload.get("msgs") or []
        messages = (
            [
                self._parse_message(item)
                for item in raw_messages[:MAX_WECHAT_MESSAGES_PER_UPDATE]
                if isinstance(item, dict)
            ]
            if isinstance(raw_messages, list)
            else []
        )
        raw_sync_buf = payload.get("get_updates_buf")
        if raw_sync_buf in (None, ""):
            sync_buf = account.sync_buf
        else:
            sync_buf = _wechat_string(raw_sync_buf, 65_536)
            if not sync_buf:
                raise RuntimeError("WeChat sync buffer is invalid or exceeds 64 KiB")
        next_timeout = _safe_int(payload.get("longpolling_timeout_ms"), timeout_ms)
        return WechatUpdate(
            _safe_int(payload.get("ret", payload.get("errcode", 0)), -1),
            _wechat_string(payload.get("errmsg") or payload.get("retmsg"), 4_096),
            sync_buf,
            max(1_000, min(next_timeout, 60_000)),
            messages,
        )

    async def send_text(
        self, account: WechatAccount, to_user_id: str, context_token: str, text: str
    ) -> None:
        if not text or len(text) > 3_800:
            raise ValueError("WeChat text chunk is empty or exceeds 3800 characters")
        message = {
            "from_user_id": account.account_id,
            "to_user_id": to_user_id,
            "client_id": f"kairocli-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
            "message_type": 2,
            "message_state": 2,
            "context_token": context_token,
            "item_list": [{"type": 1, "text_item": {"text": text}}],
        }
        await self._request(
            "POST", "ilink/bot/sendmessage", {"msg": message}, account.token, account.base_url
        )

    async def send_typing(
        self, account: WechatAccount, to_user_id: str, context_token: str, status: int
    ) -> None:
        config = await self._request(
            "POST",
            "ilink/bot/getconfig",
            {"ilink_user_id": to_user_id, "context_token": context_token},
            account.token,
            account.base_url,
        )
        ticket = _wechat_string(config.get("typing_ticket"), 16_384)
        if ticket:
            await self._request(
                "POST",
                "ilink/bot/sendtyping",
                {"ilink_user_id": to_user_id, "typing_ticket": ticket, "status": status},
                account.token,
                account.base_url,
            )

    async def notify(self, account: WechatAccount, running: bool) -> None:
        endpoint = "ilink/bot/msg/notifystart" if running else "ilink/bot/msg/notifystop"
        await self._request("POST", endpoint, {}, account.token, account.base_url)

    async def _request(
        self,
        method: str,
        endpoint: str,
        body: dict[str, Any] | None = None,
        token: str = "",
        base_url: str = "",
        request_timeout: float = 40,
    ) -> dict[str, Any]:
        import httpx

        headers = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "X-WECHAT-UIN": base64.b64encode(os.urandom(4)).decode(),
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        normalized_base = _normalize_wechat_base_url(base_url or self.base_url)
        url = normalized_base + "/" + endpoint.lstrip("/")
        _validate_wechat_request_url(url)
        try:
            request_body = (
                json.dumps(
                    body,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                if body
                else b""
            )
        except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
            raise ValueError("WeChat request body is not finite standard JSON") from exc
        if len(request_body) > MAX_WECHAT_REQUEST_BYTES:
            raise ValueError("WeChat request exceeds the 1 MiB limit")
        await self.network_policy.check(url)
        await self.network_policy.acquire()
        async with httpx.AsyncClient(timeout=request_timeout, transport=self.transport) as client:
            async with client.stream(
                method, url, headers=headers, content=request_body or None
            ) as response:
                if 300 <= response.status_code < 400:
                    raise RuntimeError("WeChat API redirects are not allowed")
                response.raise_for_status()
                response_body = bytearray()
                async for chunk in response.aiter_bytes():
                    response_body.extend(chunk)
                    if len(response_body) > MAX_WECHAT_RESPONSE_BYTES:
                        raise RuntimeError("WeChat response exceeds the 2 MiB limit")
        try:
            payload = decode_strict_json(
                response_body,
                max_bytes=MAX_WECHAT_RESPONSE_BYTES,
                max_depth=MAX_WECHAT_API_JSON_DEPTH,
                max_nodes=MAX_WECHAT_API_JSON_NODES,
            )
        except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
            raise RuntimeError("WeChat API returned invalid or unsafe JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("WeChat API response must be a JSON object")
        return payload

    @staticmethod
    def _parse_message(payload: dict[str, Any]) -> WechatMessage:
        text_parts: list[str] = []
        media_items: list[WechatMediaItem] = []
        raw_items = payload.get("item_list") or []
        if not isinstance(raw_items, list):
            raw_items = []
        for item in raw_items[:100]:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text_item"), dict):
                text_parts.append(
                    _wechat_string(
                        item["text_item"].get("text"),
                        MAX_WECHAT_RESPONSE_BYTES,
                        truncate=True,
                    )
                )
            elif isinstance(item.get("voice_item"), dict):
                text_parts.append(
                    _wechat_string(
                        item["voice_item"].get("text"),
                        MAX_WECHAT_RESPONSE_BYTES,
                        truncate=True,
                    )
                )
            elif isinstance(item.get("file_item"), dict):
                file_item = item["file_item"]
                name = _wechat_string(file_item.get("file_name"), 1_024, truncate=True) or "unknown"
                media_items.append(IlinkClient._parse_media("file", file_item, name))
                text_parts.append(f"[User sent a file: {name}]")
            elif isinstance(item.get("image_item"), dict):
                media_items.append(IlinkClient._parse_media("image", item["image_item"], ""))
        text = "\n".join(part for part in text_parts if part).strip()
        if len(text) > MAX_WECHAT_MESSAGE_CHARS:
            suffix = "\n[WeChat message truncated]"
            text = text[: MAX_WECHAT_MESSAGE_CHARS - len(suffix)] + suffix
        return WechatMessage(
            _wechat_string(
                payload.get("message_id") or payload.get("seq"),
                4_096,
                allow_int=True,
            ),
            _wechat_string(payload.get("from_user_id"), 1_024),
            _wechat_string(payload.get("context_token"), 16_384),
            text,
            tuple(media_items),
        )

    @staticmethod
    def _parse_media(media_type: str, node: dict[str, Any], file_name: str) -> WechatMediaItem:
        media = node.get("media")
        if not isinstance(media, dict):
            media = node.get("cdn_media")
        if not isinstance(media, dict):
            media = {}
        aes_key = media.get("aes_key") or node.get("aeskey") or ""
        return WechatMediaItem(
            media_type[:32],
            file_name[:1_024],
            _wechat_string(node.get("mime_type"), 256, truncate=True),
            _wechat_string(media.get("encrypt_query_param"), 16_384),
            _wechat_string(aes_key, 4_096),
        )

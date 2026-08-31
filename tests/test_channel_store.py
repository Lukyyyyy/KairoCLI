from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kairocli.channels.store import ChannelStore
from kairocli.web_auth import WebUserStore


def test_wechat_bindings_are_user_scoped_and_external_identity_is_unique(
    tmp_path: Path,
) -> None:
    database = tmp_path / "web" / "users.db"
    users = WebUserStore(database)
    alice = users.create_user("alice@example.com", "password-123")
    bob = users.create_user("bob@example.com", "password-456")
    store = ChannelStore(database)

    binding = store.save_wechat_binding(
        alice.id,
        token="secret",
        account_id="bot-a",
        external_user_id="wechat-user",
        base_url="https://ilinkai.weixin.qq.com",
        workspace=str(tmp_path),
    )

    assert store.binding_for_user(alice.id, "wechat") == binding
    assert store.wechat_credentials(binding.id).token == "secret"  # type: ignore[union-attr]
    assert store.set_enabled(binding.id, True)
    assert store.binding(binding.id).enabled is True  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="already bound"):
        store.save_wechat_binding(
            bob.id,
            token="other",
            account_id="bot-b",
            external_user_id="wechat-user",
            base_url="https://ilinkai.weixin.qq.com",
            workspace=str(tmp_path),
        )


def test_inbox_deduplicates_before_advancing_sync_state(tmp_path: Path) -> None:
    database = tmp_path / "web" / "users.db"
    users = WebUserStore(database)
    user = users.create_user("user@example.com", "password-123")
    store = ChannelStore(database)
    binding = store.save_wechat_binding(
        user.id,
        token="secret",
        account_id="bot",
        external_user_id="wechat-user",
        base_url="https://ilinkai.weixin.qq.com",
        workspace=str(tmp_path),
    )

    message = {"message_id": "message-1", "text": "hello"}
    assert store.enqueue_messages(binding.id, [message], "sync-1") == 1
    assert store.enqueue_messages(binding.id, [message], "sync-2") == 0
    assert store.wechat_credentials(binding.id).sync_buf == "sync-2"  # type: ignore[union-attr]


def test_disconnected_channel_threads_expire_after_retention(tmp_path: Path) -> None:
    database = tmp_path / "web" / "users.db"
    user = WebUserStore(database).create_user("user@example.com", "password-123")
    store = ChannelStore(database)
    binding = store.save_wechat_binding(
        user.id,
        token="secret",
        account_id="bot",
        external_user_id="wechat-user",
        base_url="https://ilinkai.weixin.qq.com",
        workspace=str(tmp_path),
    )
    store.save_thread(binding.id, str(tmp_path), "thread_test")
    disconnected_at = datetime.now(UTC)
    assert store.disconnect(binding.id)

    assert store.expired_threads(30, at=disconnected_at + timedelta(days=29)) == []
    assert store.expired_threads(30, at=disconnected_at + timedelta(days=31)) == [
        (binding.id, "thread_test", user.id)
    ]
    store.delete_thread_mapping(binding.id, "thread_test")
    assert store.expired_threads(30, at=disconnected_at + timedelta(days=31)) == []

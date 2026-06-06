"""Smoke tests for divert helpers — the subclass send() one-liner just calls these.

We test the free functions directly so we don't have to instantiate
WhatsAppCloudAdapter / TelegramAdapter (their __init__ pulls aiohttp/httpx
state and full PlatformConfig). Subclass integration is verified manually
in Task 4 (W1 dual-adapter smoke).
"""
import pytest
from unittest.mock import AsyncMock

from mindora_hooks.whatsapp_cloud_mindora import divert_whatsapp_to_outbox
from mindora_hooks.telegram_mindora import divert_telegram_to_outbox


@pytest.mark.asyncio
async def test_whatsapp_divert_calls_writer_and_returns_success():
    redis = AsyncMock(xadd=AsyncMock(return_value=b"100-0"))
    result = await divert_whatsapp_to_outbox(
        redis,
        tenant_id="testcust4",
        chat_id="6591234567",
        content="hello",
        phone_number_id="1012",
    )
    assert result.success is True
    assert result.message_id == "100-0"
    redis.xadd.assert_called_once()
    payload = redis.xadd.call_args[0][1]
    assert payload["channel"] == "whatsapp"
    assert payload["phone"] == "6591234567"
    assert payload["phone_number_id"] == "1012"
    assert payload["chat_id"] == ""
    assert payload["text"] == "hello"
    assert payload["message_id"].startswith("hermes-")


@pytest.mark.asyncio
async def test_whatsapp_divert_empty_content_skips_xadd():
    redis = AsyncMock()
    result = await divert_whatsapp_to_outbox(
        redis, tenant_id="t1", chat_id="123", content="   "
    )
    assert result.success is True
    assert result.message_id is None
    redis.xadd.assert_not_called()


@pytest.mark.asyncio
async def test_whatsapp_divert_redis_failure_returns_error():
    redis = AsyncMock(xadd=AsyncMock(side_effect=ConnectionError("redis down")))
    result = await divert_whatsapp_to_outbox(
        redis, tenant_id="t1", chat_id="123", content="hi"
    )
    assert result.success is False
    assert "redis down" in (result.error or "")


@pytest.mark.asyncio
async def test_telegram_divert_calls_writer_and_returns_success():
    redis = AsyncMock(xadd=AsyncMock(return_value=b"200-0"))
    result = await divert_telegram_to_outbox(
        redis, tenant_id="testcust4", chat_id="987654", content="hello"
    )
    assert result.success is True
    assert result.message_id == "200-0"
    redis.xadd.assert_called_once()
    payload = redis.xadd.call_args[0][1]
    assert payload["channel"] == "telegram"
    assert payload["chat_id"] == "987654"
    assert payload["phone"] == ""
    assert payload["phone_number_id"] == ""
    assert payload["text"] == "hello"


@pytest.mark.asyncio
async def test_telegram_divert_empty_content_skips_xadd():
    redis = AsyncMock()
    result = await divert_telegram_to_outbox(
        redis, tenant_id="t1", chat_id="987", content=""
    )
    assert result.success is True
    assert result.message_id is None
    redis.xadd.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_divert_redis_failure_returns_error():
    redis = AsyncMock(xadd=AsyncMock(side_effect=RuntimeError("xadd failed")))
    result = await divert_telegram_to_outbox(
        redis, tenant_id="t1", chat_id="987", content="hi"
    )
    assert result.success is False
    assert "xadd failed" in (result.error or "")


# ---------------------------------------------------------------------------
# Sidecar invariant tests: edit_message MUST NOT escape the outbox.
#
# The parent TelegramAdapter has REQUIRES_EDIT_FINALIZE=True and an
# edit_message() that POSTs Bot API directly. The parent WhatsAppCloudAdapter
# inherits the base default (success=False), which would force every streaming
# edit to look like a failed send. Both subclasses override to a no-op success.
# Use object.__new__ to skip parent __init__ (which needs env vars + redis).
# ---------------------------------------------------------------------------


def test_telegram_adapter_disables_edit_finalize():
    from mindora_hooks.telegram_mindora import TelegramMindoraAdapter
    assert TelegramMindoraAdapter.REQUIRES_EDIT_FINALIZE is False


def test_whatsapp_adapter_disables_edit_finalize():
    from mindora_hooks.whatsapp_cloud_mindora import WhatsAppCloudMindoraAdapter
    assert WhatsAppCloudMindoraAdapter.REQUIRES_EDIT_FINALIZE is False


@pytest.mark.asyncio
async def test_telegram_adapter_edit_message_is_noop():
    from mindora_hooks.telegram_mindora import TelegramMindoraAdapter
    adapter = object.__new__(TelegramMindoraAdapter)
    adapter._tenant_id = "t1"
    result = await adapter.edit_message(
        chat_id="c1", message_id="m1", content="hi", finalize=True
    )
    assert result.success is True
    assert result.message_id == "m1"


@pytest.mark.asyncio
async def test_whatsapp_adapter_edit_message_is_noop():
    from mindora_hooks.whatsapp_cloud_mindora import WhatsAppCloudMindoraAdapter
    adapter = object.__new__(WhatsAppCloudMindoraAdapter)
    adapter._tenant_id = "t1"
    result = await adapter.edit_message(
        chat_id="c1", message_id="m1", content="hi"
    )
    assert result.success is True
    assert result.message_id == "m1"


# ---------------------------------------------------------------------------
# Sidecar invariant tests: ALL non-text inherited send_* methods must be
# blocked by the @install_outbound_blockers class decorator. Phase 2 outbox
# is text-only; media + interactive prompts + drafts are Phase 3.
#
# The decorator wraps every inherited send_* method (except send,
# edit_message, send_typing) at class-creation time, so any new send_<foo>
# upstream is automatically caught.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,args", [
    # Round-3 media methods
    ("send_image", {"chat_id": "c", "image_url": "https://x/y.png"}),
    ("send_image_file", {"chat_id": "c", "image_path": "/tmp/x.png"}),
    ("send_video", {"chat_id": "c", "video_path": "https://x/y.mp4"}),
    ("send_voice", {"chat_id": "c", "audio_path": "https://x/y.ogg"}),
    ("send_document", {"chat_id": "c", "file_path": "https://x/y.pdf"}),
])
@pytest.mark.asyncio
async def test_whatsapp_media_methods_no_op(method, args):
    from mindora_hooks.whatsapp_cloud_mindora import WhatsAppCloudMindoraAdapter
    adapter = object.__new__(WhatsAppCloudMindoraAdapter)
    adapter._tenant_id = "t1"
    fn = getattr(adapter, method)
    result = await fn(**args)
    assert result.success is True
    assert result.message_id is None


@pytest.mark.parametrize("method,args", [
    # Round-3 media methods
    ("send_image", {"chat_id": "c", "image_url": "https://x/y.png"}),
    ("send_image_file", {"chat_id": "c", "image_path": "/tmp/x.png"}),
    ("send_video", {"chat_id": "c", "video_path": "https://x/y.mp4"}),
    ("send_voice", {"chat_id": "c", "audio_path": "https://x/y.ogg"}),
    ("send_document", {"chat_id": "c", "file_path": "https://x/y.pdf"}),
    ("send_animation", {"chat_id": "c", "animation_url": "https://x/y.gif"}),
])
@pytest.mark.asyncio
async def test_telegram_media_methods_no_op(method, args):
    from mindora_hooks.telegram_mindora import TelegramMindoraAdapter
    adapter = object.__new__(TelegramMindoraAdapter)
    adapter._tenant_id = "t1"
    fn = getattr(adapter, method)
    result = await fn(**args)
    assert result.success is True
    assert result.message_id is None


@pytest.mark.asyncio
async def test_telegram_send_multiple_images_no_op():
    """Catch-all wrapper returns SendResult — overrides the parent's None
    return. Acceptable: callers that only checked `result is None` would
    silently break, but the only caller surface here is hermes' run loop
    which treats SendResult(success=True) and None as both non-fatal."""
    from mindora_hooks.telegram_mindora import TelegramMindoraAdapter
    adapter = object.__new__(TelegramMindoraAdapter)
    adapter._tenant_id = "t1"
    result = await adapter.send_multiple_images(
        chat_id="c", images=[("https://x/1.png", "cap")],
    )
    assert result.success is True
    assert result.message_id is None


# ---------------------------------------------------------------------------
# Round-4 additions: interactive prompts (send_clarify, send_exec_approval,
# send_slash_confirm) and Telegram-only streaming/control surfaces
# (send_draft, send_model_picker, send_update_prompt). All POST the platform
# directly on the parent and must be neutralized.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,args", [
    ("send_clarify", {
        "chat_id": "c", "question": "pick one", "choices": ["a", "b"],
        "clarify_id": "cl1", "session_key": "sk1",
    }),
    ("send_exec_approval", {
        "chat_id": "c", "command": "rm -rf /", "session_key": "sk1",
    }),
    ("send_slash_confirm", {
        "chat_id": "c", "title": "Confirm", "message": "ok?",
        "session_key": "sk1", "confirm_id": "cf1",
    }),
    ("send_private_notice", {
        "chat_id": "c", "user_id": "u1", "content": "fyi",
    }),
    ("send_draft", {
        "chat_id": "c", "draft_id": 1, "content": "draft body",
    }),
])
@pytest.mark.asyncio
async def test_whatsapp_round4_methods_no_op(method, args):
    from mindora_hooks.whatsapp_cloud_mindora import WhatsAppCloudMindoraAdapter
    adapter = object.__new__(WhatsAppCloudMindoraAdapter)
    adapter._tenant_id = "t1"
    fn = getattr(adapter, method)
    result = await fn(**args)
    assert result.success is True
    assert result.message_id is None


@pytest.mark.parametrize("method,args", [
    ("send_clarify", {
        "chat_id": "c", "question": "pick one", "choices": ["a", "b"],
        "clarify_id": "cl1", "session_key": "sk1",
    }),
    ("send_exec_approval", {
        "chat_id": "c", "command": "rm -rf /", "session_key": "sk1",
    }),
    ("send_slash_confirm", {
        "chat_id": "c", "title": "Confirm", "message": "ok?",
        "session_key": "sk1", "confirm_id": "cf1",
    }),
    ("send_private_notice", {
        "chat_id": "c", "user_id": "u1", "content": "fyi",
    }),
    ("send_draft", {
        "chat_id": "c", "draft_id": 1, "content": "draft body",
    }),
    ("send_model_picker", {
        "chat_id": "c", "providers": [], "current_model": "m",
        "current_provider": "p", "session_key": "sk1",
        "on_model_selected": lambda *_a, **_k: None,
    }),
    ("send_update_prompt", {
        "chat_id": "c", "prompt": "update?", "default": "x",
        "session_key": "sk1",
    }),
])
@pytest.mark.asyncio
async def test_telegram_round4_methods_no_op(method, args):
    from mindora_hooks.telegram_mindora import TelegramMindoraAdapter
    adapter = object.__new__(TelegramMindoraAdapter)
    adapter._tenant_id = "t1"
    fn = getattr(adapter, method)
    result = await fn(**args)
    assert result.success is True
    assert result.message_id is None


# ---------------------------------------------------------------------------
# Guard test: send_typing must NOT be wrapped — typing/read indicators are
# ephemeral platform-side state and don't carry user-visible content.
# ---------------------------------------------------------------------------


def test_telegram_send_typing_not_wrapped():
    """The decorator's allowlist preserves send_typing as the parent method.
    Our wrapper sets __qualname__ = '<Subclass>.<method>'; an unwrapped
    inherited method keeps the parent's qualname."""
    from mindora_hooks.telegram_mindora import TelegramMindoraAdapter
    method = TelegramMindoraAdapter.send_typing
    # If the decorator had wrapped it, qualname would start with
    # "TelegramMindoraAdapter." Instead it must reference the parent class.
    assert "TelegramAdapter" in method.__qualname__
    assert not method.__qualname__.startswith("TelegramMindoraAdapter.")


def test_whatsapp_send_typing_not_wrapped():
    from mindora_hooks.whatsapp_cloud_mindora import WhatsAppCloudMindoraAdapter
    method = WhatsAppCloudMindoraAdapter.send_typing
    assert "WhatsAppCloudAdapter" in method.__qualname__
    assert not method.__qualname__.startswith("WhatsAppCloudMindoraAdapter.")


# ---------------------------------------------------------------------------
# Round-5 P2: Telegram parent's create_handoff_thread() POSTs Bot API directly
# (forum-topic creation). It's not a send_* name so the decorator misses it;
# we override explicitly to no-op (return None — matches base.py contract).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_telegram_create_handoff_thread_returns_none():
    from mindora_hooks.telegram_mindora import TelegramMindoraAdapter
    adapter = object.__new__(TelegramMindoraAdapter)
    adapter._tenant_id = "t1"
    result = await adapter.create_handoff_thread("parent-chat", "Hermes — title")
    assert result is None


def test_telegram_create_handoff_thread_is_overridden():
    """The Mindora subclass MUST override create_handoff_thread, not inherit
    TelegramAdapter's version which calls Bot.create_forum_topic() directly."""
    from mindora_hooks.telegram_mindora import TelegramMindoraAdapter
    method = TelegramMindoraAdapter.create_handoff_thread
    assert method.__qualname__.startswith("TelegramMindoraAdapter.")

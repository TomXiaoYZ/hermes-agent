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

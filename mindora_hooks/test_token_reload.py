import asyncio

import pytest
from google.api_core import exceptions as gax
from unittest.mock import AsyncMock, patch
from mindora_hooks.token_reload import handle_401, TokenReloader


@pytest.mark.asyncio
async def test_reload_returns_fresh_token():
    redis = AsyncMock()
    reloader = TokenReloader(
        tenant_id="testcust4",
        channel="whatsapp",
        sm_secret_name="mindora-testcust4-meta-system-user-token",
        redis_client=redis,
    )
    with patch.object(reloader, "_fetch_sm", new=AsyncMock(return_value="NEW_TOKEN")):
        new_token = await reloader.reload()
    assert new_token == "NEW_TOKEN"
    redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_reload_failure_sets_redis_flag():
    redis = AsyncMock()
    reloader = TokenReloader(
        tenant_id="testcust4",
        channel="whatsapp",
        sm_secret_name="mindora-testcust4-meta-system-user-token",
        redis_client=redis,
    )
    with patch.object(reloader, "_fetch_sm", new=AsyncMock(side_effect=gax.GoogleAPIError("SM down"))):
        with pytest.raises(gax.GoogleAPIError):
            await reloader.reload()
    redis.set.assert_called_once_with(
        "tenant:testcust4:meta_token_invalid", "true"
    )


@pytest.mark.asyncio
async def test_reload_failure_telegram_uses_telegram_flag_key():
    redis = AsyncMock()
    reloader = TokenReloader(
        tenant_id="testcust4",
        channel="telegram",
        sm_secret_name="mindora-testcust4-telegram-customer-bot-token",
        redis_client=redis,
    )
    with patch.object(reloader, "_fetch_sm", new=AsyncMock(side_effect=OSError("SM down"))):
        with pytest.raises(OSError):
            await reloader.reload()
    redis.set.assert_called_once_with(
        "tenant:testcust4:telegram_token_invalid", "true"
    )


@pytest.mark.asyncio
async def test_reload_does_not_swallow_cancellation():
    redis = AsyncMock()
    reloader = TokenReloader(
        tenant_id="testcust4", channel="whatsapp",
        sm_secret_name="x", redis_client=redis,
    )
    with patch.object(reloader, "_fetch_sm", new=AsyncMock(side_effect=asyncio.CancelledError())):
        with pytest.raises(asyncio.CancelledError):
            await reloader.reload()
    redis.set.assert_not_called()  # we did NOT touch Redis on cancellation


@pytest.mark.asyncio
async def test_handle_401_clears_flag_on_success_when_set():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=b"true")
    await handle_401(redis, tenant="testcust4", channel="whatsapp", outcome="success")
    redis.delete.assert_called_once_with("tenant:testcust4:meta_token_invalid")


@pytest.mark.asyncio
async def test_handle_401_no_op_on_success_when_flag_not_set():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    await handle_401(redis, tenant="testcust4", channel="whatsapp", outcome="success")
    redis.delete.assert_not_called()


@pytest.mark.asyncio
async def test_handle_401_no_op_on_non_success():
    redis = AsyncMock()
    await handle_401(redis, tenant="testcust4", channel="whatsapp", outcome="failure")
    redis.get.assert_not_called()
    redis.delete.assert_not_called()


def test_flag_key_property():
    reloader_wa = TokenReloader("t1", "whatsapp", "x", redis_client=None)
    reloader_tg = TokenReloader("t1", "telegram", "x", redis_client=None)
    assert reloader_wa.flag_key == "tenant:t1:meta_token_invalid"
    assert reloader_tg.flag_key == "tenant:t1:telegram_token_invalid"


def test_invalid_channel_rejected():
    with pytest.raises(ValueError, match="invalid channel"):
        TokenReloader("t1", "discord", "x", redis_client=None)

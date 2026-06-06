import pytest
from unittest.mock import AsyncMock
from mindora_hooks.hermes_outbox_writer import write_to_outbox


@pytest.mark.asyncio
async def test_whatsapp_envelope_shape():
    redis = AsyncMock()
    await write_to_outbox(
        redis,
        tenant_id="testcust4",
        channel="whatsapp",
        phone="+6591234567",
        phone_number_id="1012",
        chat_id=None,
        text="hi",
        message_id="m1",
        source="hermes",
    )
    redis.xadd.assert_called_once()
    args = redis.xadd.call_args
    stream_name = args[0][0]
    payload = args[0][1]
    assert stream_name == "tenant:testcust4:hermes_outbox"
    assert payload["tenant_id"] == "testcust4"
    assert payload["channel"] == "whatsapp"
    assert payload["phone"] == "+6591234567"
    assert payload["phone_number_id"] == "1012"
    assert payload["chat_id"] == ""
    assert payload["text"] == "hi"
    assert payload["message_id"] == "m1"
    assert payload["source"] == "hermes"
    assert "ts" in payload  # epoch seconds string


@pytest.mark.asyncio
async def test_telegram_envelope_shape():
    redis = AsyncMock()
    await write_to_outbox(
        redis,
        tenant_id="testcust4",
        channel="telegram",
        phone=None,
        phone_number_id=None,
        chat_id="123456",
        text="hi",
        message_id="m2",
        source="hermes",
    )
    payload = redis.xadd.call_args[0][1]
    assert payload["channel"] == "telegram"
    assert payload["chat_id"] == "123456"
    assert payload["phone"] == ""
    assert payload["phone_number_id"] == ""


@pytest.mark.asyncio
async def test_unsupported_channel_raises():
    redis = AsyncMock()
    with pytest.raises(ValueError, match="unsupported channel"):
        await write_to_outbox(
            redis,
            tenant_id="t1",
            channel="discord",
            phone=None,
            phone_number_id=None,
            chat_id="x",
            text="hi",
            message_id="m",
            source="hermes",
        )
    redis.xadd.assert_not_called()


@pytest.mark.asyncio
async def test_default_source_is_hermes():
    redis = AsyncMock()
    await write_to_outbox(
        redis,
        tenant_id="t1",
        channel="whatsapp",
        phone="+1",
        phone_number_id="x",
        chat_id=None,
        text="hi",
        message_id="m",
    )
    assert redis.xadd.call_args[0][1]["source"] == "hermes"


@pytest.mark.asyncio
async def test_returns_xadd_id():
    redis = AsyncMock(xadd=AsyncMock(return_value=b"1234567890-0"))
    result = await write_to_outbox(
        redis,
        tenant_id="t1",
        channel="whatsapp",
        phone="+1",
        phone_number_id="x",
        chat_id=None,
        text="hi",
        message_id="m",
        source="hermes",
    )
    assert result == b"1234567890-0"

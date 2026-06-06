"""Write hermes LLM replies to a Redis Stream instead of POSTing the platform.

In the Sidecar Hybrid architecture, redis-bridge is the sole Cloud API
send-owner. Hermes XADDs each reply to tenant:{id}:hermes_outbox; bridge
consumes that stream and runs the actual Meta/Telegram POST under a
single-owner Lua claim.

CONTRACT — chunking / formatting boundaries:

The redis-bridge consumer is responsible for ALL platform-specific
post-processing of envelope.text, including:
  - WhatsApp/Telegram 4096-char split (last-whitespace boundary)
  - Locale-aware fallback substitution (already in bridge sanitize stage)
  - Markdown / MarkdownV2 escaping per platform
  - URL preview suppression, link-button rendering, etc.

This module deliberately writes raw LLM text. Do NOT chunk, escape, or
truncate here — that would (a) double-process when bridge runs its own
sanitize pass, and (b) split the same string twice with conflicting
boundaries. See agents/crm/redis-bridge/internal/sender/{whatsapp,telegram}.go
for the consumer-side splitters (Phase 2 plan Tasks 7-8).
"""
import logging
import time
from typing import Any, Optional

log = logging.getLogger(__name__)

_VALID_CHANNELS = ("whatsapp", "telegram")


async def write_to_outbox(
    redis: "Any",
    *,
    tenant_id: str,
    channel: str,
    phone: Optional[str],
    phone_number_id: Optional[str],
    chat_id: Optional[str],
    text: str,
    message_id: str,
    source: str = "hermes",
) -> Any:
    """XADD a reply envelope to tenant:{id}:hermes_outbox.

    Mirrors the existing tenant:{id}:outbox envelope shape so redis-bridge
    can treat hermes_outbox as a second identical input stream.

    Returns the Redis XADD id (e.g. b"1717603200000-0").
    """
    if channel not in _VALID_CHANNELS:
        raise ValueError(
            f"unsupported channel: {channel!r} (expected one of {_VALID_CHANNELS})"
        )
    payload = {
        "tenant_id": tenant_id,
        "channel": channel,
        "phone": phone or "",
        "phone_number_id": phone_number_id or "",
        "chat_id": chat_id or "",
        "text": text,
        "message_id": message_id,
        "source": source,
        "ts": str(int(time.time())),
    }
    stream = f"tenant:{tenant_id}:hermes_outbox"
    msg_id = await redis.xadd(stream, payload)
    log.debug(
        "hermes_outbox.write tenant=%s channel=%s message_id=%s xadd_id=%s",
        tenant_id, channel, message_id, msg_id,
    )
    return msg_id

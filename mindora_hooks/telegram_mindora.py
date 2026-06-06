"""Mindora subclass of TelegramAdapter — XADDs to redis-bridge instead of calling Bot API.

The redis-bridge consumer (Phase 2 Task 5+) reads tenant:{id}:hermes_outbox
and performs the actual Telegram Bot API call under a single-owner Lua claim.
"""
import logging
import uuid
from typing import Any, Dict, Optional

from gateway.platforms.base import SendResult
from gateway.platforms.telegram import TelegramAdapter

from ._env import HERMES_OUTBOX_REDIS_URL_ENV, TENANT_ID_ENV, require_env
from .hermes_outbox_writer import write_to_outbox

log = logging.getLogger(__name__)


async def divert_telegram_to_outbox(
    redis_client: "Any",
    *,
    tenant_id: str,
    chat_id: str,
    content: str,
) -> SendResult:
    """Free function so tests don't have to instantiate the heavy parent class.

    Returns SendResult mirroring TelegramAdapter.send so hermes' state
    machine treats success/failure identically to a real Bot API call.
    """
    if not content or not content.strip():
        return SendResult(success=True, message_id=None)
    message_id = f"hermes-{uuid.uuid4()}"
    try:
        xadd_id = await write_to_outbox(
            redis_client,
            tenant_id=tenant_id,
            channel="telegram",
            phone=None,
            phone_number_id=None,
            chat_id=chat_id,  # Telegram chat_id is numeric
            text=content,
            message_id=message_id,
            source="hermes",
        )
    except Exception as exc:
        log.exception("hermes_outbox XADD failed (telegram)")
        return SendResult(success=False, error=str(exc))
    if isinstance(xadd_id, bytes):
        xadd_id = xadd_id.decode("utf-8")
    return SendResult(success=True, message_id=str(xadd_id))


class TelegramMindoraAdapter(TelegramAdapter):
    """Subclass that diverts outbound text to a Redis Stream.

    Reads two env vars (set by the deploy.yml service block in mindora-deploy):
      - TENANT_ID: tenant slug (testcust4, gft, ...)
      - HERMES_OUTBOX_REDIS_URL: redis://host:port/db
    """

    # Parent TelegramAdapter sets this True, which makes hermes' streaming
    # consumer call send() then edit_message() repeatedly with token deltas.
    # Those edit_message calls would bypass the outbox (POST Bot API directly)
    # AND choke on our XADD-style message_id ("1717603200000-0"). Disable.
    REQUIRES_EDIT_FINALIZE = False

    def __init__(self, *args, **kwargs):
        # redis.asyncio is imported lazily so unit tests of write_to_outbox
        # don't pull the redis-py dependency.
        import redis.asyncio as aioredis

        self._tenant_id = require_env(TENANT_ID_ENV)
        redis_url = require_env(HERMES_OUTBOX_REDIS_URL_ENV)
        self._redis_client = aioredis.from_url(redis_url)
        super().__init__(*args, **kwargs)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await divert_telegram_to_outbox(
            self._redis_client,
            tenant_id=self._tenant_id,
            chat_id=chat_id,
            content=content,
        )

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        # Sidecar invariant: all outbound text MUST flow through hermes_outbox.
        # The parent edit_message would otherwise POST Telegram Bot API directly,
        # bypassing the bridge's single-owner claim. We treat streaming edits as
        # no-ops; only the final send() is published to the outbox.
        log.debug(
            "mindora_edit_message_noop tenant=%s message_id=%s finalize=%s",
            getattr(self, "_tenant_id", "?"), message_id, finalize,
        )
        return SendResult(success=True, message_id=message_id)

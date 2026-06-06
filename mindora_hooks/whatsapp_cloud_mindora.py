"""Mindora subclass of WhatsAppCloudAdapter — XADDs to redis-bridge instead of POSTing Meta.

The redis-bridge consumer (Phase 2 Task 5+) reads tenant:{id}:hermes_outbox
and performs the actual Meta Cloud API POST under a single-owner Lua claim.
"""
import logging
import uuid
from typing import Any, Dict, Optional

from gateway.platforms.base import SendResult
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

from ._env import HERMES_OUTBOX_REDIS_URL_ENV, TENANT_ID_ENV, require_env
from ._media import install_outbound_blockers
from .hermes_outbox_writer import write_to_outbox

log = logging.getLogger(__name__)


async def divert_whatsapp_to_outbox(
    redis_client: "Any",
    *,
    tenant_id: str,
    chat_id: str,
    content: str,
    phone_number_id: str = "",
) -> SendResult:
    """Free function so tests don't have to instantiate the heavy parent class.

    Returns SendResult mirroring WhatsAppCloudAdapter.send so hermes' state
    machine treats success/failure identically to a real Meta POST.
    """
    if not content or not content.strip():
        return SendResult(success=True, message_id=None)
    message_id = f"hermes-{uuid.uuid4()}"
    try:
        xadd_id = await write_to_outbox(
            redis_client,
            tenant_id=tenant_id,
            channel="whatsapp",
            phone=chat_id,  # WhatsApp wa_id is the recipient phone
            phone_number_id=phone_number_id or "",
            chat_id=None,
            text=content,
            message_id=message_id,
            source="hermes",
        )
    except Exception as exc:
        log.exception("hermes_outbox XADD failed (whatsapp)")
        return SendResult(success=False, error=str(exc))
    if isinstance(xadd_id, bytes):
        xadd_id = xadd_id.decode("utf-8")
    return SendResult(success=True, message_id=str(xadd_id))


@install_outbound_blockers
class WhatsAppCloudMindoraAdapter(WhatsAppCloudAdapter):
    """Subclass that diverts outbound text to a Redis Stream.

    Reads two env vars (set by the deploy.yml service block in mindora-deploy):
      - TENANT_ID: tenant slug (testcust4, gft, ...)
      - HERMES_OUTBOX_REDIS_URL: redis://host:port/db

    The @install_outbound_blockers decorator walks the parent MRO at
    class-creation time and wraps every inherited send_* method (except
    send / edit_message / send_typing) to return a no-op success +
    WARNING log. Phase 2 outbox is text-only; non-text outbound (media,
    interactive prompts, drafts) is Phase 3.
    """

    # WhatsAppCloudAdapter inherits the base default (False), but pin it
    # explicitly so any future change in the parent doesn't silently re-enable
    # streaming edits on the sidecar (which would bypass hermes_outbox).
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
        return await divert_whatsapp_to_outbox(
            self._redis_client,
            tenant_id=self._tenant_id,
            chat_id=chat_id,
            content=content,
            phone_number_id=self._phone_number_id,
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
        # The base default returns success=False, which would force hermes to
        # treat every streaming edit as a failed send. Make it an explicit
        # success no-op so streaming consumers proceed to the final send().
        log.debug(
            "mindora_edit_message_noop tenant=%s message_id=%s finalize=%s",
            getattr(self, "_tenant_id", "?"), message_id, finalize,
        )
        return SendResult(success=True, message_id=message_id)

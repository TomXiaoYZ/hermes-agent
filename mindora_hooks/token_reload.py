"""Token reload-on-401 hook for hermes WhatsApp/Telegram adapters.

When hermes gets a 401 from Meta or Telegram:
1. Call TokenReloader.reload() to re-fetch the latest token from GCP Secret Manager
2. If re-fetch fails, the hook sets a Redis flag so control-ui can show a red prompt
3. On the next successful send, call handle_401(...outcome="success") to clear the flag
"""
import asyncio
import logging
from typing import Any, Optional

from google.api_core import exceptions as gax
from google.cloud import secretmanager

log = logging.getLogger(__name__)

_VALID_CHANNELS = ("whatsapp", "telegram")
_FLAG_SUFFIX = {
    "whatsapp": "meta_token_invalid",
    "telegram": "telegram_token_invalid",
}
_GCP_PROJECT = "agentplatform-492815"


class TokenReloader:
    def __init__(
        self,
        tenant_id: str,
        channel: str,
        sm_secret_name: str,
        # redis_client: typed as Any to avoid forcing redis-py import on test-only paths.
        redis_client: "Any",
    ):
        if channel not in _VALID_CHANNELS:
            raise ValueError(f"invalid channel: {channel!r} (expected one of {_VALID_CHANNELS})")
        self.tenant_id = tenant_id
        self.channel = channel
        self.sm_secret_name = sm_secret_name
        self.redis = redis_client
        # Lazy SM client — tests patch _fetch_sm directly.
        self._sm_client: Optional[secretmanager.SecretManagerServiceAsyncClient] = None
        self._sm_lock = asyncio.Lock()

    @property
    def flag_key(self) -> str:
        return f"tenant:{self.tenant_id}:{_FLAG_SUFFIX[self.channel]}"

    async def _fetch_sm(self) -> str:
        async with self._sm_lock:
            if self._sm_client is None:
                self._sm_client = secretmanager.SecretManagerServiceAsyncClient()
        name = f"projects/{_GCP_PROJECT}/secrets/{self.sm_secret_name}/versions/latest"
        resp = await self._sm_client.access_secret_version(request={"name": name})
        return resp.payload.data.decode("utf-8")

    async def reload(self) -> str:
        try:
            token = await self._fetch_sm()
            log.info(
                "token reload succeeded tenant=%s channel=%s",
                self.tenant_id, self.channel,
            )
            return token
        except (gax.NotFound, gax.PermissionDenied, gax.GoogleAPIError, OSError) as exc:
            log.error(
                "token reload failed tenant=%s channel=%s err=%s",
                self.tenant_id, self.channel, exc,
            )
            await self.redis.set(self.flag_key, "true")
            raise


# redis: typed as Any to avoid forcing redis-py import on test-only paths.
async def handle_401(redis: "Any", *, tenant: str, channel: str, outcome: str) -> None:
    """Clear the token_invalid Redis flag on the first successful send after recovery.

    `outcome` is "success" when the retried POST returned 2xx. Any other value is a no-op.
    """
    if outcome != "success":
        return
    if channel not in _VALID_CHANNELS:
        return
    flag_key = f"tenant:{tenant}:{_FLAG_SUFFIX[channel]}"
    if await redis.get(flag_key):
        await redis.delete(flag_key)
        log.info("cleared token_invalid flag tenant=%s channel=%s", tenant, channel)

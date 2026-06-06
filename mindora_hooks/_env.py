"""Env var name constants + actionable-error require helper for mindora hooks."""
import os

TENANT_ID_ENV = "TENANT_ID"
HERMES_OUTBOX_REDIS_URL_ENV = "HERMES_OUTBOX_REDIS_URL"


def require_env(name: str) -> str:
    """Read an env var, raising RuntimeError with an actionable message if unset."""
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(
            f"required env var {name!r} is not set; "
            f"set it in the crm-hermes container's deploy.yml service block"
        )
    return value

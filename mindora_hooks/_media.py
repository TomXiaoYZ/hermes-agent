"""Phase 2 invariant: ONLY send() and edit_message() flow through hermes_outbox.

All other inherited platform-side outbound methods (send_image, send_voice,
send_clarify, send_exec_approval, send_draft, send_model_picker, etc.)
bypass our text-only outbox contract. We auto-wrap them with a no-op +
WARNING log via the install_outbound_blockers class decorator below.

Phase 3 will design proper envelopes for media/interactive flows; until
then, missing functionality is loud-but-not-fatal.
"""
import inspect
import logging
from typing import Any, Set

log = logging.getLogger(__name__)

# Methods explicitly handled by Mindora subclasses (NOT auto-wrapped).
# - send: overridden to XADD outbox
# - edit_message: overridden to no-op SendResult
# - send_typing: passes through to platform (typing/read indicators are
#   ephemeral and don't carry user-visible content; not part of outbox contract)
_ALLOWLIST: Set[str] = {"send", "edit_message", "send_typing"}


def _blocked_method_call(adapter: Any, method_name: str, *args, **kwargs) -> Any:
    """Return a successful no-op SendResult and log the bypass attempt."""
    # Import here to avoid circular import at module load time.
    from gateway.platforms.base import SendResult

    tenant = getattr(adapter, "_tenant_id", "?")
    # Best-effort chat_id extraction — first positional or kw
    chat_id = "?"
    if args:
        chat_id = str(args[0])
    elif "chat_id" in kwargs:
        chat_id = str(kwargs["chat_id"])

    log.warning(
        "mindora_outbound_blocked tenant=%s adapter=%s method=%s chat_id=%s "
        "(Phase 2 text-only outbox; non-text outbound is Phase 3)",
        tenant, type(adapter).__name__, method_name, chat_id,
    )
    return SendResult(success=True, message_id=None)


# Backwards-compat alias for code/tests that may reference the old name.
_blocked_media_send = _blocked_method_call


def install_outbound_blockers(cls: type) -> type:
    """Class decorator that wraps every inherited send_* method (except allowlist).

    Applied to Mindora platform subclasses to neutralize any inherited
    outbound channel that isn't send() / edit_message() / send_typing().
    Walks the MRO above this class to find inherited methods we haven't
    explicitly overridden in the Mindora subclass body. Future upstream
    additions of send_<foo> are automatically caught.
    """
    own_attrs = set(cls.__dict__)  # methods defined ON the subclass directly
    seen: Set[str] = set()

    for base in cls.__mro__[1:]:  # skip cls itself; walk parents
        for name in dir(base):
            if not name.startswith("send_"):
                continue
            if name in _ALLOWLIST or name in own_attrs or name in seen:
                continue
            attr = getattr(base, name, None)
            # Only wrap async methods — defensive against any non-coroutine
            # class attribute that happens to start with "send_".
            if not inspect.iscoroutinefunction(attr):
                continue
            seen.add(name)

            # Build a wrapper that captures the method name; the default-arg
            # binding (_mn=name) freezes it so the closure doesn't see the
            # last value of the loop variable.
            method_name = name

            async def _wrapped(self, *args, _mn=method_name, **kwargs):
                return _blocked_method_call(self, _mn, *args, **kwargs)

            _wrapped.__name__ = method_name
            _wrapped.__qualname__ = f"{cls.__name__}.{method_name}"
            setattr(cls, method_name, _wrapped)

    return cls

"""Tests for MCP background-reconnect self-heal (never-give-up HTTP retry).

Spec: mindora-growth-os docs/specs/2026-06-13-hermes-mcp-reconnect-design.md
Prod incident: 2026-06-13 — a deploy recreated mcp-server and crm-hermes
simultaneously; hermes exhausted 3 initial retries in ~7s and ran with zero
MCP tools until manually restarted.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask
from tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _isolate_mcp_tool_server_map():
    """Snapshot/restore the module-level tool→server provenance map.

    _register_server_tools writes into mcp_tool._mcp_tool_server_names via
    _track_mcp_tool_server; without this, registrations leak across tests.
    """
    from tools import mcp_tool
    snapshot = dict(mcp_tool._mcp_tool_server_names)
    yield
    mcp_tool._mcp_tool_server_names.clear()
    mcp_tool._mcp_tool_server_names.update(snapshot)


def _make_mcp_tool(name: str, desc: str = ""):
    return SimpleNamespace(name=name, description=desc, inputSchema=None)


def _make_session(tools):
    """Mock MCP session whose list_tools() resolves to the given tools."""
    return SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=tools))
    )


class TestOnSessionReady:
    """Unit tests for the shared post-initialize helper."""

    @pytest.mark.asyncio
    async def test_normal_connect_skips_late_registration(self):
        """flag False -> hook inert; session + ready set; refresh untouched."""
        server = MCPServerTask("norm")
        session = _make_session([_make_mcp_tool("ping")])
        with patch.object(
            MCPServerTask, "_refresh_tools", new_callable=AsyncMock
        ) as mock_refresh:
            await server._on_session_ready(session)
        mock_refresh.assert_not_awaited()
        assert server.session is session
        assert server._ready.is_set()
        assert server._tools and server._tools[0].name == "ping"

    @pytest.mark.asyncio
    async def test_background_late_registration_fires_and_flag_sticky(self):
        """flag True + no names -> _refresh_tools registers; flag stays True."""
        server = MCPServerTask("bg")
        server._config = {"url": "http://mcp-test:1/sse"}
        server._background_pending = True
        session = _make_session([_make_mcp_tool("ping")])
        mock_registry = ToolRegistry()
        with patch("tools.registry.registry", mock_registry):
            await server._on_session_ready(session)
        assert "mcp_bg_ping" in server._registered_tool_names
        assert "mcp_bg_ping" in mock_registry.get_all_tool_names()
        # Sticky: NEVER cleared (spec §3.1 — clearing reopens the
        # double-registration race with the discovery coroutine).
        assert server._background_pending is True
        assert server._ready.is_set()

    @pytest.mark.asyncio
    async def test_already_registered_skips_refresh(self):
        """flag True but names present -> reconnect-after-blip; no refresh."""
        server = MCPServerTask("bg2")
        server._background_pending = True
        server._registered_tool_names = ["mcp_bg2_ping"]
        session = _make_session([_make_mcp_tool("ping")])
        with patch.object(
            MCPServerTask, "_refresh_tools", new_callable=AsyncMock
        ) as mock_refresh:
            await server._on_session_ready(session)
        mock_refresh.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refresh_failure_leaves_names_empty_for_retry(self):
        """_refresh_tools raising propagates; names stay empty so the next
        (re)connect attempt retries registration (spec §3.4)."""
        server = MCPServerTask("bg3")
        server._background_pending = True
        session = _make_session([_make_mcp_tool("ping")])
        with patch.object(
            MCPServerTask, "_refresh_tools",
            new_callable=AsyncMock, side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError):
                await server._on_session_ready(session)
        assert server._registered_tool_names == []
        assert server._background_pending is True


def _patch_run_http(monkeypatch, fail_times, tools=None):
    """Replace _run_http at class level with a fake transport.

    Fails `fail_times` times with ConnectionError, then connects: drives the
    REAL _on_session_ready (the code under test) and holds the connection
    until shutdown/reconnect, like a real transport.
    """
    calls = {"n": 0}
    tools = tools if tools is not None else [_make_mcp_tool("ping")]

    async def fake_run_http(self, config):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise ConnectionError(f"attempt {calls['n']} refused")
        session = _make_session(tools)
        self.initialize_result = MagicMock()
        await self._on_session_ready(session)
        await self._wait_for_lifecycle_event()

    monkeypatch.setattr(MCPServerTask, "_run_http", fake_run_http)
    return calls


def _fast_backoff(monkeypatch):
    monkeypatch.setattr("tools.mcp_tool._INITIAL_BACKOFF_SECONDS", 0.01)
    monkeypatch.setattr("tools.mcp_tool._MAX_BACKOFF_SECONDS", 0.02)


HTTP_CONFIG = {"url": "http://mcp-test:1/sse", "transport": "sse"}


class TestInitialConnectionBackgroundMode:

    @pytest.mark.asyncio
    async def test_http_exhaustion_enters_background_not_death(self, monkeypatch):
        """>3 initial failures on HTTP -> start() returns cleanly, task alive,
        flag set, _error None (spec §3.1 _error invariant)."""
        _fast_backoff(monkeypatch)
        _patch_run_http(monkeypatch, fail_times=10**9)
        server = MCPServerTask("bgmode")
        await server.start(dict(HTTP_CONFIG))   # must NOT raise
        assert server._background_pending is True
        assert server._error is None
        assert server._task is not None and not server._task.done()
        await asyncio.wait_for(server.shutdown(), timeout=5)

    @pytest.mark.asyncio
    async def test_late_connect_registers_tools(self, monkeypatch):
        """Spec test 1: 5 failures then success -> tools registered by the
        run loop without any external trigger; flag stays sticky."""
        _fast_backoff(monkeypatch)
        _patch_run_http(monkeypatch, fail_times=5)
        mock_registry = ToolRegistry()
        with patch("tools.registry.registry", mock_registry):
            server = MCPServerTask("bg_srv")
            await server.start(dict(HTTP_CONFIG))
            assert server._background_pending is True
            for _ in range(500):
                if server._registered_tool_names:
                    break
                await asyncio.sleep(0.01)
            assert "mcp_bg_srv_ping" in server._registered_tool_names
            assert "mcp_bg_srv_ping" in mock_registry.get_all_tool_names()
            assert server._background_pending is True
            await asyncio.wait_for(server.shutdown(), timeout=5)

    @pytest.mark.asyncio
    async def test_stdio_gives_up_unchanged(self, monkeypatch):
        """Spec test 3: stdio keeps today's behavior — start() raises."""
        _fast_backoff(monkeypatch)

        async def fake_run_stdio(self, config):
            raise ConnectionError("no such binary")

        monkeypatch.setattr(MCPServerTask, "_run_stdio", fake_run_stdio)
        server = MCPServerTask("stdio_srv")
        with pytest.raises(ConnectionError):
            await server.start({"command": "definitely-not-a-binary"})
        assert server._background_pending is False
        assert server._task.done()

    @pytest.mark.asyncio
    async def test_auth_error_fails_fast(self, monkeypatch):
        """Spec test 4: auth errors keep fail-fast — one attempt, raise."""
        _fast_backoff(monkeypatch)
        monkeypatch.setattr("tools.mcp_tool._is_auth_error", lambda exc: True)
        calls = _patch_run_http(monkeypatch, fail_times=10**9)
        server = MCPServerTask("auth_srv")
        with pytest.raises(ConnectionError):
            await server.start(dict(HTTP_CONFIG))
        assert calls["n"] == 1
        assert server._background_pending is False

    @pytest.mark.asyncio
    async def test_sleep_or_shutdown_wakes_on_shutdown(self):
        """Spec §3.1 shutdown-aware backoff: a 30s backoff wakes immediately
        when shutdown fires (no 10s cancel fallback needed)."""
        server = MCPServerTask("s")
        loop = asyncio.get_event_loop()

        async def set_soon():
            await asyncio.sleep(0.05)
            server._shutdown_event.set()

        t0 = loop.time()
        await asyncio.gather(server._sleep_or_shutdown(30.0), set_soon())
        assert loop.time() - t0 < 5.0

    @pytest.mark.asyncio
    async def test_shutdown_during_background_retry_is_prompt(self, monkeypatch):
        """Spec test 5/9: shutdown() of a background-retrying server
        completes promptly and the task ends without cancellation."""
        _fast_backoff(monkeypatch)
        _patch_run_http(monkeypatch, fail_times=10**9)
        server = MCPServerTask("bg_stop")
        await server.start(dict(HTTP_CONFIG))
        await asyncio.wait_for(server.shutdown(), timeout=5)
        assert server._task.done()

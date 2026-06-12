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


def _make_mcp_tool(name: str, desc: str = ""):
    return SimpleNamespace(name=name, description=desc, inputSchema=None)


def _make_session(tools):
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

"""Tests for the `vertex` provider alias added 2026-06-08.

The alias maps vertex/vertex-ai/google-vertex/google-vertex-ai to the
existing `custom` provider, reusing the ollama/vllm/llamacpp pattern.

End-to-end behavior under test:
  resolve_provider("vertex") -> "custom"   (alias normalisation)
  trust_check(base_url, "vertex") -> True   (alias-aware trust)
  resolve_runtime_provider with config.yaml provider:vertex+base_url ->
    runtime dict with provider=custom, base_url=<config base_url>.

Back-compat tests confirm openai/anthropic still resolve correctly.
"""
from unittest.mock import patch

import pytest

from hermes_cli.auth import resolve_provider
from hermes_cli.runtime_provider import (
    _config_base_url_trustworthy_for_bare_custom,
    resolve_runtime_provider,
)


# =============================================================================
# Auth-resolver layer: alias presence
# =============================================================================

class TestVertexAliasAuth:
    """resolve_provider() recognises the new vertex aliases."""

    @pytest.mark.parametrize("alias", [
        "vertex",
        "vertex-ai",
        "google-vertex",
        "google-vertex-ai",
    ])
    def test_alias_resolves_to_custom(self, alias):
        # Strip env vars so the env-presence detection at auth.py:1483-1500
        # cannot accidentally short-circuit. The alias map short-circuits
        # at auth.py:1450, then auth.py:1454-1455 returns "custom" directly.
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_provider(alias) == "custom"

    def test_back_compat_custom_still_short_circuits(self):
        # auth.py:1454-1455: "custom" returns "custom" directly.
        # Our alias additions normalise vertex/* → "custom" via the alias
        # map (line 1450) and then hit this same short-circuit. Confirms
        # the path the alias relies on is unchanged.
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_provider("custom") == "custom"

    def test_back_compat_anthropic_still_resolves(self):
        # auth.py:1456-1457: "anthropic" is in PROVIDER_REGISTRY → returns
        # "anthropic". Adding vertex alias does NOT touch this path.
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_provider("anthropic") == "anthropic"


# =============================================================================
# Trust-check layer: cfg_provider="vertex" must be trusted as custom-equivalent
# =============================================================================

class TestVertexTrustCheck:
    """runtime_provider._config_base_url_trustworthy_for_bare_custom alias-
    normalises cfg_provider via resolve_provider (lines 64-70). With the new
    alias, cfg_provider="vertex" + a non-loopback base_url should return True."""

    def test_vertex_base_url_trusted(self):
        # In-network docker-compose URL — non-loopback — would otherwise fail
        # the loopback check at line 73. Trusted only because resolve_provider
        # ("vertex") returns "custom" via the alias map (line 67).
        assert _config_base_url_trustworthy_for_bare_custom(
            "http://vertex-proxy:4010", "vertex"
        ) is True


# =============================================================================
# Integration: end-to-end runtime resolution with config.yaml fixture
# =============================================================================

class TestVertexRuntimeIntegration:
    """resolve_runtime_provider with config.yaml provider:vertex routes
    through the alias-to-custom path and returns base_url from config."""

    def test_runtime_routes_to_config_base_url(self, monkeypatch):
        # Stub _get_model_config to return a fixture matching what the
        # CRM-Hermes config.yaml will look like post-PR.
        fake_model_cfg = {
            "default": "gemini-3.5-flash",
            "provider": "vertex",
            "base_url": "http://vertex-proxy:4010",
            "api_key": "",
        }
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_model_config",
            lambda: fake_model_cfg,
        )
        # Strip env keys so the env-presence detection at auth.py:1483-1500
        # cannot accidentally short-circuit to "openrouter" or "anthropic".
        for var in [
            "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
            "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "GLM_API_KEY",
            "ZAI_API_KEY", "Z_AI_API_KEY", "KIMI_API_KEY",
            "DEEPSEEK_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)

        runtime = resolve_runtime_provider(requested="vertex")

        assert runtime["provider"] == "custom"
        assert runtime["base_url"] == "http://vertex-proxy:4010"
        # Per spec, the codebase's standard placeholder for unauthenticated
        # in-network endpoints is "no-key-required".
        assert runtime["api_key"] == "no-key-required"
        # api_mode default for OpenAI-compat endpoint:
        assert runtime["api_mode"] == "chat_completions"

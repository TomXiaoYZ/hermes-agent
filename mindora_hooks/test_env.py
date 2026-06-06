"""Tests for the env helper used by mindora hook adapters."""
import pytest

from mindora_hooks._env import (
    HERMES_OUTBOX_REDIS_URL_ENV,
    TENANT_ID_ENV,
    require_env,
)


def test_require_env_returns_value(monkeypatch):
    monkeypatch.setenv("MINDORA_TEST_VAR", "hello")
    assert require_env("MINDORA_TEST_VAR") == "hello"


def test_require_env_raises_on_unset(monkeypatch):
    monkeypatch.delenv("MINDORA_TEST_VAR", raising=False)
    with pytest.raises(RuntimeError, match="required env var 'MINDORA_TEST_VAR'"):
        require_env("MINDORA_TEST_VAR")


def test_require_env_raises_on_empty(monkeypatch):
    monkeypatch.setenv("MINDORA_TEST_VAR", "")
    with pytest.raises(RuntimeError, match="required env var 'MINDORA_TEST_VAR'"):
        require_env("MINDORA_TEST_VAR")


def test_constants_match_documented_names():
    assert TENANT_ID_ENV == "TENANT_ID"
    assert HERMES_OUTBOX_REDIS_URL_ENV == "HERMES_OUTBOX_REDIS_URL"

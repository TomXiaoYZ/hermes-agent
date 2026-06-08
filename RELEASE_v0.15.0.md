# Hermes Agent v0.15.0

**Release Date:** 2026-06-08
**Since v0.14.0:** Targeted alias addition for Vertex AI via OpenAI-compat proxy.

## What changed

- **Vertex AI provider alias.** `vertex`, `vertex-ai`, `google-vertex`, and
  `google-vertex-ai` now resolve to the existing `custom` provider pipeline
  (see `hermes_cli/auth.py` `_PROVIDER_ALIASES`). Use these in:
  - `HERMES_INFERENCE_PROVIDER=vertex` env var, or
  - `model.provider: vertex` in `~/.hermes/config.yaml`.

  Combined with `model.base_url: http://your-vertex-proxy:port` in
  config.yaml, the agent routes OpenAI-format chat-completions requests to
  any OpenAI-compatible Vertex proxy (e.g. a sidecar that fronts Vertex's
  OpenAI-compat endpoint for Gemini and translates for Anthropic Claude on
  Vertex).

- This reuses the existing `ollama` / `vllm` / `llamacpp` pattern; no new
  ProviderConfig, no new HTTP client. The runtime trust check at
  `runtime_provider.py:64-70` was already alias-aware via
  `resolve_provider(cfg_provider_norm) == "custom"`, so adding the alias
  is sufficient for end-to-end routing.

## Why

Lets fork users running their own OpenAI-compat proxy in front of Vertex AI
declare it explicitly in config.yaml without resorting to provider-presence
"bypass token" env vars (e.g. setting `ANTHROPIC_API_KEY` to a bogus value
just to satisfy `resolve_provider`'s env-detection path). The new aliases
are purely additive — existing `provider: custom` deployments are
unaffected.

## Migration

If you were already running config.yaml with `provider: custom` +
`base_url: http://vertex-proxy:port`, no action required — keep it. The
alias is for clarity in new deployments.

If you were using a bypass-token env var (`ANTHROPIC_API_KEY=<bogus>`) to
get past `resolve_provider`, you can now drop it and set
`HERMES_INFERENCE_PROVIDER=vertex` instead.

## Tests

`tests/hermes_cli/test_auth_vertex_alias.py` — 8 tests covering alias
presence (4 parametrized aliases), back-compat (custom short-circuit and
anthropic resolution), trust-check at the runtime layer, and an
integration test against a fake config.yaml fixture.

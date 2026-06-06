#!/usr/bin/env bash
set -euo pipefail
command -v jq >/dev/null || { echo "ERROR: jq required (brew install jq)" >&2; exit 1; }
DOCKER_CFG="${DOCKER_CONFIG:-$HOME/.docker}/config.json"
if [ ! -f "$DOCKER_CFG" ] || ! jq -e '.credHelpers["asia-southeast1-docker.pkg.dev"]' "$DOCKER_CFG" >/dev/null 2>&1; then
  echo "ERROR: GAR credential helper not configured. Run:" >&2
  echo "  gcloud auth configure-docker asia-southeast1-docker.pkg.dev" >&2
  exit 1
fi
cd "$(dirname "$0")/.."
SHA=$(git rev-parse --short HEAD)
IMG="asia-southeast1-docker.pkg.dev/agentplatform-492815/mindora/hermes"
if ! git diff --quiet HEAD || ! git diff --cached --quiet; then
  echo "WARNING: uncommitted changes — tag $SHA reflects HEAD only, not working tree" >&2
fi
TAG="${1:-$SHA}"
echo "Building $IMG:$TAG from crm-hermes Dockerfile..."
docker buildx build --push --platform linux/amd64 -f crm-hermes -t "$IMG:$TAG" .
echo "Pushed $IMG:$TAG"

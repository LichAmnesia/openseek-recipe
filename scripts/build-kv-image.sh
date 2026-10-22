#!/usr/bin/env bash
# Builds the KV-offload image: the pinned recipe image + the LMCache wheel, nothing else.
# Same vLLM fork, torch, EXL3/PLE patches and weights; only lmcache is added (--no-deps, so
# pip cannot swap torch or vLLM). Safe to re-run; FORCE=1 rebuilds.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_recipe
base="$(recipe_image)"
docker image inspect "$base" >/dev/null 2>&1 || die "Base image $base not found. Run scripts/setup.sh first."

if docker image inspect "$KV_IMAGE" >/dev/null 2>&1 && [[ "${FORCE:-0}" != 1 ]]; then
  ok "$KV_IMAGE already present ($(docker image inspect -f '{{.Id}}' "$KV_IMAGE" | cut -c1-19))"
  exit 0
fi

ctx="$(mktemp -d)"
# The wheel is written by root inside the container; clean it up the same way.
trap 'docker run --rm -v "$ctx:/ctx" --entrypoint sh "$base" -c "rm -rf /ctx/*" >/dev/null 2>&1; rm -rf "$ctx"' EXIT
log "Downloading lmcache==$LMCACHE_VERSION wheel inside $base (matches its Python/ABI)"
docker run --rm -v "$ctx:/out" -e HOME=/tmp --entrypoint python3 "$base" \
  -m pip download --quiet --no-deps --only-binary=:all: "lmcache==$LMCACHE_VERSION" -d /out
wheel="$(cd "$ctx" && ls lmcache-*.whl)" || die "wheel download failed"
ok "$wheel"

cat > "$ctx/Dockerfile" <<EOF
FROM $base
COPY $wheel /tmp/
RUN python3 -m pip install --no-deps /tmp/$wheel && rm /tmp/$wheel
EOF
log "Building $KV_IMAGE FROM $base"
docker build --quiet -t "$KV_IMAGE" "$ctx" >/dev/null

log "Verifying imports (native CUDA ops + MP connector + scratch-group support)"
docker run --rm --gpus "device=$GPU" -e CUDA_VISIBLE_DEVICES= -e LMCACHE_DISABLE_BANNER=1 \
  --entrypoint python3 "$KV_IMAGE" -c '
import importlib.metadata as md, inspect, torch
import lmcache.cuda_ops, lmcache.lmcache_native
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
import lmcache.integration.vllm.kv_cache_groups as g
assert "circular" in inspect.getsource(g).lower(), "this lmcache has no QSA scratch-group filter (needs >= 0.5.6)"
print("lmcache", md.version("lmcache"), "torch", torch.__version__, "vllm", md.version("vllm"))
' 2>&1 | grep -v ' INFO' | tail -1
ok "$KV_IMAGE $(docker image inspect -f '{{.Id}}' "$KV_IMAGE" | cut -c1-19) (base $(docker image inspect -f '{{.Id}}' "$base" | cut -c1-19))"

#!/usr/bin/env bash
# Shared config loader. Sourced by every script; not meant to be run directly.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export QUANT="${QUANT:-exl3-ple8}"
export PORT="${PORT:-8001}"
export GPU="${GPU:-0}"
export HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
export MTP_TOKENS="${MTP_TOKENS:-3}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
RECIPE_REPO="${RECIPE_REPO:-https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next.git}"
RECIPE_COMMIT="${RECIPE_COMMIT:-847f45b1730069eab4bdb5e29361f9ea94ac5e05}"
RECIPE_DIR="${RECIPE_DIR:-./vendor/sm12x-exl3-qwen3.8-flash-next}"
[[ "$RECIPE_DIR" = /* ]] || RECIPE_DIR="$REPO_ROOT/${RECIPE_DIR#./}"

# KV cache RAM tier (LMCache MP). docs/kv-ram-offload.md
KV_RAM_GB="${KV_RAM_GB:-64}"
KV_RAM_WATERMARK="${KV_RAM_WATERMARK:-0.95}"
KV_BLOCK_TOKENS="${KV_BLOCK_TOKENS:-3200}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$((2 * KV_BLOCK_TOKENS - 1))}"
LMCACHE_VERSION="${LMCACHE_VERSION:-0.5.6}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-6580}"
LMCACHE_METRICS_PORT="${LMCACHE_METRICS_PORT:-6590}"
LMCACHE_CONTAINER="qwen38-lmcache"
# Recipe image + LMCache wheel, built locally by scripts/build-kv-image.sh.
KV_IMAGE="${KV_IMAGE:-qwen38-rtx:lmc${LMCACHE_VERSION//./}}"

# Upstream naming: container and served model are both qwen38-$QUANT.
CONTAINER_NAME="qwen38-${QUANT}"
MODEL_ID="qwen38-${QUANT}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mfail\033[0m %s\n' "$*" >&2; exit 1; }

require_free_gpu() {
  local used
  used="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | xargs)"
  (( used < 2000 )) || die "GPU $GPU already has ${used} MiB in use. Free it first; the server claims ${GPU_MEMORY_UTILIZATION} of VRAM."
}

recipe_image() { (source "$RECIPE_DIR/platform-config.sh"; echo "$DEFAULT_IMAGE"); }

require_kv_image() {
  docker image inspect "$KV_IMAGE" >/dev/null 2>&1 || die "Image $KV_IMAGE not found. Run scripts/build-kv-image.sh (scripts/setup.sh does it too)."
}

require_recipe() {
  [[ -x "$RECIPE_DIR/start.sh" ]] || die "Upstream recipe not found at $RECIPE_DIR. Run scripts/setup.sh first."
}

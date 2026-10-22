#!/usr/bin/env bash
# Shared config loader. Sourced by every script; not meant to be run directly.
set -euo pipefail

# Legacy (no KV offload) copy. .env is read from legacy/no-offload/, the upstream recipe
# clone is shared with the main scripts at <repo>/vendor/.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TOP_ROOT="$(cd -- "$REPO_ROOT/../.." && pwd)"

# Shared settings (GPU, PORT, HF_CACHE, RECIPE_DIR, ...) from the main .env, then this
# folder's own .env on top. KV_* settings in the main .env are ignored here.
for env_file in "$TOP_ROOT/.env" "$REPO_ROOT/.env"; do
  [[ -f "$env_file" ]] || continue
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
done

export QUANT="${QUANT:-exl3-ple8}"
export PORT="${PORT:-8001}"
export GPU="${GPU:-0}"
export HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
export MTP_TOKENS="${MTP_TOKENS:-3}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
RECIPE_REPO="${RECIPE_REPO:-https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next.git}"
RECIPE_COMMIT="${RECIPE_COMMIT:-847f45b1730069eab4bdb5e29361f9ea94ac5e05}"
RECIPE_DIR="${RECIPE_DIR:-./vendor/sm12x-exl3-qwen3.8-flash-next}"
[[ "$RECIPE_DIR" = /* ]] || RECIPE_DIR="$TOP_ROOT/${RECIPE_DIR#./}"

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

require_recipe() {
  [[ -x "$RECIPE_DIR/start.sh" ]] || die "Upstream recipe not found at $RECIPE_DIR. Run scripts/setup.sh first."
}

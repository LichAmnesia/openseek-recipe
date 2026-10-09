#!/usr/bin/env bash
# Runs the upstream benchmark suite against the local server (stdlib Python only).
#   PART=core     tools 16/16, vision 1/4/16, seven workloads, orchid, C1..C16 decode (~4 min)
#   PART=context  depth sweep up to the 261,888-token boundary (long)
#   PART=all      everything (very long)
# Run on an otherwise idle server; any other traffic pollutes the numbers.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_recipe
curl -sf --max-time 3 "$BASE_URL/v1/models" >/dev/null || die "server not up (scripts/ctl.sh start)"

PART="${PART:-core}"
out="${RESULT_DIR:-$REPO_ROOT/results/$(date +%Y-%m-%d-%H%M)-local-$PART}"
gpu="$(nvidia-smi -i "$GPU" --query-gpu=name,power.max_limit --format=csv,noheader)"
log "PART=$PART  -> $out"
log "GPU: $gpu"

mkdir -p "$(dirname "$out")"
QUANT="$QUANT" MTP_TOKENS="$MTP_TOKENS" PART="$PART" BASE_URL="$BASE_URL" RESULT_DIR="$out" \
  bash "$RECIPE_DIR/scripts/benchmark-suite.sh"
echo "$gpu" > "$out/gpu.txt"
python3 "$REPO_ROOT/scripts/summarize.py" "$out"

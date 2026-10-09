#!/usr/bin/env bash
# Clones the upstream recipe at a pinned commit, pulls the pinned SM120 image,
# and downloads the checkpoint. Safe to re-run: each step is skipped if done.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

log "Upstream recipe @ ${RECIPE_COMMIT:0:12}"
if [[ ! -d "$RECIPE_DIR/.git" ]]; then
  mkdir -p "$(dirname "$RECIPE_DIR")"
  git clone "$RECIPE_REPO" "$RECIPE_DIR"
fi
git -C "$RECIPE_DIR" fetch --quiet origin || warn "fetch failed; using local clone"
git -C "$RECIPE_DIR" -c advice.detachedHead=false checkout --quiet "$RECIPE_COMMIT"
ok "$(git -C "$RECIPE_DIR" log -1 --format='%h %cs %s')"

log "Container image (~29 GB on disk)"
image="$(source "$RECIPE_DIR/platform-config.sh"; echo "$DEFAULT_IMAGE")"
if docker image inspect "$image" >/dev/null 2>&1; then
  ok "$image already present"
else
  bash "$RECIPE_DIR/pull.sh"
  ok "pulled and tagged $image"
fi

log "Checkpoint for QUANT=$QUANT (exl3-ple8 is ~120 GB, 22 shards)"
mkdir -p "$HF_CACHE"
( source "$RECIPE_DIR/model-profiles.sh"
  if [[ -f "$HF_CACHE/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/config.json" ]] \
     && ! find "$HF_CACHE/hub/$MODEL_CACHE_NAME/blobs" -name '*.incomplete' 2>/dev/null | grep -q .; then
    ok "$MODEL_REPO@${MODEL_REVISION:0:12} already downloaded"
  else
    echo "Downloading $MODEL_REPO@${MODEL_REVISION:0:12} -> $HF_CACHE (resumable; re-run if interrupted)"
    QUANT="$QUANT" HF_CACHE="$HF_CACHE" bash "$RECIPE_DIR/download.sh"
  fi )

ok "Setup done. Next: scripts/ctl.sh start"

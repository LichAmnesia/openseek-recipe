#!/usr/bin/env bash
# Clones the upstream recipe at a pinned commit, pulls the pinned SM120 image,
# and downloads the checkpoint. Safe to re-run: each step is skipped if done.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

log "Upstream recipe @ ${RECIPE_COMMIT:0:12}"
if [[ ! -d "$RECIPE_DIR/.git" ]]; then
  mkdir -p "$(dirname "$RECIPE_DIR")"
  git clone "$RECIPE_REPO" "$RECIPE_DIR"
fi
git -C "$RECIPE_DIR" diff --quiet HEAD -- || die "Upstream checkout has tracked changes; preserve them before setup."
if ! git -C "$RECIPE_DIR" cat-file -e "$RECIPE_COMMIT^{commit}" 2>/dev/null; then
  git -C "$RECIPE_DIR" fetch --quiet origin "$RECIPE_COMMIT"
fi
git -C "$RECIPE_DIR" -c advice.detachedHead=false checkout --quiet "$RECIPE_COMMIT"
ok "$(git -C "$RECIPE_DIR" log -1 --format='%h %cs %s')"
require_recipe

log "Container image (~29 GB on disk)"
image="$(recipe_image)"
if docker image inspect "$image" >/dev/null 2>&1; then
  ok "$image already present"
else
  if [[ -n "${BASE_IMAGE:-}" ]]; then docker pull "$BASE_IMAGE"
  else bash "$RECIPE_DIR/pull.sh"; fi
  ok "pulled and tagged $image"
fi

log "KV-offload image (recipe image + lmcache $LMCACHE_VERSION)"
bash "$REPO_ROOT/scripts/build-kv-image.sh"

log "Checkpoint for QUANT=$QUANT (exl3-ple8 is ~128 GB decimal, 22 weight shards)"
mkdir -p "$HF_CACHE"
if [[ -n "${MODEL_LOCK:-}" ]]; then
  log "Checking the complete pinned snapshot, then SHA256 of every file (can take several minutes)"
  docker run --rm --user "$(id -u):$(id -g)" \
    -e HF_HOME=/hf -e HF_TOKEN -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$HF_CACHE:/hf" -v "$REPO_ROOT:/openseek:ro" -w /openseek \
    --entrypoint python3 "$image" -m recipe_lib.artifacts download \
    --lock "/openseek/locks/$(basename "$MODEL_LOCK")" --cache /hf
else
 ( source "$RECIPE_DIR/model-profiles.sh"
  if [[ -f "$HF_CACHE/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/config.json" ]] \
     && ! find "$HF_CACHE/hub/$MODEL_CACHE_NAME/blobs" -name '*.incomplete' 2>/dev/null | grep -q .; then
    ok "$MODEL_REPO@${MODEL_REVISION:0:12} already downloaded"
  else
    echo "Downloading $MODEL_REPO@${MODEL_REVISION:0:12} -> $HF_CACHE (resumable; re-run if interrupted)"
    QUANT="$QUANT" HF_CACHE="$HF_CACHE" bash "$RECIPE_DIR/download.sh"
  fi )
fi

ok "Setup done. Next: ./recipe start"

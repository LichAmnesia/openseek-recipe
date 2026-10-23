#!/usr/bin/env bash
# start | stop | restart | recreate | down | status | logs [lmcache] | wait
# Two containers: qwen38-lmcache (LMCache MP server, the KV RAM tier) and the vLLM server.
# vLLM registers its KV layout with LMCache at startup, so LMCache always starts first and a
# restart of LMCache alone means vLLM must be restarted too.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

exists()  { docker container inspect "${1:-$CONTAINER_NAME}" >/dev/null 2>&1; }
running() { [[ "$(docker container inspect -f '{{.State.Running}}' "${1:-$CONTAINER_NAME}" 2>/dev/null)" == true ]]; }

validate_locked_containers() {
  [[ "${OPENSEEK_RESOLVED:-0}" == 1 ]] || return 0
  require_recipe; require_kv_image; require_checkpoint
  local expected c
  expected="$(docker image inspect -f '{{.Id}}' "$KV_IMAGE")"
  for c in "$CONTAINER_NAME" "$LMCACHE_CONTAINER"; do
    if exists "$c"; then
      [[ "$(docker container inspect -f '{{.Image}}' "$c")" == "$expected" ]] \
        || die "$c uses a different image. Use ./recipe recreate to apply the selected lock."
    fi
  done
}

wait_ready() {
  local timeout="${READY_TIMEOUT:-900}" start=$SECONDS
  log "Waiting for $BASE_URL/v1/models (first start: weight load + CUDA graph capture, ~3-5 min)"
  until curl -sf --max-time 3 "$BASE_URL/v1/models" >/dev/null; do
    running || { docker logs --tail 40 "$CONTAINER_NAME" >&2; die "container exited during startup"; }
    (( SECONDS - start < timeout )) || die "not ready after ${timeout}s (scripts/ctl.sh logs)"
    sleep 10
  done
  ok "ready in $((SECONDS - start))s: $(curl -s "$BASE_URL/v1/models" | python3 -c 'import json,sys; m=json.load(sys.stdin)["data"][0]; print(m["id"], "max_model_len", m.get("max_model_len"))')"
  docker logs "$CONTAINER_NAME" 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1 | sed 's/^/     /'
  docker logs "$CONTAINER_NAME" 2>&1 | grep -oE 'group_tokens_per_block=\[[0-9, ]+\]' | tail -1 | sed 's/^/     LMCache /'
}

wait_lmcache() {
  local start=$SECONDS
  until docker logs "$LMCACHE_CONTAINER" 2>&1 | grep -q 'cache server is running'; do
    running "$LMCACHE_CONTAINER" || { docker logs --tail 30 "$LMCACHE_CONTAINER" >&2; die "LMCache server exited"; }
    (( SECONDS - start < 120 )) || die "LMCache server not up after 120s (scripts/ctl.sh logs lmcache)"
    sleep 2
  done
  ok "LMCache $LMCACHE_VERSION on localhost:$LMCACHE_PORT, RAM tier ${KV_RAM_GB} GiB"
}

create_lmcache() {
  exists "$LMCACHE_CONTAINER" && docker rm -f "$LMCACHE_CONTAINER" >/dev/null
  log "Creating $LMCACHE_CONTAINER (RAM tier ${KV_RAM_GB} GiB, chunk ${KV_BLOCK_TOKENS}, watermark ${KV_RAM_WATERMARK})"
  docker run -d --name "$LMCACHE_CONTAINER" --network host --ipc host --gpus "device=$GPU" \
    --restart "$RESTART_POLICY" -e LMCACHE_DISABLE_BANNER=1 --entrypoint lmcache "$KV_IMAGE" server \
    --host localhost --port "$LMCACHE_PORT" \
    --http-host 127.0.0.1 --http-port "$LMCACHE_HTTP_PORT" --prometheus-port "$LMCACHE_METRICS_PORT" \
    --chunk-size "$KV_BLOCK_TOKENS" --separate-object-groups \
    --l1-size-gb "$KV_RAM_GB" --eviction-policy LRU --eviction-trigger-watermark "$KV_RAM_WATERMARK" >/dev/null
  wait_lmcache
}

create() {
  require_recipe
  require_kv_image
  exists "$LMCACHE_CONTAINER" && docker rm -f "$LMCACHE_CONTAINER" >/dev/null
  require_free_gpu
  (( MAX_BATCHED_TOKENS >= KV_BLOCK_TOKENS )) || die "MAX_BATCHED_TOKENS ($MAX_BATCHED_TOKENS) must be >= KV_BLOCK_TOKENS ($KV_BLOCK_TOKENS)"
  create_lmcache
  log "Creating $CONTAINER_NAME (QUANT=$QUANT MTP=$MTP_TOKENS ctx=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS batch=$MAX_BATCHED_TOKENS port=$PORT image=$KV_IMAGE)"
  # The explicit module path makes vLLM load LMCache's connector, not vLLM's bundled older
  # copy that only works with the hybrid KV manager off.
  local kv_cfg
  kv_cfg="$(printf '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":%s}}' "$LMCACHE_PORT")"
  IMAGE="$KV_IMAGE" bash "$RECIPE_DIR/start.sh" \
    --enable-prefix-caching --mamba-cache-mode align --kv-transfer-config "$kv_cfg" "$@" >/dev/null
  docker update --restart "$RESTART_POLICY" "$CONTAINER_NAME" >/dev/null
  ok "restart policy: $RESTART_POLICY (both containers)"
}

down() {
  exists && { log "Removing $CONTAINER_NAME"; docker rm -f "$CONTAINER_NAME" >/dev/null; }
  exists "$LMCACHE_CONTAINER" && { log "Removing $LMCACHE_CONTAINER"; docker rm -f "$LMCACHE_CONTAINER" >/dev/null; }
  return 0
}

cmd="${1:-status}"; shift || true
case "$cmd" in
  start)
    validate_locked_containers
    if running && running "$LMCACHE_CONTAINER"; then
      ok "$CONTAINER_NAME + $LMCACHE_CONTAINER already running (use recreate to apply configuration changes)"
    elif exists && exists "$LMCACHE_CONTAINER"; then
      docker stop "$CONTAINER_NAME" "$LMCACHE_CONTAINER" >/dev/null
      require_free_gpu; log "Starting existing $LMCACHE_CONTAINER + $CONTAINER_NAME"
      docker start "$LMCACHE_CONTAINER" >/dev/null; wait_lmcache
      docker start "$CONTAINER_NAME" >/dev/null
    else down; create "$@"; fi
    wait_ready ;;
  recreate)
    # Serve flags are baked into the containers at creation. Use this after editing .env.
    require_recipe; require_kv_image; require_checkpoint
    down; create "$@"; wait_ready ;;
  stop)
    exists && docker stop "$CONTAINER_NAME" >/dev/null
    exists "$LMCACHE_CONTAINER" && docker stop "$LMCACHE_CONTAINER" >/dev/null
    ok "stopped (VRAM and the ${KV_RAM_GB} GiB RAM tier released)" ;;
  restart)
    validate_locked_containers
    docker stop "$CONTAINER_NAME" >/dev/null
    docker restart "$LMCACHE_CONTAINER" >/dev/null; wait_lmcache
    docker start "$CONTAINER_NAME" >/dev/null; wait_ready ;;
  down)    down; ok "both containers removed" ;;
  wait)    wait_ready ;;
  logs)
    if [[ "${1:-}" == lmcache ]]; then docker logs --tail "${TAIL:-100}" -f "$LMCACHE_CONTAINER"
    else docker logs --tail "${TAIL:-100}" -f "$CONTAINER_NAME"; fi ;;
  status)
    for c in "$LMCACHE_CONTAINER" "$CONTAINER_NAME"; do
      if exists "$c"; then docker ps -a --filter "name=^${c}$" --format '{{.Names}}\t{{.Status}}\t{{.Image}}'
      else echo "$c: not created"; fi
    done
    if curl -sf --max-time 3 "$BASE_URL/v1/models" >/dev/null; then
      ok "API up at $BASE_URL/v1"
      curl -s --max-time 3 "$BASE_URL/metrics" | awk '
        /^vllm:kv_cache_usage_perc\{/ {u=$2}
        /^vllm:external_prefix_cache_queries_total\{/ {q+=$2} /^vllm:external_prefix_cache_hits_total\{/ {h+=$2}
        /^vllm:prefix_cache_queries_total\{/ {gq+=$2} /^vllm:prefix_cache_hits_total\{/ {gh+=$2}
        END {printf "     GPU KV usage %.1f%% | GPU prefix hits %.1f%% | RAM tier hits %.1f%% (%d tokens)\n",
             u*100, gq?gh/gq*100:0, q?h/q*100:0, h}'
    else warn "API not answering at $BASE_URL/v1"; fi
    nvidia-smi -i "$GPU" --query-gpu=memory.used,memory.total,power.draw,temperature.gpu --format=csv
    free -g | sed -n 1,2p ;;
  *) die "usage: $0 {start|stop|restart|recreate|down|status|logs [lmcache]|wait} [extra vllm args]" ;;
esac

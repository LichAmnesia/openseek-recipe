#!/usr/bin/env bash
# start | stop | restart | recreate | status | logs | wait
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

exists()  { docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; }
running() { [[ "$(docker container inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }

wait_ready() {
  local timeout="${READY_TIMEOUT:-900}" start=$SECONDS
  log "Waiting for $BASE_URL/v1/models (first start: weight load + CUDA graph capture, ~3-5 min)"
  until curl -sf --max-time 3 "$BASE_URL/v1/models" >/dev/null; do
    running || { docker logs --tail 40 "$CONTAINER_NAME" >&2; die "container exited during startup"; }
    (( SECONDS - start < timeout )) || die "not ready after ${timeout}s (scripts/ctl.sh logs)"
    sleep 10
  done
  ok "ready in $((SECONDS - start))s: $(curl -s "$BASE_URL/v1/models" | python3 -c 'import json,sys; m=json.load(sys.stdin)["data"][0]; print(m["id"], "max_model_len", m.get("max_model_len"))')"
}

create() {
  require_recipe
  # Switching back from the main (KV offload) setup: its LMCache server pins host RAM and VRAM.
  docker container inspect qwen38-lmcache >/dev/null 2>&1 && { log "Removing qwen38-lmcache (KV offload server)"; docker rm -f qwen38-lmcache >/dev/null; }
  require_free_gpu
  log "Creating $CONTAINER_NAME (QUANT=$QUANT MTP=$MTP_TOKENS ctx=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS port=$PORT)"
  bash "$RECIPE_DIR/start.sh" "$@" >/dev/null
  docker update --restart "$RESTART_POLICY" "$CONTAINER_NAME" >/dev/null
  ok "restart policy: $RESTART_POLICY"
}

cmd="${1:-status}"; shift || true
case "$cmd" in
  start)
    if running; then ok "$CONTAINER_NAME already running"
    elif exists; then require_free_gpu; log "Starting existing $CONTAINER_NAME"; docker start "$CONTAINER_NAME" >/dev/null
    else create "$@"; fi
    wait_ready ;;
  recreate)
    # Serve flags are baked into the container at creation. Use this after editing .env.
    exists && { log "Removing $CONTAINER_NAME"; docker rm -f "$CONTAINER_NAME" >/dev/null; }
    create "$@"; wait_ready ;;
  stop)    docker stop "$CONTAINER_NAME" >/dev/null && ok "stopped (VRAM released)" ;;
  restart) docker restart "$CONTAINER_NAME" >/dev/null; wait_ready ;;
  wait)    wait_ready ;;
  logs)    docker logs --tail "${TAIL:-100}" -f "$CONTAINER_NAME" ;;
  status)
    if exists; then docker ps -a --filter "name=^${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
    else echo "$CONTAINER_NAME: not created"; fi
    curl -sf --max-time 3 "$BASE_URL/v1/models" >/dev/null && ok "API up at $BASE_URL/v1" || warn "API not answering at $BASE_URL/v1"
    nvidia-smi -i "$GPU" --query-gpu=memory.used,memory.total,power.draw,temperature.gpu --format=csv ;;
  *) die "usage: $0 {start|stop|restart|recreate|status|logs|wait} [extra vllm args]" ;;
esac

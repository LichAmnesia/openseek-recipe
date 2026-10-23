#!/usr/bin/env bash
# Read-only preflight. Changes nothing; tells you what is missing.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
fails=0
bad() { printf '\033[1;31m  x\033[0m %s\n' "$*"; fails=$((fails + 1)); }

log "Host"
[[ "$(uname -s)" == Linux ]] && ok "Linux" || bad "Linux required (found $(uname -s))"
case "$(uname -m)" in
  x86_64|amd64) ok "x86_64" ;;
  *) bad "This guide targets x86_64 RTX hosts. For DGX Spark use the upstream recipe directly." ;;
esac

log "GPU"
if command -v nvidia-smi >/dev/null; then
  IFS=',' read -r name mem drv power < <(nvidia-smi -i "$GPU" --query-gpu=name,memory.total,driver_version,power.max_limit --format=csv,noheader,nounits | head -1)
  name="$(echo "$name" | xargs)"; mem="$(echo "$mem" | xargs)"; drv="$(echo "$drv" | xargs)"; power="$(echo "$power" | xargs)"
  ok "GPU $GPU: $name, ${mem} MiB, driver $drv, power cap ${power} W"
  [[ "$name" == *"RTX PRO 6000"* ]] || warn "Not an RTX PRO 6000. The pinned image targets SM120 with 96 GB; other cards are untested."
  (( ${mem%.*} >= ${MIN_GPU_VRAM_MIB:-95000} )) || bad "Need ~96 GB VRAM (found ${mem} MiB)"
  [[ "$name" == *"Max-Q"* ]] && warn "Max-Q edition: expect ~10% lower decode than the 400 W reference (see docs/benchmarks.md)."
  used="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | xargs)"
  (( used < 2000 )) || warn "${used} MiB VRAM already in use. The server wants 94% of the card; stop other GPU jobs first."
else
  bad "nvidia-smi not found (install the NVIDIA driver)"
fi

log "Memory and disk"
ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
# PLE page cache (~35-48 GiB) + OS/vLLM (~10 GiB) + pinned KV RAM tier.
need_gb=$(( KV_RAM_GB + 56 ))
(( need_gb >= ${MIN_HOST_RAM_GIB:-0} )) || need_gb="$MIN_HOST_RAM_GIB"
if (( ram_gb >= need_gb )); then ok "RAM ${ram_gb} GiB (KV RAM tier ${KV_RAM_GB} GiB + ~56 GiB PLE/system)"
elif (( ram_gb >= 64 )); then bad "RAM ${ram_gb} GiB is too small for KV_RAM_GB=${KV_RAM_GB} (needs ~${need_gb}). Use legacy/no-offload/ (see its README), or lower KV_RAM_GB; below ~1.5x the GPU KV pool (~48 GiB) the tier barely hits."
else bad "RAM ${ram_gb} GiB. Need at least 64 GiB free for the PLE table."; fi
disk_path="$HF_CACHE"
while [[ ! -d "$disk_path" && "$disk_path" != / ]]; do disk_path="$(dirname "$disk_path")"; done
free_gb=$(df -BG --output=avail "$disk_path" 2>/dev/null | tail -1 | tr -dc '0-9')
snap="$HF_CACHE/hub/models--wrldsuksgo2mars--Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1"
if [[ -n "${MODEL_LOCK:-}" ]]; then
  missing_bytes=$(PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -c \
    'from recipe_lib.artifacts import verify; from recipe_lib.config import load_lock; from pathlib import Path; import sys; print(verify(load_lock(Path(sys.argv[1])), Path(sys.argv[2]), full=False)["missing_or_bad_bytes"])' \
    "$MODEL_LOCK" "$HF_CACHE")
  reserve_gb=34  # base image plus build/cache headroom on a clean host
  if command -v docker >/dev/null && docker image inspect "$(recipe_image)" >/dev/null 2>&1; then reserve_gb=2; fi
  download_gb=$(( (missing_bytes + 1073741823) / 1073741824 ))
  needed_gb=$(( download_gb + reserve_gb ))
  if (( ${free_gb:-0} >= needed_gb )); then ok "${free_gb} GiB free; ~${download_gb} GiB missing checkpoint + ${reserve_gb} GiB image/build reserve"
  else bad "${free_gb:-?} GiB free; need ~${needed_gb} GiB for missing files + image/build reserve"; fi
elif [[ -d "$snap" ]]; then ok "Checkpoint already present under $HF_CACHE"
elif (( ${free_gb:-0} >= 160 )); then ok "${free_gb} GB free at $HF_CACHE"
else bad "Only ${free_gb:-?} GB free at $HF_CACHE. exl3-ple8 needs ~120 GB weights + ~29 GB image."; fi
[[ -n "$(find "$HF_CACHE" -maxdepth 0 -fstype nfs 2>/dev/null)" ]] && warn "HF_CACHE is on NFS. Put it on local NVMe; the PLE table is mmap'd."

log "Docker"
if command -v docker >/dev/null; then
  ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"
  if docker info >/dev/null 2>&1; then ok "daemon reachable without sudo"
  else bad "Cannot talk to the docker daemon as $(whoami). Start it and add yourself to the docker group."; fi
  if docker info 2>/dev/null | grep -qi 'nvidia'; then ok "nvidia runtime registered"
  else bad "nvidia container runtime not registered (run scripts/install-prereqs.sh)"; fi
else
  bad "docker not installed (run scripts/install-prereqs.sh)"
fi

log "Ports"
for p in "$PORT" "$LMCACHE_PORT" "$LMCACHE_HTTP_PORT" "$LMCACHE_METRICS_PORT"; do
  if command -v ss >/dev/null && ss -ltn "sport = :$p" | grep -q LISTEN; then
    warn "Port $p already in use (an existing server? check scripts/ctl.sh status)"
  else ok "port $p free"; fi
done
if command -v docker >/dev/null && docker image inspect "$KV_IMAGE" >/dev/null 2>&1; then ok "KV-offload image $KV_IMAGE present"
else warn "KV-offload image $KV_IMAGE not built yet (scripts/setup.sh builds it)"; fi

echo
if (( fails )); then die "$fails blocking issue(s)"; fi
ok "Ready. Next: scripts/setup.sh"

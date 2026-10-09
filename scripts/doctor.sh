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
  (( ${mem%.*} >= 95000 )) || bad "Need ~96 GB VRAM (found ${mem} MiB)"
  [[ "$name" == *"Max-Q"* ]] && warn "Max-Q edition: expect ~10% lower decode than the 400 W reference (see docs/benchmarks.md)."
  used="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | xargs)"
  (( used < 2000 )) || warn "${used} MiB VRAM already in use. The server wants 94% of the card; stop other GPU jobs first."
else
  bad "nvidia-smi not found (install the NVIDIA driver)"
fi

log "Memory and disk"
ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if (( ram_gb >= 120 )); then ok "RAM ${ram_gb} GiB"
elif (( ram_gb >= 64 )); then warn "RAM ${ram_gb} GiB. Works with mmap PLE, but page-cache misses will cost decode speed. 128 GB recommended."
else bad "RAM ${ram_gb} GiB. Need at least 64 GiB free for the PLE table."; fi
mkdir -p "$HF_CACHE" 2>/dev/null || true
free_gb=$(df -BG --output=avail "$HF_CACHE" 2>/dev/null | tail -1 | tr -dc '0-9')
snap="$HF_CACHE/hub/models--wrldsuksgo2mars--Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1"
if [[ -d "$snap" ]]; then ok "Checkpoint already present under $HF_CACHE"
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

log "Port"
if command -v ss >/dev/null && ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  warn "Port $PORT already in use (an existing server? check scripts/ctl.sh status)"
else ok "port $PORT free"; fi

echo
if (( fails )); then die "$fails blocking issue(s)"; fi
ok "Ready. Next: scripts/setup.sh"

#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 on two RTX PRO 6000 (96 GB each) and 128 GB of host RAM.
# Runs Diffbot's serve.sh through yoloyash's SSD-Engram overlay, with the settings measured in
# results/2026-11-17-deepseek-3bpw-dual-gpu. Foreground; Ctrl-C removes the container.
#
# Usage: API_KEY=... PACK=/path/to/diffbot-pack OVERLAY=/path/to/ssd-engram-clone ./serve.sh [parity]
#   parity = eager mode with ENGRAM_SSD_CHECK=1 and no warmup: every step re-reads sampled Engram rows
#            from the shard and raises on any difference. Use it once before trusting a new install.
set -euo pipefail
: "${API_KEY:?set API_KEY (clients send it as the Bearer token)}"
: "${PACK:?set PACK to the downloaded diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 directory}"
: "${OVERLAY:?set OVERLAY to a clone of yoloyash/dsv41-flash-1017-2x-rtx-pro-6000-ssd-engram}"
export API_KEY PACK
export PORT=${PORT:-8001}
# The cap also bounds the page cache that holds Engram rows. The measured host has 125 GiB usable and
# about 12 GiB of other load, so 108g; the overlay's default is 120g.
export CONTAINER_MEM=${CONTAINER_MEM:-108g}
# 512 VRAM slots per GPU for copies of host experts. Use 736 (the overlay's default) only when nothing
# else holds memory on either card.
export HOST_CACHE=${HOST_CACHE:-512}
# Boards without GPU peer-to-peer (`nvidia-smi topo -p2p r` shows CNS/NS between the cards): the 8-bit
# prefill all-reduce and the in-graph custom all-reduce both need P2P, so all-reduces go through NCCL
# over host memory. With P2P available, set P2P=1 to keep Diffbot's defaults.
if [[ ${P2P:-0} != 1 ]]; then
  export DS41_FP8AR=${DS41_FP8AR:-0} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
  export DS41_CAR=${DS41_CAR:-0} DS41_CAR_PCIE_IPC=${DS41_CAR_PCIE_IPC:-0}
fi
export CPU_EXPERTS=${CPU_EXPERTS:-0}          # the CPU expert runtime needs AVX-512 VNNI/VBMI
if [[ ${1:-} == parity ]]; then export ENGRAM_SSD_CHECK=1 EAGER=1 WARMUP=0; fi
[[ -x $OVERLAY/serve/serve-ssd.sh ]] || { echo "no serve/serve-ssd.sh in OVERLAY=$OVERLAY" >&2; exit 1; }
exec "$OVERLAY/serve/serve-ssd.sh"

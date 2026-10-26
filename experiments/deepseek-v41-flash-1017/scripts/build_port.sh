#!/usr/bin/env bash
# Run after the pinned diffbot base image build succeeds.
set -euo pipefail
ROOT="${DSV41_ROOT:-$HOME/src/dsv41-1017-rtx6000}"
BASE=dsv41-2p7-sm120:base
TAG=dsv41-2p7-sm120:eager-v1
NAME=dsv41-2p7-port-build
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Port build container already exists; inspect it before rerunning: $NAME" >&2
  exit 1
fi
docker run --name "$NAME" --memory=32g --memory-swap=32g --cpus=12 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v "$ROOT":/opt/port:ro --entrypoint bash "$BASE" -lc '
set -euo pipefail
VP=/usr/local/lib/python3.12/dist-packages
python3 /opt/port/scripts/patch_mixed_k.py \
  --plugin "$VP/vllm_exl3/exl3.py" \
  --helper /opt/port/source/sfxnz/docker/patch/widen_p2b_kbits.py \
  --xmoe /opt/port/recipe/kernels/xmoe_port
python3 /opt/port/scripts/patch_engram_eager.py \
  --engram "$VP/vllm/models/deepseek_v41/common/engram.py" \
  --weights "$VP/vllm/model_executor/model_loader/weight_utils.py" \
  --disk-module /opt/port/source/sfxnz/docker/patch/engram_disk.py
python3 -m py_compile "$VP/vllm_exl3/exl3.py" \
  "$VP/vllm/models/deepseek_v41/common/engram.py" \
  "$VP/vllm/model_executor/model_loader/weight_utils.py"
'
docker commit --change 'ENV DSV41_ENGRAM_DISK=1' --change 'ENV DSV41_ENGRAM_INT4=0' \
  --change 'LABEL dsv41.port=eager-v1-unvalidated' "$NAME" "$TAG"
docker rm "$NAME" >/dev/null
docker run --rm --memory=32g --memory-swap=32g --cpus=12 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_CUDA_ARCH_LIST=12.0a -e MAX_JOBS=8 \
  -v "$ROOT/recipe/kernels":/k -w /k --entrypoint python3 "$TAG" \
  build_xmoe.py xmoe_port xmoe_port
printf '%s\n' "$(date -Is) port_image_and_kernel_build_complete"

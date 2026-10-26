#!/usr/bin/env bash
set -euo pipefail
ROOT="${DSV41_ROOT:-$HOME/src/dsv41-1017-rtx6000}"
TAG=dsv41-2p7-sm120:eager-v2
NAME=dsv41-2p7-head-build
if docker image inspect "$TAG" >/dev/null 2>&1; then
  [[ $(docker image inspect -f '{{index .Config.Labels "dsv41.head_port"}}' "$TAG") == sm120-mxfp8-v1 ]]
  exit 0
fi
! docker container inspect "$NAME" >/dev/null 2>&1
docker run --name "$NAME" --memory=2g --memory-swap=2g --cpus=2 \
  -v "$ROOT":/opt/port:ro --entrypoint python3 dsv41-2p7-sm120:eager-v1 \
  /opt/port/scripts/patch_lmhead_sm120.py \
  --directory /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia
docker commit --change 'LABEL dsv41.head_port=sm120-mxfp8-v1' "$NAME" "$TAG"
docker rm "$NAME" >/dev/null

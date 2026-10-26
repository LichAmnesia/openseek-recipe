#!/usr/bin/env bash
# Foreground manual first-gate server. Never stops another GPU job/service.
set -euo pipefail
ROOT="${DSV41_ROOT:-$HOME/src/dsv41-1017-rtx6000}"
export PACK="${PACK:-$HOME/models/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-mcg-hq}"
export IMAGE=dsv41-2p7-sm120:eager-v2 NAME="${NAME:-dsv41-2p7-manual}"
export PORT=8010 SERVED=dsv41-1017-2p7-test API_KEY=local-benchmark-only
export CACHE_DIR="$ROOT/cache" CONTAINER_MEM=100g
export PLAN=plan-1gpu-bootstrap.json XMOE_EXT=xmoe_port EAGER=1 VISION=0 SPEC=none
export MAX_MODEL_LEN=262144 MAX_NUM_SEQS=1 MAX_BATCHED=512 KV_MEM=1181116006 UTIL=0.98
export EXL3_DO=0 HOST_CACHE=0 EXL3_HOST_CACHE=0 CPU_EXPERTS=0 EXL3_CPU_EXPERTS=0
export EXL3_HOST_DMA=0 EXL3_HOST_COLD_PACK='' VLLM_EXL3_MOE_KERNEL=exllamav3
export DS41_CAR=0 DS41_CAR_PCIE_IPC=0 DS41_FP8AR=0 WARMUP=0 THINKING=off
export EXTRA_ARGS='--enable-prefix-caching --load-format safetensors'
if [[ ${1:-} == --dry-run ]]; then
  printf 'root=%s image=%s name=%s mode=TP1/eager batch=512 seq=1 memory=100g swap=0 port=127.0.0.1:8010\n' "$ROOT" "$IMAGE" "$NAME"
  exit 0
fi
command -v docker >/dev/null
command -v nvidia-smi >/dev/null
docker image inspect "$IMAGE" >/dev/null
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Existing $NAME preserved; archive its logs/inspect and rename it before a new attempt" >&2
  exit 1
fi
python3 - "$ROOT" "$PACK" <<'PY'
import json,pathlib,subprocess,sys
r,model=map(pathlib.Path,sys.argv[1:])
v=json.loads((r/'logs/weight-verification.json').read_text())
assert v['ok'] and v['revision']=='a8006e99931e7d67c4d218c66543f478b521b857'
assert len(v['files'])==55 and all(f['ok'] for f in v['files'])
for f in v['files']:
    assert (model/f['file']).stat().st_size==f['expected_bytes'],f['file']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'Another GPU job is running; arrange its normal idle handoff first'
mem={k:int(v.split()[0]) for k,v in (x.split(':',1) for x in open('/proc/meminfo'))}
assert mem['MemAvailable']>100*1024**2,'Less than 100 GiB available; inspect RAM before pinning experts'
PY
python3 "$ROOT/scripts/make_bootstrap_plan.py" "$ROOT/evidence/sfxnz-config.json" \
  "$ROOT/recipe/plans/plan-1017-agent30-c512.json" "$ROOT/recipe/plans/$PLAN" --host-gib 92
python3 "$ROOT/scripts/prepare_serve.py" "$ROOT/recipe"
mkdir -p "$CACHE_DIR"
exec bash "$ROOT/recipe/serve/serve-1gpu-eager.sh"

# DeepSeek-V4.1-Flash-1017 2.7bpw: experimental RTX PRO 6000 recipe

Measured **2026-10-26** on one RTX PRO 6000 Blackwell **Max-Q 96 GB / 300 W**,
128 GB host RAM and local NVMe. **10.64 tok/s decode; 5.31 s TTFT** for
1550 input / 256 output tokens, three C1 trials. Arithmetic and a parsed weather
tool call passed. This eager port misses the original speed stop lines
(decode <20 tok/s; warmed agent-shaped TTFT >6 s). KL, SWE-bench, real coding
traffic and full 256K quality acceptance remain untested.

![Measured DeepSeek eager throughput and memory observations](assets/deepseek-1017-2.7bpw-card.png)

[Download the 1600×900 card](assets/deepseek-1017-2.7bpw-card.png) ·
[Editable HTML source](assets/deepseek-1017-2.7bpw-card.html).
The card summarizes the archived run; generating it did not rerun the GPU benchmark.

The main `./recipe` CLI remains the qualified Qwen adapter. This manual experiment
has its own [locked sources and patches](../experiments/deepseek-v41-flash-1017/),
[measured evidence](../results/2026-10-26-deepseek-2.7bpw-eager/SUMMARY.md),
and [license notices](../experiments/deepseek-v41-flash-1017/THIRD_PARTY_NOTICES.md).
The original GPU run passed; the portable packaging has offline checks, not a
second fresh-machine GPU qualification. The separate
[HF distribution](https://huggingface.co/LichAmnesia/DeepSeek-V4.1-Flash-1017-OpenSeek-EXL3)
contains the attributed sfxnz weights and a frozen portable snapshot. Its card
documents downloading this release and using the mapped verification manifest;
the original upstream download below remains valid. Container images are local
builds, not public registry artifacts.

## Where the GPU memory, RAM and SSD go

In plain English, this run uses **95 GiB GPU memory + about 100 GiB host RAM + NVMe SSD** together.

| Location | Amount | What lives there |
|---|---:|---|
| GPU memory | **95 GiB** sampled peak | About **79 GiB** of expert weights in the allocation plan, plus other model layers, KV cache and compute workspace |
| Host RAM | About **100 GiB** container peak, including loading/file cache | About **92 GiB** for **8,220 experts**, plus runtime buffers and file cache |
| NVMe SSD | About **405 GB** for the complete checkpoint | Includes about **203 GB of Engram tables**, retained on SSD and read on demand |

The RAM tier stores expert weights; **the GPU still performs their computation**,
accessing those weights over PCIe. CPU-expert compute was disabled in this run.
The expert trellis allocation plan was 78.8599 GiB on GPU and 91.9995 GiB in RAM;
these components exclude small scales and are separate from the measured total peaks.
The GPU's total also includes the 1.1 GiB KV allocation and other runtime memory.

**16.39 GiB in the figure is remaining available host RAM.** The host reported
125.0604 GiB total; total minus minimum available was 108.6697 GiB (about109 GiB),
including the OS and other workloads. The model container's measured peak was
about100 GiB, including file cache and earlier loading. Peaks are observed independently.

The full checkpoint is 405.05 GB (377.23 GiB); its two Engram shards total
203.07 GB (189.13 GiB), already included in that total. Other weight files also
remain on SSD after loading. Docker images/build caches require additional space.
Engram rows are read on demand from the NVMe-backed files; the OS can cache rows.
Measured container **swap was0**: SSD participates through explicit model-table reads.
These are the archived run's measurements and allocation plan, not a new benchmark.

## What was required

| Input / setting | Fixed value |
|---|---|
| Checkpoint | `sfxnz/DeepSeek-V4.1-Flash-1017-EXL3@a8006e99931e7d67c4d218c66543f478b521b857` |
| Diffbot recipe | `2c3797578722747dad3593531fd447b1d3d4c732` |
| sfxnz helper source | `19e0c56620a7b9748fd901cdf3ef533a0f6ed917` |
| Base image | Digest in `experiments/deepseek-v41-flash-1017/runtime-lock.json` |
| Local serving image | `dsv41-2p7-sm120:eager-v2`; measured image ID in that lock |
| Mode | TP1, eager; CUDA graphs, DSpark, vision, host cache, CPU experts and DMA disabled |
| Host experts | 8220 / 91.9995 GiB trellis, deterministic bootstrap; extra IDs unprofiled |
| Engram / context | FP8 tables on NVMe; FP8 KV 1.1 GiB; max context 262144 |
| Scheduling / limits | batch512, seq1, container100g, container swap0 |

Patches cover mixed gate/up versus down bit widths and K0/K4 MCG kernels;
explicit disk Engram allocation/loader-skip/lookup; and MXFP8 head namespace,
FP8/scale allocation and sm120 W8A16 compute. Stock loading failed with
`There is no module or parameter named 'lm_head' in DeepseekV41ForCausalLM`:
the VL suffix corrupts `lm_head.weight` and leaves its scale at the wrong root.
Increasing RAM does not fix that error. `EXL3_HOST_COLD_PACK` must be **empty**;
the string `0` is interpreted as a path. Graphs and speculative paths are not
validated by these patches.

## 1. Prepare on x86_64 Linux

Prerequisites: Docker with NVIDIA Container Toolkit, CUDA13-capable driver,
Python3.10+, `git`, `curl`, `nvidia-smi`, and NVMe space for **405050217567 bytes**
of weights plus images, build and cache space. Measured driver: 610.57.04.
Do not run this beside another GPU workload or a 64 GiB LMCache tier.

From this repository root:

```bash
export DSV41_ROOT="$HOME/.local/share/openseek/dsv41-2p7-eager"
export PACK="$HOME/models/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-mcg-hq"
EXP="$PWD/experiments/deepseek-v41-flash-1017"
python3 "$EXP/prepare.py" --root "$DSV41_ROOT" --dry-run
python3 "$EXP/prepare.py" --root "$DSV41_ROOT"
```

Preparation fetches 152 small source/config files at fixed revisions and checks
SHA256, stages the local helpers, and prepares the runtime-K kernel source.
It downloads no weights, runs no Docker/GPU commands, and refuses to overwrite
an existing differing source/helper. Keep a separate work root for a new build.

## 2. Download and verify

Use the tested downloader versions in an isolated environment (if not already
available); `uv` is only needed for this environment creation:

```bash
uv venv "$DSV41_ROOT/download-venv" --python 3.12
uv pip install --python "$DSV41_ROOT/download-venv/bin/python" \
  huggingface-hub==2.0.0 hf-xet==1.6.0
export HF_BIN="$DSV41_ROOT/download-venv/bin/hf"
bash "$DSV41_ROOT/scripts/download_model.sh" | tee "$DSV41_ROOT/logs/download.log"
```

55 files, including 48 weight shards, are checked against `model-files.json`:
sizes for every file and published LFS SHA256 for weight shards. Do not count
temporary files as complete weights. Completed files are reusable; the tested
HF2 downloader does **not** resume an incomplete UUID temporary file. Never run
competing downloaders or repeatedly restart the two ~101.5GB Engram shards.
An eight-worker/Xet-HP trial did not improve this host's sustained download.

If the checkpoint already exists, verify it without another download:

```bash
python3 "$DSV41_ROOT/scripts/verify_weights.py" \
  "$DSV41_ROOT/evidence/sfxnz-model-info.json" "$PACK" \
  "$DSV41_ROOT/logs/weight-verification.json"
```

## 3. Build

First inspect existing build containers; retain failure logs/inspect before a
retry. The upstream base build uses apt packages that are not fully locked, so
new image IDs may differ from the recorded local image ID. It is not pullable
from a public registry.

```bash
if ! docker image inspect dsv41-2p7-sm120:base >/dev/null 2>&1; then
  TAG=dsv41-2p7-sm120:base BUILD_CONTAINER="openseek-dsv41-base-$(date +%s)" \
    BUILD_MEM=64g BUILD_CPUS=16 MAX_JOBS=8 \
    bash "$DSV41_ROOT/recipe/docker/build.sh" | tee "$DSV41_ROOT/logs/base-build.log"
fi
if ! docker image inspect dsv41-2p7-sm120:eager-v1 >/dev/null 2>&1; then
  bash "$DSV41_ROOT/scripts/build_port.sh" | tee "$DSV41_ROOT/logs/port-build.log"
fi
bash "$DSV41_ROOT/scripts/build_lmhead_port.sh" | tee "$DSV41_ROOT/logs/head-build.log"
```

Reusing an existing tag means reusing that build. Preserve its image inspection
and rerun the numeric gates; a tag alone is not a content-integrity receipt.

## 4. Hand off the GPU and run numeric gates

Wait for existing work to finish, then use its normal lifecycle entry point.
For the Qwen recipe: `./recipe stop` (or its existing managed entry point) stops
both Qwen and LMCache. Preserve model files, stopped containers and training
checkpoints. Check `nvidia-smi` and `/proc/meminfo` before pinning experts.

On **lich-6000 only**, the existing HeyDreaming integration can check fresh idle
cloud state before a normal service stop. It uses the worker's own environment;
credentials and cloud-specific code are intentionally outside this OSS bundle:

```bash
set -a; . "$HOME/video-worker/worker.env"; set +a
"$HOME/src/LTX-2/.venv/bin/python" \
  "$HOME/src/dsv41-1017-rtx6000/scripts/check_video_worker.py" --require-idle
sudo systemctl stop video-worker.service
```

If it reports busy/stale, wait and check again. Do not force-kill a generating
video. Other hosts should use their own workload's idle/normal-stop mechanism.

```bash
probe() {
  docker run --rm --gpus device=0 --memory=12g --memory-swap=12g --cpus=6 \
    -e VLLM_EXL3_MOE_KERNEL=exllamav3 \
    -v "$DSV41_ROOT":/opt/port -v "$DSV41_ROOT/recipe":/opt/recipe:ro \
    -v "$PACK":"$PACK":ro --entrypoint python3 dsv41-2p7-sm120:eager-v2 "$@"
}
probe /opt/port/scripts/probe_disk_reader.py \
  /opt/port/source/sfxnz/docker/patch/engram_disk.py /opt/port/logs/disk-probe.json
probe /opt/port/scripts/probe_moe_kernel.py \
  /opt/port/recipe/kernels/build/xmoe_port /opt/port/logs/moe-probe.json
probe /opt/port/scripts/probe_lmhead_sm120.py /opt/port "$PACK"
```

These check disk rows, mixed-K GPU/UVA numerics and actual checkpoint head rows.
They are prerequisites, not model quality acceptance. Restore the prior service
on a failed gate after these containers have exited.

## 5. Serve, connect and measure

```bash
bash "$EXP/serve.sh" --dry-run
bash "$EXP/serve.sh"  # foreground; Ctrl-C stops this model, retaining its container
```

The wrapper refuses another GPU compute process, <100GiB available host RAM,
missing/failed weight verification, changed file sizes, or an existing same-name
container. It does not stop services. The manual container is
`dsv41-2p7-manual`, restart policy `no`, localhost8010, model
`dsv41-1017-2p7-test`, key `local-benchmark-only`. For the original lich-6000
installation, set `DSV41_ROOT="$HOME/src/dsv41-1017-rtx6000"` and reuse its
verified weights/images directly; no download/build/old systemd-gate restart is
needed. The previous automatic follow-up is paused.

The original load took about14 minutes. **48/48 shards is an iterator milestone,
not API readiness.** From a second terminal on the GPU host, wait for
`curl -f -H 'Authorization: Bearer local-benchmark-only' http://127.0.0.1:8010/v1/models`.
Use an SSH tunnel for remote OpenSeek TUI access; this profile binds localhost.

To repeat the archived reference harness (ongoing performance tooling belongs
to OpenSeek Perf / `osp`), with the same exported `DSV41_ROOT`:

```bash
RUN="$DSV41_ROOT/logs/manual-$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$RUN"
printf '{"stage":"correctness_and_speed"}\n' > "$RUN/pipeline-state.json"
python3 "$DSV41_ROOT/scripts/resource_monitor.py" --container dsv41-2p7-manual \
  --state "$RUN/pipeline-state.json" --output "$RUN/resources.jsonl" &
MONITOR_PID=$!
python3 "$DSV41_ROOT/scripts/benchmark_api.py" --output "$RUN/benchmark.json"
kill "$MONITOR_PID"; wait "$MONITOR_PID" || true
python3 "$DSV41_ROOT/scripts/summarize_measurements.py" "$RUN"
```

Record benchmark failure even if the resource summarizer runs. The harness
requires real arithmetic/tool calls and four full256-token streams, measures
TTFT and decode separately, and tests34K warmed prefix+6K new tokens. Repeated
raw completion text is a latency/throughput workload, not a coding-quality test.
With absent API `cached_tokens`, record only the verified cache-counter increase.

## 6. Release GPU and restore video

Stop this foreground model, preserve logs/inspect, and confirm
`docker inspect -f '{{.State.Running}}' dsv41-2p7-manual` is `false` before
restarting video. On lich-6000, `sudo systemctl start video-worker.service`, then
rerun the worker check above **without `--require-idle`** to verify a fresh
healthy heartbeat (busy is valid after restoration). Restore on either success
or failure. Keep unrelated jobs and training checkpoints untouched.

The automated measured run used a root watchdog bound to one systemd gate
InvocationID and verified image ownership before cleanup. This manual bundle
does not install that host-specific watchdog. Arrange an equivalent supervisor
before leaving a shared production GPU unattended.

# OpenSeek Recipe

Reproducible recipes for running open-weight models on local hardware: pinned
checkpoints and runtimes, preflight checks, serving, client integration, and
measurements you can inspect and repeat.

**This repository is the recipe manager.** Use OpenSeek TUI for chat and coding;
future performance/monitoring integration belongs to SparkDash. The repository
entry point is `./recipe`, so it does not replace your `openseek` command.

**First validated recipe:** Qwen3.8-Flash-Next on one RTX PRO 6000 Blackwell Max-Q
96 GB. The supported setup is described below. Additional models, hardware,
and project-produced quantizations are planned, not validated support. See the
[CLI/configuration guide](docs/CLI.md) and [roadmap](docs/ROADMAP.md).

Previously published as `qwen38-flash-next-rtx6000`.

## Qwen3.8-Flash-Next on one RTX PRO 6000

Run **Qwen3.8-Flash-Next** (125B MoE, 6B active, 262K context, vision, tool calling) on a
single **RTX PRO 6000 Blackwell 96 GB** with vLLM. You get an OpenAI-compatible endpoint
that coding agents (OpenCode, Codex CLI, anything OpenAI-compatible) can drive.

Measured on a **300 W Max-Q** card:

| | |
|---|---|
| Decode, C1 (code / math) | **181 / 186 tok/s** |
| Decode, seven-workload weighted | **134.4 tok/s** (−9% vs the 400 W reference) |
| Decode, 16 concurrent | **595 tok/s** aggregate |
| TTFT | **85–102 ms** |
| SWE-bench Pro (50-instance stratified subset) | **66.0%** (33/50), Qwen's card says 62.5 |
| Tool calling / vision | 16/16 / 1, 4 and 16 images pass |
| VRAM / power | 92.8 GB / 300 W |
| Long agent session evicted from GPU, back from the RAM tier | **0.64 s** instead of 19.5 s re-prefill (130K tokens) |

All numbers and raw files: [docs/benchmarks.md](docs/benchmarks.md),
[docs/swebench-pro.md](docs/swebench-pro.md), [docs/kv-ram-offload.md](docs/kv-ram-offload.md),
[results/](results/).

## What this repo is

The hard engineering is in [`tpurtell/sm12x-exl3-qwen3.8-flash-next`](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next)
(Apache-2.0): a patched vLLM image for SM120, EXL3 K4.25 experts, the PLE n-gram table
mmap'd from disk, and tuned MTP. This repo pins that recipe and adds:

- a **preflight check** that reads the machine before anything gets downloaded
- **one-command lifecycle** (`start / stop / recreate / logs / status`) with a restart policy
- a **64 GiB host-RAM tier for the KV cache** (LMCache), on by default. Agent sessions pushed
  out of the GPU KV pool come back over PCIe instead of being re-prefilled. See
  [docs/kv-ram-offload.md](docs/kv-ram-offload.md); the GPU-only setup is in
  [legacy/no-offload](legacy/no-offload/README.md)
- a **smoke test** that checks a real tool call, reasoning and the Responses API, not just "HTTP 200"
- **client configs** for OpenCode and Codex CLI
- **first-party Max-Q 300 W numbers**. Every number published before this came from 400 W or 600 W cards.

## Requirements

| | Minimum | Tested |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell 96 GB (Workstation, Max-Q or Server) | Max-Q Workstation, 300 W |
| Driver | `nvidia-open`, CUDA 13 capable | 610.57 |
| Host RAM | 128 GB (PLE page cache + 64 GiB KV tier). 64 GB works only with [legacy/no-offload](legacy/no-offload/README.md) | 128 GB |
| Disk | Plan ~160 GiB (~172 GB) on local NVMe for the 119.5 GiB checkpoint, image and build reserve | 990 EVO Plus 4 TB |
| OS | x86_64 Linux with Docker + NVIDIA Container Toolkit | Arch (Omarchy), kernel 7.2 |
| Host Python | Python 3.10+; standard library only for `./recipe` | CLI checks on macOS; GPU recipe targets Linux |

## Quickstart

On the GPU box:

```bash
git clone https://github.com/LichAmnesia/openseek-recipe.git
cd openseek-recipe
./recipe models list
./recipe models select qwen38-flash-next  # saves selection locally; does not change a running service
./recipe install --dry-run               # inspect the selected model, runtime and commands
./recipe install                         # preflight, pinned images, checkpoint download + SHA256
./recipe start                           # starts LMCache then vLLM; waits for the API
./recipe smoke                           # chat, reasoning, tools, Responses API and streaming
./recipe connection                      # endpoint/model JSON for OpenSeek TUI or another client
```

If preflight reports missing Docker/NVIDIA prerequisites, run
`scripts/install-prereqs.sh` and retry. The installer does not perform system-wide
package upgrades automatically. Only the listed Qwen model/profile is currently
registered; there is no placeholder OpenSeek-owned checkpoint.

Model selection lives in ignored `.recipe/config.json`. Local flags/environment
and the legacy `.env` remain available for paths, ports and resource settings;
see [configuration precedence](docs/CLI.md#configuration). Runtime commits,
base image digest, the LMCache wheel hash, and all **40 checkpoint files** are
pinned in `locks/`. The selected checkpoint totals **128.33 GB decimal** (~119.5
GiB), including 22 weight shards. An intact cache is reused; installation still
hash-checks every file and may take several minutes without downloading again.

For an existing deployment using the old unlabelled image, install the locked
image first and explicitly use `./recipe recreate` when ready to switch. A
plain `start`/`restart` refuses a mismatched image. Changing configuration also
requires `recreate`; selecting a model alone never switches a running service.

Expected smoke test:

```text
[PASS] GET /v1/models          ...  qwen38-exl3-ple8 max_model_len=262144
[PASS] chat (thinking off)     ...  'SMOKE_OK'
[PASS] reasoning (thinking on) ...  answer='391' reasoning_chars=...
[PASS] tool call               ...  get_weather({"city": "Tokyo", "unit": "c"})
[PASS] POST /v1/responses      ...  status=completed text='RESPONSES_OK'
[PASS] streaming               ...  TTFT ~100 ms ...
6/6 passed
```

The endpoint is `http://<gpu-host>:8001/v1` and the model id is `qwen38-exl3-ple8`. No
API key is needed. If other machines can't connect, open the port in your firewall (see
[troubleshooting](docs/troubleshooting.md)).

## Connect a coding agent

See [clients/README.md](clients/README.md). In short:

```bash
# OpenCode
OPENCODE_CONFIG=clients/opencode.json opencode -m rtx6000/qwen38-exl3-ple8

# Codex CLI (copy the profile to ~/.codex/rtx6000.config.toml first)
codex -p rtx6000
```

Edit `YOUR_GPU_HOST` in both files first.

## Day-2 operations

```bash
scripts/ctl.sh status       # both containers, API up/down, GPU vs RAM-tier hit rates, VRAM / RAM
scripts/ctl.sh logs         # follow vLLM logs (`logs lmcache` for the KV tier)
scripts/ctl.sh stop         # frees all 93 GB of VRAM and the 64 GiB RAM tier within seconds
scripts/ctl.sh start        # restarts the existing containers (LMCache first)
scripts/ctl.sh recreate     # apply .env changes (serve flags are fixed at container creation)
scripts/ctl.sh down         # remove both containers
PART=core scripts/bench.sh  # rerun the benchmark suite (~4 min) and print a summary
python3 scripts/kv_bench.py run all   # KV-tier bench: eviction/restore, 9-session rotation, decode (~12 min)
```

Both containers are created with `--restart unless-stopped`. They come back after a reboot,
but stay down after a manual `stop`.

## The serve configuration

This is what `scripts/ctl.sh start` runs. First the KV tier:

```bash
lmcache server --host localhost --port 6555 --chunk-size 3200 --separate-object-groups \
  --l1-size-gb 64 --eviction-policy LRU --eviction-trigger-watermark 0.95
```

then vLLM through the upstream `start.sh`, on `qwen38-rtx:lmc056` (the recipe image plus
the `lmcache==0.5.6` wheel, built by `scripts/build-kv-image.sh`):

```bash
vllm serve <exl3-ple8 snapshot> --served-model-name qwen38-exl3-ple8 --port 8001 \
  --tensor-parallel-size 1 --distributed-executor-backend mp \
  --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 6399 \
  --gpu-memory-utilization 0.94 --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching --mamba-cache-mode align \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":6555}}'
```

The container also gets `VLLM_PLE_MMAP=1` and related env from the recipe. Why each part
matters:

| Choice | Why |
|---|---|
| `exl3-ple8` profile | EXL3 K4.25 experts on GPU. The 51B-param n-gram (PLE) table is FP8 and **mmap'd from NVMe**, so the host only needs page cache, not 95 GB of pinned RAM. Best fit for a 128 GB box |
| FP8 KV cache | KV pool of 842K tokens, 3.2× the full 262K context, with 16 slots |
| MTP 3 | 52% draft acceptance on mixed work and 97% on repetitive output. This is where the 100→200 tok/s spread comes from |
| `qwen3_coder` tool parser + `qwen3` reasoning parser | Tool calls arrive as proper `tool_calls` and thinking goes to `reasoning_content`. Coding agents need both |
| LMCache MP connector, 64 GiB RAM tier | GPU pool ~0.8M tokens + RAM tier ~1.25M tokens. Evicted sessions restore in ~0.6 s instead of a full re-prefill. Only LMCache ≥ 0.5.6 handles this model's QSA scratch ring; vLLM native offload in this image does not. [Details](docs/kv-ram-offload.md) |
| `max-num-batched-tokens 6399` | 2×3200−1. Must be ≥ the 3200-token unified block size; 2N−1 lets decodes share a step with one prefill block |
| `gpu-memory-utilization 0.94` | Leaves ~5 GB for CUDA context and graphs. Don't run anything else on the card |

## Things I measured or hit that aren't in the upstream README

1. **The checkpoint is 119.5 GiB (128.33 GB), not 48 GiB.** The "~48 GiB" in the upstream
   profile table is the PLE table alone. Plan ~160 GiB for weights, image and build reserve.
2. **Max-Q costs about 10%, not 25%.** On 25% less power, decode drops 9% (C1 weighted)
   and 9.8% (C16).
3. **Max-Q runs an untuned MoE kernel.** B12x has no profile under the Max-Q device name
   and falls back to a heuristic. The die and SM count (188) are the same as the 400 W
   card, so there is probably more to recover.
4. **A tok/s number without a workload doesn't tell you much.** The same server does
   102 tok/s on multilingual prose and 215 tok/s on repetitive output, because MTP
   acceptance runs from 52% to 97%.
5. **SWE-bench Pro has three harness traps that fail silently.** One of them burned 186
   model calls before it showed up. Write-up and fixes: [docs/swebench-pro.md](docs/swebench-pro.md).
6. **A KV RAM tier smaller than the GPU pool does nothing.** LMCache's tier also holds what
   is already on the GPU, so 32 GiB (~0.6M tokens) measured ~0 hits against a ~0.8M GPU pool.
   64 GiB (~1.25M tokens) cut a 9-session agent rotation from 142 s to 10.7 s per round.

## Credits

- Serving recipe, patched vLLM image, benchmark suite: [tpurtell/sm12x-exl3-qwen3.8-flash-next](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next) (Apache-2.0)
- EXL3 K4.25 + FP8 PLE checkpoint: [wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1](https://huggingface.co/wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1)
- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

**This repo does not redistribute any model weights or container images.** `scripts/setup.sh`
downloads them from their original sources. Both the Qwen model and the EXL3 quant are
listed on Hugging Face under `license: other`. Read and accept those terms yourself,
especially before any commercial use.

The scripts in this repo are MIT-licensed. The upstream recipe is cloned at run time
into `vendor/` and is not redistributed here.

---

中文说明见 [README.zh-CN.md](README.zh-CN.md)。

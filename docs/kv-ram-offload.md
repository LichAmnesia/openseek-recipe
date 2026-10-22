# KV cache in host RAM (LMCache)

Coding agents resend their whole conversation every turn. With 150–200K-token sessions,
a few agents fill the ~0.8M-token GPU KV pool, the prefix cache evicts in LRU order, and
every turn re-prefills the full context (~19 s per 130K tokens on a Max-Q card). The main
scripts add a **64 GiB host-RAM tier** behind the GPU pool, so an evicted session comes
back over PCIe instead of being recomputed.

| | GPU only | + 64 GiB RAM tier |
|---|---|---|
| 130K-token session after eviction | 19.5 s | **0.64 s** |
| 9 × 108K sessions, round-robin, per round | 142 s | **10.7 s** |
| Cold prefill | – | ~2% slower |
| Decode | ~127 tok/s | ~127 tok/s |

Raw numbers: [results/2026-10-21-maxq-kv-ram-offload](../results/2026-10-21-maxq-kv-ram-offload/SUMMARY.md).

## What runs

```text
qwen38-lmcache     lmcache server --chunk-size 3200 --separate-object-groups \
                     --l1-size-gb 64 --eviction-policy LRU --eviction-trigger-watermark 0.95
                   localhost:6555 (zmq), 127.0.0.1:6580 (http), :6590 (prometheus)

qwen38-exl3-ple8   the recipe's vllm serve, image qwen38-rtx:lmc056, plus
                     --enable-prefix-caching --mamba-cache-mode align
                     --max-num-batched-tokens 6399
                     --kv-transfer-config '{"kv_connector":"LMCacheMPConnector",
                       "kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector",
                       "kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":6555}}'
```

`qwen38-rtx:lmc056` is the pinned recipe image with one addition: the `lmcache==0.5.6`
wheel installed with `--no-deps`. vLLM, torch, the EXL3/PLE patches and the weights are
unchanged (same EXL3 K4.25 experts, FP8 PLE, FP8 KV). `scripts/build-kv-image.sh` builds
it and checks the imports.

## Why this exact setup

The model is a hybrid: Gated-DeltaNet linear-attention layers plus Qwen Sparse Attention
(QSA) layers. QSA keeps a small per-request scratch ring for keys whose compression group
is still open. That ring is not prefix-cacheable, and two older paths trip on it:

| Path | Result in this image |
|---|---|
| vLLM native offload (`--kv-offloading-backend native`) | `AssertionError: tokens_per_block=8 not divisible by tokens_per_hash=3200`. Fixed upstream in vLLM #57145 (0.30.0), not in the recipe's vLLM fork |
| LMCache 0.5.4 `LMCacheConnectorV1` | vLLM disables the hybrid KV manager, then `Failed to promote local KV cache specs to one unified type` |
| **LMCache ≥ 0.5.6 `LMCacheMPConnector`** | Works. Skips the scratch group (LMCache #5042), keeps the hybrid manager on |

At startup the vLLM log should show
`group_tokens_per_block=[3200, 0, 3200, 3200, 3200, 3200]`. The `0` is the scratch ring,
skipped on purpose.

Settings that have to match:

- `--chunk-size` = vLLM's unified block size N. The log prints it as
  `Setting attention block size to N tokens`. **N = 3200 with MTP 3.** Re-read it if you
  change `MTP_TOKENS` or the KV dtype and set `KV_BLOCK_TOKENS`.
- `--max-num-batched-tokens` ≥ N. 2N−1 = 6399 is LMCache's recommendation for GDN hybrids:
  it lets decodes share a step with one prefill block and still snapshots the GDN state
  at every block boundary.
- `--mamba-cache-mode align` (vLLM picks it by default with prefix caching on this model;
  set explicitly anyway) and `--separate-object-groups` on the server.
- Use the explicit `kv_connector_module_path`. Without it vLLM may load its bundled older MP
  connector, which only works with the hybrid manager off.
- Prefix hits happen in 3200-token blocks. The last partial block of a prompt is always
  recomputed.

## Sizing the RAM tier

Per token the tier stores 52,032 bytes: 13 GDN layers (13,312 B), 13 QSA compressed-index
layers (832 B) and 37 QSA KV layers (37,888 B).

| `KV_RAM_GB` | Usable tokens (×0.95) | vs GPU pool (~0.78–0.85M) |
|---|---|---|
| 32 | ~0.63M | smaller: **measured ~0 hits** |
| 48 | ~0.94M | about the same |
| **64** | **~1.25M** | 1.5× |

**The tier is inclusive.** It also stores what is already on the GPU. A working set that
overflows the GPU pool overflows a smaller tier too, and LRU evicts everything. Make it
clearly larger than the GPU pool or don't bother.

**The ceiling is host RAM.** The FP8 PLE table is mmap'd and lives in page cache (vLLM
worker RSS ~35 GB). On a 128 GB host, 64 GiB of pinned tier leaves ~30 GB available and
swap grew from 13 to 20 GB in testing. Keep it at or below ~70 GiB. `scripts/doctor.sh`
checks for `KV_RAM_GB + 56` GiB.

## Day-2

```bash
scripts/ctl.sh status        # both containers, GPU KV usage, GPU vs RAM-tier hit rates
scripts/ctl.sh logs lmcache  # LMCache server log
scripts/ctl.sh restart       # restarts LMCache, then vLLM (vLLM must re-register)
scripts/ctl.sh down          # remove both containers
python3 scripts/kv_bench.py my-run all > results/<dir>/my-run.json   # ~12 min, idle server
```

Prometheus counters on `:8001/metrics`: `vllm:external_prefix_cache_hits_total` /
`vllm:external_prefix_cache_queries_total` for the RAM tier, `vllm:prefix_cache_*` for
the GPU.

## Known limits

- **First long request after a restart is slow** (130K tokens: ~2 min instead of ~19 s).
  Pinning the tier evicts the PLE table from page cache, and it faults back in from NVMe.
- If only `qwen38-lmcache` restarts (crash, OOM), vLLM's KV registration is gone. Run
  `scripts/ctl.sh restart`.
- Not validated: restore after MTP rejects a draft at a block boundary, many sessions
  restoring at the same time, image/video KV. LMCache marks MTP + this model as
  "not yet validated" too. Our checks were a code-word answer before and after a RAM
  restore (identical) and unchanged decode speed.
- Cached and fresh runs are not guaranteed bit-exact (GDN has no batch-invariant mode).

To run without the tier, use [legacy/no-offload](../legacy/no-offload/README.md).

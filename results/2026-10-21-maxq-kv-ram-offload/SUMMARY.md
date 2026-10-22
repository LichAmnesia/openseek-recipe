# 2026-10-21 · Max-Q 300 W · KV cache RAM tier A/B

Card: RTX PRO 6000 Blackwell Max-Q, 300 W, driver 610.57. Host: Core Ultra 9 285K,
128 GB DDR5 (125 GiB usable). Profile exl3-ple8, FP8 KV, MTP 3, 262K context.
Bench: [`scripts/kv_bench.py`](../../scripts/kv_bench.py), sequential requests, temperature 0,
idle server.

| Config | Image | Batch | RAM tier |
|---|---|---|---|
| no-offload | `qwen38-rtx:local` | 8192 | none |
| kv-ram-32gib | `qwen38-rtx:lmc056` | 6399 | LMCache 0.5.6, 32 GiB, watermark 0.8 |
| kv-ram-64gib | `qwen38-rtx:lmc056` | 6399 | LMCache 0.5.6, 64 GiB, watermark 0.95 |

**g1**: session A (130,062 tokens, code word at the top) cold → resend → six unrelated
135K prompts push it out of the GPU pool → resend A.
**rotate**: 9 sessions × ~108K tokens, 3 rounds round-robin, each turn appends one line
(working set ~976K tokens > GPU pool).

| | no-offload | kv-ram-32gib | kv-ram-64gib |
|---|---|---|---|
| GPU KV pool (tokens) | 834,788 | 778,796 | 847,514 |
| A after eviction, TTFT | 19.53 s | 19.85 s | **0.64 s** |
| A GPU hit, TTFT | 1.11 s | 0.62 s | 0.77 s |
| Cold 135K prefill (×6) | 18.98–20.15 s | 19.59–20.44 s | 19.88–20.49 s |
| rotate round 1 | 142.2 s | 143.8 s | 144.1 s |
| rotate rounds 2 / 3 | 142.1 / 142.2 s | 143.6 / 143.8 s | **10.7 / 10.6 s** |
| rotate RAM-tier hit tokens | – | 0 | 1,843,200 |
| Code word, cold / after restore | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| Decode, short prompt (×3) | 125.4–127.7 tok/s | – | 126.2–127.8 tok/s |
| Decode, 130K context (×2) | 124.1–128.6 tok/s | – | 125.7–129.4 tok/s |
| Preemptions | 0 | 0 | 0 |

Decode numbers are from a separate run of the same prompts (now `kv_bench.py … decode`),
not stored in `raw/`. The `no-offload` g1 answers were empty in the first run (thinking went
to `reasoning_content`); `no-offload-g1-answers.json` is the rerun with thinking off.

Notes:
- The 32 GiB tier holds ~0.5M usable tokens, less than the GPU pool. The tier is inclusive,
  so a working set that overflows the GPU also overflows it: ~0 hits.
- First long request after a restart with the 64 GiB tier took 127.7 s (32 GiB: 42.7 s,
  no-offload: 18.5 s). Pinning 64 GiB pushes the PLE table out of page cache. Later cold
  prefills are back to normal.

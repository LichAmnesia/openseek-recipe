# Benchmarks: RTX PRO 6000 Blackwell Max-Q (300 W)

Measured 2026-09-21 with the upstream recipe's own `scripts/benchmark-suite.sh PART=core`,
so the workloads match the upstream README's. The hardware does not: upstream used a
**400 W** card; this is the **300 W Max-Q** edition. Raw files are in
[`results/2026-09-21-maxq-300w/raw/`](../results/2026-09-21-maxq-300w/raw/).

## Setup

| | |
|---|---|
| GPU | RTX PRO 6000 Blackwell Max-Q Workstation, 96 GB, 300 W hard cap, driver 610.57 |
| Host | Core Ultra 9 285K (24C), 128 GB DDR5, Samsung 990 EVO Plus 4 TB NVMe, Arch Linux |
| Profile | `exl3-ple8`: EXL3 K4.25 experts, FP8 PLE table mmap'd from NVMe |
| Serve | MTP 3, FP8 KV cache, 262,144 context, 16 slots, `gpu-memory-utilization 0.94` |
| Recipe | `tpurtell/sm12x-exl3-qwen3.8-flash-next` @ `847f45b`, image digest `sha256:0f3fdb9e…` |

## Headline vs the 400 W reference

| Metric | Max-Q 300 W (this repo) | 400 W reference (upstream README, same profile) | Delta |
|---|---:|---:|---:|
| C1 seven-workload weighted decode | **134.4 tok/s** | 147.78 | **−9.0%** |
| C16 aggregate decode | **595.2 tok/s** | 659.75 | **−9.8%** |
| API tool constraints | 16/16 | 16/16 | = |
| Vision 1 / 4 / 16 images | pass | pass | = |

You give up 25% of the power budget and lose about 10% of the throughput.

One more reason the gap may be smaller than it looks: on Max-Q the B12x MoE kernel logs
`profile 'nvidia.rtx.pro.6000.blackwell.max-q' does not cover the query` and falls back to
a heuristic config. The Max-Q has the same GB202 die and 188 SMs as the 400/600 W cards,
so the tuned profile is probably usable. That hasn't been tested yet.

## Seven workloads (C1, 3 timed runs each)

| Case | decode tok/s (median) | TTFT ms | quality contract |
|---|---:|---:|:-:|
| math | 186.1 | 100 | 3/3 |
| code (`merge_intervals`) | 181.0 | 96 | 3/3 |
| structured-json | 170.4 | 102 | 3/3 |
| hello | 162.4 | 84 | 3/3 |
| topic | 126.2 | 99 | 3/3 |
| fable | 106.5 | 101 | 1/3 |
| multilingual | 102.3 | 99 | 2/3 |

TTFT stays flat at 85–102 ms for every workload.

The `fable` and `multilingual` contract misses are word-count and format checks (for
example "fable has 177 words, outside 140..170"). Upstream says contract compliance varies
across profiles. There was no same-workload baseline on other hardware, so these misses
can't be blamed on the card.

## MTP speculative decoding explains the spread

| Suite | Draft tokens | Accepted | Acceptance | Decode |
|---|---:|---:|---:|---:|
| seven (mixed real prompts) | 3,831 | 2,013 | **52.5%** | 102–186 tok/s |
| orchid (repeat one word 100×) | 2,766 | 2,686 | **97.1%** | **215.5 tok/s** |

With 3 draft tokens per step, tok/s mostly depends on how predictable the output is. Math
and code are predictable, so they decode fast. Free prose and multilingual text are not,
so they decode slowly. A single "Flash-Next does N tok/s" number means little unless it
names the workload.

## Concurrency (256 output tokens, 3 runs per point)

| Concurrency | aggregate tok/s (median) | per-stream | TTFT ms |
|---:|---:|---:|---:|
| 1 | 104.5 | 104.5 | 85 |
| 2 | 170.9 | 85.5 | 99 |
| 4 | 205.0 ⚠️ | 51.3 | 207 |
| 8 | 431.1 | 53.9 | 152 |
| 16 | **595.2** | 37.2 | 317 |

⚠️ C4 ranged from 52 to 256 tok/s across its three runs. C8 and C16 are tight (±3%), so
the C4 spread looks like a one-off stall. Re-run it before quoting that number.

The C1 figure here (104.5) is lower than the seven-workload numbers because this test
samples free prose, where MTP acceptance is lowest.

## Resources at peak

| | |
|---|---|
| VRAM | 92,825 / 97,887 MiB (94.8%) |
| Power | 299.9 W (at the 300 W cap the whole run) |
| GPU temp | 83 °C (blower) |
| KV cache pool | 842,423 tokens = 3.2× the 262,144 context |
| Host RAM | ~20 GiB used + ~107 GiB page cache (PLE mmap, clean and reclaimable) |
| Token embeddings | 1.18 GiB pinned host RAM × 2 (target + MTP draft) |
| Cold start | ~3 min 16 s (weights 31 s with a warm page cache, graph capture ~15 s) |

## Not measured yet

- `PART=prefill`, `context`, `retrieval`, `coding`. The full-context boundary test
  (261,888 input tokens) hasn't been run on Max-Q.
- Aliasing the 400 W B12x tuning profile to the Max-Q name.
- `--max-num-batched-tokens` above 2048. vLLM warns that MTP slots crowd the 2048 budget.

Reproduce:

```bash
PART=core scripts/bench.sh
```

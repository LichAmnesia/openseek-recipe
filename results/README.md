# Results

| Run | Card | Profile | Notes |
|---|---|---|---|
| [2026-09-21-maxq-300w](2026-09-21-maxq-300w/SUMMARY.md) | RTX PRO 6000 Max-Q, 300 W | exl3-ple8, MTP 3 | `PART=core` + SWE-bench Pro 50-subset (no KV offload) |
| [2026-10-21-maxq-kv-ram-offload](2026-10-21-maxq-kv-ram-offload/SUMMARY.md) | RTX PRO 6000 Max-Q, 300 W | exl3-ple8, MTP 3, LMCache 0.5.6 | `scripts/kv_bench.py`: no tier vs 32 GiB vs 64 GiB RAM tier |

`raw/` holds the unmodified output of the upstream `benchmark-suite.sh` (paths anonymised).
The `raw/*.jsonl` files include the benchmark prompts from the upstream suite
([tpurtell/sm12x-exl3-qwen3.8-flash-next](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next),
Apache-2.0) together with the model's responses. `swebench-pro/` holds only instance IDs,
test counts and pass/fail verdicts. No dataset content and no patches are included.

Regenerate the summary: `scripts/summarize.py results/<run>/raw`.

Have a different card (400 W / 600 W / Server edition) or a different profile? A PR with
your `results/<date>-<card>/` directory is very welcome; include `gpu.txt`.

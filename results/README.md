# Results

| Run | Card | Profile | Notes |
|---|---|---|---|
| [2026-09-21-maxq-300w](2026-09-21-maxq-300w/SUMMARY.md) | RTX PRO 6000 Max-Q, 300 W | exl3-ple8, MTP 3 | `PART=core` + SWE-bench Pro 50-subset |

`raw/` holds the unmodified output of the upstream `benchmark-suite.sh` (paths anonymised).
Regenerate the summary: `scripts/summarize.py results/<run>/raw`.

Have a different card (400 W / 600 W / Server edition) or a different profile? A PR with
your `results/<date>-<card>/` directory is very welcome; include `gpu.txt`.

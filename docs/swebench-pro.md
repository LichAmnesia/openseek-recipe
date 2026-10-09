# SWE-bench Pro on one 300 W card: 66.0%

**33 / 50 resolved (66.0%), Wilson 95% CI [52.2%, 77.6%].** The model ran locally on one
RTX PRO 6000 Max-Q at 300 W. No API calls were made, so the run cost $0 beyond
electricity.

Qwen's model card reports **62.5** on SWE-bench Pro for the full-precision model. The
local EXL3 K4.25 quant lands inside that range.

## Setup

| | |
|---|---|
| Harness | [`scaleapi/SWE-bench_Pro-os`](https://github.com/scaleapi/SWE-bench_Pro-os), official `swe_bench_pro_eval.py --use_local_docker` |
| Dataset | `ScaleAI/SWE-bench_Pro` test split (731 instances) |
| Sample | 50 instances, stratified by repo, largest-remainder, `random.Random(20260921)` |
| Scaffold | `mini-swe-agent` (Scale fork), `step_limit 250`, 6 parallel workers |
| Model | `qwen38-exl3-ple8`, `temperature 0.0`, `max_tokens 4096` |

Instance IDs: [`results/.../swebench-pro/sample50_ids.txt`](../results/2026-09-21-maxq-300w/swebench-pro/sample50_ids.txt).
Per-instance verdicts: [`verify_rows.json`](../results/2026-09-21-maxq-300w/swebench-pro/verify_rows.json).

A verdict counts as resolved only when **every** `fail_to_pass` and `pass_to_pass` test
passes. All 50 verdicts were re-derived independently from the harness's
per-instance output: 33 strict, 33 harness, 0 disagreements.

## Per repo

| Repo | Resolved |
|---|---:|
| qutebrowser | 5/5 |
| vuls | 4/4 |
| navidrome | 4/4 |
| tutanota | 1/1 |
| element-web | 3/4 |
| ansible | 5/7 |
| openlibrary | 4/6 |
| NodeBB | 2/3 |
| teleport | 3/5 |
| flipt | 2/6 |
| protonmail/webclients | 0/5 |

Each repo has 1 to 7 instances, which is too few to say anything about how well the
model handles a particular repo.

## The 66% is a lower bound

In 13 of the 50 instances, the target `fail_to_pass` tests never appeared in the
executed output. Six of those ran zero tests. The harness scores all of them as failures.
All 5 protonmail instances fall in this group, so its 0/5 is almost certainly an
environment problem rather than five model failures.

Excluding those 13 gives 33/37 = 89%. That figure is biased upward, so don't quote it.
The defensible number is 66%, and on a fully working harness the true rate is likely
somewhere between 66% and 89%.

## Run stats

| | |
|---|---|
| Wall clock | 1 h 55 min, 6 workers |
| Exit status | 50/50 submitted, none hit the step limit |
| Agent steps | 3,141 total (median 62, max 223 per instance) |
| Non-empty patches | 49/50 |
| GPU during run | ~300 W, 100% util, ~197 tok/s generation across 5–6 concurrent requests |
| Prefix-cache hit rate | **67.4%**. The agent re-sends its growing history, which is why 6-way concurrency works |

## Three harness traps (each fails silently)

**1. `load_dataset` rejects a bare `.json` file.** It raises a misleading
`FileNotFoundError` even though the path exists. Fix: write a directory containing
`test.jsonl`.

**2. litellm has no price for a local model, so every instance dies at step 0.**
Fix: `export MSWEA_COST_TRACKING=ignore_errors`.

**3. Image `ENTRYPOINT` vs `sleep`, and `/testbed` vs `/app`.** This is the expensive one.
The SWE-bench Pro images declare `ENTRYPOINT ["/bin/bash"]`. mini-swe-agent runs
`docker run -d <image> sleep 2h`, which becomes `/bin/bash sleep 2h`. The container exits
immediately, but `docker run -d` still returns exit code 0. The agent then reads every
failed `docker exec` as task feedback and grinds toward the step limit. The first attempt
burned 186 model calls before this surfaced. The images also use `/app`, while
mini-swe-agent assumes SWE-bench classic's `/testbed`.

Fix, in your mini-swe-agent config:

```yaml
environment:
  cwd: "/app"
  run_args: ["--rm", "--entrypoint", ""]
```

Also replace `/testbed` with `/app` in the prompt templates so the model gets the right
path.

## Caveats

This is one subset of n=50 out of 731, one scaffold and one seed, with no repeat runs.
Don't compare it with the 2025 launch leaderboard (~23% for the frontier models of that
time) as if the two were measured at the same time.

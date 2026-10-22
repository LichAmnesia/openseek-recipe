# Legacy: no KV offload

The original setup from before 2026-10-21: the stock recipe image, GPU-only prefix cache,
`--max-num-batched-tokens 2048`. The main scripts at the repo root now run the
[KV RAM offload](../../docs/kv-ram-offload.md) version by default. Use this folder only if
your host has less than ~120 GB of RAM or you want to A/B against the old behaviour.

```bash
# from the repo root, after scripts/setup.sh
cp legacy/no-offload/.env.example legacy/no-offload/.env   # optional: overrides the main .env
legacy/no-offload/scripts/ctl.sh recreate   # removes qwen38-lmcache if present, then starts
legacy/no-offload/scripts/ctl.sh status
```

It creates the same container name (`qwen38-exl3-ple8`) on the same port, so only one of
the two setups runs at a time. To go back to the main setup: `scripts/ctl.sh recreate`.
`scripts/doctor.sh`, `scripts/setup.sh`, `scripts/smoke_test.py` and `scripts/bench.sh` at
the repo root work for both.

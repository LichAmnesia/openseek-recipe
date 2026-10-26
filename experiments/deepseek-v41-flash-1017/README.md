# Experimental DeepSeek 1017 2.7bpw port

[Complete runbook](../../docs/deepseek-v41-flash-1017.md) ·
[Measured results](../../results/2026-10-26-deepseek-2.7bpw-eager/SUMMARY.md) ·
[Licenses](THIRD_PARTY_NOTICES.md)

The real single-card TP1/eager first gate measured 10.64 tok/s and 5.31 s TTFT.
Speed targets were missed and full quality acceptance is incomplete. This
standalone experiment is not registered in the Qwen-specific main recipe CLI.

```bash
python3 experiments/deepseek-v41-flash-1017/prepare.py --root "$HOME/.local/share/openseek/dsv41-2p7-eager" --dry-run
bash experiments/deepseek-v41-flash-1017/serve.sh --dry-run
```

`runtime-lock.json` holds 152 pinned/hash-checked source/config files and the
measured local image identity; `model-files.json` holds the 55-file checkpoint
manifest. Sources and weights are fetched from their original publishers.
Portable packaging has offline checks; a fresh host/image still needs numeric
and API qualification. See the runbook for build, shared-GPU handoff, measurement
and video restoration on both successful and failed attempts.

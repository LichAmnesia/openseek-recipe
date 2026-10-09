# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | nvidia runtime not registered. `scripts/install-prereqs.sh`, then restart docker |
| `permission denied ... docker.sock` | You were just added to the `docker` group. Open a new login shell |
| Container exits during startup, `CUDA out of memory` | Something else holds VRAM. `nvidia-smi`, stop it. The server claims 94% of the card |
| Startup takes 10+ min the first time | Cold page cache: 120 GB of weights read from disk. Later starts with warm cache take ~3 min |
| `b12x policy fallback: moe.decode is using a heuristic ... max-q` | Expected on the Max-Q edition: no tuned profile name match. Costs some speed, not correctness |
| `max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings` | Expected with MTP 3. Raise `MAX_BATCHED_TOKENS` in `.env` + `scripts/ctl.sh recreate` if you care about high-concurrency throughput |
| Edited `.env`, nothing changed | Serve flags are baked in at container creation. `scripts/ctl.sh recreate` |
| Server down after reboot | Created before restart policy was set, or you `docker stop`ped it (`unless-stopped` respects that). `scripts/ctl.sh start` |
| LAN clients time out, localhost works | Host firewall. Example: `sudo ufw allow from 192.168.1.0/24 to any port 8001 proto tcp` |
| Model answers but never calls tools | Server must have `--enable-auto-tool-choice --tool-call-parser qwen3_coder` (the recipe sets both). Check with `scripts/smoke_test.py` |
| HTTP 404 `model ... does not exist` | Use the id from `/v1/models` (`qwen38-exl3-ple8`), not the Hugging Face repo name. In OpenCode the `rtx6000/` prefix is client-side only |
| Swap grows a little during benchmarks | Cold anonymous pages pushed out under page-cache pressure from the 120 GB checkpoint. ~1 GiB observed, no throughput dip |
| Slow decode on a 64 GB RAM host | PLE table (~48 GiB FP8) is mmap'd; page-cache misses dominate. 128 GB RAM recommended |
| Download is slow / rate-limited | Anonymous HF pulls are throttled. Log in on the host (`huggingface-cli login`) before `scripts/setup.sh` |

Wrong-host sanity check: the `/v1/models` response should show `max_model_len: 262144`.

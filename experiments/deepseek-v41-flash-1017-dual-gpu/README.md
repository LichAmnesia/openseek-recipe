# DeepSeek 1017 on two RTX PRO 6000 and 128 GB of RAM

[Runbook](../../docs/deepseek-v41-flash-1017-dual-gpu.md) ·
[Measured results](../../results/2026-11-17-deepseek-3bpw-dual-gpu/SUMMARY.md)

Serves DeepSeek-V4.1-Flash-1017 on 2× RTX PRO 6000 Blackwell Max-Q (96 GB, 300 W each) with 128 GB of host RAM.
Measured 2026-11-17: **72 tok/s on one stream, about 99 tok/s across four** on code prompts. The single-GPU experiment
in [`../deepseek-v41-flash-1017`](../deepseek-v41-flash-1017/) measured 10.64 tok/s.

Nothing here is a fork. It is three upstream pieces and the settings this board needs:

| Piece | Source | Pinned at |
|---|---|---|
| 3-bit EXL3 weights, vLLM image, kernels, `serve.sh` | [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) | `6110453b901e4b51da66f539f3802b5c7f18e8e8` |
| Engram tables on NVMe instead of pinned RAM | [yoloyash/dsv41-flash-1017-2x-rtx-pro-6000-ssd-engram](https://github.com/yoloyash/dsv41-flash-1017-2x-rtx-pro-6000-ssd-engram) | `ca6491e` |
| Launcher and probes | this directory | |

```bash
API_KEY=change-me PACK=$HOME/models/dsv41-1017-3bpw OVERLAY=$HOME/src/ssd-engram ./serve.sh parity   # once
API_KEY=change-me PACK=$HOME/models/dsv41-1017-3bpw OVERLAY=$HOME/src/ssd-engram ./serve.sh
API_KEY=change-me python3 smoke.py
```

`serve.sh` differs from the overlay's defaults in four ways: port 8001, `CONTAINER_MEM=108g`, `HOST_CACHE=512`, and
the peer-to-peer features off (`DS41_FP8AR=0 NCCL_P2P_DISABLE=1 DS41_CAR=0 DS41_CAR_PCIE_IPC=0`). Set `P2P=1` on a
board where `nvidia-smi topo -p2p r` reports `OK` between the cards.

| File | What it does |
|---|---|
| `serve.sh` | Launcher. `serve.sh parity` runs the eager byte-check mode |
| `smoke.py` | Four requests: Chinese answer, code, 6K-token prompt, tool call |
| `bench/practical.py` | Speed by workload, plus coding / math / long-context checks verified by code |
| `bench/concurrency.py` | 1 to 8 streams with a cached 20K-token context each |
| `bench/prose_probe.py` | Decode speed on Chinese and English prose |

The decode and prefill numbers come from Diffbot's own `recipe/bench/bench-fixed.py`, unchanged.

Weights are MIT (Diffbot). The image Diffbot's scripts build includes an AGPL-3.0 `vllm-exl3` patch; see the
pack's `recipe/THIRD_PARTY_NOTICES.md`. The overlay is MIT + Apache-2.0. The files in this directory are MIT.

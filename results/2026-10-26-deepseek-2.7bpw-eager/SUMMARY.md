# DeepSeek 1017 2.7bpw first gate — 2026-10-26

One RTX PRO 6000 Blackwell Max-Q96GB / 300W, host128GB, local NVMe.
TP1/eager, batch512, seq1, FP8KV1.1GiB, NVMe Engram, 8220 host experts /
92GiB trellis. No graphs, DSpark, vision, host cache, CPU experts or DMA.
Extra cold expert IDs are unprofiled. [Reproduction](../../docs/deepseek-v41-flash-1017.md).

| Workload | Input/output tokens | Decode tok/s | TTFT s |
|---|---:|---:|---:|
| C1 trial1 | 1550/256 | 10.5752 | 8.4490 |
| C1 trial2 | 1550/256 | 10.6374 | 5.3096 |
| C1 trial3 | 1550/256 | 10.6641 | 5.2358 |
| **C1 median** | **1550/256** | **10.6374** | **5.3096** |
| 34K prefix warmed +6K new | 40000/256 | 10.9388 | 16.2935 |

Decode=255/(last text event−first text event), local SSE. All four streams:
256 completion tokens, 256 text events, finish_reason=length. First-trial latency
is included. Raw repeated-text completions are not useful-task performance.
Arithmetic42 and parsed get_weather(San Francisco) smoke passed.
Prefix hits increased33920; API cached_tokens absent. Some cache hits verified,
API amount unverified. Not actual OpenCode traffic.

| Actual resource observation | Value |
|---|---:|
| Inference GPU sampled peak | 97280MiB =95.00GiB |
| Kernel cgroup peak, includes cache/earlier load | 100.000008GiB |
| Inference sampled current peak | 99.9990GiB |
| Inference anonymous-memory peak | 96.2308GiB |
| Host minimum MemAvailable / total | 16.3907/125.0604GiB |
| Host total minus minimum available (not model RSS) | 108.6697GiB |
| Container swap / OOM / OOM kills | 0/0/0 |

1074 raw1-second samples, 233 in correctness/speed phase. Memory max-limit events
968412 are not OOM events; independent peaks should not be summed.

**Below the original speed gates**: C1<20tok/s; warmed agent-shaped TTFT>6s.
This is an experimental port, not a qualified resident replacement. No short100
KL, SWE-bench, 256K quality, graphs or optimized routing acceptance.

The first load failed on the MXFP8 head namespace/scale allocation, not OOM.
The repaired eager-v2 run finished01:03:24 PDT; model exited0 and was retained;
video restored/verified01:03:47 and independently checked healthy01:11:03.
Qwen containers/checkpoints preserved; training was not restarted.

[benchmark.json](benchmark.json), [measurements.json](measurements.json),
[resources.jsonl](resources.jsonl), [provenance.json](provenance.json).
Host/model and cgroup paths are anonymized; numeric values are unchanged.
The measured local image ID and immutable model/recipe revisions are in
provenance. A new build is not assumed byte-identical or GPU-qualified.

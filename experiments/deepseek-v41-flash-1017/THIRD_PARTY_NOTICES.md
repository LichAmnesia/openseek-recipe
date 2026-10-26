# Provenance and licenses

This is an experimental first-gate port, not a new quantization or model release.

- Model: DeepSeek-V4.1-Flash-1017, quantized by sfxnz. Original checkpoint and
  its notices remain at the pinned Hugging Face repository in `model-files.json`.
- Serving recipe: Diffbot, MIT, HF commit
  `2c3797578722747dad3593531fd447b1d3d4c732`. Its LICENSE and third-party notices
  are fetched intact with the other hash-locked sources.
- Mixed-bit loader edits derive from sfxnz `widen_p2b_kbits.py`, commit
  `19e0c56620a7b9748fd901cdf3ef533a0f6ed917`, and modify the AGPL-3.0-only
  vllm-exl3 host-expert port. **`scripts/patch_mixed_k.py` and its resulting
  plugin modifications are AGPL-3.0-only**, not covered by the root MIT license.
  See [LICENSE-AGPL-3.0.txt](LICENSE-AGPL-3.0.txt). The generated kernel source
  retains upstream ExLlamaV3 notices.
- Disk Engram and head layout/mapping derive from the pinned sfxnz helpers.
  The source LICENSE is fetched alongside them. The head compute path uses
  Diffbot's `sm120_dense.py`; no Spark b12x runtime is bundled here.
- Remaining orchestration, checks and measurement helpers use this repository's
  MIT license. No weights or container images are redistributed.

The local measured image ID records evidence; it is not a public registry
artifact. Base digest and source revisions are pinned, but apt/compiler package
artifacts are not fully locked. A rebuild may differ and needs its own gates.

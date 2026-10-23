---
read_when:
  - Adding a model, hardware target, published checkpoint, or release workflow.
  - Changing the scope of OpenSeek Recipe beyond its validated Qwen deployment.
---

# OpenSeek Recipe roadmap

OpenSeek Recipe packages reproducible local inference deployments. Its first
validated combination is Qwen3.8-Flash-Next EXL3 K4.25 with FP8 PLE, a patched
SM120 vLLM runtime, and one RTX PRO 6000 Max-Q 96 GB with 128 GB host RAM.

The `./recipe` entry point, JSON model/profile configuration, immutable locks,
checkpoint integrity verification and offline regression gates are implemented
in 0.1.0. Later milestones below remain planned.

This is a recipe repository. **OpenSeek TUI owns chat/coding. SparkDash owns
future readiness/resource/KV/performance visualization and diagnostics.**
The recipe exports connection metadata and runs the chosen backend.

## Reference and scope

[Project Maya](https://github.com/mw00/project-maya/tree/25deec505b525212e05faa58697d9d2b7d17fdb3)
combines model-specific engines, installation, an API server, a dashboard,
evaluation and quantization tools, and
[downloadable model artifacts](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF/tree/d1fc01009f21e26157acb6858c315d0365483b72).
Use that complete installation-to-measurement experience as a reference.

OpenSeek Recipe can build that experience on established runtimes. An independently
implemented inference engine is a separate research project, justified only by
measured needs that existing runtimes cannot meet.

The current Qwen deployment uses GPU expert weights, host-backed PLE pages, and
a host-RAM prefix-cache tier. Its KV restore results are not evidence of Maya's
GPU/RAM/SSD expert placement or active within-request KV streaming.

## 1. Unified entry point, configuration and locks (0.1.0)

Implemented: `./recipe models/config/install/start/smoke/connection/verify/lock`,
recipe/profile separation, 40-file checkpoint lock, runtime image/commit lock,
LMCache wheel hash, local installation receipt, and CPU-only CI. Existing script
entry points remain compatible. See [the CLI guide](CLI.md).

Tests cover config precedence, immutable-lock rejection, real shell installer
orchestration with a local Git fixture and fake Docker boundary, interrupted
install/retry, cached reuse, and corruption detection. Those gates do not replace
fresh GPU-host qualification; record that separately before claiming it passes.

Follow-up release engineering:

Deliverables:

- Release metadata, a changelog, an explicitly selected version, and a recovery
  path to the previous known-good configuration/image without redownloading
  weights or silently replacing a running service.

Acceptance: on a clean target host, setup reaches an API that passes chat,
reasoning, a real tool call, streaming, and Responses checks. A repeated setup
reuses verified downloads. An interrupted setup recovers. A deliberately corrupt
shard is detected. Capture exact versions and raw output in the release receipt.

## 2. Installation, model selection and external-client integration

- A supported-model/hardware table distinguishing locally validated,
  community-reported, experimental, and unsupported combinations.
- Setup recommends only measured profiles. Hardware checks do not imply an
  automatic performance tuner exists.
- Keep installation progress, selection and actionable prerequisite/download
  errors in this repo. Reuse OpenSeek TUI for conversation; export its connection
  metadata instead of implementing a second TUI or browser chat.
- Keep performance, VRAM/RAM/KV monitoring, dashboards and deeper diagnostics
  with SparkDash. Agree its integration contract later; do not build a parallel
  collector or dashboard in the recipe repository.
- Keep the OpenAI-compatible API and tested client configurations. Add an
  Anthropic-compatible gateway only with protocol and tool-result tests;
  changing a URL alone is not protocol compatibility.
- CPU-only CI for CLI/config/download behavior; separately recorded GPU
  qualification for each supported recipe. Add structured community reports
  with model revision, runtime, hardware, context, concurrency, and raw results.

Acceptance: another user can clone, select a real published model, install a
compatible locked recipe, and use it from OpenSeek TUI. Selecting a model alone
does not stop or silently replace an existing service. SparkDash integration is
an independent later gate.

## 3. Choose the Hugging Face artifact strategy

There are three different outputs:

| Output | Required content | Accurate description |
|---|---|---|
| Deployment card | README, pinned recipe, source links, runtime lock, measurements | Documentation; weights download from their original publisher |
| Redistribution of an existing checkpoint | Verified redistribution terms, exact source revisions and hashes, notices, complete checkpoint assets | Attributed mirror/distribution; no claim of new quantization |
| Project-produced checkpoint | Reproducible conversion/quantization, calibration provenance, held-out evaluation, complete release files | An OpenSeek quantization or transformed checkpoint, with its actual authorship and lineage |

The current project has the first output. A Maya-style model repository with real
weight downloads requires the second or third. A card-only page cannot provide
the same downloadable checkpoint experience.

Keep the artifact format tied to a tested loader. The existing EXL3/FP8-PLE
checkpoint does not need to become GGUF to match the reference's presentation.
Do not label an EXL3 checkpoint as GGUF or assume a format conversion preserves
quantization, numerical behavior, tool formatting, or runtime compatibility.

For a project-produced quantization:

1. Pin the source checkpoint, tokenizer, conversion/quantization tool, and
   runtime. Confirm supported model architecture, storage, and memory first.
2. Publish authorized calibration data provenance and a frozen held-out set.
   Keep calibration and evaluation inputs separate.
3. Compare against the pinned source using NLL/KL or next-token agreement where
   feasible, plus code, tool/JSON, reasoning, multilingual, and long-context
   tasks. Next-token agreement is not task accuracy.
4. Report disk size, peak RAM/VRAM, cold load, TTFT, prefill, decode, concurrency,
   and task success under named hardware and settings.
5. Publish shards, config, tokenizer/processor, needed auxiliary components,
   file hashes, source lineage, license notices, and a tested launch command.

Do not advertise a size/quality tier until its artifact exists and its gates
pass. Source model scores and another publisher's measurements remain their
results, not qualification of a new OpenSeek artifact.

## 4. Expand one measured combination at a time

Choose a second model or hardware target after the first release works on a
second installation. Record explicit adapter requirements and separate results
for each combination. Windows, AMD, smaller GPUs, multiple GPUs, and new model
families need their own qualification.

Do not promise Maya's hardware range or engine performance based on this
repository's name, directory structure, or use of an existing backend.

## Naming migration

- Public repository: `LichAmnesia/openseek-recipe`; display name: OpenSeek Recipe.
- Update current clone commands, badges, homepage/HF links, release examples,
  and local Git remotes. Preserve historical result files and their identities.
- Keep the existing served model ID and container/image names during the brand
  migration. Rename runtime resources only with a separate compatibility plan.
- GitHub redirects ordinary old repository URLs and Git operations. GitHub Pages
  and references to a repository used as a GitHub Action require separate checks.
  Do not reuse the old repository name for a different project while redirects
  are needed. See [GitHub's rename documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository).

Initial migration changes documentation and repository identity. No GPU service
restart, weight download, model conversion, or new supported-target claim is
part of the rename.

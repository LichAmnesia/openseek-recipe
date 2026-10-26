# Changelog

## Unreleased

- Document the measured DeepSeek-V4.1-Flash-1017 2.7bpw TP1/eager experiment,
  including source/checkpoint locks, mixed-bit/disk-Engram/MXFP8-head patches,
  preparation and serving commands, numeric gates and shared-GPU handoff.
- Preserve benchmark/resource evidence and unpassed speed/quality boundaries;
  keep the experiment separate from the qualified Qwen CLI adapter.

## 0.1.0

- Rename the repository to OpenSeek Recipe, retaining its validated Qwen recipe.
- Add `./recipe` for selection, configuration, installation, lifecycle, integrity
  checks, API smoke tests, and client connection metadata.
- Separate recipe, hardware profile, and immutable checkpoint/runtime locks.
- Pin 40 checkpoint files, the runtime commit/image digest, and LMCache wheel hash.
- Add install-retry/cache-corruption regression tests and CPU-only CI.
- Keep chat in OpenSeek TUI and future monitoring/performance work in SparkDash.

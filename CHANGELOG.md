# Changelog

## 0.1.0

- Rename the repository to OpenSeek Recipe, retaining its validated Qwen recipe.
- Add `./recipe` for selection, configuration, installation, lifecycle, integrity
  checks, API smoke tests, and client connection metadata.
- Separate recipe, hardware profile, and immutable checkpoint/runtime locks.
- Pin 40 checkpoint files, the runtime commit/image digest, and LMCache wheel hash.
- Add install-retry/cache-corruption regression tests and CPU-only CI.
- Keep chat in OpenSeek TUI and future monitoring/performance work in SparkDash.

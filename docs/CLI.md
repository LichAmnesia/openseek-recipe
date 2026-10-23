# Recipe CLI and configuration

Python 3.10+, standard library only. Use `./recipe` from a Git clone; listing,
configuration, connection metadata, verification, and API smoke tests work on
client machines. GPU installation/lifecycle commands target x86_64 Linux.

The command does not install a global `openseek` binary. OpenSeek TUI owns chat;
SparkDash owns future monitoring/performance integration.

## Commands

| Command | Effect |
|---|---|
| `./recipe --help`, `help install`, `--version` | Offline help/version |
| `./recipe models list` | Registered, real model recipes |
| `./recipe models show MODEL` | Recipe, supported profile, immutable lock |
| `./recipe models select MODEL` | Save local selection; no service switch |
| `./recipe config init` | Create local config; refuse overwrite |
| `./recipe config show` | Effective configuration and its lock |
| `./recipe lock` | Export selected model/runtime versions and lock hash |
| `./recipe install --dry-run` | Preview without downloads, writes or processes |
| `./recipe install` | Preflight, pinned vendor/image/wheel, checkpoint download and hashes |
| `./recipe doctor` | Existing prerequisite checks; no package installation |
| `./recipe start`, `stop`, `restart`, `recreate`, `down`, `wait` | Existing two-container lifecycle; down preserves weights |
| `./recipe logs [server\|lmcache]` | Follow existing container logs |
| `./recipe smoke` | Real API checks, including tool calling |
| `./recipe connection` | Client endpoint, served model ID and protocol as JSON |
| `./recipe verify` | Full local SHA256 check of every locked file |
| `./recipe verify --fast` | Size/completeness only; not a content-integrity verdict |
| `./recipe bench` | Existing upstream benchmark integration |
| `./recipe status` | Existing basic service output; no new monitoring platform |
| `./recipe check` | Offline unit/integration fixtures, shell syntax, lock and dry-run gates |

`--model`, `--profile`, `--config`, `--port`, `--gpu`, `--hf-cache`, and
`--base-url` may appear before or after a subcommand. Data commands support
`--json` (object-oriented commands already print JSON). Mutating script actions
stream their existing logs and reject `--json`; use `--dry-run --json` to inspect
their plan. `--dry-run` never starts containers, downloads, or writes config.

Examples:

```bash
./recipe config init --port 8002 --hf-cache /mnt/nvme/huggingface
./recipe config show --json
./recipe install --model qwen38-flash-next --dry-run
./recipe install
./recipe start
./recipe smoke --base-url http://gpu-host:8002/v1
./recipe connection --base-url http://gpu-host:8002
./recipe verify --fast
./recipe lock > selected-lock.json
```

## Configuration

Highest precedence first:

1. Invocation flags.
2. Process environment, including `OPENSEEK_RECIPE_MODEL` and
   `OPENSEEK_RECIPE_PROFILE` for selection.
3. Legacy repository `.env` for paths, ports and serving settings.
4. Local `.recipe/config.json` (`--config` or `OPENSEEK_RECIPE_CONFIG` selects another file).
5. Registered model recipe, its hardware profile, and defaults.

The immutable lock is checked independently: local overrides cannot change the
upstream commit/repository, base image digest, LMCache version, or derived image
tag. A reviewed lock change is required. The qualified MTP draft length and
hybrid block size are also validated.

```json
{
  "schema_version": 1,
  "model": "qwen38-flash-next",
  "profile": "rtx-pro-6000-kv64",
  "env": {
    "GPU": "0",
    "PORT": "8001",
    "HF_CACHE": "/mnt/nvme/huggingface"
  }
}
```

Relative filesystem paths resolve against the clone, not the caller's current
working directory. Legacy `.env` supports simple `KEY=value`, quoted spaces and
basic variable expansion; shell commands/substitutions are rejected by the CLI.
Empty `MAX_BATCHED_TOKENS` derives `2 * KV_BLOCK_TOKENS - 1`. API and LMCache
ports must be distinct. Model selection does not change a running service.

HF authentication uses the downloader's normal HF token/cache mechanism;
`HF_TOKEN` is forwarded by environment name when present, never printed or
saved in exported configuration. Do not put credentials in JSON, `.env`, URLs,
command-line flags or checked-in manifests.

## Files and installation guarantees

- `recipes/*.json`: model identity, backend, lock ID, supported profiles and
  model-level settings.
- `profiles/*.json`: qualified hardware settings and evidence locations.
- `locks/*.lock.json`: exact HF commit, per-file sizes/hashes, upstream runtime
  commit, base image digest, and the exact LMCache wheel/hash.
- `.recipe/config.json`: local selection/settings; ignored by Git.
- `.recipe/installed-lock.json`: successful-install receipt; ignored by Git.

The recipe pins 40 checkpoint files. Missing files are downloaded through
`huggingface_hub` in the digest-pinned runtime image. SDK downloads use the HF
cache and can be retried after interruption. A complete cache avoids the SDK
network call; full SHA256 checks still run. A `config.json` alone is insufficient.
Wrong-size files fail before download; same-size corruption fails the hash check.
Damaged shared cache files are not silently overwritten: move them aside after
checking which snapshots reference them, then retry.

The derived KV image has a lock-specific tag and label. Installation preserves
tracked edits in a user-supplied upstream checkout by refusing to overwrite
them. Existing legacy `scripts/*.sh` entry points remain available for previous
users; **the unified CLI is the version-locked contract**. Direct legacy scripts
retain their `.env` behavior and do not acquire the full lock guarantees.

`start` and `restart` refuse existing containers using another image. To migrate
an old deployment, run `install`, inspect the plan/config, then explicitly run
`recreate` to replace its containers. After changing local serving settings,
`recreate` is also required. This command stops the old service and creates the
new containers; it is not an automatic zero-downtime deployment.

## Registering another HF release

Add an actual published checkpoint, complete immutable lock, a compatible
backend adapter, and a qualified hardware profile. The current backend binds to
the pinned upstream Qwen profile and rejects another checkpoint identity until
its loader path is adapted and verified. Editing `repo_id` alone is insufficient.
Own-published weights will use the same registry, after artifact and loader
qualification. Do not register a placeholder HF repository or relabel another
publisher's quantization as new OpenSeek quantization work.

## Validation and errors

`./recipe check` exercises config precedence, lock rejection, dispatcher behavior,
interrupted install/retry, intact cache reuse, and content-corruption detection.
Installer tests use a real local Git fixture and fake the Docker/network boundary;
they do not establish clean-host GPU installation or CUDA correctness.

`./recipe smoke --base-url http://HOST:8001` checks the running API. It proves
protocol compatibility on that service; it does not qualify a newly installed
image, an entire model benchmark, or another hardware configuration.

Exit codes: `0` success; `1` failed verification/operational error; `2` invalid
usage/configuration; `130` Ctrl-C. Delegated scripts preserve their nonzero exit
codes. Errors go to stderr; data goes to stdout. Config/help/list/lock do not
perform network calls or telemetry.

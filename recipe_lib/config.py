"""Resolve data-only configuration without evaluating shell input."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "qwen38-flash-next"
SETTINGS = {
    "GPU", "PORT", "HF_CACHE", "RECIPE_DIR", "RUNTIME_CACHE", "BASE_URL",
    "MAX_MODEL_LEN", "MAX_NUM_SEQS", "GPU_MEMORY_UTILIZATION", "MTP_TOKENS",
    "KV_RAM_GB", "KV_RAM_WATERMARK", "KV_BLOCK_TOKENS", "MAX_BATCHED_TOKENS",
    "LMCACHE_PORT", "LMCACHE_HTTP_PORT", "LMCACHE_METRICS_PORT", "RESTART_POLICY",
    "READY_TIMEOUT", "PART", "RESULT_DIR", "TAIL", "QUANT",
}
LOCKED_SETTINGS = {"RECIPE_REPO", "RECIPE_COMMIT", "LMCACHE_VERSION", "BASE_IMAGE", "KV_IMAGE"}


class ConfigError(ValueError):
    """Expected configuration error, shown without a traceback."""


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a JSON object in {path}")
    return data


def named(root: Path, folder: str, name: str, suffix: str = ".json") -> dict:
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ConfigError(f"Invalid {folder} id: {name!r}")
    return read_json(root / folder / (name + suffix))


def load_lock(path: Path) -> dict:
    lock = read_json(path)
    try:
        if lock["schema_version"] != 1:
            raise ValueError("unsupported schema_version")
        model, runtime = lock["model"], lock["runtime"]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model["repo_id"]):
            raise ValueError("invalid model repo_id")
        for revision in (model["revision"], runtime["recipe_commit"]):
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("revisions must be full 40-character commit hashes")
        if not re.fullmatch(r"[\w./:-]+@sha256:[0-9a-f]{64}", runtime["base_image"]):
            raise ValueError("base_image must use an immutable sha256 digest")
        if not runtime["recipe_repo"].startswith("https://github.com/"):
            raise ValueError("recipe_repo must use GitHub HTTPS")
        wheel = runtime["lmcache"]
        if "/" in wheel["wheel"] or not wheel["wheel"].endswith(".whl"):
            raise ValueError("invalid LMCache wheel filename")
        if not re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"]):
            raise ValueError("invalid LMCache wheel hash")
        seen = set()
        for f in model["files"]:
            p = PurePosixPath(f["path"])
            if p.is_absolute() or ".." in p.parts or not p.parts or str(p) != f["path"]:
                raise ValueError("unsafe checkpoint file path")
            if f["path"] in seen or type(f["size"]) is not int or f["size"] < 0:
                raise ValueError("duplicate file or invalid size")
            if not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]):
                raise ValueError("invalid checkpoint SHA256")
            seen.add(f["path"])
        if not seen:
            raise ValueError("empty checkpoint file list")
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid lock {path}: {exc}") from exc
    return lock


def dotenv(path: Path, environ: dict) -> dict:
    """Accept simple legacy KEY=value lines; never execute substitutions."""
    if not path.exists():
        return {}
    values = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Z_][A-Z_0-9]*", key.strip()):
            raise ConfigError(f"{path}:{number}: expected KEY=value")
        key = key.strip()
        if key not in SETTINGS | LOCKED_SETTINGS:
            raise ConfigError(f"{path}:{number}: unsupported setting {key}")
        if "$(" in value or "`" in value:
            raise ConfigError(f"{path}:{number}: shell execution is not supported")
        try:
            parts = shlex.split(value, comments=True)
        except ValueError as exc:
            raise ConfigError(f"{path}:{number}: {exc}") from exc
        if len(parts) > 1:
            raise ConfigError(f"{path}:{number}: quote values containing spaces")
        value = parts[0] if parts else ""
        scope = {**environ, **values}
        value = re.sub(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))",
                       lambda m: scope.get(m[1] or m[2], ""), value)
        values[key] = value
    return values


def config_path(root: Path, args: dict, environ: dict) -> Path:
    path = args.get("config") or environ.get("OPENSEEK_RECIPE_CONFIG")
    return Path(path).expanduser().resolve() if path else root / ".recipe/config.json"


def resolve(args: dict, root: Path = ROOT, environ: dict | None = None) -> dict:
    environ = dict(os.environ if environ is None else environ)
    path = config_path(root, args, environ)
    if args.get("config") and not path.exists() and args.get("command") != "config":
        raise ConfigError(f"Config not found: {path}. Use recipe config init --config PATH.")
    user = read_json(path) if path.exists() else {}
    if set(user) - {"schema_version", "model", "profile", "env"}:
        raise ConfigError(f"Unknown config keys in {path}")
    if user.get("schema_version", 1) != 1 or not isinstance(user.get("env", {}), dict):
        raise ConfigError("Config requires schema_version 1 and an env object")
    legacy = dotenv(root / ".env", environ)
    model_id = args.get("model") or environ.get("OPENSEEK_RECIPE_MODEL") or user.get("model", DEFAULT_MODEL)
    recipe = named(root, "recipes", model_id)
    saved_profile = user.get("profile") if user.get("model", DEFAULT_MODEL) == model_id else None
    profile_id = args.get("profile") or environ.get("OPENSEEK_RECIPE_PROFILE") or saved_profile or recipe["default_profile"]
    if profile_id not in recipe["profiles"]:
        raise ConfigError(f"Profile {profile_id!r} is not supported by {model_id}")
    profile = named(root, "profiles", profile_id)
    lock_path = root / "locks" / (recipe["lock"] + ".lock.json")
    lock = load_lock(lock_path)
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    env = {"GPU": "0", "PORT": "8001", "HF_CACHE": str(Path.home() / ".cache/huggingface"),
           "RECIPE_DIR": str(root / "vendor/sm12x-exl3-qwen3.8-flash-next"),
           "LMCACHE_PORT": "6555", "LMCACHE_HTTP_PORT": "6580", "LMCACHE_METRICS_PORT": "6590",
           "RESTART_POLICY": "unless-stopped", **recipe["env"], **profile["env"]}
    overrides = {**user.get("env", {}), **legacy, **{k: v for k, v in environ.items() if k in SETTINGS | LOCKED_SETTINGS}}
    overrides.update({k.upper(): str(args[k]) for k in ("gpu", "port", "hf_cache", "base_url") if k in args})
    runtime = lock["runtime"]
    fixed = {"RECIPE_REPO": runtime["recipe_repo"], "RECIPE_COMMIT": runtime["recipe_commit"],
             "BASE_IMAGE": runtime["base_image"], "LMCACHE_VERSION": runtime["lmcache"]["version"],
             "KV_IMAGE": "openseek-qwen38:lmc" + runtime["lmcache"]["version"].replace(".", "") + "-" + lock_sha[:12]}
    for key, value in overrides.items():
        if key not in SETTINGS | LOCKED_SETTINGS or type(value) not in (str, int, float):
            raise ConfigError(f"Unsupported config setting: {key}")
        value = str(value)
        if key in fixed and value != fixed[key]:
            raise ConfigError(f"{key} is version-locked. Change the reviewed lock, not local config.")
        env[key] = value
    env.update(fixed)
    if env["QUANT"] != recipe["env"]["QUANT"] or env["MTP_TOKENS"] != recipe["env"]["MTP_TOKENS"]:
        raise ConfigError("QUANT and MTP_TOKENS must match the qualified model recipe")
    for key in ("HF_CACHE", "RECIPE_DIR", "RUNTIME_CACHE", "RESULT_DIR"):
        if key in env:
            p = Path(env[key]).expanduser()
            env[key] = str((root / p).resolve() if not p.is_absolute() else p)
    if not env.get("MAX_BATCHED_TOKENS"):
        env["MAX_BATCHED_TOKENS"] = str(2 * int(env["KV_BLOCK_TOKENS"]) - 1)
    for key in ("PORT", "LMCACHE_PORT", "LMCACHE_HTTP_PORT", "LMCACHE_METRICS_PORT"):
        if not env[key].isdigit() or not 1 <= int(env[key]) <= 65535:
            raise ConfigError(f"{key} must be a port in 1..65535")
    ports = [env[k] for k in ("PORT", "LMCACHE_PORT", "LMCACHE_HTTP_PORT", "LMCACHE_METRICS_PORT")]
    if len(set(ports)) != len(ports):
        raise ConfigError("API and LMCache ports must be distinct")
    for key in ("MAX_MODEL_LEN", "MAX_NUM_SEQS", "KV_RAM_GB", "KV_BLOCK_TOKENS", "MAX_BATCHED_TOKENS"):
        if not env[key].isdigit() or int(env[key]) < 1:
            raise ConfigError(f"{key} must be a positive integer")
    if not env["GPU"].isdigit():
        raise ConfigError("GPU must be a nonnegative device index")
    if int(env["MAX_BATCHED_TOKENS"]) < int(env["KV_BLOCK_TOKENS"]):
        raise ConfigError("MAX_BATCHED_TOKENS must be >= KV_BLOCK_TOKENS")
    if int(env["KV_BLOCK_TOKENS"]) != 3200 or int(env["MAX_MODEL_LEN"]) > 262144:
        raise ConfigError("This qualified MTP/FP8 profile requires 3200-token blocks and context <=262144")
    for key in ("GPU_MEMORY_UTILIZATION", "KV_RAM_WATERMARK"):
        try:
            valid = 0 < float(env[key]) < 1
        except ValueError:
            valid = False
        if not valid:
            raise ConfigError(f"{key} must be between 0 and 1")
    env.setdefault("BASE_URL", "http://127.0.0.1:" + env["PORT"])
    env["BASE_URL"] = env["BASE_URL"].rstrip("/").removesuffix("/v1")
    parsed = urlparse(env["BASE_URL"])
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.query or parsed.fragment or parsed.path:
        raise ConfigError("BASE_URL must be an HTTP(S) origin without credentials, query, or path")
    env.update({"OPENSEEK_RESOLVED": "1", "MODEL_LOCK": str(lock_path), "LOCK_SHA256": lock_sha,
                "LOCKED_MODEL_REPO": lock["model"]["repo_id"], "LOCKED_MODEL_REVISION": lock["model"]["revision"],
                "LMCACHE_WHEEL": runtime["lmcache"]["wheel"], "LMCACHE_WHEEL_SHA256": runtime["lmcache"]["sha256"],
                "MIN_GPU_VRAM_MIB": str(profile["requirements"]["gpu_vram_mib"]),
                "MIN_HOST_RAM_GIB": str(profile["requirements"]["host_ram_gib"])})
    return {"version": (root / "VERSION").read_text().strip(), "model": model_id, "profile": profile_id,
            "recipe": recipe, "hardware": profile, "lock": lock, "lock_sha256": lock_sha,
            "config_path": str(path), "env": env}


def save_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)

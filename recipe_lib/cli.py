"""A single entry point for recipe selection, installation and lifecycle scripts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

from . import artifacts
from .config import ROOT, ConfigError, config_path, named, read_json, resolve, save_config


def options(parser: argparse.ArgumentParser) -> None:
    for flag, help_text in (
        ("model", "Recipe id (recipe models list)"), ("profile", "Qualified hardware profile id"),
        ("config", "Local JSON config path"), ("base-url", "Server origin for smoke/client connection"),
        ("gpu", "GPU device index"), ("port", "Local API port"), ("hf-cache", "Hugging Face home/cache path"),
    ):
        parser.add_argument("--" + flag, default=argparse.SUPPRESS, help=help_text)
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Machine output for data or dry-run commands")
    parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS, help="Preview without changing files or running scripts")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="recipe", description="Install and run pinned local-model recipes. Chat: OpenSeek TUI; monitoring: SparkDash.",
                                 epilog="Example: ./recipe models list; ./recipe install --dry-run; ./recipe install; ./recipe start")
    options(ap)
    ap.add_argument("--version", action="version", version="OpenSeek Recipe " + (ROOT / "VERSION").read_text().strip())
    subs = ap.add_subparsers(dest="command")
    for command, description in (
        ("models", "List, inspect or select model recipes"), ("config", "Inspect or initialize local configuration"),
        ("lock", "Print the selected immutable model/runtime lock"), ("connection", "Print the endpoint and model for a chat client"),
        ("install", "Preflight, install pinned runtime, download and hash-check checkpoint"),
        ("doctor", "Check installation prerequisites"), ("start", "Start selected recipe (install first)"),
        ("check", "Run offline developer checks (no GPU or downloads)"),
        ("stop", "Stop both containers"), ("restart", "Restart LMCache and vLLM in order"),
        ("recreate", "Recreate containers with the selected configuration"), ("down", "Remove containers, preserving weights"),
        ("status", "Existing basic service status (advanced monitoring belongs to SparkDash)"),
        ("logs", "Follow server logs"), ("wait", "Wait for API readiness"),
        ("smoke", "Run real API compatibility checks"), ("bench", "Run existing benchmark scripts"),
        ("verify", "Verify every checkpoint file against the selected lock"),
    ):
        sub = subs.add_parser(command, help=description, description=description)
        options(sub)
        if command == "models":
            sub.add_argument("action", nargs="?", default="list", choices=("list", "show", "select"))
            sub.add_argument("id", nargs="?")
        if command == "config":
            sub.add_argument("action", nargs="?", default="show", choices=("show", "init"))
        if command == "logs":
            sub.add_argument("target", nargs="?", choices=("server", "lmcache"), default="server")
        if command == "verify":
            sub.add_argument("--fast", action="store_true", help="Check sizes only, not SHA256 integrity")
    return ap


def emit(data, as_json: bool = False) -> None:
    if as_json or isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2))
    else:
        print(data)


def plan(command: str, config: dict, args: dict, root: Path = ROOT) -> list[list[str]]:
    script = lambda name: str(root / "scripts" / name)
    if command == "install":
        return [["bash", script("doctor.sh")], ["bash", script("setup.sh")]]
    if command == "doctor":
        return [["bash", script("doctor.sh")]]
    if command == "check":
        return [["bash", script("check.sh")]]
    if command == "smoke":
        return [[sys.executable, script("smoke_test.py"), "--base-url", config["env"]["BASE_URL"],
                 "--model", config["recipe"]["served_model"]]]
    if command == "bench":
        return [["bash", script("bench.sh")]]
    argv = ["bash", script("ctl.sh"), command]
    if command == "logs" and args.get("target") == "lmcache":
        argv.append("lmcache")
    return [argv]


def run(argv: list[str] | None = None, root: Path = ROOT) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "-":
        raise ConfigError("Chat lives in OpenSeek TUI (openseek). Use ./recipe connection for its local endpoint.")
    if argv and argv[0] == "help":
        argv = argv[1:] + ["--help"]
    ap = parser()
    # Help must work even if other arguments/configuration are invalid.
    if "--help" in argv or "-h" in argv:
        for action in ap._actions:
            if isinstance(action, argparse._SubParsersAction):
                target = next((a for a in argv if a in action.choices), None)
                (action.choices[target] if target else ap).print_help()
                return 0
    args = vars(ap.parse_args(argv))
    command = args.get("command")
    if command is None:
        ap.print_help()
        return 0
    if command == "models" and args["action"] == "list":
        models = [read_json(p) for p in sorted((root / "recipes").glob("*.json"))]
        if args.get("json"):
            emit(models, True)
        else:
            for model in models:
                print(f"{model['id']}\t{model['name']}\t{model['status']}")
        return 0
    if command == "models" and args["action"] in ("show", "select"):
        if not args.get("id"):
            raise ConfigError("Use recipe models show/select MODEL_ID (see recipe models list)")
        args["model"] = args["id"]
    cfg = resolve(args, root)
    if command == "models" and args["action"] == "show":
        emit({"recipe": cfg["recipe"], "hardware": cfg["hardware"], "lock": cfg["lock"]}, args.get("json", False))
        return 0
    if command == "config" or (command == "models" and args["action"] == "select"):
        if command == "config" and args["action"] == "show":
            emit(cfg, args.get("json", False))
            return 0
        path = Path(cfg["config_path"])
        if command == "config" and path.exists():
            raise ConfigError(f"Config already exists: {path}. Edit it or select a model; it was not overwritten.")
        previous = read_json(path) if path.exists() else {}
        data = {"schema_version": 1, "model": cfg["model"], "profile": cfg["profile"], "env": previous.get("env", {})}
        for key in ("gpu", "port", "hf_cache", "base_url"):
            if key in args:
                data["env"][key.upper()] = cfg["env"][key.upper()]
        if not args.get("dry_run"):
            save_config(path, data)
        emit({"config": str(path), "selection": data, "dry_run": args.get("dry_run", False)}, True)
        return 0
    if command == "lock":
        emit({"version": cfg["version"], "model": cfg["model"], "profile": cfg["profile"],
              "lock_sha256": cfg["lock_sha256"], "lock": cfg["lock"]}, True)
        return 0
    if command == "connection":
        emit({"base_url": cfg["env"]["BASE_URL"] + "/v1", "model": cfg["recipe"]["served_model"],
              "protocol": "openai-compatible", "api_key_required": False,
              "chat_client": "OpenSeek TUI (separate project)"}, True)
        return 0
    if command == "verify":
        if args.get("dry_run"):
            emit({"snapshot": str(artifacts.snapshot_path(cfg["lock"], Path(cfg["env"]["HF_CACHE"]))),
                  "files": len(cfg["lock"]["model"]["files"]), "mode": "size-only" if args["fast"] else "sha256"}, True)
            return 0
        result = artifacts.verify(cfg["lock"], Path(cfg["env"]["HF_CACHE"]), full=not args["fast"], progress=not args.get("json"))
        emit(result, True)
        return 0 if result["ok"] else 1
    commands = plan(command, cfg, args, root)
    if args.get("dry_run"):
        emit({"model": cfg["model"], "profile": cfg["profile"], "commands": commands,
              "env": cfg["env"], "checkpoint_bytes": sum(f["size"] for f in cfg["lock"]["model"]["files"]),
              "dry_run": True}, True)
        return 0
    if args.get("json"):
        raise ConfigError("This action streams its script output. Use --dry-run --json to inspect its plan.")
    if command in {"install", "doctor", "start", "stop", "restart", "recreate", "down", "status", "logs", "wait", "bench"}:
        if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
            raise ConfigError("This recipe runs on x86_64 Linux with an RTX PRO 6000. Listing/config/lock/connection/verify/smoke work on clients.")
    environ = {**os.environ, **cfg["env"]}
    for cmd in commands:
        print("+ " + shlex.join(cmd), file=sys.stderr, flush=True)
        result = subprocess.run(cmd, cwd=root, env=environ)
        if result.returncode:
            return result.returncode if result.returncode > 0 else 128 - result.returncode
    if command == "install":
        save_config(Path(cfg["config_path"]).parent / "installed-lock.json",
                    {"version": cfg["version"], "model": cfg["model"], "profile": cfg["profile"],
                     "lock_sha256": cfg["lock_sha256"], "env": cfg["env"]})
        print("Installation verified. Next: ./recipe start, then ./recipe smoke.")
    return 0


def main() -> int:
    try:
        return run()
    except (ConfigError, OSError, ValueError) as exc:
        print(f"recipe: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, ConfigError) else 1
    except KeyboardInterrupt:
        print("recipe: interrupted", file=sys.stderr)
        return 130

#!/usr/bin/env python3
"""Fetch hash-locked small sources and stage the measured experimental port.

No weights, Docker, GPU operations, package installation or service changes.
Existing differing files are preserved and rejected. --dry-run is offline.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import urllib.request

BUNDLE = Path(__file__).resolve().parent


def checked_write(path, data, expected):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Source checksum mismatch: {path.name}")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing differing file preserved: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download-part")
    temporary.write_bytes(data)
    temporary.replace(path)


def validate(lock):
    paths = set()
    for item in lock["source_files"]:
        path = PurePosixPath(item["path"])
        if path.is_absolute() or ".." in path.parts or str(path) != item["path"] or str(path) in paths:
            raise ValueError("Unsafe or duplicate source path")
        paths.add(str(path))
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid source SHA256")
        prefixes = (
            "https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/resolve/" + lock["recipe_revision"] + "/",
            "https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-1017-EXL3/resolve/" + lock["model_revision"] + "/",
            "https://raw.githubusercontent.com/sfxnz/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-vLLM-2x-DGX-Spark/" + lock["sfxnz_helper_revision"] + "/",
        )
        if not item["url"].startswith(prefixes):
            raise ValueError("Source URL is not pinned to an allowed origin/revision")


def prepare(root, lock):
    validate(lock)
    def fetch(item):
        target = root / item["path"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Existing differing file preserved: {target}")
            return
        with urllib.request.urlopen(item["url"], timeout=90) as response:
            data = response.read()
        checked_write(target, data, item["sha256"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, lock["source_files"]))
    for source in sorted((BUNDLE / "scripts").glob("*")):
        if source.is_file() and source.suffix in (".py", ".sh"):
            data = source.read_bytes()
            checked_write(root / "scripts" / source.name, data, hashlib.sha256(data).hexdigest())
    data = (BUNDLE / "model-files.json").read_bytes()
    checked_write(root / "evidence/sfxnz-model-info.json", data, hashlib.sha256(data).hexdigest())
    port = root / "recipe/kernels/xmoe_port"
    if not port.exists():
        shutil.copytree(root / "recipe/kernels/xmoe", port)
    spec = importlib.util.spec_from_file_location("dsv41_patch", root / "scripts/patch_mixed_k.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.kernel_patch(port)
    (root / "logs").mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    lock = json.loads((BUNDLE / "runtime-lock.json").read_text())
    validate(lock)
    if not args.dry_run:
        prepare(args.root.expanduser().resolve(), lock)
    print(json.dumps({"root": str(args.root), "source_files": len(lock["source_files"]),
                      "model_revision": lock["model_revision"], "dry_run": args.dry_run}))


if __name__ == "__main__":
    main()

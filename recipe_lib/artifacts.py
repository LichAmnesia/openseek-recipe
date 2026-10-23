"""Pinned snapshot download and content verification; stdlib except the downloader."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from .config import ConfigError, load_lock


def snapshot_path(lock: dict, cache: Path) -> Path:
    model = lock["model"]
    return cache / "hub" / ("models--" + model["repo_id"].replace("/", "--")) / "snapshots" / model["revision"]


def verify(lock: dict, cache: Path, full: bool = True, progress: bool = False) -> dict:
    snapshot = snapshot_path(lock, cache)
    failures = []
    total = 0
    for index, entry in enumerate(lock["model"]["files"], 1):
        path = snapshot / entry["path"]
        if progress:
            print(f"[{index}/{len(lock['model']['files'])}] {entry['path']}", file=sys.stderr, flush=True)
        try:
            if path.stat().st_size != entry["size"]:
                failures.append({"file": entry["path"], "reason": "size mismatch"})
                continue
            if full:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != entry["sha256"]:
                    failures.append({"file": entry["path"], "reason": "SHA256 mismatch"})
                    continue
            total += entry["size"]
        except FileNotFoundError:
            failures.append({"file": entry["path"], "reason": "missing"})
    return {"ok": not failures, "mode": "sha256" if full else "size-only",
            "snapshot": str(snapshot), "checked_files": len(lock["model"]["files"]),
            "verified_bytes": total, "missing_or_bad_bytes": sum(f["size"] for f in lock["model"]["files"]) - total,
            "failures": failures}


def download(lock: dict, cache: Path, snapshot_download=None) -> dict:
    """Download only missing files; never silently repair shared cached bytes."""
    before = verify(lock, cache, full=False)
    corrupt = [f for f in before["failures"] if f["reason"] != "missing"]
    if corrupt:
        raise ConfigError(f"Cached files have wrong sizes: {corrupt}. Move damaged files aside, then retry.")
    if before["failures"]:
        if snapshot_download is None:
            from huggingface_hub import snapshot_download
        model = lock["model"]
        snapshot_download(repo_id=model["repo_id"], revision=model["revision"], cache_dir=str(cache / "hub"),
                          allow_patterns=[f["path"] for f in model["files"]])
    result = verify(lock, cache, full=True, progress=True)
    if not result["ok"]:
        raise ConfigError(f"Checkpoint integrity failed: {result['failures']}. Move damaged files aside and retry.")
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "verify"))
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    try:
        lock = load_lock(args.lock)
        result = download(lock, args.cache) if args.action == "download" else verify(lock, args.cache)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    except (ConfigError, OSError) as exc:
        print(f"recipe: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

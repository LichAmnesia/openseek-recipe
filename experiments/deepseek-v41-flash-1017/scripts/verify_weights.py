#!/usr/bin/env python3
"""Verify pinned Hub file sizes and published LFS SHA256s without loading tensors."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    manifest, directory, output = map(Path, sys.argv[1:])
    info = json.loads(manifest.read_text())
    results = []
    for item in info["siblings"]:
        name = item["rfilename"]
        path = directory / name
        row = {"file": name, "expected_bytes": item.get("size")}
        if not path.is_file():
            row.update(ok=False, error="missing")
        else:
            row["actual_bytes"] = path.stat().st_size
            row["ok"] = row["actual_bytes"] == row["expected_bytes"]
            expected = item.get("lfs", {}).get("sha256")
            if expected:
                h = hashlib.sha256()
                with path.open("rb") as src:
                    for chunk in iter(lambda: src.read(16 * 1024 * 1024), b""):
                        h.update(chunk)
                row.update(sha256=h.hexdigest(), expected_sha256=expected)
                row["ok"] = row["ok"] and h.hexdigest() == expected
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "revision": info["sha"],
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "ok": all(r["ok"] for r in results),
        "files": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Make a separate first-gate TP1 launcher, preserving the pinned source script."""
import argparse
from pathlib import Path

from patch_mixed_k import replace_once


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recipe", type=Path)
    args = ap.parse_args()
    original = args.recipe / "serve/serve.sh"
    src = original.read_text()
    src = replace_once(src, "--tensor-parallel-size 2 --host 0.0.0.0", "--tensor-parallel-size 1 --host 127.0.0.1", "TP1/loopback")
    src = replace_once(src, "-e DSV41_ENGRAM_DISK=0 -e DSV41_ENGRAM_INT4=1", "-e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_INT4=0 -e DSV41_TP1_MIXED_K=1", "disk FP8 tables")
    src = replace_once(src, 'docker run --rm --name "$NAME"', 'docker run --restart no --name "$NAME"', "retain failed test container")
    src = replace_once(src, 'docker rm -f "$NAME" >/dev/null 2>&1;', 'docker stop -t 30 "$NAME" >/dev/null 2>&1;', "retain inspect/log evidence")
    target = args.recipe / "serve/serve-1gpu-eager.sh"
    target.write_text(src)
    print(target)


if __name__ == "__main__":
    main()

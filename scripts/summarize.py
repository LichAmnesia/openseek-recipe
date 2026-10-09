#!/usr/bin/env python3
"""Print a markdown summary of an upstream benchmark-suite result directory.

Usage: scripts/summarize.py <result-dir>
Reads whichever of seven.jsonl, orchid.jsonl, clients.json, api-tools.jsonl,
vision.json exist. Stdlib only.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def contract_passed(contract: dict) -> bool:
    # seven uses quality_contract_passed; orchid uses pass.
    return bool(contract.get("quality_contract_passed", contract.get("pass")))


def workloads(path: Path) -> None:
    rows = jsonl(path)
    timed = [r for r in rows if r.get("record") == "measurement" and r.get("timed")]
    summary = next((r for r in rows if r.get("record") == "summary"), {})
    mtp = next((r["timed_suite_delta"] for r in rows if r.get("record") == "mtp_after"), None)
    print(f"\n### {path.stem}\n")
    if "weighted_decode_tps" in summary:
        print(f"Weighted decode: **{summary['weighted_decode_tps']:.1f} tok/s**\n")
    print("| Case | decode tok/s (median) | TTFT ms (median) | quality contract |")
    print("|---|---:|---:|:-:|")
    cases: dict[str, list[dict]] = {}
    for r in timed:
        cases.setdefault(r["case"], []).append(r)
    for case, rs in sorted(cases.items(), key=lambda kv: -statistics.median(x["decode_tps"] for x in kv[1])):
        tps = statistics.median(x["decode_tps"] for x in rs)
        ttft = statistics.median(x["ttft_seconds"] for x in rs) * 1000
        passed = sum(1 for x in rs if contract_passed(x.get("contract", {})))
        print(f"| {case} | {tps:.1f} | {ttft:.0f} | {passed}/{len(rs)} |")
    if mtp:
        print(
            f"\nMTP acceptance: **{mtp['acceptance_rate'] * 100:.1f}%** "
            f"({mtp['accepted_tokens']:.0f}/{mtp['draft_tokens']:.0f} draft tokens, "
            f"mean acceptance length {mtp['mean_acceptance_length']:.2f})"
        )


def clients(path: Path) -> None:
    data = json.loads(path.read_text())
    print("\n### Decode scaling (clients)\n")
    print(f"{data.get('output_tokens_per_sequence')} output tokens per request, MTP={data.get('mtp_tokens')}\n")
    print("| Concurrency | aggregate tok/s (median) | min | max | per-stream | TTFT ms (median) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for p in data["points"]:
        agg = p["aggregate_decode_tokens_per_second"]
        c = p["concurrency"]
        ttfts = [rr["ttft_ms"] for run in p["runs"] for rr in run["request_results"]]
        print(
            f"| {c} | {agg['median']:.1f} | {agg['min']:.1f} | {agg['max']:.1f} | "
            f"{agg['median'] / c:.1f} | {statistics.median(ttfts):.0f} |"
        )


def tools(path: Path) -> None:
    rows = [r for r in jsonl(path) if r.get("record") == "measurement"]
    passed = sum(1 for r in rows if r.get("passed"))
    print(f"\nAPI tool constraints: **{passed}/{len(rows)}**")


def vision(path: Path) -> None:
    data = json.loads(path.read_text())
    counts = data.get("supported_image_counts", {})
    res = ", ".join(f"{k} img {'pass' if v.get('passed') else 'FAIL'}" for k, v in counts.items())
    print(f"Vision: {res}")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    d = Path(sys.argv[1])
    if not d.is_dir():
        print(f"not a directory: {d}", file=sys.stderr)
        return 2
    gpu = d / "gpu.txt"
    print(f"## Results: {d.parent.name if d.name == 'raw' else d.name}")
    if gpu.exists():
        print(f"\nGPU: {gpu.read_text().strip()}")
    for name in ("seven.jsonl", "orchid.jsonl"):
        if (d / name).exists():
            workloads(d / name)
    if (d / "clients.json").exists():
        clients(d / "clients.json")
    if (d / "api-tools.jsonl").exists():
        tools(d / "api-tools.jsonl")
    if (d / "vision.json").exists():
        vision(d / "vision.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Combine actual API timing and memory samples without inventing missing data."""
import argparse
import json
import statistics
from pathlib import Path


def max_value(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return max(values) if values else None


def min_value(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return min(values) if values else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", type=Path)
    args = ap.parse_args()
    rows = []
    path = args.logs / "resources.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    benchmark_file = args.logs / "benchmark.json"
    benchmark = json.loads(benchmark_file.read_text()) if benchmark_file.exists() else {}
    measured = [r for r in benchmark.get("speed", []) if r.get("valid_speed_sample")]
    c1 = [r for r in measured if r["case"].startswith("c1_trial_")]
    phases = sorted({r.get("phase", "unknown") for r in rows})
    memory = {}
    for phase in phases + ["all_observed_phases"]:
        group = rows if phase == "all_observed_phases" else [r for r in rows if r.get("phase") == phase]
        memory[phase] = {
            "samples": len(group),
            "gpu_sampled_peak_mib": max_value(group, "gpu_memory_used_mib"),
            "container_cgroup_peak_bytes": max_value(group, "container_memory_peak_bytes"),
            "container_sampled_peak_current_bytes_including_file_cache": max_value(group, "container_memory_current_bytes"),
            "container_sampled_peak_anon_bytes": max_value(group, "container_anon_bytes"),
            "container_sampled_peak_file_cache_bytes": max_value(group, "container_file_bytes"),
            "container_sampled_peak_swap_bytes": max_value(group, "container_swap_bytes"),
            "host_minimum_MemAvailable_bytes": min_value(group, "host_MemAvailable_bytes"),
            "host_total_bytes": max_value(group, "host_MemTotal_bytes"),
        }
    report = {
        "api_benchmark_ok": benchmark.get("ok"),
        "api_benchmark_error": benchmark.get("error"),
        "mode": benchmark.get("mode"),
        "c1_median_observed_decode_tok_s": statistics.median(r["observed_decode_tok_s"] for r in c1) if c1 else None,
        "c1_median_ttft_s": statistics.median(r["ttft_s"] for r in c1) if c1 else None,
        "valid_api_samples": measured,
        "memory": memory,
        "notes": [
            "Null means not measured, never zero.",
            "GPU peaks are sampled; cgroup memory.peak is kernel-maintained and includes page cache.",
            "Per-phase cgroup peak can include earlier startup allocations; sampled current/anon/file values are phase-local.",
            "Whole-host MemAvailable includes all workloads; container cgroup accounting isolates this server.",
            "A low eager/bootstrap score is not an optimized CUDA-graph throughput result.",
        ],
    }
    (args.logs / "measurements.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

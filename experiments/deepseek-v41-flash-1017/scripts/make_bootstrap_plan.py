#!/usr/bin/env python3
"""Capacity-bounded bootstrap only; added experts have no measured routing rank."""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("original_plan", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--host-gib", type=float, default=94)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    tc = cfg["text_config"]
    q = cfg["quantization_config"]
    original = json.loads(args.original_plan.read_text())
    hidden, inter = tc["hidden_size"], tc["moe_intermediate_size"]
    assert hidden % 128 == inter % 128 == 0
    num, count = tc["num_hidden_layers"], tc["n_routed_experts"]
    costs, plan = {}, {}
    for layer in range(num):
        key = f"language_model.model.layers.{layer}.ffn.experts"
        ku, kd = q["expert_bits"][str(layer)]
        costs[key] = hidden * inter * (2 * ku + kd) // 8
        plan[key] = set(original.get(key, []))
        assert all(0 <= e < count for e in plan[key])
    current = sum(len(v) * costs[k] for k, v in plan.items())
    budget = int(args.host_gib * 1024**3)
    if current > budget:
        raise ValueError("Original cold plan already exceeds requested host budget")
    rng = random.Random(1017)
    candidates = [(k, e) for k in plan for e in range(count) if e not in plan[k]]
    rng.shuffle(candidates)
    for key, expert in candidates:
        if current + costs[key] <= budget:
            plan[key].add(expert)
            current += costs[key]
    result = {k: sorted(v) for k, v in plan.items()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    total = sum(costs.values()) * count
    summary = {
        "kind": "unprofiled_bootstrap", "seed": 1017,
        "host_experts": sum(map(len, result.values())),
        "expert_trellis_total_gib": total / 1024**3,
        "host_trellis_gib": current / 1024**3,
        "gpu_trellis_gib": (total - current) / 1024**3,
        "notes": "Excludes small scale vectors, dense layers, activations, KV and OS. Extra cold expert IDs are unprofiled. No fit or speed claim.",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

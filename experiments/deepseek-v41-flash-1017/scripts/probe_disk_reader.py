#!/usr/bin/env python3
"""Numerically compare disk FP8/UE8M0 rows to an independent tensor reference."""
import importlib.util
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import torch


def main():
    module, output = map(Path, sys.argv[1:])
    spec = importlib.util.spec_from_file_location("disk_under_test", module)
    disk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(disk)
    torch.manual_seed(1017)
    n, dim = 65, 256
    weights = (torch.randn(n, dim) * 2).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 134, (n, dim // 32), dtype=torch.uint8)
    names = ("layers.0.engram.embed.weight", "layers.0.engram.embed.scale")
    data = weights.view(torch.uint8).numpy().tobytes() + scales.numpy().tobytes()
    split = weights.numel()
    header = {
        names[0]: {"dtype": "F8_E4M3", "shape": [n, dim], "data_offsets": [0, split]},
        names[1]: {"dtype": "U8", "shape": [n, dim // 32], "data_offsets": [split, len(data)]},
    }
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        raw = json.dumps(header).encode()
        raw += b" " * ((-len(raw)) % 8)
        shard = root / "tiny.safetensors"
        shard.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: shard.name for name in names}}))
        table = disk.DiskEngramTable(str(root), 0, dim, 32)
        results = []
        for count in (1, 8, 80):
            rows = torch.randint(0, n, (count,), dtype=torch.int64)
            if count > 1:
                rows[-1] = rows[0]
            owned = torch.arange(count) % 3 != 1
            actual = table.gather_dequant(rows, owned)
            multiplier = torch.pow(2.0, scales[rows].to(torch.float32) - 127)
            expected = (weights.to(torch.float32)[rows].view(count, -1, 32) * multiplier[:, :, None]).view(count, dim)
            expected[~owned] = 0
            expected = expected.to(torch.bfloat16)
            equal = torch.equal(actual, expected)
            results.append({"rows": count, "bf16_bitwise_equal": equal})
            if not equal:
                raise AssertionError(results[-1])
        # Corruption must fail visibly instead of returning partial/uninitialized rows.
        os.truncate(shard, len(struct.pack("<Q", len(raw)) + raw) + 1)
        try:
            table.gather_dequant(torch.tensor([64]), torch.tensor([True]))
        except OSError:
            short_read_rejected = True
        else:
            raise AssertionError("Truncated shard was accepted")
        table.pool.shutdown()
        os.close(table.w_fd)
        os.close(table.s_fd)
    report = {"ok": True, "scope": "synthetic disk row reader only; not full-model/graph validation", "torch": torch.__version__, "cases": results, "short_read_rejected": short_read_rejected}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

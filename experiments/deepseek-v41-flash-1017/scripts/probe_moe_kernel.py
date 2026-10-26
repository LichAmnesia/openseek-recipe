#!/usr/bin/env python3
"""Compare ported xmoe to pinned stock ExLlamaV3, including UVA cold rows."""
import importlib
import json
import math
import sys
from pathlib import Path

import torch
import exllamav3_ext as reference


def case(extension, ku, kd, host, tokens=2, hidden=5120, inter=2304):
    torch.manual_seed(1017)
    device = "cuda"
    experts, topk = 4, 2
    keep = []
    ptrs = []
    for projection, bits in (("gate", ku), ("up", ku), ("down", kd)):
        in_f, out_f = (inter, hidden) if projection == "down" else (hidden, inter)
        trellis = torch.randint(-32768, 32767, (experts, in_f // 16, out_f // 16, bits * 16), dtype=torch.int16)
        if host:
            trellis = trellis.pin_memory()
            from vllm_exl3.exl3 import _uva_alias
            alias = _uva_alias(trellis)
        else:
            trellis = trellis.to(device)
            alias = trellis
        suh = torch.ones(experts, in_f, dtype=torch.float16, device=device)
        svh = torch.full((experts, out_f), 1 / math.sqrt(in_f), dtype=torch.float16, device=device)
        keep.extend((trellis, alias, suh, svh))
        ptrs.extend(torch.tensor([t[i].data_ptr() for i in range(experts)], dtype=torch.int64, device=device) for t in (alias, suh, svh))
    x = (torch.randn(tokens, hidden, device=device) * 0.1).half()
    ids = torch.stack((torch.arange(tokens, device=device) % experts, (torch.arange(tokens, device=device) + 1) % experts), dim=1).flatten()
    order = ids.argsort()
    counts = torch.bincount(ids, minlength=experts + 1).long()
    sorted_tokens = torch.arange(tokens, device=device).repeat_interleave(topk)[order].long()
    weights = torch.tensor([0.6, 0.4], dtype=torch.float16, device=device).repeat(tokens)[order]
    concurrency = min(reference.exl3_moe_max_concurrency(0), extension.exl3_moe_max_concurrency(0))
    temps = [torch.empty(concurrency, 128, width, dtype=torch.float16, device=device) for width in (hidden, hidden, inter, inter)]
    args = [x, None, counts, sorted_tokens, weights, *temps, 0, ku, ku, kd, *ptrs, True, False, True, False, True, False, 0.0]
    def call(module):
        out = torch.zeros(tokens, hidden, dtype=torch.float32, device=device)
        args[1] = out
        fn = module.exl3_moe
        doc = fn.__doc__ or ""
        force = getattr(module, "xmoe_set_force_group", None)
        if force is not None:
            force(int(host))
        try:
            if "num_active" in doc or "arg29" in doc or doc.count("arg") >= 30:
                fn(*args, -1)
            else:
                fn(*args)
        finally:
            if force is not None:
                force(0)
        torch.cuda.synchronize()
        return out
    expected = call(reference)
    actual = call(extension)
    if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("non-finite synthetic MoE output")
    reference_norm = torch.linalg.vector_norm(expected)
    if reference_norm.item() <= 1e-8:
        raise AssertionError("Reference output has no signal; numeric parity would be inconclusive")
    error = torch.linalg.vector_norm(actual - expected) / reference_norm
    result = {"ku": ku, "kd": kd, "host_uva": host, "tokens": tokens, "reference_l2": reference_norm.item(), "relative_l2": error.item(), "max_abs": (actual - expected).abs().max().item()}
    if result["relative_l2"] > 0.003:
        raise AssertionError(result)
    return result


def main():
    build_dir, output = map(Path, sys.argv[1:])
    sys.path.insert(0, str(build_dir))
    port = importlib.import_module("xmoe_port")
    cases = [case(port, ku, kd, host, tokens) for ku, kd in ((2, 2), (2, 3), (3, 3), (3, 4)) for host in (False, True) for tokens in (2, 65)]
    report = {"ok": True, "scope": "synthetic MoE numeric parity; not model speed/quality", "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "cases": cases}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

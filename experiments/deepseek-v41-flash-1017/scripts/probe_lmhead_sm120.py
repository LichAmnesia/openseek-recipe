#!/usr/bin/env python3
"""Real checkpoint head rows, actual ParallelLMHead loader, BF16 reference.

This narrow numeric probe is not full-model correctness or speed evidence.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.models.deepseek_v41.nvidia.lmhead_sm120 import configure_lm_head
from vllm.models.deepseek_v41.nvidia.model import _make_deepseek_v4_weights_mapper
from vllm.models.deepseek_v41.nvidia.vl_model import _make_deepseek_v4_vl_weights_mapper


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("model", type=Path)
    args = ap.parse_args()
    for dtype in ("fp4", "fp8"):
        for maker, prefix in ((_make_deepseek_v4_weights_mapper, ""),
                              (_make_deepseek_v4_vl_weights_mapper, "language_model.")):
            mapper = maker(dtype, "weight_scale")
            for key in ("lm_head.weight", "lm_head.weight_scale"):
                assert mapper.map_name(key) == prefix + key
            assert mapper.map_name("head.weight") == prefix + "lm_head.weight"
    idx = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(args.model / idx["lm_head.weight"], framework="pt", device="cpu") as f:
        w = f.get_slice("lm_head.weight")[:1024, :]
        s = f.get_slice("lm_head.weight_scale")[:1024, :]
    assert w.shape == (1024, 5120) and s.shape == (1024, 160)
    cfg = SimpleNamespace(model_config=SimpleNamespace(model=str(args.model), enforce_eager=True,
                          dtype=torch.bfloat16, head_dtype=torch.bfloat16),
                          parallel_config=SimpleNamespace(tensor_parallel_size=1))
    with torch.device("cuda"):
        layer = ParallelLMHead(1024, 5120, params_dtype=torch.bfloat16, disable_tp=True)
    configure_lm_head(layer, cfg)
    layer.weight.weight_loader(layer.weight, w)
    layer.weight_scale.weight_loader(layer.weight_scale, s)
    assert torch.equal(layer.weight.cpu().view(torch.uint8), w.view(torch.uint8))
    assert torch.equal(layer.weight_scale.cpu(), s)
    layer.quant_method.process_weights_after_loading(layer)
    ref_w = (w.cuda().float().view(1024, 160, 32) *
             torch.exp2(s.cuda().float() - 127).unsqueeze(-1)).reshape(1024, 5120).bfloat16()
    rows = []
    for tokens in (1, 17, 65, 512):
        torch.manual_seed(1017 + tokens)
        x = (torch.randn(tokens, 5120, device="cuda") * .1).bfloat16()
        ref = torch.mm(x, ref_w.t()).float()
        actual = layer.quant_method.apply(layer, x).float()
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all() and ref.norm() > 0
        rel = ((actual - ref).norm() / ref.norm()).item()
        row = {"tokens": tokens, "relative_l2": rel, "reference_norm": ref.norm().item()}
        assert rel < .003, row
        rows.append(row)
    result = {"ok": True, "scope": "1024 real checkpoint head rows; actual TP1 loader and sm120 W8A16; not full-model acceptance", "cases": rows}
    (args.root / "logs/lmhead-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

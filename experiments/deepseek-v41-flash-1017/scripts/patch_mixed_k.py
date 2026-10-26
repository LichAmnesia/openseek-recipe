#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Initial correctness port onto diffbot 2c379757, not an optimized final runtime.

Plugin modifications inherit diffbot's AGPL-3.0-only patch license. The selected
loader edits derive from sfxnz's widen_p2b_kbits.py. Fail if source anchors drift.
Mixed projections initially use ExLlamaV3's runtime-K group kernel; decode-once
and the optional CPU/DMA/cache paths must remain disabled for this first gate.
"""
import argparse
import ast
import json
import re
import runpy
from pathlib import Path


def replace_once(src, old, new, label):
    count = src.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected exactly one anchor, got {count}")
    return src.replace(old, new, 1)


def plugin_patch(src, helper):
    if "DSV41_TP1_MIXED_K" in src:
        return src
    selected = {
        "Exl3Config expert_bits", "expert_bits_for_prefix", "get_quant_method MoE K",
        "Exl3MoEMethod K_d", "layer K attributes", "fused state K (native only)",
        "fused state K", "exl3_moe K triple", "fat path K_d", "fat path down K",
    }
    edits = runpy.run_path(str(helper))["PY_EDITS"]
    for old, new, label in edits:
        if label in selected:
            src = replace_once(src, old, new, label)
    src = replace_once(
        src, '    k = int(getattr(layer, "_exl3_k", 4))\n    args = (',
        '    k = int(getattr(layer, "_exl3_k", 4))\n'
        '    k_d = int(getattr(layer, "_exl3_k_d", k))\n    args = (', "fused down K",
    )
    src = replace_once(
        src,
        'layer._exl3_host2 = _HostExpertTrellis(num_experts, _cold, (out_tiles, in_tiles, k_words), _dev, host_row_shape=(out_tiles, in_tiles, _ckw))',
        'layer._exl3_host2 = _HostExpertTrellis(num_experts, _cold, (out_tiles, in_tiles, self.bits_d * 16), _dev, host_row_shape=(out_tiles, in_tiles, self.bits_d * 16))',
        "host down allocation",
    )
    src = replace_once(
        src, '        if _cold:\n            w2_trellis = Parameter(',
        '        k_words = self.bits_d * 16\n        if _cold:\n            w2_trellis = Parameter(',
        "GPU down allocation",
    )
    for name in ("lc", "la"):
        src = replace_once(
            src, f'{name}[10], {name}[11], {name}[12] = ck, ck, ck',
            f'{name}[10], {name}[11], {name}[12] = ck, ck, k_d',
            f"{name} cold K triple",
        )
    src += '\n# DSV41_TP1_MIXED_K: first-gate paths only; fail closed on unsupported optimizations.\n'
    src += '''if os.environ.get("DSV41_TP1_MIXED_K", "0") == "1":
    if os.environ.get("EXL3_HOST_COLD_PACK", ""):
        raise RuntimeError("Initial mixed-K port requires an empty EXL3_HOST_COLD_PACK; string 0 is a path")
    for _flag in ("EXL3_CPU_EXPERTS", "EXL3_HOST_DMA", "EXL3_DO", "EXL3_HOST_CACHE"):
        if os.environ.get(_flag, "0") not in ("", "0"):
            raise RuntimeError(f"Initial mixed-K port requires {_flag}=0")
    if get_moe_kernel_backend() == "native":
        raise RuntimeError("Initial mixed-K port requires the exllamav3 backend")
'''
    ast.parse(src)
    return src


def kernel_patch(root):
    cu = root / "quant/exl3_moe.cu"
    src = cu.read_text()
    if "DSV41_TP1_RUNTIME_K" in src:
        return
    for name in ("exl3_moe_kernel_instances", "exl3_moe_kernel_instances_mt"):
        pattern = rf"(fp_exl3_moe_kernel {name}\[\] =\s*\{{)(.*?)(\n\}};)"
        match = re.search(pattern, src, re.S)
        if not match:
            raise ValueError(f"missing array {name}")
        body = re.sub(r"//[^\n]*", "", match.group(2))
        values = [v.strip() for v in body.split(",") if v.strip()]
        assert len(values) == 36, (name, len(values))
        suffix = "_mt" if name.endswith("_mt") else ""
        for k in (0, 4):
            for j, n in enumerate((128, 256)):
                assert values[4 * k + j] == "nullptr"
                values[4 * k + j] = f"exl3_moe_kernel_k{k}_n{n}_cb1{suffix}()"
        body = "\n    // DSV41_TP1_RUNTIME_K: MCG K=0 runtime dispatch plus uniform K=4.\n"
        body += "\n".join("    " + ", ".join(values[i:i + 4]) + ("," if i < 32 else "") for i in range(0, 36, 4))
        src = src[:match.start(2)] + body + src[match.end(2):]
    src = src.replace("kernel variant not built (K=2/3 MCG, K=3 MUL1)", "kernel variant not built (K=0/2/3/4 MCG, K=3 MUL1)")
    cu.write_text(src)
    hdr = root / "quant/comp_units/exl3_moe_instances.cuh"
    text = hdr.read_text()
    text += "\n// DSV41_TP1_RUNTIME_K\n"
    for k in (0, 4):
        text += f"fp_exl3_moe_kernel exl3_moe_kernel_k{k}_n128_cb1_mt();\n"
        text += f"fp_exl3_moe_kernel exl3_moe_kernel_k{k}_n256_cb1_mt();\n"
        unit = '#include "exl3_moe_instances.cuh"\n#include "../exl3_moe_kernel.cuh"\n\n'
        for n in (128, 256):
            for suffix, extra in (("", ""), ("_mt", ", true")):
                unit += f"fp_exl3_moe_kernel exl3_moe_kernel_k{k}_n{n}_cb1{suffix}() {{ return exl3_moe_kernel<{k}, {n}, 1{extra}>; }}\n"
        (root / f"quant/comp_units/exl3_moe_inst_k{k}_cb1.cu").write_text(unit)
    hdr.write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plugin", type=Path, required=True)
    ap.add_argument("--helper", type=Path, required=True)
    ap.add_argument("--xmoe", type=Path, required=True)
    args = ap.parse_args()
    args.plugin.write_text(plugin_patch(args.plugin.read_text(), args.helper))
    kernel_patch(args.xmoe)
    print(json.dumps({"plugin": str(args.plugin), "xmoe": str(args.xmoe), "syntax_ok": True}))


if __name__ == "__main__":
    main()

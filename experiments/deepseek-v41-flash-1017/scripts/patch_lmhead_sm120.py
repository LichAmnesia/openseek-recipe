#!/usr/bin/env python3
"""Patch the pinned 5555a2d8 model and VL mapper for sfxnz's MXFP8 head."""
import argparse
import ast
import shutil
from pathlib import Path
from patch_mixed_k import replace_once


def patch_model(src):
    if "DSV41_MXFP8_HEAD_PORT" in src:
        return src
    src = replace_once(src, "        orig_to_new_regex=scale_regex,",
        '        # DSV41_MXFP8_HEAD_PORT: regex precedes the stock head.weight suffix.\n'
        '        orig_to_new_regex={**scale_regex,\n'
        '            re.compile(r"^lm_head\\.weight$"): "head.weight",\n'
        '            re.compile(r"^lm_head\\.weight_scale$"): "lm_head.weight_scale"},',
        "text head key mapping")
    src = replace_once(src,
        '                prefix=maybe_prefix(prefix, "lm_head"),\n            )\n        else:',
        '                prefix=maybe_prefix(prefix, "lm_head"),\n            )\n'
        '            from .lmhead_sm120 import configure_lm_head\n'
        '            configure_lm_head(self.lm_head, vllm_config)\n        else:',
        "MXFP8 head allocation")
    ast.parse(src)
    return src


def patch_vl(src):
    if "DSV41_MXFP8_HEAD_PORT" in src:
        return src
    src = replace_once(src, "import torch\n", "import re\nimport torch\n", "regex import")
    src = replace_once(src, "        orig_to_new_regex=base.orig_to_new_regex,",
        '        # DSV41_MXFP8_HEAD_PORT: scale must live under language_model.\n'
        '        orig_to_new_regex={**base.orig_to_new_regex,\n'
        '            re.compile(r"^lm_head\\.weight_scale$"): "language_model.lm_head.weight_scale"},',
        "VL head scale mapping")
    ast.parse(src)
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, required=True)
    args = ap.parse_args()
    for name, patch in (("model.py", patch_model), ("vl_model.py", patch_vl)):
        path = args.directory / name
        path.write_text(patch(path.read_text()))
    shutil.copy2(Path(__file__).with_name("lmhead_sm120.py"), args.directory / "lmhead_sm120.py")
    print("MXFP8 head namespace, allocation and sm120 compute patched")


if __name__ == "__main__":
    main()

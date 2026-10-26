#!/usr/bin/env python3
"""Port sfxnz DiskEngramTable to the exact 5555a2d8 vLLM layout, eager first.

This first gate deliberately rejects CUDA graph capture. Full graph support is
a separate change; setting DSV41_ENGRAM_DISK alone must never be called success.
"""
import argparse
import ast
import shutil
from pathlib import Path

from patch_mixed_k import replace_once


def patch_engram(src):
    if "DSV41_DISK_EAGER_PORT" in src:
        return src
    src = replace_once(src, "import torch\n", "import torch\nfrom .engram_disk import DiskEngramTable, engram_disk_enabled\n", "disk import")
    src = replace_once(
        src, "        use_thp: bool = False,\n    ) -> None:\n        super().__init__()\n        assert head_sizes",
        "        use_thp: bool = False,\n        model_dir: str | None = None,\n        layer_id: int | None = None,\n    ) -> None:\n        super().__init__()\n        self.disk = None\n        assert head_sizes",
        "embedding constructor",
    )
    anchor = "        weight, scales = self._allocate_weights()\n        self.weight = nn.Parameter(weight, requires_grad=False)"
    replacement = '''        # DSV41_DISK_EAGER_PORT: no full Engram table allocation or checkpoint load.
        if engram_disk_enabled():
            if self.cpu_offload or self.dp_shared_memory:
                raise RuntimeError("Disk Engram first gate requires TP1/DP1 without CPU table offload")
            if model_dir is None or layer_id is None:
                raise RuntimeError("Disk Engram requires the pinned local model path and layer ID")
            if os.environ.get("DSV41_ENGRAM_INT4", "0") != "0":
                raise RuntimeError("This disk reader handles the sfxnz FP8 tables, not int4 tables")
            self.disk = DiskEngramTable(model_dir, layer_id, dim, block_size)
            self.register_buffer("weight", torch.zeros(1, dim, dtype=torch.float8_e4m3fn, device="cpu"), persistent=False)
            self.register_buffer("weight_scale_inv", torch.zeros(1, dim // block_size, dtype=torch.uint8, device="cpu"), persistent=False)
            logger.info("DSV41 disk Engram eager port: layer %s, table stays on NVMe", layer_id)
            return
        weight, scales = self._allocate_weights()
        self.weight = nn.Parameter(weight, requires_grad=False)'''
    src = replace_once(src, anchor, replacement, "skip full table allocation")
    src = replace_once(
        src, "        weight, scales = self._storage()\n        ids_stride_t, ids_stride_h = indices.stride()",
        "        if self.disk is not None:\n            self._disk_lookup(indices, out)\n            return\n        weight, scales = self._storage()\n        ids_stride_t, ids_stride_h = indices.stride()",
        "lookup dispatch",
    )
    method = '''    def _disk_lookup(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Disk Engram first gate requires --enforce-eager")
        if out.dtype != torch.bfloat16:
            raise RuntimeError("Disk Engram first gate only supports BF16 staging")
        t = indices.shape[0]
        ids = indices.detach().to("cpu", dtype=torch.int64)
        local = ids[:, self.head_start:self.head_start + self.part_n_hash_cols]
        if local.shape[1] != self.part_n_hash_cols:
            raise RuntimeError("Disk Engram first gate requires complete TP1 hash heads")
        rows = local.reshape(-1)
        owned = (rows >= self.vocab_start_idx) & (rows < self.vocab_end_idx)
        file_rows = torch.where(owned, rows, torch.zeros_like(rows))
        values = self.disk.gather_dequant(file_rows, owned)
        out[:t].copy_(values.view(t, self.part_n_hash_cols, self.dim))

'''
    # There is only one such class forward signature in this module.
    src = replace_once(src, "    def forward(self, indices: torch.Tensor) -> torch.Tensor:\n", method + "    def forward(self, indices: torch.Tensor) -> torch.Tensor:\n", "disk lookup implementation")
    src = replace_once(
        src, "            cpu_offload=engram_config.cpu_offload,\n            dp_shared_memory=bool(engram_config.dp_shared_memory),\n            use_thp=engram_config.use_thp,",
        "            cpu_offload=False if engram_disk_enabled() else engram_config.cpu_offload,\n            dp_shared_memory=False if engram_disk_enabled() else bool(engram_config.dp_shared_memory),\n            use_thp=engram_config.use_thp,\n            model_dir=get_current_vllm_config().model_config.model,\n            layer_id=layout.layer_ids[layer_hash_index],",
        "pass model path and layer ID",
    )
    ast.parse(src)
    return src


def patch_weights(src):
    if "DSV41_DISK_EAGER_PORT" in src:
        return src
    src = replace_once(
        src, "                    param = f.get_tensor(name)\n                    yield name, param",
        '                    # DSV41_DISK_EAGER_PORT: do not fault 203 GB of lookup tables into RAM.\n'
        '                    if os.environ.get("DSV41_ENGRAM_DISK", "0") == "1" and name.endswith((".engram.embed.weight", ".engram.embed.scale")):\n'
        '                        continue\n                    param = f.get_tensor(name)\n                    yield name, param',
        "weight iterator skip",
    )
    ast.parse(src)
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engram", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--disk-module", type=Path, required=True)
    args = ap.parse_args()
    eng = patch_engram(args.engram.read_text())
    weights = patch_weights(args.weights.read_text())
    shutil.copy2(args.disk_module, args.engram.parent / "engram_disk.py")
    args.engram.write_text(eng)
    args.weights.write_text(weights)
    print("disk Engram allocation/iterator/lookup patched; eager-only first gate")


if __name__ == "__main__":
    main()

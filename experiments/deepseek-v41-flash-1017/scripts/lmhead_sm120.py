"""TP1 eager MXFP8 head, preserving the pack's FP8 values and E8M0 scales.

Parameter layout/loader follows sfxnz lmhead_mxfp8.py at 19e0c566;
compute uses the pinned diffbot sm120_dense W8A16 path, not Spark b12x.
"""
import importlib.util
import json
from pathlib import Path

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import set_weight_attrs

_dense = None


def dense():
    global _dense
    if _dense is None:
        spec = importlib.util.spec_from_file_location(
            "dsv41_head_dense", "/opt/recipe/kernels/sm120_dense.py")
        _dense = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_dense)
    return _dense


class Mxfp8HeadMethod(QuantizeMethodBase):
    def create_weights(self, *args, **kwargs):
        raise RuntimeError("MXFP8 head parameters must be created by configure_lm_head")

    def process_weights_after_loading(self, layer):
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.dtype == torch.uint8
        assert layer.weight_scale.shape == (layer.weight.shape[0], layer.weight.shape[1] // 32)
        layer.weight.data = layer.weight.data.contiguous()
        layer.weight_scale.data = layer.weight_scale.data.contiguous()

    def apply(self, layer, x, bias=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("This MXFP8 head port is validated only in eager mode")
        return dense().linear(x, layer.weight, layer.weight_scale, bias,
                              gemv_max=32, chunk_max=256)


def configure_lm_head(layer, vllm_config):
    cfg = vllm_config.model_config
    index = Path(cfg.model) / "model.safetensors.index.json"
    keys = json.loads(index.read_text())["weight_map"]
    if "lm_head.weight_scale" not in keys:
        return
    assert "lm_head.weight" in keys, "Incomplete MXFP8 head checkpoint"
    assert vllm_config.parallel_config.tensor_parallel_size == 1
    assert cfg.enforce_eager, "MXFP8 head first gate requires eager mode"
    assert cfg.head_dtype == cfg.dtype == torch.bfloat16
    n, k = layer.weight.shape
    assert k % 32 == 0
    device = layer.weight.device
    for name, shape, dtype in (
        ("weight", (n, k), torch.float8_e4m3fn),
        ("weight_scale", (n, k // 32), torch.uint8),
    ):
        param = Parameter(torch.empty(shape, dtype=dtype, device=device), requires_grad=False)
        set_weight_attrs(param, {"input_dim": 1, "output_dim": 0,
                                 "weight_loader": layer.weight_loader})
        setattr(layer, name, param)
    layer.quant_method = Mxfp8HeadMethod()
    print(f"dsv41: MXFP8 lm_head sm120 W8A16 eager enabled {(n, k)}", flush=True)

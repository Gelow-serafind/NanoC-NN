from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from nanoc_onnx_converter.exporter import convert_model
from nanoc_onnx_converter.model import ConversionOptions


def _make_gemm_model(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])
    weight = numpy_helper.from_array(
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        name="dense.weight",
    )
    bias = numpy_helper.from_array(
        np.array([0.01, 0.02, 0.03], dtype=np.float32),
        name="dense.bias",
    )
    gemm = helper.make_node(
        "Gemm",
        inputs=["input", "dense.weight", "dense.bias"],
        outputs=["logits"],
        name="dense",
        transB=1,
    )
    graph = helper.make_graph([gemm], "test_graph", [x], [y], [weight, bias])
    model = helper.make_model(
        graph,
        producer_name="nanoc-test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_convert_model_writes_complete_output_folder(tmp_path: Path) -> None:
    model_path = tmp_path / "gemm.onnx"
    out_dir = tmp_path / "export"
    _make_gemm_model(model_path)

    model_info = convert_model(
        ConversionOptions(model_path=model_path, out_dir=out_dir, prefix="unit")
    )

    assert len(model_info.nodes) == 1
    assert {item.name for item in model_info.initializers} == {"dense.weight", "dense.bias"}
    assert (out_dir / "README.md").exists()
    assert (out_dir / "conversion_report.txt").exists()
    assert (out_dir / "model_summary.md").exists()
    assert (out_dir / "model_graph.json").exists()
    assert (out_dir / "weights.h").exists()

    graph = json.loads((out_dir / "model_graph.json").read_text(encoding="utf-8"))
    assert graph["nodes"][0]["op_type"] == "Gemm"
    assert graph["nodes"][0]["input_shapes"]["dense.weight"] == [3, 2]
    assert graph["nodes"][0]["output_shapes"] == {"logits": [1, 3]}

    weights_h = (out_dir / "weights.h").read_text(encoding="utf-8")
    assert "static const float unit_dense_weight[6]" in weights_h
    assert "#define UNIT_DENSE_BIAS_SHAPE {3}" in weights_h

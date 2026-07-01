from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from nanoc_nn.pipeline.cli import main as pipeline_main
from nanoc_nn.pipeline.runner import safe_prefix


def test_safe_prefix_is_c_identifier_friendly() -> None:
    assert safe_prefix("01 demo-model") == "m_01_demo_model"
    assert safe_prefix("model.onnx") == "model_onnx"


def test_root_pipeline_cli_creates_two_stage_output(tmp_path: Path) -> None:
    model_path = tmp_path / "demo.onnx"
    _make_demo_model(model_path)

    status = pipeline_main(
        [
            "--model",
            str(model_path),
            "--target",
            "cortex-m4",
            "--no-compile",
        ]
    )

    output_root = tmp_path / "demo-nanoc-cmsis"
    assert status == 0
    assert (output_root / "converter-output" / "model_graph.json").exists()
    assert (output_root / "cmsis-codegen" / "src" / "model.c").exists()
    assert (output_root / "cmsis-codegen" / "reports" / "op_mapping.md").exists()
    assert (output_root / "pipeline_report.md").exists()


def _make_demo_model(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    weight = numpy_helper.from_array(
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        name="fc.weight",
    )
    bias = numpy_helper.from_array(
        np.array([0.01, -0.01], dtype=np.float32),
        name="fc.bias",
    )
    gemm = helper.make_node(
        "Gemm",
        ["input", "fc.weight", "fc.bias"],
        ["output"],
        name="fc",
        transB=1,
    )
    graph = helper.make_graph([gemm], "demo", [x], [y], [weight, bias])
    model = helper.make_model(
        graph,
        producer_name="nanoc-test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)

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


def test_root_pipeline_cli_rejects_float_model_by_default(tmp_path: Path) -> None:
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
    assert status == 2
    assert (output_root / "converter-output" / "model_graph.json").exists()
    assert (output_root / "cmsis-codegen" / "src" / "model.c").exists()
    assert (output_root / "cmsis-codegen" / "reports" / "op_mapping.md").exists()
    assert (output_root / "pipeline_report.md").exists()
    report = (output_root / "pipeline_report.md").read_text(encoding="utf-8")
    assert "Codegen 状态：`blocked`" in report


def test_root_pipeline_cli_accepts_qdq_int8_fc_model(tmp_path: Path) -> None:
    model_path = tmp_path / "qdq_fc.onnx"
    _make_qdq_fc_model(model_path)

    status = pipeline_main(
        [
            "--model",
            str(model_path),
            "--layout",
            "NHWC",
            "--target",
            "cortex-m4",
            "--no-compile",
        ]
    )

    output_root = tmp_path / "qdq_fc-nanoc-cmsis"
    assert status == 0
    model_c = (output_root / "cmsis-codegen" / "src" / "model.c").read_text(
        encoding="utf-8"
    )
    assert "arm_fully_connected_s8(" in model_c
    assert "codegen status: ok" not in model_c
    report = (output_root / "pipeline_report.md").read_text(encoding="utf-8")
    assert "Codegen 状态：`ok`" in report


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


def _make_qdq_fc_model(path: Path) -> None:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    value_infos = [
        helper.make_tensor_value_info("input.q", TensorProto.INT8, [1, 4]),
        helper.make_tensor_value_info("input.dq", TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info("fc.weight.dq", TensorProto.FLOAT, [2, 4]),
        helper.make_tensor_value_info("fc.out", TensorProto.FLOAT, [1, 2]),
        helper.make_tensor_value_info("output.q", TensorProto.INT8, [1, 2]),
    ]
    initializers = [
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), name="input.scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="input.zero_point"),
        numpy_helper.from_array(np.array(0.05, dtype=np.float32), name="fc.weight.scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="fc.weight.zero_point"),
        numpy_helper.from_array(np.array(0.2, dtype=np.float32), name="output.scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="output.zero_point"),
        numpy_helper.from_array(
            np.array([[1, -2, 3, 4], [-5, 6, -7, 8]], dtype=np.int8),
            name="fc.weight.q",
        ),
        numpy_helper.from_array(np.array([0.1, -0.2], dtype=np.float32), name="fc.bias"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "input.scale", "input.zero_point"],
            ["input.q"],
            name="input_quant",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input.q", "input.scale", "input.zero_point"],
            ["input.dq"],
            name="input_dequant",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["fc.weight.q", "fc.weight.scale", "fc.weight.zero_point"],
            ["fc.weight.dq"],
            name="weight_dequant",
        ),
        helper.make_node(
            "Gemm",
            ["input.dq", "fc.weight.dq", "fc.bias"],
            ["fc.out"],
            name="fc",
            transB=1,
        ),
        helper.make_node(
            "QuantizeLinear",
            ["fc.out", "output.scale", "output.zero_point"],
            ["output.q"],
            name="output_quant",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output.q", "output.scale", "output.zero_point"],
            ["output"],
            name="output_dequant",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq_fc",
        [input_tensor],
        [output_tensor],
        initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="nanoc-test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)

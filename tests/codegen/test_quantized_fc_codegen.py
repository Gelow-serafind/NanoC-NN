from __future__ import annotations

# ruff: noqa: E402, I001

import json
import shutil
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from nanoc_nn.codegen.generator import generate_project
from nanoc_nn.codegen.model import CodegenOptions
from nanoc_nn.converter.exporter import convert_model
from nanoc_nn.converter.model import ConversionOptions


def test_qdq_fully_connected_generates_real_cmsis_call(tmp_path: Path) -> None:
    model_path = tmp_path / "qdq_fc.onnx"
    export_dir = tmp_path / "converter-output"
    codegen_dir = tmp_path / "cmsis-codegen"
    _make_qdq_fc_model(model_path)

    convert_model(
        ConversionOptions(
            model_path=model_path,
            out_dir=export_dir,
            layout="NHWC",
            prefix="unit",
        )
    )

    graph = json.loads((export_dir / "model_graph.json").read_text(encoding="utf-8"))
    assert graph["quantization"]["nodes"]["fc"]["cmsis_nn"]["api"] == (
        "arm_fully_connected_s8"
    )
    assert graph["quantization"]["weights"]["fc.weight.q"]["values"] == [1, -2, 3, 4, -5, 6, -7, 8]

    result = generate_project(CodegenOptions(input_dir=export_dir, out_dir=codegen_dir))

    assert result.status == "ok"
    model_c = (codegen_dir / "src" / "model.c").read_text(encoding="utf-8")
    model_weights = (codegen_dir / "include" / "model_weights.h").read_text(encoding="utf-8")
    assert "arm_fully_connected_s8_get_buffer_size(&filter_dims)" in model_c
    assert "arm_fully_connected_s8(" in model_c
    assert "static const int8_t nanoc_fc_weights[8]" in model_weights
    assert "static const int32_t nanoc_fc_bias[2]" in model_weights

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no host C compiler available")
    subprocess.run(
        [
            cc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-I",
            str(codegen_dir / "include"),
            str(codegen_dir / "src" / "model.c"),
            str(codegen_dir / "src" / "main.c"),
            "-o",
            str(tmp_path / "smoke"),
        ],
        check=True,
    )


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

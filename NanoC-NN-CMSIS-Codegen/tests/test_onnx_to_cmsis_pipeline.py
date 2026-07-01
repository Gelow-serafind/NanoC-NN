from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def test_pipeline_creates_converter_and_codegen_folders_next_to_onnx(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "tiny_model.onnx"
    _make_tiny_model(model_path)
    script = Path(__file__).resolve().parents[1] / "tools" / "onnx_to_cmsis_pipeline.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(model_path),
            "--target",
            "cortex-m4",
            "--sram-budget",
            "64K",
            "--flash-budget",
            "256K",
            "--no-compile",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output_root = tmp_path / "tiny_model-nanoc-cmsis"
    assert (output_root / "converter-output" / "model_graph.json").exists()
    assert (output_root / "converter-output" / "weights.h").exists()
    assert (output_root / "cmsis-codegen" / "include" / "model.h").exists()
    assert (output_root / "cmsis-codegen" / "src" / "model.c").exists()
    assert (output_root / "cmsis-codegen" / "reports" / "codegen_report.txt").exists()
    assert (output_root / "pipeline_report.md").exists()

    report = (output_root / "pipeline_report.md").read_text(encoding="utf-8")
    assert "converter-output/" in report
    assert "cmsis-codegen/" in report


def _make_tiny_model(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
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
        ["logits"],
        name="fc",
        transB=1,
    )
    graph = helper.make_graph([gemm], "tiny", [x], [y], [weight, bias])
    model = helper.make_model(
        graph,
        producer_name="nanoc-test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)

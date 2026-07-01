from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nanoc_nn.codegen.generator import generate_project
from nanoc_nn.codegen.model import CodegenOptions


def test_generate_project_from_legacy_converter_output(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    out_dir = tmp_path / "generated"
    _write_converter_output(export_dir)

    result = generate_project(
        CodegenOptions(
            input_dir=export_dir,
            out_dir=out_dir,
            sram_budget="64K",
            flash_budget="256K",
        )
    )

    assert result.status == "blocked"
    assert len(result.mappings) == 2
    assert result.mappings[0].status == "blocked"
    assert result.mappings[1].status == "fused"
    assert (out_dir / "include" / "model.h").exists()
    assert (out_dir / "src" / "model.c").exists()
    assert (out_dir / "reports" / "op_mapping.md").exists()

    report = (out_dir / "reports" / "codegen_report.txt").read_text(encoding="utf-8")
    assert "status: blocked" in report
    assert "quantization_issues: 1" in report


def test_generated_c_project_is_c99_smoke_compilable(tmp_path: Path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no host C compiler available")

    export_dir = tmp_path / "export"
    out_dir = tmp_path / "generated"
    _write_converter_output(export_dir)
    generate_project(CodegenOptions(input_dir=export_dir, out_dir=out_dir))

    subprocess.run(
        [
            cc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-I",
            str(out_dir / "include"),
            str(out_dir / "src" / "model.c"),
            str(out_dir / "src" / "main.c"),
            "-o",
            str(tmp_path / "smoke"),
        ],
        check=True,
    )


def _write_converter_output(export_dir: Path) -> None:
    export_dir.mkdir(parents=True)
    graph = {
        "model_path": "unit.onnx",
        "ir_version": 9,
        "opsets": {"": 17},
        "producer_name": "unit",
        "producer_version": "",
        "graph_name": "unit_graph",
        "layout": "NCHW",
        "inputs": [
            {
                "name": "input",
                "elem_type": "FLOAT",
                "shape": [1, 4],
                "raw_shape": [1, 4],
                "dynamic_axes": [],
                "is_dynamic": False,
            }
        ],
        "outputs": [
            {
                "name": "relu_out",
                "elem_type": "FLOAT",
                "shape": [1, 2],
                "raw_shape": [1, 2],
                "dynamic_axes": [],
                "is_dynamic": False,
            }
        ],
        "initializers": [
            {
                "name": "fc.weight",
                "elem_type": "FLOAT",
                "shape": [2, 4],
                "element_count": 8,
                "c_name": "unit_fc_weight",
                "role": "parameter",
                "is_float32": True,
                "is_c_exportable": True,
            },
            {
                "name": "fc.bias",
                "elem_type": "FLOAT",
                "shape": [2],
                "element_count": 2,
                "c_name": "unit_fc_bias",
                "role": "parameter",
                "is_float32": True,
                "is_c_exportable": True,
            },
        ],
        "nodes": [
            {
                "index": 0,
                "name": "fc",
                "op_type": "Gemm",
                "inputs": ["input", "fc.weight", "fc.bias"],
                "outputs": ["fc_out"],
                "attributes": {"transB": 1},
                "normalized_attributes": {"alpha": 1.0, "beta": 1.0, "transB": 1},
                "input_shapes": {"input": [1, 4], "fc.weight": [2, 4], "fc.bias": [2]},
                "output_shapes": {"fc_out": [1, 2]},
                "weights": ["fc.weight", "fc.bias"],
                "status": "supported",
            },
            {
                "index": 1,
                "name": "relu",
                "op_type": "Relu",
                "inputs": ["fc_out"],
                "outputs": ["relu_out"],
                "attributes": {},
                "normalized_attributes": {},
                "input_shapes": {"fc_out": [1, 2]},
                "output_shapes": {"relu_out": [1, 2]},
                "weights": [],
                "status": "supported",
            },
        ],
        "warnings": [],
        "errors": [],
        "unsupported_ops": [],
        "c_exportable_initializers": ["fc.weight", "fc.bias"],
    }
    (export_dir / "model_graph.json").write_text(
        json.dumps(graph, indent=2) + "\n",
        encoding="utf-8",
    )
    (export_dir / "conversion_report.txt").write_text("unit report\n", encoding="utf-8")
    (export_dir / "weights.h").write_text(
        "\n".join(
            [
                "#ifndef UNIT_WEIGHTS_H",
                "#define UNIT_WEIGHTS_H",
                "static const float unit_fc_weight[8] = {0};",
                "static const float unit_fc_bias[2] = {0};",
                "#endif",
                "",
            ]
        ),
        encoding="utf-8",
    )


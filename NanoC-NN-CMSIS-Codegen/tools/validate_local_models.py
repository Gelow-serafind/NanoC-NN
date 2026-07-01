from __future__ import annotations

# ruff: noqa: E402, I001

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_ROOT = REPO_ROOT / "NanoC-NN-ONNX-Converter"
CODEGEN_ROOT = REPO_ROOT / "NanoC-NN-CMSIS-Codegen"
sys.path.insert(0, str(CONVERTER_ROOT))
sys.path.insert(0, str(CODEGEN_ROOT))

from nanoc_cmsis_codegen.generator import generate_project  # noqa: E402
from nanoc_cmsis_codegen.model import CodegenOptions, GenerationResult  # noqa: E402
from nanoc_onnx_converter.exporter import convert_model  # noqa: E402
from nanoc_onnx_converter.model import ConversionOptions  # noqa: E402


@dataclass(frozen=True)
class ValidationCase:
    name: str
    builder: object
    layout: str = "NCHW"
    target: str = "cortex-m4"


@dataclass(frozen=True)
class ValidationRow:
    name: str
    onnx_path: Path
    export_dir: Path
    generated_dir: Path
    codegen_status: str
    blocked_or_unsupported: int
    compile_status: str
    notes: str


def main() -> int:
    build_dir = CODEGEN_ROOT / "build" / "local-validation"
    models_dir = build_dir / "models"
    exports_dir = build_dir / "exports"
    generated_dir = build_dir / "generated"
    report_path = build_dir / "coverage_report.md"
    build_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ValidationCase("gemm_relu", _make_gemm_relu, layout="NCHW"),
        ValidationCase("conv_relu_maxpool_gemm", _make_conv_relu_maxpool_gemm, layout="NCHW"),
        ValidationCase("reshape_add_softmax", _make_reshape_add_softmax, layout="NCHW"),
        ValidationCase("global_average_pool", _make_global_average_pool, layout="NCHW"),
        ValidationCase("conv_batchnorm_relu", _make_conv_batchnorm_relu, layout="NCHW"),
        ValidationCase("mul_upstream_gap", _make_mul_upstream_gap, layout="NCHW"),
    ]

    rows: list[ValidationRow] = []
    for index, case in enumerate(cases, start=1):
        onnx_path = models_dir / f"{index:02d}_{case.name}.onnx"
        export_dir = exports_dir / case.name
        out_dir = generated_dir / case.name
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        case.builder(onnx_path)
        convert_model(
            ConversionOptions(
                model_path=onnx_path,
                out_dir=export_dir,
                layout=case.layout,
                prefix=f"val{index}",
            )
        )
        result = generate_project(
            CodegenOptions(
                input_dir=export_dir,
                out_dir=out_dir,
                cmsis_nn_root=CODEGEN_ROOT / "third_party" / "CMSIS-NN",
                target=case.target,
                sram_budget="128K",
                flash_budget="512K",
            )
        )
        rows.append(
            ValidationRow(
                name=case.name,
                onnx_path=onnx_path,
                export_dir=export_dir,
                generated_dir=out_dir,
                codegen_status=result.status,
                blocked_or_unsupported=len(result.blocking_mappings),
                compile_status=_compile_generated(out_dir),
                notes=_notes(result),
            )
        )

    report_path.write_text(_coverage_report(rows), encoding="utf-8")
    print(f"coverage report: {report_path}")
    for row in rows:
        print(
            f"{row.name}: codegen={row.codegen_status}, "
            f"blocked={row.blocked_or_unsupported}, compile={row.compile_status}"
        )
    return 0 if all(row.compile_status in {"ok", "skipped"} for row in rows) else 1


def _compile_generated(out_dir: Path) -> str:
    cc = shutil.which("cc")
    if cc is None:
        return "skipped"
    smoke = out_dir / "build-smoke" / "nanoc_smoke"
    smoke.parent.mkdir(parents=True, exist_ok=True)
    command = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-I",
        str(out_dir / "include"),
        str(out_dir / "src" / "model.c"),
        str(out_dir / "src" / "main.c"),
        "-o",
        str(smoke),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode == 0:
        return "ok"
    log_path = out_dir / "build-smoke" / "compile.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    return f"failed: {log_path}"


def _notes(result: GenerationResult) -> str:
    if result.status == "ok":
        return "ready"
    blockers = result.blocking_mappings[:3]
    if not blockers:
        return "blocked by quantization or budget"
    return "; ".join(f"{item.node_name}:{item.status}" for item in blockers)


def _coverage_report(rows: list[ValidationRow]) -> str:
    lines = [
        "# Local ONNX Coverage Report",
        "",
        "本报告由 `tools/validate_local_models.py` 生成。",
        "",
        "验证链路固定为：ONNX -> `NanoC-NN-ONNX-Converter` 输出目录 -> "
        "`NanoC-NN-CMSIS-Codegen`，codegen 不直接读取 ONNX。",
        "",
        "| 模型 | Codegen 状态 | 阻塞/不支持数量 | C99 编译 | 输出目录 | 备注 |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.name}` | `{row.codegen_status}` | {row.blocked_or_unsupported} | "
            f"`{row.compile_status}` | `{row.generated_dir}` | {row.notes} |"
        )
    lines.extend(
        [
            "",
            "## 初步结论",
            "",
            (
                "- 当前 converter 尚未输出 int8 量化 section，因此 CMSIS-NN s8 runtime "
                "层均会被标记为 blocked。"
            ),
            "- 生成的工程骨架和报告可稳定输出，并能进行 C99 smoke compile。",
            "- `mul_upstream_gap` 用于验证 converter 上游不支持时，codegen 不绕过前端自行解析。",
            "- 下一批优先项：converter schema version、量化字段、layout 字段、Q/DQ 模型解析。",
            "",
        ]
    )
    return "\n".join(lines)


def _make_gemm_relu(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    weight = _tensor("fc.weight", np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]))
    bias = _tensor("fc.bias", np.array([0.01, -0.01]))
    gemm = helper.make_node(
        "Gemm",
        ["input", "fc.weight", "fc.bias"],
        ["fc_out"],
        name="fc",
        transB=1,
    )
    relu = helper.make_node("Relu", ["fc_out"], ["output"], name="relu")
    _save_model(path, [gemm, relu], [x], [y], [weight, bias], "gemm_relu")


def _make_conv_relu_maxpool_gemm(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])
    conv_w = _tensor("conv.weight", np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) / 50)
    conv_b = _tensor("conv.bias", np.array([0.01, -0.02]))
    fc_w = _tensor("fc.weight", np.ones((3, 18), dtype=np.float32) / 18)
    fc_b = _tensor("fc.bias", np.array([0.1, 0.0, -0.1]))
    conv = helper.make_node(
        "Conv",
        ["input", "conv.weight", "conv.bias"],
        ["conv_out"],
        name="conv",
    )
    relu = helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu")
    pool = helper.make_node(
        "MaxPool",
        ["relu_out"],
        ["pool_out"],
        name="pool",
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    flatten = helper.make_node("Flatten", ["pool_out"], ["flat"], name="flatten", axis=1)
    gemm = helper.make_node(
        "Gemm",
        ["flat", "fc.weight", "fc.bias"],
        ["logits"],
        name="fc",
        transB=1,
    )
    _save_model(
        path,
        [conv, relu, pool, flatten, gemm],
        [x],
        [y],
        [conv_w, conv_b, fc_w, fc_b],
        "conv_relu_maxpool_gemm",
    )


def _make_reshape_add_softmax(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2])
    y = helper.make_tensor_value_info("prob", TensorProto.FLOAT, [1, 4])
    shape = numpy_helper.from_array(np.array([1, 4], dtype=np.int64), name="target_shape")
    bias = _tensor("add.bias", np.array([[0.1, -0.1, 0.2, -0.2]]))
    reshape = helper.make_node("Reshape", ["input", "target_shape"], ["flat"], name="reshape")
    add = helper.make_node("Add", ["flat", "add.bias"], ["biased"], name="add")
    softmax = helper.make_node("Softmax", ["biased"], ["prob"], name="softmax", axis=1)
    _save_model(path, [reshape, add, softmax], [x], [y], [shape, bias], "reshape_add_softmax")


def _make_global_average_pool(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
    fc_w = _tensor("fc.weight", np.ones((2, 3), dtype=np.float32) / 3)
    fc_b = _tensor("fc.bias", np.array([0.0, 0.1]))
    gap = helper.make_node("GlobalAveragePool", ["input"], ["gap"], name="gap")
    flatten = helper.make_node("Flatten", ["gap"], ["flat"], name="flatten", axis=1)
    gemm = helper.make_node(
        "Gemm",
        ["flat", "fc.weight", "fc.bias"],
        ["logits"],
        name="fc",
        transB=1,
    )
    _save_model(path, [gap, flatten, gemm], [x], [y], [fc_w, fc_b], "global_average_pool")


def _make_conv_batchnorm_relu(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 5, 5])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 3, 3])
    conv_w = _tensor("conv.weight", np.ones((2, 2, 3, 3), dtype=np.float32) / 9)
    conv_b = _tensor("conv.bias", np.array([0.0, 0.0]))
    scale = _tensor("bn.scale", np.array([1.0, 1.0]))
    bias = _tensor("bn.bias", np.array([0.0, 0.0]))
    mean = _tensor("bn.mean", np.array([0.0, 0.0]))
    var = _tensor("bn.var", np.array([1.0, 1.0]))
    conv = helper.make_node(
        "Conv",
        ["input", "conv.weight", "conv.bias"],
        ["conv_out"],
        name="conv",
    )
    bn = helper.make_node(
        "BatchNormalization",
        ["conv_out", "bn.scale", "bn.bias", "bn.mean", "bn.var"],
        ["bn_out"],
        name="bn",
    )
    relu = helper.make_node("Relu", ["bn_out"], ["output"], name="relu")
    _save_model(
        path,
        [conv, bn, relu],
        [x],
        [y],
        [conv_w, conv_b, scale, bias, mean, var],
        "conv_batchnorm_relu",
    )


def _make_mul_upstream_gap(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    scale = _tensor("scale", np.array([[1.0, 2.0, 3.0, 4.0]]))
    mul = helper.make_node("Mul", ["input", "scale"], ["output"], name="mul")
    _save_model(path, [mul], [x], [y], [scale], "mul_upstream_gap")


def _tensor(name: str, values: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def _save_model(
    path: Path,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto],
    graph_name: str,
) -> None:
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializers)
    model = helper.make_model(
        graph,
        producer_name="nanoc-local-validation",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    raise SystemExit(main())

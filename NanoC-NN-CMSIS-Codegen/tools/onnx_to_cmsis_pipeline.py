from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_ROOT = REPO_ROOT / "NanoC-NN-ONNX-Converter"
CODEGEN_ROOT = REPO_ROOT / "NanoC-NN-CMSIS-Codegen"
sys.path.insert(0, str(CONVERTER_ROOT))
sys.path.insert(0, str(CODEGEN_ROOT))

from nanoc_cmsis_codegen.generator import generate_project
from nanoc_cmsis_codegen.model import CodegenOptions, GenerationResult
from nanoc_onnx_converter.exporter import convert_model
from nanoc_onnx_converter.model import ConversionOptions


@dataclass(frozen=True)
class PipelineOptions:
    model_path: Path
    output_root: Path
    layout: str
    prefix: str
    batch_size: int
    cmsis_nn_root: Path | None
    cmsis_path: Path | None
    cmsis_version: str | None
    target: str
    backend: str | None
    sram_budget: str | None
    flash_budget: str | None
    strict: bool
    verbose: bool
    compile_smoke: bool

    @property
    def converter_dir(self) -> Path:
        return self.output_root / "converter-output"

    @property
    def codegen_dir(self) -> Path:
        return self.output_root / "cmsis-codegen"


@dataclass(frozen=True)
class PipelineResult:
    options: PipelineOptions
    codegen_result: GenerationResult
    compile_status: str
    compile_log: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnx-to-cmsis-pipeline",
        description=(
            "Run NanoC-NN ONNX Converter and then CMSIS-NN Codegen for one ONNX model."
        ),
    )
    parser.add_argument("--model", required=True, help="input ONNX model path")
    parser.add_argument(
        "--out-root",
        help=(
            "pipeline output directory; default is <onnx-dir>/<onnx-stem>-nanoc-cmsis"
        ),
    )
    parser.add_argument("--layout", default="NCHW", help="converter layout label")
    parser.add_argument("--prefix", help="C symbol prefix; default uses ONNX file stem")
    parser.add_argument("--batch-size", type=int, default=1, help="fixed dynamic batch size")
    parser.add_argument(
        "--cmsis-nn-root",
        default=str(CODEGEN_ROOT / "third_party" / "CMSIS-NN"),
        help="CMSIS-NN source or installation path",
    )
    parser.add_argument(
        "--cmsis-path",
        help="CMSIS repository path that provides CMSIS/Core/Include",
    )
    parser.add_argument("--cmsis-version", help="target CMSIS-NN version label")
    parser.add_argument("--target", default="cortex-m4", help="target Cortex-M profile")
    parser.add_argument("--backend", choices=["scalar", "dsp", "mve"], help="CMSIS-NN backend")
    parser.add_argument("--sram-budget", help="target SRAM budget, e.g. 128K")
    parser.add_argument("--flash-budget", help="target Flash budget, e.g. 512K")
    parser.add_argument("--strict", action="store_true", help="fail on blocked/unsupported cases")
    parser.add_argument("--verbose", action="store_true", help="print detailed diagnostics")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip host C99 smoke compile of generated sources",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        parser.exit(status=2, message=f"error: ONNX model does not exist: {model_path}\n")
    if not model_path.is_file():
        parser.exit(status=2, message=f"error: ONNX model path is not a file: {model_path}\n")

    output_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else model_path.parent / f"{model_path.stem}-nanoc-cmsis"
    )
    options = PipelineOptions(
        model_path=model_path,
        output_root=output_root,
        layout=args.layout,
        prefix=args.prefix or _safe_prefix(model_path.stem),
        batch_size=args.batch_size,
        cmsis_nn_root=Path(args.cmsis_nn_root).expanduser().resolve()
        if args.cmsis_nn_root
        else None,
        cmsis_path=Path(args.cmsis_path).expanduser().resolve() if args.cmsis_path else None,
        cmsis_version=args.cmsis_version,
        target=args.target,
        backend=args.backend,
        sram_budget=args.sram_budget,
        flash_budget=args.flash_budget,
        strict=args.strict,
        verbose=args.verbose,
        compile_smoke=not args.no_compile,
    )

    try:
        result = run_pipeline(options)
    except Exception as exc:  # noqa: BLE001 - CLI boundary.
        parser.exit(status=1, message=f"error: pipeline failed: {exc}\n")

    print(f"pipeline output: {result.options.output_root}")
    print(f"converter output: {result.options.converter_dir}")
    print(f"cmsis-codegen output: {result.options.codegen_dir}")
    print(f"codegen status: {result.codegen_result.status}")
    print(f"c99 smoke compile: {result.compile_status}")
    print(f"pipeline report: {result.options.output_root / 'pipeline_report.md'}")
    if options.verbose:
        for mapping in result.codegen_result.mappings:
            print(
                f"- {mapping.index}:{mapping.node_name} {mapping.onnx_op} "
                f"=> {mapping.status} ({mapping.cmsis_action})"
            )
    return 0


def run_pipeline(options: PipelineOptions) -> PipelineResult:
    options.output_root.mkdir(parents=True, exist_ok=True)
    convert_model(
        ConversionOptions(
            model_path=options.model_path,
            out_dir=options.converter_dir,
            layout=options.layout,
            prefix=options.prefix,
            batch_size=options.batch_size,
            strict=options.strict,
            verbose=options.verbose,
        )
    )
    codegen_result = generate_project(
        CodegenOptions(
            input_dir=options.converter_dir,
            out_dir=options.codegen_dir,
            cmsis_nn_root=options.cmsis_nn_root,
            cmsis_path=options.cmsis_path,
            cmsis_version=options.cmsis_version,
            target=options.target,
            backend=options.backend,
            sram_budget=options.sram_budget,
            flash_budget=options.flash_budget,
            strict=options.strict,
            verbose=options.verbose,
        )
    )
    compile_status, compile_log = (
        _compile_generated(options.codegen_dir) if options.compile_smoke else ("skipped", None)
    )
    result = PipelineResult(
        options=options,
        codegen_result=codegen_result,
        compile_status=compile_status,
        compile_log=compile_log,
    )
    _write_pipeline_report(result)
    return result


def _compile_generated(codegen_dir: Path) -> tuple[str, Path | None]:
    cc = shutil.which("cc")
    if cc is None:
        return "skipped: no host cc", None
    smoke_dir = codegen_dir / "build-smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    log_path = smoke_dir / "compile.log"
    command = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-I",
        str(codegen_dir / "include"),
        str(codegen_dir / "src" / "model.c"),
        str(codegen_dir / "src" / "main.c"),
        "-o",
        str(smoke_dir / "nanoc_smoke"),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode == 0:
        return "ok", log_path
    return f"failed: {completed.returncode}", log_path


def _write_pipeline_report(result: PipelineResult) -> None:
    options = result.options
    codegen = result.codegen_result
    lines = [
        "# NanoC-NN ONNX 到 CMSIS-NN Pipeline 报告",
        "",
        "## 输入输出",
        "",
        f"- ONNX：`{options.model_path}`",
        f"- 输出根目录：`{options.output_root}`",
        f"- Converter 产物：`{options.converter_dir}`",
        f"- CMSIS-NN Codegen 产物：`{options.codegen_dir}`",
        "",
        "## 状态",
        "",
        f"- Codegen 状态：`{codegen.status}`",
        f"- 阻塞/不支持节点数量：`{len(codegen.blocking_mappings)}`",
        f"- C99 smoke compile：`{result.compile_status}`",
        f"- Compile log：`{result.compile_log or 'none'}`",
        "",
        "## 关键报告",
        "",
        f"- `{options.codegen_dir / 'reports' / 'codegen_report.txt'}`",
        f"- `{options.codegen_dir / 'reports' / 'op_mapping.md'}`",
        f"- `{options.codegen_dir / 'reports' / 'memory_plan.md'}`",
        f"- `{options.codegen_dir / 'reports' / 'quantization.md'}`",
        f"- `{options.codegen_dir / 'reports' / 'firmware_integration.md'}`",
        "",
        "## 目录结构",
        "",
        "```text",
        f"{options.output_root.name}/",
        "├── converter-output/",
        "│   ├── model_graph.json",
        "│   ├── model_summary.md",
        "│   ├── conversion_report.txt",
        "│   └── weights.h",
        "├── cmsis-codegen/",
        "│   ├── include/",
        "│   ├── src/",
        "│   ├── reports/",
        "│   └── CMakeLists.txt",
        "└── pipeline_report.md",
        "```",
        "",
        "## 说明",
        "",
        "- 本脚本只是编排 converter 和 codegen；codegen 仍然只消费 converter 输出目录。",
        "- 当前 converter 未提供 int8 量化 section 时，CMSIS-NN runtime 层会报告为 `blocked`。",
        "- C99 smoke compile 通过只代表生成工程骨架语法可编译，不代表真实 int8 推理已可用。",
        "",
    ]
    (options.output_root / "pipeline_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _safe_prefix(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    prefix = "".join(chars).strip("_")
    if not prefix:
        prefix = "nanoc"
    if prefix[0].isdigit():
        prefix = f"m_{prefix}"
    return prefix


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nanoc_nn.codegen.generator import generate_project
from nanoc_nn.codegen.model import CodegenOptions, GenerationResult
from nanoc_nn.converter.exporter import convert_model
from nanoc_nn.converter.model import ConversionOptions


@dataclass(frozen=True)
class PipelineOptions:
    model_path: Path
    output_root: Path
    layout: str = "NCHW"
    prefix: str = "nanoc"
    batch_size: int = 1
    cmsis_nn_root: Path | None = None
    cmsis_path: Path | None = None
    cmsis_version: str | None = None
    target: str = "cortex-m4"
    backend: str | None = None
    sram_budget: str | None = None
    flash_budget: str | None = None
    strict: bool = False
    verbose: bool = False
    compile_smoke: bool = True

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
        compile_generated(options.codegen_dir) if options.compile_smoke else ("skipped", None)
    )
    result = PipelineResult(
        options=options,
        codegen_result=codegen_result,
        compile_status=compile_status,
        compile_log=compile_log,
    )
    write_pipeline_report(result)
    return result


def compile_generated(codegen_dir: Path) -> tuple[str, Path | None]:
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


def write_pipeline_report(result: PipelineResult) -> None:
    options = result.options
    codegen = result.codegen_result
    quantization_note = (
        "- Converter 已提供量化 section；若 codegen 仍为 `blocked`，请查看 "
        "`quantization.md`、`unsupported_ops.md` 和 `op_mapping.md` 的具体节点原因。"
        if codegen.model_graph.has_quantization
        else "- Converter 未提供 int8 量化 section 时，CMSIS-NN runtime 层会报告为 `blocked`。"
    )
    runtime_note = (
        "- Codegen 状态为 `ok` 时，生成物已包含当前支持范围内的真实 CMSIS-NN 调用。"
        if codegen.status == "ok"
        else "- 当前状态不是 `ok`，CLI 默认按交付失败处理；C99 smoke compile 通过也只代表"
        "生成工程语法可编译，不代表 blocked 节点已可真实推理。"
    )
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
        "- 本流程由根级 `nanoc_nn` 包编排 converter 和 codegen。",
        "- codegen 仍然只消费 converter 输出目录，不直接读取 ONNX。",
        quantization_note,
        runtime_note,
        "",
    ]
    (options.output_root / "pipeline_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def safe_prefix(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    prefix = "".join(chars).strip("_")
    if not prefix:
        prefix = "nanoc"
    if prefix[0].isdigit():
        prefix = f"m_{prefix}"
    return prefix

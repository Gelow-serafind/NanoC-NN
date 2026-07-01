from __future__ import annotations

from pathlib import Path

from .loader import load_codegen_input
from .mapper import map_model
from .memory import plan_memory
from .model import CodegenError, CodegenOptions, GenerationResult, MemoryPlan, OpMapping
from .quantization import analyze_quantization
from .renderer import render_text
from .report import write_reports


def generate_project(options: CodegenOptions) -> GenerationResult:
    codegen_input = load_codegen_input(options.input_dir)
    mappings = map_model(codegen_input.model_graph)
    quantization_issues = analyze_quantization(codegen_input.model_graph, mappings)
    weights_header_size = len(codegen_input.weights_header.encode("utf-8"))
    memory_plan = plan_memory(
        codegen_input.model_graph,
        mappings,
        options,
        weights_header_size=weights_header_size,
    )
    status = _status(mappings, quantization_issues, memory_plan)
    if options.strict and status != "ok":
        raise CodegenError(
            "strict mode rejected generation because mappings, quantization, or budgets are blocked"
        )

    options.out_dir.mkdir(parents=True, exist_ok=True)
    generated_files = _write_project_files(
        options=options,
        mappings=mappings,
        memory_plan=memory_plan,
        status=status,
        weights_header=codegen_input.weights_header,
    )
    result = GenerationResult(
        status=status,
        input_dir=options.input_dir,
        out_dir=options.out_dir,
        model_graph=codegen_input.model_graph,
        validation_issues=codegen_input.validation_issues,
        mappings=mappings,
        quantization_issues=quantization_issues,
        memory_plan=memory_plan,
        generated_files=generated_files,
    )
    generated_files.extend(write_reports(result, options))
    return result


def _status(
    mappings: list[OpMapping],
    quantization_issues: list,
    memory_plan: MemoryPlan,
) -> str:
    if any(item.status == "unsupported" for item in mappings):
        return "unsupported"
    if any(item.status == "blocked" for item in mappings) or quantization_issues:
        return "blocked"
    if memory_plan.sram_budget_status == "over_budget":
        return "blocked"
    if memory_plan.flash_budget_status == "over_budget":
        return "blocked"
    return "ok"


def _write_project_files(
    *,
    options: CodegenOptions,
    mappings: list[OpMapping],
    memory_plan: MemoryPlan,
    status: str,
    weights_header: str,
) -> list[Path]:
    out_dir = options.out_dir
    generated = {
        out_dir / "include" / "model.h": _model_h(memory_plan),
        out_dir / "include" / "model_weights.h": _model_weights_h(memory_plan, weights_header),
        out_dir / "src" / "model.c": _model_c(mappings, memory_plan, status),
        out_dir / "src" / "main.c": _main_c(),
        out_dir / "CMakeLists.txt": _cmake(options),
    }
    if weights_header:
        generated[out_dir / "include" / "converter_weights.h"] = weights_header

    written: list[Path] = []
    for path, content in generated.items():
        render_text(path, content)
        written.append(path)
    return written


def _model_h(memory_plan: MemoryPlan) -> str:
    input_bytes = max(1, sum(item.size_bytes for item in memory_plan.input_buffers))
    output_bytes = max(1, sum(item.size_bytes for item in memory_plan.output_buffers))
    activation_a = max(1, memory_plan.activation_buffers[0].size_bytes)
    activation_b = max(1, memory_plan.activation_buffers[1].size_bytes)
    scratch = max(1, memory_plan.max_scratch_bytes)
    return "\n".join(
        [
            "#ifndef NANOC_MODEL_H",
            "#define NANOC_MODEL_H",
            "",
            "#include <stdint.h>",
            "#include <stddef.h>",
            "",
            f"#define NANOC_MODEL_INPUT_BYTES {input_bytes}u",
            f"#define NANOC_MODEL_OUTPUT_BYTES {output_bytes}u",
            f"#define NANOC_MODEL_ACTIVATION_A_BYTES {activation_a}u",
            f"#define NANOC_MODEL_ACTIVATION_B_BYTES {activation_b}u",
            f"#define NANOC_MODEL_SCRATCH_BYTES {scratch}u",
            f"#define NANOC_MODEL_ESTIMATED_SRAM_BYTES {memory_plan.total_sram_bytes}u",
            f"#define NANOC_MODEL_ESTIMATED_FLASH_BYTES {memory_plan.total_flash_bytes}u",
            "",
            "typedef enum nanoc_status_t {",
            "    NANOC_STATUS_OK = 0,",
            "    NANOC_STATUS_BLOCKED = 2,",
            "    NANOC_STATUS_UNSUPPORTED = 3",
            "} nanoc_status_t;",
            "",
            "const char *nanoc_model_status(void);",
            "int nanoc_model_run(const int8_t *input, int8_t *output);",
            "",
            "#endif /* NANOC_MODEL_H */",
            "",
        ]
    )


def _model_weights_h(memory_plan: MemoryPlan, weights_header: str) -> str:
    include_converter = '#include "converter_weights.h"' if weights_header else "/* no weights.h */"
    return "\n".join(
        [
            "#ifndef NANOC_MODEL_WEIGHTS_H",
            "#define NANOC_MODEL_WEIGHTS_H",
            "",
            "#include <stdint.h>",
            "",
            f"#define NANOC_MODEL_WEIGHT_FLASH_BYTES {memory_plan.weight_flash_bytes}u",
            include_converter,
            "",
            "#endif /* NANOC_MODEL_WEIGHTS_H */",
            "",
        ]
    )


def _model_c(
    mappings: list[OpMapping],
    memory_plan: MemoryPlan,
    status: str,
) -> str:
    return_code = {
        "ok": "NANOC_STATUS_OK",
        "blocked": "NANOC_STATUS_BLOCKED",
        "unsupported": "NANOC_STATUS_UNSUPPORTED",
    }.get(status, "NANOC_STATUS_BLOCKED")
    lines = [
        "#include \"model.h\"",
        "#include \"model_weights.h\"",
        "",
        "#ifndef NANOC_ENABLE_CMSIS_NN",
        "#define NANOC_ENABLE_CMSIS_NN 0",
        "#endif",
        "",
        "#if NANOC_ENABLE_CMSIS_NN",
        "#include \"arm_nnfunctions.h\"",
        "#endif",
        "",
        "static int8_t nanoc_activation_a[NANOC_MODEL_ACTIVATION_A_BYTES];",
        "static int8_t nanoc_activation_b[NANOC_MODEL_ACTIVATION_B_BYTES];",
        "static int8_t nanoc_scratch[NANOC_MODEL_SCRATCH_BYTES];",
        "",
        "const char *nanoc_model_status(void)",
        "{",
        f"    return \"{status}\";",
        "}",
        "",
        "int nanoc_model_run(const int8_t *input, int8_t *output)",
        "{",
        "    (void)input;",
        "    (void)output;",
        "    (void)nanoc_activation_a;",
        "    (void)nanoc_activation_b;",
        "    (void)nanoc_scratch;",
        "",
        "    /*",
        "     * First-version generated execution trace.",
        "     * CMSIS-NN calls are emitted only after converter provides complete int8",
        "     * quantization fields and layout requirements are satisfied.",
        "     */",
    ]
    for mapping in mappings:
        lines.extend(_mapping_comment(mapping))
    lines.extend(
        [
            "",
            f"    return {return_code};",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _mapping_comment(mapping: OpMapping) -> list[str]:
    return [
        f"    /* node {mapping.index}: {mapping.node_name}",
        f"     * ONNX op: {mapping.onnx_op}",
        f"     * codegen status: {mapping.status}",
        f"     * CMSIS-NN action: {mapping.cmsis_action}",
        f"     * reason: {mapping.reason}",
        "     */",
    ]


def _main_c() -> str:
    return "\n".join(
        [
            "#include \"model.h\"",
            "",
            "static int8_t input_buffer[NANOC_MODEL_INPUT_BYTES];",
            "static int8_t output_buffer[NANOC_MODEL_OUTPUT_BYTES];",
            "",
            "int main(void)",
            "{",
            "    return nanoc_model_run(input_buffer, output_buffer);",
            "}",
            "",
        ]
    )


def _cmake(options: CodegenOptions) -> str:
    cmsis_nn_root = _cmake_path(options.cmsis_nn_root)
    cmsis_path = _cmake_path(options.cmsis_path)
    lines = [
        "cmake_minimum_required(VERSION 3.16)",
        "project(nanoc_cmsis_generated C)",
        "",
        "set(CMAKE_C_STANDARD 99)",
        "set(CMAKE_C_STANDARD_REQUIRED ON)",
        "",
        "add_library(nanoc_model STATIC",
        "    src/model.c",
        ")",
        "",
        "target_include_directories(nanoc_model PUBLIC",
        "    include",
        ")",
    ]
    if cmsis_nn_root:
        lines.extend(
            [
                "",
                f"set(NANOC_CMSIS_NN_ROOT \"{cmsis_nn_root}\" CACHE PATH \"CMSIS-NN root\")",
                "target_include_directories(nanoc_model PUBLIC",
                "    ${NANOC_CMSIS_NN_ROOT}/Include",
                ")",
            ]
        )
    if cmsis_path:
        lines.extend(
            [
                "",
                f"set(NANOC_CMSIS_PATH \"{cmsis_path}\" CACHE PATH \"CMSIS root\")",
                "target_include_directories(nanoc_model PUBLIC",
                "    ${NANOC_CMSIS_PATH}/CMSIS/Core/Include",
                ")",
            ]
        )
    lines.extend(
        [
            "",
            "add_executable(nanoc_cmsis_smoke",
            "    src/main.c",
            ")",
            "target_link_libraries(nanoc_cmsis_smoke PRIVATE nanoc_model)",
            "",
        ]
    )
    return "\n".join(lines)


def _cmake_path(path: Path | None) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/")

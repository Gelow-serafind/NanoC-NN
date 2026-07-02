from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .loader import load_codegen_input
from .mapper import map_model
from .memory import plan_memory
from .model import (
    CodegenError,
    CodegenOptions,
    GenerationResult,
    MemoryPlan,
    ModelGraph,
    OpMapping,
)
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
        model_graph=codegen_input.model_graph,
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
    model_graph: ModelGraph,
    mappings: list[OpMapping],
    memory_plan: MemoryPlan,
    status: str,
    weights_header: str,
) -> list[Path]:
    out_dir = options.out_dir
    generated = {
        out_dir / "include" / "model.h": _model_h(memory_plan),
        out_dir / "include" / "model_weights.h": _model_weights_h(
            model_graph,
            memory_plan,
            weights_header,
        ),
        out_dir / "src" / "model.c": _model_c(model_graph, mappings, memory_plan, status),
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


def _model_weights_h(
    graph: ModelGraph,
    memory_plan: MemoryPlan,
    weights_header: str,
) -> str:
    include_converter = '#include "converter_weights.h"' if weights_header else "/* no weights.h */"
    lines = [
        "#ifndef NANOC_MODEL_WEIGHTS_H",
        "#define NANOC_MODEL_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define NANOC_MODEL_WEIGHT_FLASH_BYTES {memory_plan.weight_flash_bytes}u",
        include_converter,
        "",
    ]
    lines.extend(_quantized_weight_declarations(graph))
    lines.extend(["#endif /* NANOC_MODEL_WEIGHTS_H */", ""])
    return "\n".join(lines)


def _model_c(
    graph: ModelGraph,
    mappings: list[OpMapping],
    memory_plan: MemoryPlan,
    status: str,
) -> str:
    fc_layers = _fc_layers(graph, mappings) if status == "ok" else []
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
    ]
    if fc_layers:
        lines.extend(_cmsis_fc_run_body(fc_layers))
        lines.extend(
            [
                "#else",
                "    (void)input;",
                "    (void)output;",
                "    (void)nanoc_activation_a;",
                "    (void)nanoc_activation_b;",
                "    (void)nanoc_scratch;",
                "    return NANOC_STATUS_BLOCKED;",
                "#endif",
            ]
        )
    else:
        lines.extend(
            [
                "    (void)input;",
                "    (void)output;",
                "    (void)nanoc_activation_a;",
                "    (void)nanoc_activation_b;",
                "    (void)nanoc_scratch;",
                "",
                "    /*",
                "     * Generated execution trace.",
                "     * Real CMSIS-NN calls are emitted only for nodes whose converter",
                "     * quantization fields and renderer support are complete.",
                "     */",
            ]
        )
        for mapping in mappings:
            lines.extend(_mapping_comment(mapping))
        lines.extend(
            [
                "",
                f"    return {return_code};",
            ]
        )
    lines.extend(["}", ""])
    return "\n".join(lines)


def _cmsis_fc_run_body(layers: list[dict[str, Any]]) -> list[str]:
    lines = [
        "#if NANOC_ENABLE_CMSIS_NN",
        "    const int8_t *current_input = input;",
        "    int8_t *current_output = output;",
        "    arm_cmsis_nn_status cmsis_status;",
        "",
    ]
    for layer_index, layer in enumerate(layers):
        input_size = layer["input_size"]
        output_size = layer["output_size"]
        prefix = layer["symbol"]
        is_last = layer_index == len(layers) - 1
        if not is_last:
            target = "nanoc_activation_a" if layer_index % 2 == 0 else "nanoc_activation_b"
            current_output = target
        else:
            current_output = "output"
        lines.extend(
            [
                f"    /* node {layer['index']}: {layer['name']} -> arm_fully_connected_s8 */",
                "    {",
                "        cmsis_nn_context ctx;",
                "        cmsis_nn_fc_params fc_params;",
                "        cmsis_nn_per_tensor_quant_params quant_params;",
                f"        cmsis_nn_dims input_dims = {{1, 1, 1, {input_size}}};",
                f"        cmsis_nn_dims filter_dims = {{{input_size}, 1, 1, {output_size}}};",
                f"        cmsis_nn_dims bias_dims = {{1, 1, 1, {output_size}}};",
                f"        cmsis_nn_dims output_dims = {{1, 1, 1, {output_size}}};",
                "",
                "        ctx.buf = nanoc_scratch;",
                "        ctx.size = arm_fully_connected_s8_get_buffer_size(&filter_dims);",
                "        if (ctx.size > NANOC_MODEL_SCRATCH_BYTES) {",
                "            return NANOC_STATUS_BLOCKED;",
                "        }",
                f"        fc_params.input_offset = {layer['input_offset']};",
                f"        fc_params.filter_offset = {layer['filter_offset']};",
                f"        fc_params.output_offset = {layer['output_offset']};",
                f"        fc_params.activation.min = {layer['activation_min']};",
                f"        fc_params.activation.max = {layer['activation_max']};",
                f"        quant_params.multiplier = {layer['multiplier']};",
                f"        quant_params.shift = {layer['shift']};",
                "",
                "        cmsis_status = arm_fully_connected_s8(",
                "            &ctx,",
                "            &fc_params,",
                "            &quant_params,",
                "            &input_dims,",
                "            current_input,",
                "            &filter_dims,",
                f"            {prefix}_weights,",
                "            &bias_dims,",
                f"            {prefix}_bias,",
                "            &output_dims,",
                f"            {current_output});",
                "        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {",
                "            return NANOC_STATUS_BLOCKED;",
                "        }",
                "    }",
                "",
            ]
        )
        if not is_last:
            lines.append(f"    current_input = {current_output};")
            lines.append("")
    lines.append("    return NANOC_STATUS_OK;")
    return lines


def _quantized_weight_declarations(graph: ModelGraph) -> list[str]:
    layers = _fc_layers(
        graph,
        [
            OpMapping(
                index=node.index,
                node_name=node.name,
                onnx_op=node.op_type,
                converter_status=node.converter_status,
                status="wrapper_api",
                cmsis_action="arm_fully_connected_s8",
                reason="quantized weight declaration",
                inputs=node.inputs,
                outputs=node.outputs,
                input_shapes=node.input_shapes,
                output_shapes=node.output_shapes,
            )
            for node in graph.nodes
            if node.op_type in {"Gemm", "MatMul"}
        ],
    )
    if not layers:
        return []
    lines = ["/* Quantized CMSIS-NN runtime weights. */"]
    for layer in layers:
        symbol = layer["symbol"]
        weight_values = _int_values(layer["weight_values"])
        bias_values = _int_values(layer["bias_values"] or [0] * layer["output_size"])
        lines.extend(
            [
                f"/* ONNX node: {layer['name']} */",
                f"#define {symbol.upper()}_WEIGHT_SIZE {len(weight_values)}u",
                f"#define {symbol.upper()}_BIAS_SIZE {len(bias_values)}u",
                f"static const int8_t {symbol}_weights[{len(weight_values)}] = {{",
                *_format_int_array(weight_values),
                "};",
                f"static const int32_t {symbol}_bias[{len(bias_values)}] = {{",
                *_format_int_array(bias_values),
                "};",
                "",
            ]
        )
    return lines


def _fc_layers(graph: ModelGraph, mappings: list[OpMapping]) -> list[dict[str, Any]]:
    quantization = graph.quantization or {}
    node_quant = quantization.get("nodes", {})
    weights = quantization.get("weights", {})
    if not isinstance(node_quant, dict) or not isinstance(weights, dict):
        return []
    node_by_name = {node.name: node for node in graph.nodes}
    layers: list[dict[str, Any]] = []
    for mapping in mappings:
        if mapping.status not in {"wrapper_api", "direct_api"}:
            continue
        if mapping.onnx_op not in {"Gemm", "MatMul"}:
            continue
        node = node_by_name.get(mapping.node_name)
        quant = node_quant.get(mapping.node_name)
        if node is None or not isinstance(quant, dict):
            continue
        cmsis_nn = quant.get("cmsis_nn", {})
        quant_weights = quant.get("weights", {})
        if not isinstance(cmsis_nn, dict) or not isinstance(quant_weights, dict):
            continue
        if any(field not in cmsis_nn for field in _CMSIS_FC_REQUIRED_FIELDS):
            continue
        weight_name = str(quant_weights.get("weight", ""))
        weight_info = weights.get(weight_name, {})
        if not isinstance(weight_info, dict):
            continue
        weight_shape = weight_info.get("shape", [])
        if not isinstance(weight_shape, list) or len(weight_shape) != 2:
            continue
        output_size, input_size = int(weight_shape[0]), int(weight_shape[1])
        values = weight_info.get("values", [])
        if not isinstance(values, list) or len(values) != output_size * input_size:
            continue
        layers.append(
            {
                "index": mapping.index,
                "name": mapping.node_name,
                "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
                "input_size": input_size,
                "output_size": output_size,
                "weight_values": values,
                "bias_values": quant_weights.get("bias_values", []),
                "input_offset": int(cmsis_nn["input_offset"]),
                "filter_offset": int(cmsis_nn["filter_offset"]),
                "output_offset": int(cmsis_nn["output_offset"]),
                "multiplier": int(cmsis_nn["multiplier"]),
                "shift": int(cmsis_nn["shift"]),
                "activation_min": int(cmsis_nn["activation_min"]),
                "activation_max": int(cmsis_nn["activation_max"]),
            }
        )
    return layers


_CMSIS_FC_REQUIRED_FIELDS = (
    "input_offset",
    "filter_offset",
    "output_offset",
    "multiplier",
    "shift",
    "activation_min",
    "activation_max",
)


def _c_symbol(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]", "_", value)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    if not text:
        text = "nanoc_node"
    if text[0].isdigit():
        text = f"_{text}"
    return text


def _int_values(values: object) -> list[int]:
    if not isinstance(values, list):
        return []
    return [int(item) for item in values]


def _format_int_array(values: list[int], line_width: int = 12) -> list[str]:
    if not values:
        return ["    0,"]
    lines: list[str] = []
    for start in range(0, len(values), line_width):
        chunk = values[start : start + line_width]
        lines.append("    " + ", ".join(str(value) for value in chunk) + ",")
    return lines


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

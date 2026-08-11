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
    InitializerSpec,
    MemoryPlan,
    ModelGraph,
    OpMapping,
    element_count_from_shape,
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
    status = _status(codegen_input.model_graph, mappings, quantization_issues, memory_plan)
    if options.strict and status != "ok":
        raise CodegenError(
            "strict mode rejected generation because mappings, quantization, rendering, or target budgets are not deliverable"
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
    graph: ModelGraph,
    mappings: list[OpMapping],
    quantization_issues: list,
    memory_plan: MemoryPlan,
) -> str:
    if _reference_runtime_enabled(graph) and _reference_runtime_supported(graph):
        if memory_plan.sram_budget_status == "over_budget":
            return "oversize"
        if memory_plan.flash_budget_status == "over_budget":
            return "oversize"
        return "ok"
    if any(item.status == "unsupported" for item in mappings):
        return "unsupported"
    if any(item.status == "blocked" for item in mappings) or quantization_issues:
        return "blocked"
    if not _renderer_is_complete(graph, mappings):
        return "blocked"
    if memory_plan.sram_budget_status == "over_budget":
        return "oversize"
    if memory_plan.flash_budget_status == "over_budget":
        return "oversize"
    return "ok"


def _renderer_is_complete(graph: ModelGraph, mappings: list[OpMapping]) -> bool:
    generated_runtime_mappings = [
        mapping
        for mapping in mappings
        if mapping.status in {"wrapper_api", "direct_api"}
        and mapping.onnx_op
        in {
            "Abs",
            "Add",
            "AveragePool",
            "Ceil",
            "Clip",
            "Concat",
            "Conv",
            "Div",
            "Exp",
            "Floor",
            "Gather",
            "Gemm",
            "GlobalAveragePool",
            "LeakyRelu",
            "Log",
            "MatMul",
            "Max",
            "MaxPool",
            "Min",
            "Mul",
            "Neg",
            "Pad",
            "Pow",
            "QLinearAdd",
            "QLinearConv",
            "QLinearGlobalAveragePool",
            "QLinearMatMul",
            "Reciprocal",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceMax",
            "ReduceMean",
            "ReduceMin",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
            "Round",
            "Sign",
            "Sigmoid",
            "Slice",
            "Softmax",
            "Sqrt",
            "Sub",
            "Tanh",
            "Transpose",
            "Unsqueeze",
            "Where",
            "Equal",
            "Greater",
            "Less",
            "And",
            "Or",
            "Not",
            "Xor",
            "GreaterOrEqual",
            "LessOrEqual",
            "Erf",
            "Softplus",
            "Softsign",
            "HardSwish",
            "Elu",
            "Selu",
            "HardSigmoid",
            "ThresholdedRelu",
            "Celu",
            "PRelu",
        }
    ]
    generated_layers = _runtime_layers(graph, generated_runtime_mappings)
    return len(generated_layers) == len(generated_runtime_mappings)


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
        out_dir / "include" / "model.h": _model_h(model_graph, memory_plan),
        out_dir / "include" / "model_weights.h": _model_weights_h(
            model_graph,
            memory_plan,
            weights_header,
        ),
        out_dir / "src" / "model.c": _model_c(model_graph, mappings, memory_plan, status),
        out_dir / "src" / "main.c": _main_c(),
        out_dir / "CMakeLists.txt": _cmake(options),
        out_dir / "firmware_integration.md": _firmware_integration(options, memory_plan, status),
    }
    if weights_header:
        generated[out_dir / "include" / "converter_weights.h"] = weights_header

    written: list[Path] = []
    for path, content in generated.items():
        render_text(path, content)
        written.append(path)
    return written


def _model_h(graph: ModelGraph, memory_plan: MemoryPlan) -> str:
    input_bytes = max(1, sum(item.size_bytes for item in memory_plan.input_buffers))
    output_bytes = max(1, sum(item.size_bytes for item in memory_plan.output_buffers))
    input_floats = max(1, sum((tensor.element_count or 0) for tensor in graph.inputs))
    output_floats = max(1, sum((tensor.element_count or 0) for tensor in graph.outputs))
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
            f"#define NANOC_MODEL_INPUT_FLOATS {input_floats}u",
            f"#define NANOC_MODEL_OUTPUT_FLOATS {output_floats}u",
            f"#define NANOC_MODEL_ACTIVATION_A_BYTES {activation_a}u",
            f"#define NANOC_MODEL_ACTIVATION_B_BYTES {activation_b}u",
            f"#define NANOC_MODEL_SCRATCH_BYTES {scratch}u",
            f"#define NANOC_MODEL_ESTIMATED_SRAM_BYTES {memory_plan.total_sram_bytes}u",
            f"#define NANOC_MODEL_ESTIMATED_FLASH_BYTES {memory_plan.total_flash_bytes}u",
            "",
            "typedef enum nanoc_status_t {",
            "    NANOC_STATUS_OK = 0,",
            "    NANOC_STATUS_BLOCKED = 2,",
            "    NANOC_STATUS_UNSUPPORTED = 3,",
            "    NANOC_STATUS_OVERSIZE = 4",
            "} nanoc_status_t;",
            "",
            "const char *nanoc_model_status(void);",
            "int nanoc_model_run(const int8_t *input, int8_t *output);",
            "int nanoc_model_run_float(const float * const *inputs, float *output);",
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
    if status == "ok" and _reference_runtime_enabled(graph) and _reference_runtime_supported(graph):
        return _reference_model_c(graph, memory_plan, status)
    runtime_layers = _runtime_layers(graph, mappings) if status == "ok" else []
    tensor_buffers = _tensor_buffer_declarations(graph, runtime_layers)
    flatten_transforms = _flatten_layout_transforms(graph)
    return_code = {
        "ok": "NANOC_STATUS_OK",
        "blocked": "NANOC_STATUS_BLOCKED",
        "unsupported": "NANOC_STATUS_UNSUPPORTED",
        "oversize": "NANOC_STATUS_OVERSIZE",
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
        "#include <math.h>",
        "#include \"arm_nnfunctions.h\"",
        "#include \"arm_nnsupportfunctions.h\"",
        "#endif",
        "",
        "#include <string.h>",
        "",
        "static int8_t nanoc_activation_a[NANOC_MODEL_ACTIVATION_A_BYTES];",
        "static int8_t nanoc_activation_b[NANOC_MODEL_ACTIVATION_B_BYTES];",
        "static int8_t nanoc_scratch[NANOC_MODEL_SCRATCH_BYTES];",
    ]
    # Declare NHWC input buffer if the model input is multi-channel 4-D
    input_shape = _graph_input_shape(graph)
    output_shape = _graph_output_shape(graph)
    need_input_xpose = (
        input_shape is not None
        and len(input_shape) == 4
        and isinstance(input_shape[1], int)
        and input_shape[1] > 1
    )
    if need_input_xpose:
        input_count = element_count_from_shape(input_shape) or max(1, memory_plan.input_buffers[0].size_bytes)
        lines.append(f"static int8_t nanoc_input_nhwc[{input_count}u];")
    need_output_xpose = _needs_nchw_to_nhwc_boundary_transpose(output_shape)
    if need_output_xpose:
        output_count = element_count_from_shape(output_shape) or max(1, memory_plan.output_buffers[0].size_bytes)
        lines.append(f"static int8_t nanoc_output_nhwc[{output_count}u];")
    lines.extend(tensor_buffers)
    if _needs_stable_softmax(runtime_layers):
        lines.extend(_stable_softmax_helper())
    if _needs_static_indexing_helpers(runtime_layers):
        lines.extend(_static_indexing_helpers())
    lines.extend(
        [
        "",
        "const char *nanoc_model_status(void)",
        "{",
        f"    return \"{status}\";",
        "}",
        "",
        "int nanoc_model_run(const int8_t *input, int8_t *output)",
        "{",
        ]
    )
    if runtime_layers:
        lines.extend(_cmsis_tensor_runtime_run_body_with_transpose(graph, runtime_layers))
        void_lines = [
            "#else",
            "    (void)input;",
            "    (void)output;",
            "    (void)nanoc_activation_a;",
            "    (void)nanoc_activation_b;",
            "    (void)nanoc_scratch;",
        ]
        if need_input_xpose:
            void_lines.append("    (void)nanoc_input_nhwc;")
        if need_output_xpose:
            void_lines.append("    (void)nanoc_output_nhwc;")
        tensor_symbols = _tensor_symbol_map(graph, runtime_layers)
        for sym in tensor_symbols.values():
            void_lines.append(f"    (void){sym};")
        for sym in _concat_requant_symbols(runtime_layers):
            void_lines.append(f"    (void){sym};")
        for transform in flatten_transforms.values():
            void_lines.append(f"    (void){transform['symbol']};")
        void_lines.extend(
            [
                "    return NANOC_STATUS_BLOCKED;",
                "#endif",
            ]
        )
        lines.extend(void_lines)
    elif status == "ok":
        copy_lines = _folded_runtime_copy_body(graph)
        if copy_lines:
            lines.extend(copy_lines)
        else:
            lines.extend(
                [
                    "    (void)input;",
                    "    (void)output;",
                    "    (void)nanoc_activation_a;",
                    "    (void)nanoc_activation_b;",
                    "    (void)nanoc_scratch;",
                    "",
                    "    return NANOC_STATUS_BLOCKED;",
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


def _folded_runtime_copy_body(graph: ModelGraph) -> list[str]:
    if not graph.inputs or not graph.outputs:
        return []
    aliases = _tensor_aliases(graph)
    model_inputs = {tensor.name for tensor in graph.inputs}
    output_name = graph.outputs[0].name
    source_name = _resolve_tensor_alias(output_name, aliases)
    if source_name not in model_inputs:
        return []
    input_size = _graph_input_size(graph)
    output_size = _graph_output_size(graph)
    if input_size != output_size:
        return []
    lines = [
        "    (void)nanoc_activation_a;",
        "    (void)nanoc_activation_b;",
        "    (void)nanoc_scratch;",
    ]
    input_shape = _graph_input_shape(graph)
    if _needs_nchw_to_nhwc_boundary_transpose(input_shape):
        lines.append("    (void)nanoc_input_nhwc;")
    output_shape = _graph_output_shape(graph)
    if _needs_nchw_to_nhwc_boundary_transpose(output_shape):
        lines.append("    (void)nanoc_output_nhwc;")
    lines.extend(
        [
            "",
            f"    memcpy(output, input, {output_size}u);",
            "    return NANOC_STATUS_OK;",
        ]
    )
    return lines


def _cmsis_runtime_run_body(layers: list[dict[str, Any]]) -> list[str]:
    lines = [
        "#if NANOC_ENABLE_CMSIS_NN",
        "    const int8_t *current_input = input;",
        "    int8_t *current_output = output;",
        "    arm_cmsis_nn_status cmsis_status;",
        "",
    ]
    for layer_index, layer in enumerate(layers):
        is_last = layer_index == len(layers) - 1
        if not is_last:
            target = "nanoc_activation_a" if layer_index % 2 == 0 else "nanoc_activation_b"
            current_output = target
        else:
            current_output = "output"
        if layer["kind"] == "conv":
            lines.extend(_cmsis_conv_call(layer, current_output))
        elif layer["kind"] == "depthwise":
            lines.extend(_cmsis_depthwise_call(layer, current_output))
        elif layer["kind"] == "pool":
            lines.extend(_cmsis_pool_call(layer, current_output))
        elif layer["kind"] == "add":
            lines.extend(_cmsis_add_call(layer, current_output))
        elif layer["kind"] == "mul":
            lines.extend(_cmsis_mul_call(layer, current_output))
        elif layer["kind"] in {
            "equal",
            "greater",
            "less",
            "greaterorequal",
            "lessorequal",
        }:
            lines.extend(_generated_compare_call(layer, current_output))
        elif layer["kind"] in {"and", "or", "not", "xor"}:
            lines.extend(_generated_bool_logic_call(layer, current_output))
        elif layer["kind"] == "where":
            lines.extend(_generated_where_call(layer, current_output))
        elif layer["kind"] in {"sub", "div", "min", "max", "pow", "prelu"}:
            lines.extend(_generated_binary_call(layer, current_output))
        elif layer["kind"] == "abs":
            lines.extend(_generated_abs_call(layer, current_output))
        elif layer["kind"] in {
            "sigmoid",
            "tanh",
            "leakyrelu",
            "clip",
            "neg",
            "sqrt",
            "reciprocal",
            "exp",
            "log",
            "floor",
            "ceil",
            "round",
            "sign",
            "erf",
            "softplus",
            "softsign",
            "hardswish",
            "elu",
            "selu",
            "hardsigmoid",
            "thresholdedrelu",
            "celu",
        }:
            lines.extend(_generated_unary_call(layer, current_output))
        elif layer["kind"] in {
            "reducemean",
            "reducesum",
            "reducemax",
            "reducemin",
            "reduceprod",
            "reducel1",
            "reducel2",
            "reducelogsum",
            "reducelogsumexp",
            "reducesumsquare",
        }:
            lines.extend(_generated_reduce_call(layer, current_output))
        elif layer["kind"] == "unsqueeze":
            lines.extend(_generated_unsqueeze_call(layer, current_output))
        elif layer["kind"] == "pad":
            lines.extend(_generated_pad_call(layer, current_output))
        elif layer["kind"] == "slice":
            lines.extend(_generated_slice_call(layer, current_output))
        elif layer["kind"] == "gather":
            lines.extend(_generated_gather_call(layer, current_output))
        elif layer["kind"] == "transpose":
            lines.extend(_cmsis_transpose_call(layer, current_output))
        elif layer["kind"] == "concat":
            lines.extend(_cmsis_concat_call(layer, current_output))
        elif layer["kind"] == "softmax":
            lines.extend(_cmsis_softmax_call(layer, current_output))
        else:
            lines.extend(_cmsis_fc_call(layer, current_output))
        if not is_last:
            lines.append(f"    current_input = {current_output};")
            lines.append("")
    lines.append("    return NANOC_STATUS_OK;")
    return lines


def _cmsis_tensor_runtime_run_body_with_transpose(
    graph: ModelGraph,
    layers: list[dict[str, Any]],
) -> list[str]:
    """Wrap _cmsis_tensor_runtime_run_body with NCHW↔NHWC boundary transposes.

    The external model_run() interface uses NCHW (ONNX convention).
    CMSIS-NN internally uses NHWC.  This function adds:
      - Input:  NCHW → NHWC transpose before first layer
      - Output: NHWC → NCHW transpose after last layer
    Transposes are only emitted for 4-D tensors with C > 1.
    """
    # Determine if input/output need 4-D transposes
    input_shape = None
    output_shape = None
    if graph.inputs:
        input_shape = _tensor_shape(graph, graph.inputs[0].name)
    if graph.outputs:
        output_shape = _tensor_shape(graph, graph.outputs[0].name)

    uses_nhwc_runtime = any(
        layer.get("kind") in {"conv", "depthwise", "pool", "concat"} for layer in layers
    )
    need_input_xpose = uses_nhwc_runtime and _needs_nchw_to_nhwc_boundary_transpose(input_shape)
    need_output_xpose = uses_nhwc_runtime and _needs_nchw_to_nhwc_boundary_transpose(output_shape)

    input_expr_overrides: dict[str, str] = {}
    output_expr_overrides: dict[str, str] = {}
    if need_input_xpose and graph.inputs:
        input_expr_overrides[graph.inputs[0].name] = "nanoc_input_nhwc"
    if need_output_xpose and graph.outputs:
        output_expr_overrides[graph.outputs[0].name] = "nanoc_output_nhwc"

    body = _cmsis_tensor_runtime_run_body(
        graph,
        layers,
        input_expr_overrides=input_expr_overrides,
        output_expr_overrides=output_expr_overrides,
    )
    if not body:
        return body

    if not need_input_xpose and not need_output_xpose:
        return body

    # body[0] = "#if NANOC_ENABLE_CMSIS_NN"
    # We insert transposes right after the #if line.
    result = body[:2]  # keep the #if and cmsis_status lines

    input_size = element_count_from_shape(input_shape) if input_shape else 0

    if need_input_xpose:
        n, c, h, w = [int(d) for d in input_shape]
        result.append(f"    /* NCHW→NHWC input transpose ({n}x{c}x{h}x{w}) */")
        result.append(f"    for (int _ni = 0; _ni < {n}; ++_ni)")
        result.append(f"        for (int _hi = 0; _hi < {h}; ++_hi)")
        result.append(f"            for (int _wi = 0; _wi < {w}; ++_wi)")
        result.append(f"                for (int _ci = 0; _ci < {c}; ++_ci)")
        result.append(
            f"                    nanoc_input_nhwc[_ni * {h * w * c} + _hi * {w * c} + _wi * {c} + _ci] = "
            f"input[_ni * {c * h * w} + _ci * {h * w} + _hi * {w} + _wi];"
        )
        result.append("")

    # Insert the body (skip the #if and cmsis_status lines). When an output
    # transpose is needed, move the success return after the transpose.
    body_tail = body[2:]
    success_return = None
    if need_output_xpose and body_tail and body_tail[-1] == "    return NANOC_STATUS_OK;":
        success_return = body_tail.pop()
    result.extend(body_tail)

    if need_output_xpose:
        # Find the output expression from the last layer
        last_layer = layers[-1]
        last_outputs = last_layer.get("output_tensors", [])
        n, c, h, w = [int(d) for d in output_shape]
        last_expr = "output"
        if last_outputs:
            aliases = _tensor_aliases(graph)
            tensor_symbols = _tensor_symbol_map(graph, layers)
            model_outputs = {t.name for t in graph.outputs}
            resolved = _resolve_tensor_alias(str(last_outputs[0]), aliases)
            if resolved not in model_outputs:
                last_expr = tensor_symbols.get(resolved, last_expr)
            else:
                last_expr = output_expr_overrides.get(resolved, "output")

        result.append(f"    /* NHWC→NCHW output transpose ({n}x{c}x{h}x{w}) */")
        result.append(f"    for (int _ni = 0; _ni < {n}; ++_ni)")
        result.append(f"        for (int _ci = 0; _ci < {c}; ++_ci)")
        result.append(f"            for (int _hi = 0; _hi < {h}; ++_hi)")
        result.append(f"                for (int _wi = 0; _wi < {w}; ++_wi)")
        result.append(
            f"                    output[_ni * {c * h * w} + _ci * {h * w} + _hi * {w} + _wi] = "
            f"{last_expr}[_ni * {h * w * c} + _hi * {w * c} + _wi * {c} + _ci];"
        )
    if success_return is not None:
        result.append(success_return)

    return result


def _needs_nchw_to_nhwc_boundary_transpose(shape: list[int] | None) -> bool:
    return (
        shape is not None
        and len(shape) == 4
        and isinstance(shape[1], int)
        and shape[1] > 1
    )


def _tensor_shape(graph: ModelGraph, name: str) -> list[int] | None:
    """Look up a tensor's NCHW shape from the graph."""
    for node in graph.nodes:
        for shape in (node.output_shapes.get(name), node.input_shapes.get(name)):
            if shape and all(isinstance(d, int) for d in shape):
                return [int(d) for d in shape]
    for tensor in [*graph.inputs, *graph.outputs]:
        if tensor.name == name and tensor.shape:
            return [int(d) if isinstance(d, int) else 0 for d in tensor.shape]
    return None


def _cmsis_tensor_runtime_run_body(
    graph: ModelGraph,
    layers: list[dict[str, Any]],
    *,
    input_expr_overrides: dict[str, str] | None = None,
    output_expr_overrides: dict[str, str] | None = None,
) -> list[str]:
    aliases = _tensor_aliases(graph)
    flatten_transforms = _flatten_layout_transforms(graph)
    tensor_symbols = _tensor_symbol_map(graph, layers)
    model_inputs = {tensor.name for tensor in graph.inputs}
    model_outputs = {tensor.name for tensor in graph.outputs}
    model_input_exprs = _model_input_expr_map(graph)
    input_expr_overrides = input_expr_overrides or {}
    output_expr_overrides = output_expr_overrides or {}
    last_output_expr = "output"

    def tensor_expr(name: str) -> str | None:
        resolved = _resolve_tensor_alias(name, aliases)
        if resolved in model_inputs:
            return input_expr_overrides.get(resolved, model_input_exprs.get(resolved, "input"))
        if resolved in model_outputs:
            return output_expr_overrides.get(resolved, "output")
        return tensor_symbols.get(resolved)

    lines = [
        "#if NANOC_ENABLE_CMSIS_NN",
        "    arm_cmsis_nn_status cmsis_status;",
        "    (void)nanoc_activation_a;",
        "    (void)nanoc_activation_b;",
        "",
    ]
    for layer in layers:
        outputs = layer.get("output_tensors", [])
        if not outputs:
            return []
        output_name = str(outputs[0])
        output_expr = tensor_expr(output_name) or tensor_symbols.get(output_name)
        if output_expr is None:
            return []
        input_tensors = [str(name) for name in layer.get("input_tensors", [])]
        constant_idx = layer.get("constant_input_index")
        constant_symbol = layer.get("constant_symbol")
        input_exprs = []
        for i, name in enumerate(input_tensors):
            if layer.get("kind") == "where" and i == 0 and layer.get("condition_symbol"):
                input_exprs.append(str(layer["condition_symbol"]))
            elif constant_idx is not None and i == constant_idx and constant_symbol:
                input_exprs.append(constant_symbol)
            elif name in flatten_transforms:
                transform = flatten_transforms[name]
                source_expr = tensor_expr(str(transform["source"]))
                if source_expr is None:
                    return []
                lines.extend(_flatten_transform_call(transform, source_expr))
                input_exprs.append(str(transform["symbol"]))
            else:
                expr = tensor_expr(name)
                if expr is None:
                    return []
                input_exprs.append(expr)
        layer = dict(layer)
        if input_exprs:
            layer["input_expr"] = input_exprs[0]
        if layer["kind"] == "add" and len(input_exprs) >= 2:
            layer["input_1_expr"] = input_exprs[0]
            if not constant_symbol:
                layer["input_2_expr"] = input_exprs[1]
        if layer["kind"] == "mul" and len(input_exprs) >= 2:
            layer["input_1_expr"] = input_exprs[0]
            if not constant_symbol:
                layer["input_2_expr"] = input_exprs[1]
        if layer["kind"] in {
            "sub",
            "div",
            "min",
            "max",
            "pow",
            "prelu",
            "equal",
            "greater",
            "less",
            "greaterorequal",
            "lessorequal",
            "and",
            "or",
            "xor",
        } and len(input_exprs) >= 2:
            layer["input_1_expr"] = input_exprs[0]
            if not constant_symbol:
                layer["input_2_expr"] = input_exprs[1]
        if layer["kind"] == "not" and len(input_exprs) >= 1:
            layer["input_1_expr"] = input_exprs[0]
        if layer["kind"] == "where" and len(input_exprs) >= 3:
            layer["condition_expr"] = input_exprs[0]
            layer["x_expr"] = input_exprs[1]
            if not constant_symbol:
                layer["y_expr"] = input_exprs[2]
        if layer["kind"] == "concat":
            layer["input_exprs"] = input_exprs
        if layer["kind"] == "conv":
            lines.extend(_cmsis_conv_call(layer, output_expr))
        elif layer["kind"] == "depthwise":
            lines.extend(_cmsis_depthwise_call(layer, output_expr))
        elif layer["kind"] == "pool":
            lines.extend(_cmsis_pool_call(layer, output_expr))
        elif layer["kind"] == "add":
            lines.extend(_cmsis_add_call(layer, output_expr))
        elif layer["kind"] == "mul":
            lines.extend(_cmsis_mul_call(layer, output_expr))
        elif layer["kind"] in {"sub", "div", "min", "max", "pow", "prelu"}:
            lines.extend(_generated_binary_call(layer, output_expr))
        elif layer["kind"] in {
            "equal",
            "greater",
            "less",
            "greaterorequal",
            "lessorequal",
        }:
            lines.extend(_generated_compare_call(layer, output_expr))
        elif layer["kind"] in {"and", "or", "not", "xor"}:
            lines.extend(_generated_bool_logic_call(layer, output_expr))
        elif layer["kind"] == "where":
            lines.extend(_generated_where_call(layer, output_expr))
        elif layer["kind"] == "abs":
            lines.extend(_generated_abs_call(layer, output_expr))
        elif layer["kind"] in {
            "sigmoid",
            "tanh",
            "leakyrelu",
            "clip",
            "neg",
            "sqrt",
            "reciprocal",
            "exp",
            "log",
            "floor",
            "ceil",
            "round",
            "sign",
            "erf",
            "softplus",
            "softsign",
            "hardswish",
            "elu",
            "selu",
            "hardsigmoid",
            "thresholdedrelu",
            "celu",
        }:
            lines.extend(_generated_unary_call(layer, output_expr))
        elif layer["kind"] in {
            "reducemean",
            "reducesum",
            "reducemax",
            "reducemin",
            "reduceprod",
            "reducel1",
            "reducel2",
            "reducelogsum",
            "reducelogsumexp",
            "reducesumsquare",
        }:
            lines.extend(_generated_reduce_call(layer, output_expr))
        elif layer["kind"] == "unsqueeze":
            lines.extend(_generated_unsqueeze_call(layer, output_expr))
        elif layer["kind"] == "pad":
            lines.extend(_generated_pad_call(layer, output_expr))
        elif layer["kind"] == "slice":
            lines.extend(_generated_slice_call(layer, output_expr))
        elif layer["kind"] == "gather":
            lines.extend(_generated_gather_call(layer, output_expr))
        elif layer["kind"] == "transpose":
            lines.extend(_cmsis_transpose_call(layer, output_expr))
        elif layer["kind"] == "concat":
            lines.extend(_cmsis_concat_call(layer, output_expr))
        elif layer["kind"] == "softmax":
            lines.extend(_cmsis_softmax_call(layer, output_expr))
        else:
            lines.extend(_cmsis_fc_call(layer, output_expr))
        last_output_expr = output_expr

    graph_output_name = graph.outputs[0].name if graph.outputs else ""
    output_expr = tensor_expr(graph_output_name) if graph.outputs else None
    output_expr = output_expr or last_output_expr
    output_is_transpose_override = (
        bool(graph_output_name)
        and graph_output_name in output_expr_overrides
        and output_expr == output_expr_overrides[graph_output_name]
    )
    if output_expr != "output" and not output_is_transpose_override:
        output_size = _graph_output_size(graph)
        lines.extend(
            [
                f"    for (size_t nanoc_i = 0; nanoc_i < {output_size}u; ++nanoc_i) {{",
                f"        output[nanoc_i] = {output_expr}[nanoc_i];",
                "    }",
                "",
            ]
        )
    lines.append("    return NANOC_STATUS_OK;")
    return lines


def _tensor_buffer_declarations(
    graph: ModelGraph,
    layers: list[dict[str, Any]],
) -> list[str]:
    if not layers:
        return []
    model_outputs = {tensor.name for tensor in graph.outputs}
    aliases = _tensor_aliases(graph)
    declared: dict[str, int] = {}
    lines: list[str] = []
    for layer in layers:
        for name in layer.get("output_tensors", []):
            resolved = _resolve_tensor_alias(str(name), aliases)
            if resolved in model_outputs or resolved in declared:
                continue
            size = _tensor_element_count(graph, resolved)
            if size <= 0:
                size = _layer_output_element_count(layer)
            if size <= 0:
                continue
            declared[resolved] = size
            lines.append(f"static int8_t {_tensor_symbol(resolved)}[{size}u];")
        if layer.get("kind") == "concat":
            for input_index, requant in enumerate(layer.get("input_requantize", [])):
                if not isinstance(requant, dict) or not requant.get("required"):
                    continue
                symbol = str(
                    requant.get(
                        "symbol",
                        f"{layer.get('symbol', 'nanoc_concat')}_requant_{input_index}",
                    )
                )
                if symbol in declared:
                    continue
                dims = layer.get("input_dims", [])
                if input_index >= len(dims):
                    continue
                size = 1
                for dim in dims[input_index]:
                    size *= int(dim)
                if size <= 0:
                    continue
                declared[symbol] = size
                lines.append(f"static int8_t {symbol}[{size}u];")
    for tensor_name, transform in _flatten_layout_transforms(graph).items():
        symbol = str(transform["symbol"])
        size = int(transform["size"])
        if tensor_name in declared or size <= 0:
            continue
        declared[tensor_name] = size
        lines.append(f"static int8_t {symbol}[{size}u];")
    return lines


def _concat_requant_symbols(layers: list[dict[str, Any]]) -> list[str]:
    symbols: list[str] = []
    for layer in layers:
        if layer.get("kind") != "concat":
            continue
        for requant in layer.get("input_requantize", []):
            if (
                isinstance(requant, dict)
                and requant.get("required")
                and requant.get("symbol")
            ):
                symbols.append(str(requant["symbol"]))
    return symbols


def _needs_stable_softmax(layers: list[dict[str, Any]]) -> bool:
    return any(
        layer.get("kind") == "softmax" and bool(layer.get("stable_softmax"))
        for layer in layers
    )


def _needs_static_indexing_helpers(layers: list[dict[str, Any]]) -> bool:
    return any(layer.get("kind") in {"pad", "slice", "gather"} for layer in layers)


def _static_indexing_helpers() -> list[str]:
    return [
        "",
        "#if NANOC_ENABLE_CMSIS_NN",
        "static void nanoc_unravel_index(size_t index, int rank, const int *shape, int *coords)",
        "{",
        "    for (int axis = rank - 1; axis >= 0; --axis) {",
        "        int dim = shape[axis];",
        "        coords[axis] = dim > 0 ? (int)(index % (size_t)dim) : 0;",
        "        if (dim > 0) {",
        "            index /= (size_t)dim;",
        "        }",
        "    }",
        "}",
        "",
        "static size_t nanoc_ravel_index(int rank, const int *shape, const int *coords)",
        "{",
        "    size_t offset = 0u;",
        "    for (int axis = 0; axis < rank; ++axis) {",
        "        offset = offset * (size_t)shape[axis] + (size_t)coords[axis];",
        "    }",
        "    return offset;",
        "}",
        "#endif",
        "",
    ]


def _stable_softmax_helper() -> list[str]:
    return [
        "",
        "#if NANOC_ENABLE_CMSIS_NN",
        "static void nanoc_softmax_s8_stable(const int8_t *input,",
        "                                     int32_t num_rows,",
        "                                     int32_t row_size,",
        "                                     float input_scale,",
        "                                     int8_t *output)",
        "{",
        "    for (int32_t row = 0; row < num_rows; ++row) {",
        "        const int8_t *row_in = input + (size_t)row * (size_t)row_size;",
        "        int8_t *row_out = output + (size_t)row * (size_t)row_size;",
        "        int8_t max_q = row_in[0];",
        "        int32_t max_col = 0;",
        "        for (int32_t col = 1; col < row_size; ++col) {",
        "            if (row_in[col] > max_q) {",
        "                max_q = row_in[col];",
        "                max_col = col;",
        "            }",
        "        }",
        "        float sum = 0.0f;",
        "        for (int32_t col = 0; col < row_size; ++col) {",
        "            sum += expf(((float)row_in[col] - (float)max_q) * input_scale);",
        "        }",
        "        if (sum <= 0.0f) {",
        "            for (int32_t col = 0; col < row_size; ++col) {",
        "                row_out[col] = -128;",
        "            }",
        "            continue;",
        "        }",
        "        for (int32_t col = 0; col < row_size; ++col) {",
        "            float prob = expf(((float)row_in[col] - (float)max_q) * input_scale) / sum;",
        "            int32_t q = (int32_t)(prob * 256.0f - 128.0f + 0.5f);",
        "            if (q > 127) { q = 127; }",
        "            if (q < -128) { q = -128; }",
        "            row_out[col] = (int8_t)q;",
        "        }",
        "        int8_t best = row_out[max_col];",
        "        for (int32_t col = 0; col < row_size; ++col) {",
        "            if (col != max_col && row_out[col] >= best && best < 127) {",
        "                best = (int8_t)(row_out[col] + 1);",
        "            }",
        "        }",
        "        row_out[max_col] = best;",
        "    }",
        "}",
        "#endif",
        "",
    ]


def _tensor_symbol_map(
    graph: ModelGraph,
    layers: list[dict[str, Any]],
) -> dict[str, str]:
    aliases = _tensor_aliases(graph)
    result: dict[str, str] = {}
    model_outputs = {tensor.name for tensor in graph.outputs}
    for layer in layers:
        for name in layer.get("output_tensors", []):
            resolved = _resolve_tensor_alias(str(name), aliases)
            if resolved not in model_outputs:
                result.setdefault(resolved, _tensor_symbol(resolved))
    return result


def _tensor_aliases(graph: ModelGraph) -> dict[str, str]:
    aliases: dict[str, str] = {}
    alias_ops = {
        "DequantizeLinear",
        "Dropout",
        "Flatten",
        "QuantizeLinear",
        "Relu",
        "Reshape",
        "Squeeze",
    }
    for node in graph.nodes:
        if node.op_type not in alias_ops or not node.inputs:
            continue
        for output_name in node.outputs:
            aliases[output_name] = node.inputs[0]
    return aliases


def _resolve_tensor_alias(name: str, aliases: dict[str, str]) -> str:
    seen: set[str] = set()
    current = name
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current


def _resolve_tensor_alias_path(name: str, aliases: dict[str, str]) -> list[str]:
    seen: set[str] = set()
    path = [name]
    current = name
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
        path.append(current)
    return path


def _output_flows_through_relu(graph: ModelGraph, output_name: str) -> bool:
    """Return true when a tensor is folded through Q/DQ into Relu."""
    consumers: dict[str, list[Any]] = {}
    for node in graph.nodes:
        for input_name in node.inputs:
            consumers.setdefault(str(input_name), []).append(node)

    queue = [output_name]
    seen: set[str] = set()
    passthrough_ops = {"QuantizeLinear", "DequantizeLinear", "Relu"}
    while queue:
        tensor_name = queue.pop(0)
        if tensor_name in seen:
            continue
        seen.add(tensor_name)
        for consumer in consumers.get(tensor_name, []):
            if consumer.op_type == "Relu":
                return True
            if consumer.op_type not in passthrough_ops:
                continue
            queue.extend(str(name) for name in consumer.outputs)
    return False


def _flatten_layout_transforms(graph: ModelGraph) -> dict[str, dict[str, Any]]:
    """Find alias tensors that need NHWC memory flattened back to ONNX NCHW order."""
    aliases = _tensor_aliases(graph)
    reshape_by_output: dict[str, dict[str, Any]] = {}
    for node in graph.nodes:
        if node.op_type not in {"Flatten", "Reshape"} or not node.inputs or not node.outputs:
            continue
        source = node.inputs[0]
        output = node.outputs[0]
        source_shape = node.input_shapes.get(source, [])
        output_shape = node.output_shapes.get(output, [])
        if (
            len(source_shape) not in {3, 4}
            or len(output_shape) not in {1, 2}
            or not all(isinstance(dim, int) and dim > 0 for dim in source_shape)
        ):
            continue
        source_count = element_count_from_shape(source_shape)
        output_count = element_count_from_shape(output_shape)
        if source_count is None or output_count is None or source_count != output_count:
            continue
        reshape_by_output[output] = {
            "source": source,
            "source_shape": [int(dim) for dim in source_shape],
            "size": int(output_count),
        }

    if not reshape_by_output:
        return {}

    tensor_names: set[str] = set()
    for node in graph.nodes:
        tensor_names.update(str(name) for name in node.inputs)
        tensor_names.update(str(name) for name in node.outputs)
    tensor_names.update(tensor.name for tensor in graph.inputs)
    tensor_names.update(tensor.name for tensor in graph.outputs)

    transforms: dict[str, dict[str, Any]] = {}
    for tensor_name in tensor_names:
        path = _resolve_tensor_alias_path(tensor_name, aliases)
        for reshape_output, transform in reshape_by_output.items():
            if reshape_output not in path:
                continue
            item = dict(transform)
            item["symbol"] = _tensor_symbol(tensor_name)
            transforms[tensor_name] = item
            break
    return transforms


def _flatten_transform_call(transform: dict[str, Any], source_expr: str) -> list[str]:
    source_shape = [int(dim) for dim in transform["source_shape"]]
    target = str(transform["symbol"])
    if len(source_shape) == 3:
        n, c, w = source_shape
        return [
            f"    /* NCW flatten for {transform['source']} -> {target} */",
            f"    for (int _ni = 0; _ni < {n}; ++_ni)",
            f"        for (int _ci = 0; _ci < {c}; ++_ci)",
            f"            for (int _wi = 0; _wi < {w}; ++_wi)",
            (
                f"                {target}[_ni * {c * w} + _ci * {w} + _wi] = "
                f"{source_expr}[_ni * {w * c} + _wi * {c} + _ci];"
            ),
            "",
        ]
    n, c, h, w = source_shape
    return [
        f"    /* NCHW flatten for {transform['source']} -> {target} */",
        f"    for (int _ni = 0; _ni < {n}; ++_ni)",
        f"        for (int _ci = 0; _ci < {c}; ++_ci)",
        f"            for (int _hi = 0; _hi < {h}; ++_hi)",
        f"                for (int _wi = 0; _wi < {w}; ++_wi)",
        (
            f"                    {target}[_ni * {c * h * w} + _ci * {h * w} + "
            f"_hi * {w} + _wi] = {source_expr}[_ni * {h * w * c} + "
            f"_hi * {w * c} + _wi * {c} + _ci];"
        ),
        "",
    ]


def _tensor_symbol(name: str) -> str:
    return _c_symbol(f"nanoc_tensor_{name}")


def _tensor_element_count(graph: ModelGraph, tensor_name: str) -> int:
    for node in graph.nodes:
        for shape in (node.output_shapes.get(tensor_name), node.input_shapes.get(tensor_name)):
            if shape:
                count = element_count_from_shape(shape)
                if count is not None:
                    return count
    for tensor in [*graph.inputs, *graph.outputs]:
        if tensor.name == tensor_name and tensor.element_count is not None:
            return tensor.element_count
    return 0


def _layer_output_element_count(layer: dict[str, Any]) -> int:
    if layer["kind"] in {"conv", "depthwise", "pool", "concat"}:
        count = 1
        for dim in layer.get("output_dims", []):
            count *= int(dim)
        return count
    if layer["kind"] == "fc":
        return int(layer.get("output_size", 0))
    if layer["kind"] in {
        "add",
        "mul",
        "sub",
        "div",
        "min",
        "max",
        "pow",
        "prelu",
        "equal",
        "greater",
        "less",
        "greaterorequal",
        "lessorequal",
        "and",
        "or",
        "not",
        "xor",
        "where",
    }:
        return int(layer.get("block_size", 0))
    if layer["kind"] in {
        "abs",
        "sigmoid",
        "tanh",
        "leakyrelu",
        "clip",
        "neg",
        "sqrt",
        "reciprocal",
        "exp",
        "log",
        "floor",
        "ceil",
        "round",
        "sign",
        "erf",
        "softplus",
        "softsign",
        "hardswish",
        "elu",
        "selu",
        "hardsigmoid",
        "thresholdedrelu",
        "celu",
        "reducemean",
        "reducesum",
        "reducemax",
        "reducemin",
        "reduceprod",
        "reducel1",
        "reducel2",
        "reducelogsum",
        "reducelogsumexp",
        "reducesumsquare",
        "unsqueeze",
        "pad",
        "slice",
        "gather",
    }:
        return int(layer.get("block_size", 0))
    if layer["kind"] == "transpose":
        return int(layer.get("block_size", 0))
    if layer["kind"] == "softmax":
        return int(layer.get("num_rows", 0)) * int(layer.get("row_size", 0))
    return 0


def _graph_input_shape(graph: ModelGraph) -> list[int] | None:
    """Return the NCHW shape of the first model input, or None."""
    if not graph.inputs:
        return None
    shape = graph.inputs[0].shape
    if shape and all(isinstance(d, int) for d in shape):
        return [int(d) for d in shape]
    return None


def _graph_output_shape(graph: ModelGraph) -> list[int] | None:
    """Return the NCHW shape of the first model output, or None."""
    if not graph.outputs:
        return None
    shape = graph.outputs[0].shape
    if shape and all(isinstance(d, int) for d in shape):
        return [int(d) for d in shape]
    return None


def _model_input_expr_map(graph: ModelGraph) -> dict[str, str]:
    """Map each graph input to its packed offset in nanoc_model_run(input, output)."""
    result: dict[str, str] = {}
    offset = 0
    for tensor in graph.inputs:
        result[tensor.name] = "input" if offset == 0 else f"(input + {offset}u)"
        offset += int(tensor.element_count or 0)
    return result


def _graph_output_size(graph: ModelGraph) -> int:
    if not graph.outputs:
        return 1
    return graph.outputs[0].element_count or 1


def _graph_input_size(graph: ModelGraph) -> int:
    if not graph.inputs:
        return 1
    return graph.inputs[0].element_count or 1


def _cmsis_fc_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_size = layer["input_size"]
    output_size = layer["output_size"]
    prefix = layer["symbol"]
    input_expr = layer.get("input_expr", "current_input")
    per_channel = isinstance(layer["multiplier"], list)
    api = "arm_fully_connected_per_channel_s8" if per_channel else "arm_fully_connected_s8"
    quant_type = (
        "cmsis_nn_per_channel_quant_params"
        if per_channel
        else "cmsis_nn_per_tensor_quant_params"
    )
    multiplier_expr = (
        f"(int32_t *){prefix}_multiplier" if per_channel else str(layer["multiplier"])
    )
    shift_expr = f"(int32_t *){prefix}_shift" if per_channel else str(layer["shift"])
    return [
        f"    /* node {layer['index']}: {layer['name']} -> {api} */",
        "    {",
        "        cmsis_nn_context ctx;",
        "        cmsis_nn_fc_params fc_params;",
        f"        {quant_type} quant_params;",
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
        f"        quant_params.multiplier = {multiplier_expr};",
        f"        quant_params.shift = {shift_expr};",
        "",
        f"        cmsis_status = {api}(",
        "            &ctx,",
        "            &fc_params,",
        "            &quant_params,",
        "            &input_dims,",
        f"            {input_expr},",
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


def _cmsis_conv_call(layer: dict[str, Any], current_output: str) -> list[str]:
    prefix = layer["symbol"]
    input_expr = layer.get("input_expr", "current_input")
    input_n, input_h, input_w, input_c = layer["input_dims"]
    output_n, output_h, output_w, output_c = layer["output_dims"]
    filter_h, filter_w = layer["kernel_shape"]
    stride_h, stride_w = layer["stride"]
    pad_h, pad_w = layer["padding"]
    dilation_h, dilation_w = layer["dilation"]
    return [
        f"    /* node {layer['index']}: {layer['name']} -> arm_convolve_wrapper_s8 */",
        "    {",
        "        cmsis_nn_context ctx;",
        "        cmsis_nn_conv_params conv_params;",
        "        cmsis_nn_per_channel_quant_params quant_params;",
        f"        cmsis_nn_dims input_dims = {{{input_n}, {input_h}, {input_w}, {input_c}}};",
        f"        cmsis_nn_dims filter_dims = {{{output_c}, {filter_h}, {filter_w}, {input_c}}};",
        f"        cmsis_nn_dims bias_dims = {{1, 1, 1, {output_c}}};",
        f"        cmsis_nn_dims output_dims = {{{output_n}, {output_h}, {output_w}, {output_c}}};",
        "",
        f"        conv_params.input_offset = {layer['input_offset']};",
        f"        conv_params.output_offset = {layer['output_offset']};",
        f"        conv_params.stride.h = {stride_h};",
        f"        conv_params.stride.w = {stride_w};",
        f"        conv_params.padding.h = {pad_h};",
        f"        conv_params.padding.w = {pad_w};",
        f"        conv_params.dilation.h = {dilation_h};",
        f"        conv_params.dilation.w = {dilation_w};",
        f"        conv_params.activation.min = {layer['activation_min']};",
        f"        conv_params.activation.max = {layer['activation_max']};",
        f"        quant_params.multiplier = (int32_t *){prefix}_multiplier;",
        f"        quant_params.shift = (int32_t *){prefix}_shift;",
        "",
        "        ctx.buf = nanoc_scratch;",
        "        ctx.size = arm_convolve_wrapper_s8_get_buffer_size(",
        "            &conv_params, &input_dims, &filter_dims, &output_dims);",
        "        if (ctx.size > NANOC_MODEL_SCRATCH_BYTES) {",
        "            return NANOC_STATUS_BLOCKED;",
        "        }",
        "",
        "        cmsis_status = arm_convolve_wrapper_s8(",
        "            &ctx,",
        "            &conv_params,",
        "            &quant_params,",
        "            &input_dims,",
        f"            {input_expr},",
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


def _cmsis_depthwise_call(layer: dict[str, Any], current_output: str) -> list[str]:
    prefix = layer["symbol"]
    input_expr = layer.get("input_expr", "current_input")
    input_n, input_h, input_w, input_c = layer["input_dims"]
    output_n, output_h, output_w, output_c = layer["output_dims"]
    filter_h, filter_w = layer["kernel_shape"]
    stride_h, stride_w = layer["stride"]
    pad_h, pad_w = layer["padding"]
    dilation_h, dilation_w = layer["dilation"]
    return [
        f"    /* node {layer['index']}: {layer['name']} -> arm_depthwise_conv_wrapper_s8 */",
        "    {",
        "        cmsis_nn_context ctx;",
        "        cmsis_nn_dw_conv_params dw_conv_params;",
        "        cmsis_nn_per_channel_quant_params quant_params;",
        f"        cmsis_nn_dims input_dims = {{{input_n}, {input_h}, {input_w}, {input_c}}};",
        f"        cmsis_nn_dims filter_dims = {{1, {filter_h}, {filter_w}, {output_c}}};",
        f"        cmsis_nn_dims bias_dims = {{1, 1, 1, {output_c}}};",
        f"        cmsis_nn_dims output_dims = {{{output_n}, {output_h}, {output_w}, {output_c}}};",
        "",
        f"        dw_conv_params.input_offset = {layer['input_offset']};",
        f"        dw_conv_params.output_offset = {layer['output_offset']};",
        f"        dw_conv_params.stride.h = {stride_h};",
        f"        dw_conv_params.stride.w = {stride_w};",
        f"        dw_conv_params.padding.h = {pad_h};",
        f"        dw_conv_params.padding.w = {pad_w};",
        f"        dw_conv_params.dilation.h = {dilation_h};",
        f"        dw_conv_params.dilation.w = {dilation_w};",
        f"        dw_conv_params.ch_mult = {layer['channel_multiplier']};",
        f"        dw_conv_params.activation.min = {layer['activation_min']};",
        f"        dw_conv_params.activation.max = {layer['activation_max']};",
        f"        quant_params.multiplier = (int32_t *){prefix}_multiplier;",
        f"        quant_params.shift = (int32_t *){prefix}_shift;",
        "",
        "        ctx.buf = nanoc_scratch;",
        "        ctx.size = arm_depthwise_conv_wrapper_s8_get_buffer_size(",
        "            &dw_conv_params, &input_dims, &filter_dims, &output_dims);",
        "        if (ctx.size > NANOC_MODEL_SCRATCH_BYTES) {",
        "            return NANOC_STATUS_BLOCKED;",
        "        }",
        "",
        "        cmsis_status = arm_depthwise_conv_wrapper_s8(",
        "            &ctx,",
        "            &dw_conv_params,",
        "            &quant_params,",
        "            &input_dims,",
        f"            {input_expr},",
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


def _cmsis_pool_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_expr = layer.get("input_expr", "current_input")
    input_n, input_h, input_w, input_c = layer["input_dims"]
    output_n, output_h, output_w, output_c = layer["output_dims"]
    kernel_h, kernel_w = layer["kernel_shape"]
    stride_h, stride_w = layer["stride"]
    pad_h, pad_w = layer["padding"]
    api = layer["api"]
    lines = [
        f"    /* node {layer['index']}: {layer['name']} -> {api} */",
        "    {",
        "        cmsis_nn_context ctx;",
        "        cmsis_nn_pool_params pool_params;",
        f"        cmsis_nn_dims input_dims = {{{input_n}, {input_h}, {input_w}, {input_c}}};",
        f"        cmsis_nn_dims filter_dims = {{1, {kernel_h}, {kernel_w}, 1}};",
        f"        cmsis_nn_dims output_dims = {{{output_n}, {output_h}, {output_w}, {output_c}}};",
        "",
        f"        pool_params.stride.h = {stride_h};",
        f"        pool_params.stride.w = {stride_w};",
        f"        pool_params.padding.h = {pad_h};",
        f"        pool_params.padding.w = {pad_w};",
        f"        pool_params.activation.min = {layer['activation_min']};",
        f"        pool_params.activation.max = {layer['activation_max']};",
        "        ctx.buf = nanoc_scratch;",
    ]
    if api == "arm_avgpool_s8":
        lines.extend(
            [
                f"        ctx.size = arm_avgpool_s8_get_buffer_size({output_w}, {input_c});",
                "        if (ctx.size > NANOC_MODEL_SCRATCH_BYTES) {",
                "            return NANOC_STATUS_BLOCKED;",
                "        }",
            ]
        )
    else:
        lines.append("        ctx.size = 0;")
    lines.extend(
        [
            "",
            f"        cmsis_status = {api}(",
            "            &ctx,",
            "            &pool_params,",
            "            &input_dims,",
            f"            {input_expr},",
            "            &filter_dims,",
            "            &output_dims,",
            f"            {current_output});",
            "        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {",
            "            return NANOC_STATUS_BLOCKED;",
            "        }",
        ]
    )
    if layer.get("requantize_output"):
        lines.extend(
            [
                f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer.get('output_elements', 0))}u; ++nanoc_i) {{",
                (
                    f"            int32_t nanoc_v = (int32_t){current_output}[nanoc_i] - "
                    f"{int(layer['requantize_input_zero_point'])};"
                ),
                (
                    f"            nanoc_v = arm_nn_requantize(nanoc_v, "
                    f"{int(layer['requantize_multiplier'])}, {int(layer['requantize_shift'])}) + "
                    f"{int(layer['requantize_output_zero_point'])};"
                ),
                f"            if (nanoc_v > {int(layer['activation_max'])}) {{ nanoc_v = {int(layer['activation_max'])}; }}",
                f"            if (nanoc_v < {int(layer['activation_min'])}) {{ nanoc_v = {int(layer['activation_min'])}; }}",
                f"            {current_output}[nanoc_i] = (int8_t)nanoc_v;",
                "        }",
            ]
        )
    lines.extend(["    }", ""])
    return lines


def _cmsis_add_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_1 = layer.get("input_1_expr", "current_input")
    input_2 = layer.get("constant_symbol") or "current_input"
    if not layer.get("constant_symbol"):
        input_2 = layer.get("input_2_expr", input_2)
    return [
        f"    /* node {layer['index']}: {layer['name']} -> arm_elementwise_add_s8 */",
        "    {",
        "        cmsis_status = arm_elementwise_add_s8(",
        f"            {input_1},",
        f"            {input_2},",
        f"            {layer['input_1_offset']},",
        f"            {layer['input_1_multiplier']},",
        f"            {layer['input_1_shift']},",
        f"            {layer['input_2_offset']},",
        f"            {layer['input_2_multiplier']},",
        f"            {layer['input_2_shift']},",
        f"            {layer['left_shift']},",
        f"            {current_output},",
        f"            {layer['output_offset']},",
        f"            {layer['output_multiplier']},",
        f"            {layer['output_shift']},",
        f"            {layer['activation_min']},",
        f"            {layer['activation_max']},",
        f"            {layer['block_size']});",
        "        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {",
        "            return NANOC_STATUS_BLOCKED;",
        "        }",
        "    }",
        "",
    ]


def _cmsis_mul_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_1 = layer.get("input_1_expr", "current_input")
    input_2 = layer.get("constant_symbol") or "current_input"
    if not layer.get("constant_symbol"):
        input_2 = layer.get("input_2_expr", input_2)
    return [
        f"    /* node {layer['index']}: {layer['name']} -> arm_elementwise_mul_s8 */",
        "    {",
        "        cmsis_status = arm_elementwise_mul_s8(",
        f"            {input_1},",
        f"            {input_2},",
        f"            {layer['input_1_offset']},",
        f"            {layer['input_2_offset']},",
        f"            {current_output},",
        f"            {layer['output_offset']},",
        f"            {layer['output_multiplier']},",
        f"            {layer['output_shift']},",
        f"            {layer['activation_min']},",
        f"            {layer['activation_max']},",
        f"            {layer['block_size']});",
        "        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {",
        "            return NANOC_STATUS_BLOCKED;",
        "        }",
        "    }",
        "",
    ]


def _generated_abs_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_expr = layer.get("input_expr", "current_input")
    return [
        f"    /* node {layer['index']}: {layer['name']} -> generated_c_abs_s8 */",
        "    {",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer['block_size'])}u; ++nanoc_i) {{",
        f"            int32_t nanoc_v = (int32_t){input_expr}[nanoc_i];",
        "            if (nanoc_v < 0) {",
        "                nanoc_v = nanoc_v == -128 ? 127 : -nanoc_v;",
        "            }",
        f"            if (nanoc_v > {int(layer['activation_max'])}) {{ nanoc_v = {int(layer['activation_max'])}; }}",
        f"            if (nanoc_v < {int(layer['activation_min'])}) {{ nanoc_v = {int(layer['activation_min'])}; }}",
        f"            {current_output}[nanoc_i] = (int8_t)nanoc_v;",
        "        }",
        "    }",
        "",
    ]


def _generated_binary_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_1 = layer.get("input_1_expr", "current_input")
    input_2 = layer.get("constant_symbol") or layer.get("input_2_expr", "current_input")
    op = str(layer["kind"])
    api = f"generated_c_{op}_s8"
    guard_lines: list[str] = []
    if op == "div":
        guard_lines = [
            "            float nanoc_v = 0.0f;",
            "            if (nanoc_b > 0.0000001f || nanoc_b < -0.0000001f) {",
            "                nanoc_v = nanoc_a / nanoc_b;",
            "            }",
        ]
    elif op == "sub":
        guard_lines = ["            float nanoc_v = nanoc_a - nanoc_b;"]
    elif op == "min":
        guard_lines = ["            float nanoc_v = nanoc_a < nanoc_b ? nanoc_a : nanoc_b;"]
    elif op == "max":
        guard_lines = ["            float nanoc_v = nanoc_a > nanoc_b ? nanoc_a : nanoc_b;"]
    elif op == "pow":
        guard_lines = [
            "            float nanoc_v = 0.0f;",
            "            if (nanoc_a >= 0.0f) {",
            "                nanoc_v = powf(nanoc_a, nanoc_b);",
            "            }",
        ]
    elif op == "prelu":
        guard_lines = ["            float nanoc_v = nanoc_a > 0.0f ? nanoc_a : nanoc_a * nanoc_b;"]
    else:
        return []
    return [
        f"    /* node {layer['index']}: {layer['name']} -> {api} */",
        "    {",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer['block_size'])}u; ++nanoc_i) {{",
        (
            f"            float nanoc_a = ((float){input_1}[nanoc_i] - "
            f"{_c_float_literal(float(layer['input_1_zero_point']))}) * "
            f"{_c_float_literal(float(layer['input_1_scale']))};"
        ),
        (
            f"            float nanoc_b = ((float){input_2}[nanoc_i] - "
            f"{_c_float_literal(float(layer['input_2_zero_point']))}) * "
            f"{_c_float_literal(float(layer['input_2_scale']))};"
        ),
        *guard_lines,
        (
            f"            float nanoc_qf = nanoc_v / {_c_float_literal(float(layer['output_scale']))} + "
            f"{_c_float_literal(float(layer['output_zero_point']))};"
        ),
        "            int32_t nanoc_q = (int32_t)nearbyintf(nanoc_qf);",
        f"            if (nanoc_q > {int(layer['activation_max'])}) {{ nanoc_q = {int(layer['activation_max'])}; }}",
        f"            if (nanoc_q < {int(layer['activation_min'])}) {{ nanoc_q = {int(layer['activation_min'])}; }}",
        f"            {current_output}[nanoc_i] = (int8_t)nanoc_q;",
        "        }",
        "    }",
        "",
    ]


def _generated_compare_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_1 = layer.get("input_1_expr", "current_input")
    input_2 = layer.get("constant_symbol") or layer.get("input_2_expr", "current_input")
    op = str(layer["kind"])
    api = f"generated_c_{op}_bool"
    if op == "equal":
        compare_line = "            int nanoc_cond = nanoc_a == nanoc_b;"
    elif op == "greater":
        compare_line = "            int nanoc_cond = nanoc_a > nanoc_b;"
    elif op == "less":
        compare_line = "            int nanoc_cond = nanoc_a < nanoc_b;"
    elif op == "greaterorequal":
        compare_line = "            int nanoc_cond = nanoc_a >= nanoc_b;"
    elif op == "lessorequal":
        compare_line = "            int nanoc_cond = nanoc_a <= nanoc_b;"
    else:
        return []
    return [
        f"    /* node {layer['index']}: {layer['name']} -> {api} */",
        "    {",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer['block_size'])}u; ++nanoc_i) {{",
        (
            f"            int32_t nanoc_a = (int32_t){input_1}[nanoc_i] - "
            f"{int(layer['input_1_zero_point'])};"
        ),
        (
            f"            int32_t nanoc_b = (int32_t){input_2}[nanoc_i] - "
            f"{int(layer['input_2_zero_point'])};"
        ),
        compare_line,
        f"            {current_output}[nanoc_i] = (int8_t)(nanoc_cond ? 1 : 0);",
        "        }",
        "    }",
        "",
    ]


def _generated_bool_logic_call(layer: dict[str, Any], current_output: str) -> list[str]:
    op = str(layer["kind"])
    api = f"generated_c_{op}_bool"
    if op == "not":
        input_exprs = [layer.get("input_1_expr", "current_input")]
    else:
        input_exprs = [
            layer.get("input_1_expr", "current_input"),
            layer.get("input_2_expr", "current_input"),
        ]
    if op == "and":
        logic_line = "            int nanoc_cond = nanoc_a && nanoc_b;"
    elif op == "or":
        logic_line = "            int nanoc_cond = nanoc_a || nanoc_b;"
    elif op == "not":
        logic_line = "            int nanoc_cond = !nanoc_a;"
    elif op == "xor":
        logic_line = "            int nanoc_cond = (nanoc_a != 0) != (nanoc_b != 0);"
    else:
        return []
    lines = [
        f"    /* node {layer['index']}: {layer['name']} -> {api} */",
        "    {",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer['block_size'])}u; ++nanoc_i) {{",
    ]
    for index, expr in enumerate(input_exprs):
        lines.append(f"            int32_t nanoc_{chr(ord('a') + index)} = ({expr}[nanoc_i] != 0) ? 1 : 0;")
    lines.append(logic_line)
    lines.append(f"            {current_output}[nanoc_i] = (int8_t)(nanoc_cond ? 1 : 0);")
    lines.extend(["        }", "    }", ""])
    return lines


def _generated_where_call(layer: dict[str, Any], current_output: str) -> list[str]:
    condition = layer.get("condition_symbol") or layer.get("condition_expr", "current_input")
    x_expr = layer.get("x_expr", "current_input")
    y_expr = layer.get("constant_symbol") or layer.get("y_expr", "current_input")
    return [
        f"    /* node {layer['index']}: {layer['name']} -> generated_c_where_s8 */",
        "    {",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer['block_size'])}u; ++nanoc_i) {{",
        f"            const int8_t nanoc_selected_q = {condition}[nanoc_i] ? {x_expr}[nanoc_i] : {y_expr}[nanoc_i];",
        "            float nanoc_v;",
        f"            if ({condition}[nanoc_i]) {{",
        (
            f"                nanoc_v = ((float)nanoc_selected_q - {int(layer['x_zero_point'])}) * "
            f"{_c_float_literal(float(layer['x_scale']))};"
        ),
        "            } else {",
        (
            f"                nanoc_v = ((float)nanoc_selected_q - {int(layer['y_zero_point'])}) * "
            f"{_c_float_literal(float(layer['y_scale']))};"
        ),
        "            }",
        (
            f"            float nanoc_qf = nanoc_v / {_c_float_literal(float(layer['output_scale']))} + "
            f"{_c_float_literal(float(layer['output_zero_point']))};"
        ),
        "            int32_t nanoc_q = (int32_t)nearbyintf(nanoc_qf);",
        f"            if (nanoc_q > {int(layer['activation_max'])}) {{ nanoc_q = {int(layer['activation_max'])}; }}",
        f"            if (nanoc_q < {int(layer['activation_min'])}) {{ nanoc_q = {int(layer['activation_min'])}; }}",
        f"            {current_output}[nanoc_i] = (int8_t)nanoc_q;",
        "        }",
        "    }",
        "",
    ]


def _generated_unary_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_expr = layer.get("input_expr", "current_input")
    kind = str(layer["kind"])
    api = f"generated_c_{kind}_s8"
    if kind == "sigmoid":
        value_lines = ["            float nanoc_v = 1.0f / (1.0f + expf(-nanoc_x));"]
    elif kind == "tanh":
        value_lines = ["            float nanoc_v = tanhf(nanoc_x);"]
    elif kind == "leakyrelu":
        value_lines = [
            (
                "            float nanoc_v = nanoc_x >= 0.0f ? nanoc_x : "
                f"nanoc_x * {_c_float_literal(float(layer.get('alpha', 0.01)))};"
            )
        ]
    elif kind == "clip":
        value_lines = ["            float nanoc_v = nanoc_x;"]
        if "clip_min" in layer:
            value_lines.append(
                f"            if (nanoc_v < {_c_float_literal(float(layer['clip_min']))}) "
                f"{{ nanoc_v = {_c_float_literal(float(layer['clip_min']))}; }}"
            )
        if "clip_max" in layer:
            value_lines.append(
                f"            if (nanoc_v > {_c_float_literal(float(layer['clip_max']))}) "
                f"{{ nanoc_v = {_c_float_literal(float(layer['clip_max']))}; }}"
            )
    elif kind == "neg":
        value_lines = ["            float nanoc_v = -nanoc_x;"]
    elif kind == "sqrt":
        value_lines = [
            "            float nanoc_v = nanoc_x <= 0.0f ? 0.0f : sqrtf(nanoc_x);"
        ]
    elif kind == "reciprocal":
        value_lines = [
            "            float nanoc_v = 0.0f;",
            "            if (nanoc_x > 0.0000001f || nanoc_x < -0.0000001f) {",
            "                nanoc_v = 1.0f / nanoc_x;",
            "            }",
        ]
    elif kind == "exp":
        value_lines = ["            float nanoc_v = expf(nanoc_x);"]
    elif kind == "log":
        value_lines = [
            "            float nanoc_v = -128.0f;",
            "            if (nanoc_x > 0.0000001f) {",
            "                nanoc_v = logf(nanoc_x);",
            "            }",
        ]
    elif kind == "floor":
        value_lines = ["            float nanoc_v = floorf(nanoc_x);"]
    elif kind == "ceil":
        value_lines = ["            float nanoc_v = ceilf(nanoc_x);"]
    elif kind == "round":
        value_lines = ["            float nanoc_v = nearbyintf(nanoc_x);"]
    elif kind == "sign":
        value_lines = [
            "            float nanoc_v = 0.0f;",
            "            if (nanoc_x > 0.0f) { nanoc_v = 1.0f; }",
            "            if (nanoc_x < 0.0f) { nanoc_v = -1.0f; }",
        ]
    elif kind == "erf":
        value_lines = ["            float nanoc_v = erff(nanoc_x);"]
    elif kind == "softplus":
        value_lines = ["            float nanoc_v = log1pf(expf(nanoc_x));"]
    elif kind == "softsign":
        value_lines = ["            float nanoc_v = nanoc_x / (1.0f + fabsf(nanoc_x));"]
    elif kind == "hardswish":
        value_lines = [
            "            float nanoc_v = nanoc_x * fminf(fmaxf(nanoc_x + 3.0f, 0.0f), 6.0f) / 6.0f;"
        ]
    elif kind == "elu":
        value_lines = [
            (
                "            float nanoc_v = nanoc_x > 0.0f ? nanoc_x : "
                f"{_c_float_literal(float(layer.get('alpha', 1.0)))} * (expf(nanoc_x) - 1.0f);"
            )
        ]
    elif kind == "selu":
        value_lines = [
            (
                f"            float nanoc_v = {_c_float_literal(float(layer.get('gamma', 1.0507)))} * "
                f"(nanoc_x > 0.0f ? nanoc_x : "
                f"{_c_float_literal(float(layer.get('alpha', 1.67326)))} * (expf(nanoc_x) - 1.0f));"
            )
        ]
    elif kind == "hardsigmoid":
        value_lines = [
            (
                f"            float nanoc_v = fminf(fmaxf("
                f"{_c_float_literal(float(layer.get('alpha', 0.2)))} * nanoc_x + "
                f"{_c_float_literal(float(layer.get('beta', 0.5)))}, 0.0f), 1.0f);"
            )
        ]
    elif kind == "thresholdedrelu":
        value_lines = [
            (
                f"            float nanoc_v = nanoc_x > {_c_float_literal(float(layer.get('alpha', 1.0)))} "
                f"? nanoc_x : 0.0f;"
            )
        ]
    elif kind == "celu":
        value_lines = [
            (
                f"            float nanoc_v = fmaxf(0.0f, nanoc_x) + fminf(0.0f, "
                f"{_c_float_literal(float(layer.get('alpha', 1.0)))} * "
                f"(expf(nanoc_x / {_c_float_literal(float(layer.get('alpha', 1.0)))}) - 1.0f));"
            )
        ]
    else:
        return []
    return [
        f"    /* node {layer['index']}: {layer['name']} -> {api} */",
        "    {",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer['block_size'])}u; ++nanoc_i) {{",
        (
            f"            float nanoc_x = ((float){input_expr}[nanoc_i] - "
            f"{_c_float_literal(float(layer['input_zero_point']))}) * "
            f"{_c_float_literal(float(layer['input_scale']))};"
        ),
        *value_lines,
        (
            f"            float nanoc_qf = nanoc_v / {_c_float_literal(float(layer['output_scale']))} + "
            f"{_c_float_literal(float(layer['output_zero_point']))};"
        ),
        "            int32_t nanoc_q = (int32_t)nearbyintf(nanoc_qf);",
        f"            if (nanoc_q > {int(layer['activation_max'])}) {{ nanoc_q = {int(layer['activation_max'])}; }}",
        f"            if (nanoc_q < {int(layer['activation_min'])}) {{ nanoc_q = {int(layer['activation_min'])}; }}",
        f"            {current_output}[nanoc_i] = (int8_t)nanoc_q;",
        "        }",
        "    }",
        "",
    ]


def _generated_reduce_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_expr = layer.get("input_expr", "current_input")
    input_shape = [int(item) for item in layer["input_shape"]]
    rows = input_shape[0]
    cols = input_shape[1]
    kind = str(layer["kind"])
    api = f"generated_c_{kind}_s8"
    if kind == "reducemax":
        init_line = "            float nanoc_v = -3.402823466e+38f;"
        update_line = "                if (nanoc_x > nanoc_v) { nanoc_v = nanoc_x; }"
        finish_lines: list[str] = []
    elif kind == "reducemin":
        init_line = "            float nanoc_v = 3.402823466e+38f;"
        update_line = "                if (nanoc_x < nanoc_v) { nanoc_v = nanoc_x; }"
        finish_lines = []
    elif kind == "reduceprod":
        init_line = "            float nanoc_v = 1.0f;"
        update_line = "                nanoc_v *= nanoc_x;"
        finish_lines = []
    elif kind == "reducel1":
        init_line = "            float nanoc_v = 0.0f;"
        update_line = "                nanoc_v += nanoc_x < 0.0f ? -nanoc_x : nanoc_x;"
        finish_lines = []
    elif kind == "reducel2":
        init_line = "            float nanoc_v = 0.0f;"
        update_line = "                nanoc_v += nanoc_x * nanoc_x;"
        finish_lines = ["            nanoc_v = sqrtf(nanoc_v);"]
    elif kind == "reducelogsum":
        init_line = "            float nanoc_v = 0.0f;"
        update_line = "                nanoc_v += nanoc_x;"
        finish_lines = ["            nanoc_v = logf(nanoc_v);"]
    elif kind == "reducelogsumexp":
        init_line = "            float nanoc_v = 0.0f;"
        update_line = "                nanoc_v += expf(nanoc_x);"
        finish_lines = ["            nanoc_v = logf(nanoc_v);"]
    elif kind == "reducesumsquare":
        init_line = "            float nanoc_v = 0.0f;"
        update_line = "                nanoc_v += nanoc_x * nanoc_x;"
        finish_lines = []
    else:
        init_line = "            float nanoc_v = 0.0f;"
        update_line = "                nanoc_v += nanoc_x;"
        finish_lines = (
            [f"            nanoc_v = nanoc_v / {_c_float_literal(float(cols))};"]
            if kind == "reducemean"
            else []
        )
    return [
        f"    /* node {layer['index']}: {layer['name']} -> {api} */",
        "    {",
        f"        for (size_t nanoc_row = 0u; nanoc_row < {rows}u; ++nanoc_row) {{",
        init_line,
        f"            for (size_t nanoc_col = 0u; nanoc_col < {cols}u; ++nanoc_col) {{",
        (
            f"                float nanoc_x = ((float){input_expr}[nanoc_row * {cols}u + nanoc_col] - "
            f"{_c_float_literal(float(layer['input_zero_point']))}) * "
            f"{_c_float_literal(float(layer['input_scale']))};"
        ),
        update_line,
        "            }",
        *finish_lines,
        (
            f"            float nanoc_qf = nanoc_v / {_c_float_literal(float(layer['output_scale']))} + "
            f"{_c_float_literal(float(layer['output_zero_point']))};"
        ),
        "            int32_t nanoc_q = (int32_t)nearbyintf(nanoc_qf);",
        f"            if (nanoc_q > {int(layer['activation_max'])}) {{ nanoc_q = {int(layer['activation_max'])}; }}",
        f"            if (nanoc_q < {int(layer['activation_min'])}) {{ nanoc_q = {int(layer['activation_min'])}; }}",
        f"            {current_output}[nanoc_row] = (int8_t)nanoc_q;",
        "        }",
        "    }",
        "",
    ]


def _generated_unsqueeze_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_expr = layer.get("input_expr", "current_input")
    return [
        f"    /* node {layer['index']}: {layer['name']} -> generated_c_unsqueeze_s8 */",
        "    {",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer['block_size'])}u; ++nanoc_i) {{",
        f"            {current_output}[nanoc_i] = {input_expr}[nanoc_i];",
        "        }",
        "    }",
        "",
    ]


def _generated_pad_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_expr = layer.get("input_expr", "current_input")
    input_shape = [int(item) for item in layer["input_shape"]]
    output_shape = [int(item) for item in layer["output_shape"]]
    pads = [int(item) for item in layer["pads"]]
    rank = len(input_shape)
    return [
        f"    /* node {layer['index']}: {layer['name']} -> generated_c_pad_s8 */",
        "    {",
        f"        const int nanoc_input_shape[{rank}] = {{{_c_int_list(input_shape)}}};",
        f"        const int nanoc_output_shape[{rank}] = {{{_c_int_list(output_shape)}}};",
        f"        const int nanoc_pads[{rank * 2}] = {{{_c_int_list(pads)}}};",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer['block_size'])}u; ++nanoc_i) {{",
        f"            {current_output}[nanoc_i] = (int8_t){int(layer['constant_q'])};",
        "        }",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {_shape_size(input_shape)}u; ++nanoc_i) {{",
        "            int nanoc_coords[8] = {0};",
        "            int nanoc_out_coords[8] = {0};",
        f"            nanoc_unravel_index(nanoc_i, {rank}, nanoc_input_shape, nanoc_coords);",
        f"            for (int nanoc_axis = 0; nanoc_axis < {rank}; ++nanoc_axis) {{",
        "                nanoc_out_coords[nanoc_axis] = nanoc_coords[nanoc_axis] + nanoc_pads[nanoc_axis];",
        "            }",
        f"            size_t nanoc_out_i = nanoc_ravel_index({rank}, nanoc_output_shape, nanoc_out_coords);",
        f"            {current_output}[nanoc_out_i] = {input_expr}[nanoc_i];",
        "        }",
        "    }",
        "",
    ]


def _generated_slice_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_expr = layer.get("input_expr", "current_input")
    input_shape = [int(item) for item in layer["input_shape"]]
    output_shape = [int(item) for item in layer["output_shape"]]
    starts = [int(item) for item in layer["starts"]]
    axes = [int(item) for item in layer["axes"]]
    steps = [int(item) for item in layer["steps"]]
    rank = len(input_shape)
    param_count = len(starts)
    return [
        f"    /* node {layer['index']}: {layer['name']} -> generated_c_slice_s8 */",
        "    {",
        f"        const int nanoc_input_shape[{rank}] = {{{_c_int_list(input_shape)}}};",
        f"        const int nanoc_output_shape[{rank}] = {{{_c_int_list(output_shape)}}};",
        f"        const int nanoc_starts[{param_count}] = {{{_c_int_list(starts)}}};",
        f"        const int nanoc_axes[{param_count}] = {{{_c_int_list(axes)}}};",
        f"        const int nanoc_steps[{param_count}] = {{{_c_int_list(steps)}}};",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer['block_size'])}u; ++nanoc_i) {{",
        "            int nanoc_out_coords[8] = {0};",
        "            int nanoc_in_coords[8] = {0};",
        f"            nanoc_unravel_index(nanoc_i, {rank}, nanoc_output_shape, nanoc_out_coords);",
        f"            for (int nanoc_axis = 0; nanoc_axis < {rank}; ++nanoc_axis) {{",
        "                nanoc_in_coords[nanoc_axis] = nanoc_out_coords[nanoc_axis];",
        "            }",
        f"            for (int nanoc_p = 0; nanoc_p < {param_count}; ++nanoc_p) {{",
        "                int nanoc_axis = nanoc_axes[nanoc_p];",
        "                nanoc_in_coords[nanoc_axis] = nanoc_starts[nanoc_p] + nanoc_out_coords[nanoc_axis] * nanoc_steps[nanoc_p];",
        "            }",
        f"            size_t nanoc_in_i = nanoc_ravel_index({rank}, nanoc_input_shape, nanoc_in_coords);",
        f"            {current_output}[nanoc_i] = {input_expr}[nanoc_in_i];",
        "        }",
        "    }",
        "",
    ]


def _generated_gather_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_expr = layer.get("input_expr", "current_input")
    input_shape = [int(item) for item in layer["input_shape"]]
    output_shape = [int(item) for item in layer["output_shape"]]
    indices = [int(item) for item in layer["indices"]]
    rank = len(input_shape)
    axis = int(layer["axis"])
    return [
        f"    /* node {layer['index']}: {layer['name']} -> generated_c_gather_s8 */",
        "    {",
        f"        const int nanoc_input_shape[{rank}] = {{{_c_int_list(input_shape)}}};",
        f"        const int nanoc_output_shape[{rank}] = {{{_c_int_list(output_shape)}}};",
        f"        const int nanoc_indices[{len(indices)}] = {{{_c_int_list(indices)}}};",
        f"        for (size_t nanoc_i = 0u; nanoc_i < {int(layer['block_size'])}u; ++nanoc_i) {{",
        "            int nanoc_out_coords[8] = {0};",
        "            int nanoc_in_coords[8] = {0};",
        f"            nanoc_unravel_index(nanoc_i, {rank}, nanoc_output_shape, nanoc_out_coords);",
        f"            for (int nanoc_axis = 0; nanoc_axis < {rank}; ++nanoc_axis) {{",
        "                nanoc_in_coords[nanoc_axis] = nanoc_out_coords[nanoc_axis];",
        "            }",
        f"            nanoc_in_coords[{axis}] = nanoc_indices[nanoc_out_coords[{axis}]];",
        f"            size_t nanoc_in_i = nanoc_ravel_index({rank}, nanoc_input_shape, nanoc_in_coords);",
        f"            {current_output}[nanoc_i] = {input_expr}[nanoc_in_i];",
        "        }",
        "    }",
        "",
    ]


def _cmsis_transpose_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_expr = layer.get("input_expr", "current_input")
    input_n, input_h, input_w, input_c = layer["input_dims"]
    output_n, output_h, output_w, output_c = layer["output_dims"]
    perm_values = ", ".join(f"{int(item)}u" for item in layer["perm"])
    prefix = layer["symbol"]
    return [
        f"    /* node {layer['index']}: {layer['name']} -> arm_transpose_s8 */",
        "    {",
        f"        const uint32_t {prefix}_perm[{len(layer['perm'])}] = {{{perm_values}}};",
        f"        const cmsis_nn_dims input_dims = {{{input_n}, {input_h}, {input_w}, {input_c}}};",
        f"        const cmsis_nn_dims output_dims = {{{output_n}, {output_h}, {output_w}, {output_c}}};",
        f"        const cmsis_nn_transpose_params transpose_params = {{{len(layer['perm'])}, {prefix}_perm}};",
        "        cmsis_status = arm_transpose_s8(",
        f"            {input_expr},",
        f"            {current_output},",
        "            &input_dims,",
        "            &output_dims,",
        "            &transpose_params);",
        "        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {",
        "            return NANOC_STATUS_BLOCKED;",
        "        }",
        "    }",
        "",
    ]


def _cmsis_concat_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_exprs = layer.get("input_exprs", [])
    input_dims = layer["input_dims"]
    output_n, output_h, output_w, output_c = layer["output_dims"]
    axis = int(layer["axis"])
    if layer.get("layout") == "NHWC":
        cmsis_axis = {0: "w", 1: "y", 2: "x", 3: "z"}.get(axis)
    else:
        cmsis_axis = {0: "w", 1: "z", 2: "y", 3: "x"}.get(axis)
    if cmsis_axis is None:
        return []
    lines = [
        f"    /* node {layer['index']}: {layer['name']} -> arm_concatenation_s8_{cmsis_axis} */",
        "    {",
        "        uint32_t concat_offset = 0u;",
    ]
    requantize_inputs = layer.get("input_requantize", [])
    for input_index, dims in enumerate(input_dims):
        input_n, input_h, input_w, input_c = dims
        input_expr = input_exprs[input_index] if input_index < len(input_exprs) else "current_input"
        requant = (
            requantize_inputs[input_index]
            if input_index < len(requantize_inputs) and isinstance(requantize_inputs[input_index], dict)
            else {}
        )
        if requant.get("required"):
            temp_symbol = requant.get("symbol", f"nanoc_concat_requant_{layer['index']}_{input_index}")
            input_count = int(input_n) * int(input_h) * int(input_w) * int(input_c)
            lines.extend(
                [
                    f"        for (size_t nanoc_i = 0u; nanoc_i < {input_count}u; ++nanoc_i) {{",
                    (
                        f"            int32_t nanoc_v = (int32_t){input_expr}[nanoc_i] - "
                        f"{int(requant['input_zero_point'])};"
                    ),
                    (
                        f"            nanoc_v = arm_nn_requantize(nanoc_v, "
                        f"{int(requant['multiplier'])}, {int(requant['shift'])}) + "
                        f"{int(requant['output_zero_point'])};"
                    ),
                    "            if (nanoc_v > 127) { nanoc_v = 127; }",
                    "            if (nanoc_v < -128) { nanoc_v = -128; }",
                    f"            {temp_symbol}[nanoc_i] = (int8_t)nanoc_v;",
                    "        }",
                ]
            )
            input_expr = str(temp_symbol)
        if cmsis_axis == "z":
            lines.extend(
                [
                    "        /* NHWC channel concat. arm_concatenation_s8_z copies channel-contiguous blocks. */",
                    f"        for (uint32_t n = 0u; n < {input_n}u; ++n) {{",
                    f"            for (uint32_t y = 0u; y < {input_h}u; ++y) {{",
                    f"                for (uint32_t x = 0u; x < {input_w}u; ++x) {{",
                    f"                    for (uint32_t z = 0u; z < {input_c}u; ++z) {{",
                    (
                        f"                        size_t src = ((size_t)n * {input_h * input_w * input_c}u) + "
                        f"((size_t)y * {input_w * input_c}u) + ((size_t)x * {input_c}u) + z;"
                    ),
                    (
                        f"                        size_t dst = ((size_t)n * {input_h * input_w * output_c}u) + "
                        f"((size_t)y * {input_w * output_c}u) + ((size_t)x * {output_c}u) + concat_offset + z;"
                    ),
                    f"                        {current_output}[dst] = {input_expr}[src];",
                    "                    }",
                    "                }",
                    "            }",
                    "        }",
                ]
            )
            lines.append(f"        concat_offset += {input_c}u;")
            continue
        axis_extent = {
            "x": input_w,
            "y": input_h,
            "z": input_c,
            "w": input_n,
        }[cmsis_axis]
        if cmsis_axis == "w":
            lines.extend(
                [
                    f"        arm_concatenation_s8_w({input_expr},",
                    f"                                 {input_w}, {input_h}, {input_c}, {input_n},",
                    f"                                 {current_output},",
                    "                                 concat_offset);",
                ]
            )
        else:
            output_extent = {
                "x": output_w,
                "y": output_h,
                "z": output_c,
            }[cmsis_axis]
            lines.extend(
                [
                    f"        arm_concatenation_s8_{cmsis_axis}({input_expr},",
                    f"                                 {input_w}, {input_h}, {input_c}, {input_n},",
                    f"                                 {current_output},",
                    f"                                 {output_extent},",
                    "                                 concat_offset);",
                ]
            )
        lines.append(f"        concat_offset += {axis_extent}u;")
    lines.extend(["    }", ""])
    return lines


def _cmsis_softmax_call(layer: dict[str, Any], current_output: str) -> list[str]:
    input_expr = layer.get("input_expr", "current_input")
    if layer.get("stable_softmax"):
        return [
            f"    /* node {layer['index']}: {layer['name']} -> stable s8 softmax fallback for large row_size; arm_softmax_s8 is unsafe here */",
            "    {",
            "        nanoc_softmax_s8_stable(",
            f"            {input_expr},",
            f"            {layer['num_rows']},",
            f"            {layer['row_size']},",
            f"            {_c_float_literal(float(layer['input_scale']))},",
            f"            {current_output});",
            "    }",
            "",
        ]
    return [
        f"    /* node {layer['index']}: {layer['name']} -> arm_softmax_s8 */",
        "    {",
        "        arm_softmax_s8(",
        f"            {input_expr},",
        f"            {layer['num_rows']},",
        f"            {layer['row_size']},",
        f"            {layer['multiplier']},",
        f"            {layer['shift']},",
        f"            {layer['diff_min']},",
        f"            {current_output});",
        "    }",
        "",
    ]


def _quantized_weight_declarations(graph: ModelGraph) -> list[str]:
    layers = _runtime_layers(graph, _synthetic_runtime_mappings(graph))
    if not layers:
        return []
    lines = ["/* Quantized CMSIS-NN runtime weights. */"]
    for layer in layers:
        symbol = layer["symbol"]
        if "condition_values" in layer:
            condition_values = _int_values(layer["condition_values"])
            condition_symbol = str(layer.get("condition_symbol", f"{symbol}_condition"))
            lines.extend(
                [
                    f"/* ONNX node: {layer['name']} static bool condition */",
                    f"#define {condition_symbol.upper()}_SIZE {len(condition_values)}u",
                    f"static const int8_t {condition_symbol}[{len(condition_values)}] = {{",
                    *_format_int_array(condition_values),
                    "};",
                    "",
                ]
            )
        if "weight_values" not in layer:
            continue
        weight_values = _int_values(layer["weight_values"])
        if layer["kind"] in {
            "add",
            "mul",
            "sub",
            "div",
            "min",
            "max",
            "pow",
            "prelu",
            "equal",
            "greater",
            "less",
            "greaterorequal",
            "lessorequal",
            "where",
        }:
            lines.extend(
                [
                    f"/* ONNX node: {layer['name']} constant input */",
                    f"#define {symbol.upper()}_CONST_SIZE {len(weight_values)}u",
                    f"static const int8_t {symbol}_weights[{len(weight_values)}] = {{",
                    *_format_int_array(weight_values),
                    "};",
                    "",
                ]
            )
            continue
        bias_values = _int_values(layer["bias_values"] or [0] * layer["output_channels"])
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
            ]
        )
        if layer["kind"] in {"conv", "depthwise"} or isinstance(layer.get("multiplier"), list):
            multiplier_values = _int_values(layer["multiplier"])
            shift_values = _int_values(layer["shift"])
            lines.extend(
                [
                    f"static const int32_t {symbol}_multiplier[{len(multiplier_values)}] = {{",
                    *_format_int_array(multiplier_values),
                    "};",
                    f"static const int32_t {symbol}_shift[{len(shift_values)}] = {{",
                    *_format_int_array(shift_values),
                    "};",
                ]
            )
        lines.append("")
    return lines


def _synthetic_runtime_mappings(graph: ModelGraph) -> list[OpMapping]:
    mappings: list[OpMapping] = []
    for node in graph.nodes:
        if node.op_type not in {
            "Add",
            "And",
            "Ceil",
            "Conv",
            "Concat",
            "Equal",
            "Elu",
            "Erf",
            "Exp",
            "Floor",
            "Gemm",
            "Gather",
            "Greater",
            "GreaterOrEqual",
            "HardSigmoid",
            "HardSwish",
            "Less",
            "LessOrEqual",
            "Log",
            "MatMul",
            "Max",
            "Min",
            "Mul",
            "Clip",
            "Celu",
            "LeakyRelu",
            "Neg",
            "Not",
            "Or",
            "Xor",
            "PRelu",
            "Pad",
            "Pow",
            "QLinearAdd",
            "QLinearConv",
            "QLinearMatMul",
            "Reciprocal",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceMax",
            "ReduceMean",
            "ReduceMin",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
            "Round",
            "Selu",
            "Sign",
            "Sigmoid",
            "Slice",
            "Softplus",
            "Softsign",
            "Sqrt",
            "Sub",
            "Div",
            "Tanh",
            "ThresholdedRelu",
            "Unsqueeze",
            "Where",
        }:
            continue
        if node.op_type in {"Conv", "QLinearConv"}:
            action = (
                "arm_depthwise_conv_wrapper_s8"
                if _is_depthwise_conv_node(node)
                else "arm_convolve_wrapper_s8"
            )
        elif node.op_type in {"Add", "QLinearAdd"}:
            action = "arm_elementwise_add_s8"
        elif node.op_type == "Mul":
            action = "arm_elementwise_mul_s8"
        elif node.op_type in {"Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "And", "Or", "Not", "Xor"}:
            action = f"generated_c_{node.op_type.lower()}_bool"
        elif node.op_type in {
            "Sub",
            "Div",
            "Sigmoid",
            "Tanh",
            "LeakyRelu",
            "Clip",
            "Neg",
            "Sqrt",
            "Reciprocal",
            "Exp",
            "Log",
            "Floor",
            "Ceil",
            "Round",
            "Sign",
            "Erf",
            "Softplus",
            "Softsign",
            "HardSwish",
            "Elu",
            "Selu",
            "HardSigmoid",
            "ThresholdedRelu",
            "Celu",
            "PRelu",
            "Min",
            "Max",
            "Pow",
            "ReduceMax",
            "ReduceMean",
            "ReduceMin",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
            "Where",
            "Unsqueeze",
            "Pad",
            "Slice",
            "Gather",
        }:
            action = f"generated_c_{node.op_type.lower()}_s8"
        else:
            action = "arm_fully_connected_s8"
        mappings.append(
            OpMapping(
                index=node.index,
                node_name=node.name,
                onnx_op=node.op_type,
                converter_status=node.converter_status,
                status="wrapper_api",
                cmsis_action=action,
                reason="quantized weight declaration",
                inputs=node.inputs,
                outputs=node.outputs,
                input_shapes=node.input_shapes,
                output_shapes=node.output_shapes,
            )
        )
    return mappings


def _runtime_layers(graph: ModelGraph, mappings: list[OpMapping]) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for mapping in mappings:
        if mapping.status not in {"wrapper_api", "direct_api"}:
            continue
        if mapping.onnx_op in {"Conv", "QLinearConv"}:
            if "depthwise" in mapping.cmsis_action:
                layer = _depthwise_layer(graph, mapping)
            else:
                layer = _conv_layer(graph, mapping)
        elif mapping.onnx_op in {
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
            "QLinearGlobalAveragePool",
        }:
            layer = _pool_layer(graph, mapping)
        elif mapping.onnx_op == "Softmax":
            layer = _softmax_layer(graph, mapping)
        elif mapping.onnx_op == "Abs":
            layer = _abs_layer(graph, mapping)
        elif mapping.onnx_op in {"Gemm", "MatMul", "QLinearMatMul"}:
            layer = _fc_layer(graph, mapping)
        elif mapping.onnx_op in {"Add", "QLinearAdd"}:
            layer = _add_layer(graph, mapping)
        elif mapping.onnx_op == "Mul":
            layer = _mul_layer(graph, mapping)
        elif mapping.onnx_op in {"Sub", "Div", "Min", "Max", "Pow", "PRelu"}:
            layer = _generated_binary_layer(graph, mapping)
        elif mapping.onnx_op in {"Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"}:
            layer = _generated_compare_layer(graph, mapping)
        elif mapping.onnx_op in {"And", "Or", "Not", "Xor"}:
            layer = _bool_logic_layer(graph, mapping)
        elif mapping.onnx_op == "Where":
            layer = _where_layer(graph, mapping)
        elif mapping.onnx_op in {
            "Sigmoid",
            "Tanh",
            "LeakyRelu",
            "Clip",
            "Neg",
            "Sqrt",
            "Reciprocal",
            "Exp",
            "Log",
            "Floor",
            "Ceil",
            "Round",
            "Sign",
            "Erf",
            "Softplus",
            "Softsign",
            "HardSwish",
            "Elu",
            "Selu",
            "HardSigmoid",
            "ThresholdedRelu",
            "Celu",
        }:
            layer = _generated_unary_layer(graph, mapping)
        elif mapping.onnx_op in {
            "ReduceMean",
            "ReduceSum",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceSumSquare",
        }:
            layer = _reduce_mean_layer(graph, mapping)
        elif mapping.onnx_op == "Unsqueeze":
            layer = _unsqueeze_layer(graph, mapping)
        elif mapping.onnx_op == "Pad":
            layer = _pad_layer(graph, mapping)
        elif mapping.onnx_op == "Slice":
            layer = _slice_layer(graph, mapping)
        elif mapping.onnx_op == "Gather":
            layer = _gather_layer(graph, mapping)
        elif mapping.onnx_op == "Transpose":
            layer = _transpose_layer(graph, mapping)
        elif mapping.onnx_op == "Concat":
            layer = _concat_layer(graph, mapping)
        else:
            layer = None
        if layer is not None:
            layers.append(layer)
    return layers


def _fc_layers(graph: ModelGraph, mappings: list[OpMapping]) -> list[dict[str, Any]]:
    return [layer for layer in (_fc_layer(graph, mapping) for mapping in mappings) if layer]


def _fc_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quantization = graph.quantization or {}
    node_quant = quantization.get("nodes", {})
    weights = quantization.get("weights", {})
    if not isinstance(node_quant, dict) or not isinstance(weights, dict):
        return None
    node_by_name = {node.name: node for node in graph.nodes}
    node = node_by_name.get(mapping.node_name)
    quant = node_quant.get(mapping.node_name)
    if node is None or not isinstance(quant, dict):
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    quant_weights = quant.get("weights", {})
    if not isinstance(cmsis_nn, dict) or not isinstance(quant_weights, dict):
        return None
    if any(field not in cmsis_nn for field in _CMSIS_FC_REQUIRED_FIELDS):
        return None
    weight_name = str(quant_weights.get("weight", ""))
    weight_info = weights.get(weight_name, {})
    if not isinstance(weight_info, dict):
        return None
    weight_shape = weight_info.get("shape", [])
    if not isinstance(weight_shape, list) or len(weight_shape) != 2:
        return None
    output_size, input_size = int(weight_shape[0]), int(weight_shape[1])
    values = weight_info.get("values", [])
    if not isinstance(values, list) or len(values) != output_size * input_size:
        return None
    return {
        "kind": "fc",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]] if node.inputs else [],
        "output_tensors": list(node.outputs),
        "input_size": input_size,
        "output_size": output_size,
        "output_channels": output_size,
        "weight_values": values,
        "bias_values": quant_weights.get("bias_values", []),
        "input_offset": int(cmsis_nn["input_offset"]),
        "filter_offset": int(cmsis_nn["filter_offset"]),
        "output_offset": int(cmsis_nn["output_offset"]),
        "multiplier": cmsis_nn["multiplier"],
        "shift": cmsis_nn["shift"],
        "activation_min": int(cmsis_nn["activation_min"]),
        "activation_max": int(cmsis_nn["activation_max"]),
    }


def _conv_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quantization = graph.quantization or {}
    node_quant = quantization.get("nodes", {})
    weights = quantization.get("weights", {})
    if not isinstance(node_quant, dict) or not isinstance(weights, dict):
        return None
    node_by_name = {node.name: node for node in graph.nodes}
    node = node_by_name.get(mapping.node_name)
    quant = node_quant.get(mapping.node_name)
    if node is None or not isinstance(quant, dict):
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    quant_weights = quant.get("weights", {})
    if not isinstance(cmsis_nn, dict) or not isinstance(quant_weights, dict):
        return None
    if any(field not in cmsis_nn for field in _CMSIS_CONV_REQUIRED_FIELDS):
        return None
    if int(cmsis_nn.get("groups", 1)) != 1:
        return None
    if _pair(cmsis_nn.get("dilation", [1, 1]), default=[1, 1]) != [1, 1]:
        return None
    weight_name = str(quant_weights.get("weight", ""))
    weight_info = weights.get(weight_name, {})
    if not isinstance(weight_info, dict):
        return None
    weight_shape = weight_info.get("shape", [])
    values = weight_info.get("values", [])
    if not isinstance(weight_shape, list) or len(weight_shape) not in {3, 4}:
        return None
    if len(weight_shape) == 3:
        output_channels, input_channels, kernel_w = [int(item) for item in weight_shape]
        kernel_h = 1
    else:
        output_channels, input_channels, kernel_h, kernel_w = [
            int(item) for item in weight_shape
        ]
    if (
        not isinstance(values, list)
        or len(values) != output_channels * input_channels * kernel_h * kernel_w
    ):
        return None
    input_shape = _first_shape(node.input_shapes, node.inputs)
    output_shape = _first_shape(node.output_shapes, node.outputs)
    input_dims = _activation_dims(input_shape, graph.layout)
    output_dims = _activation_dims(output_shape, graph.layout)
    if input_dims is None or output_dims is None:
        return None
    activation_min = int(cmsis_nn["activation_min"])
    if node.outputs and _output_flows_through_relu(graph, str(node.outputs[0])):
        activation_min = max(activation_min, int(cmsis_nn["output_offset"]))
    return {
        "kind": "conv",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]] if node.inputs else [],
        "output_tensors": list(node.outputs),
        "input_dims": input_dims,
        "output_dims": output_dims,
        "kernel_shape": [kernel_h, kernel_w],
        "output_channels": output_channels,
        "weight_values": _reorder_conv_weight_to_ohwi(_int_values(values), weight_shape),
        "bias_values": quant_weights.get("bias_values", []),
        "multiplier": cmsis_nn["multiplier"],
        "shift": cmsis_nn["shift"],
        "input_offset": int(cmsis_nn["input_offset"]),
        "output_offset": int(cmsis_nn["output_offset"]),
        "activation_min": activation_min,
        "activation_max": int(cmsis_nn["activation_max"]),
        "stride": _pair(cmsis_nn.get("stride", [1, 1]), default=[1, 1]),
        "padding": _pair(cmsis_nn.get("padding", [0, 0]), default=[0, 0]),
        "dilation": _pair(cmsis_nn.get("dilation", [1, 1]), default=[1, 1]),
    }


def _depthwise_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quantization = graph.quantization or {}
    node_quant = quantization.get("nodes", {})
    weights = quantization.get("weights", {})
    if not isinstance(node_quant, dict) or not isinstance(weights, dict):
        return None
    node = _node_by_name(graph, mapping.node_name)
    quant = node_quant.get(mapping.node_name)
    if node is None or not isinstance(quant, dict):
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    quant_weights = quant.get("weights", {})
    if not isinstance(cmsis_nn, dict) or not isinstance(quant_weights, dict):
        return None
    if any(field not in cmsis_nn for field in _CMSIS_CONV_REQUIRED_FIELDS):
        return None
    if _pair(cmsis_nn.get("dilation", [1, 1]), default=[1, 1]) != [1, 1]:
        return None
    weight_name = str(quant_weights.get("weight", ""))
    weight_info = weights.get(weight_name, {})
    if not isinstance(weight_info, dict):
        return None
    weight_shape = weight_info.get("shape", [])
    values = weight_info.get("values", [])
    if not isinstance(weight_shape, list) or len(weight_shape) != 4:
        return None
    output_channels, in_per_group, kernel_h, kernel_w = [int(item) for item in weight_shape]
    group = int(cmsis_nn.get("groups", 1))
    if group <= 1 or in_per_group != 1 or output_channels % group != 0:
        return None
    input_shape = _first_shape(node.input_shapes, node.inputs)
    output_shape = _first_shape(node.output_shapes, node.outputs)
    input_dims = _activation_dims(input_shape, graph.layout)
    output_dims = _activation_dims(output_shape, graph.layout)
    if input_dims is None or output_dims is None:
        return None
    input_channels = int(input_dims[3])
    if input_channels != group:
        return None
    if (
        not isinstance(values, list)
        or len(values) != output_channels * in_per_group * kernel_h * kernel_w
    ):
        return None
    activation_min = int(cmsis_nn["activation_min"])
    if node.outputs and _output_flows_through_relu(graph, str(node.outputs[0])):
        activation_min = max(activation_min, int(cmsis_nn["output_offset"]))
    return {
        "kind": "depthwise",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]] if node.inputs else [],
        "output_tensors": list(node.outputs),
        "input_dims": input_dims,
        "output_dims": output_dims,
        "kernel_shape": [kernel_h, kernel_w],
        "output_channels": output_channels,
        "channel_multiplier": output_channels // input_channels,
        "weight_values": _reorder_depthwise_oihw_to_1hwo(_int_values(values), weight_shape),
        "bias_values": quant_weights.get("bias_values", []),
        "multiplier": cmsis_nn["multiplier"],
        "shift": cmsis_nn["shift"],
        "input_offset": int(cmsis_nn["input_offset"]),
        "output_offset": int(cmsis_nn["output_offset"]),
        "activation_min": activation_min,
        "activation_max": int(cmsis_nn["activation_max"]),
        "stride": _pair(cmsis_nn.get("stride", [1, 1]), default=[1, 1]),
        "padding": _pair(cmsis_nn.get("padding", [0, 0]), default=[0, 0]),
        "dilation": _pair(cmsis_nn.get("dilation", [1, 1]), default=[1, 1]),
    }


def _pool_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    if any(field not in cmsis_nn for field in _CMSIS_POOL_REQUIRED_FIELDS):
        return None
    input_shape = _first_shape(node.input_shapes, node.inputs)
    output_shape = _first_shape(node.output_shapes, node.outputs)
    input_dims = _activation_dims(input_shape, graph.layout)
    output_dims = _activation_dims(output_shape, graph.layout)
    if input_dims is None or output_dims is None:
        return None
    return {
        "kind": "pool",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]] if node.inputs else [],
        "output_tensors": list(node.outputs),
        "api": str(cmsis_nn["api"]),
        "input_dims": input_dims,
        "output_dims": output_dims,
        "kernel_shape": _pair(cmsis_nn.get("kernel_shape", [1, 1]), default=[1, 1]),
        "stride": _pair(cmsis_nn.get("stride", [1, 1]), default=[1, 1]),
        "padding": _pair(cmsis_nn.get("padding", [0, 0]), default=[0, 0]),
        "activation_min": int(cmsis_nn["activation_min"]),
        "activation_max": int(cmsis_nn["activation_max"]),
        "requantize_output": bool(cmsis_nn.get("requantize_output", False)),
        "requantize_multiplier": int(cmsis_nn.get("requantize_multiplier", 0)),
        "requantize_shift": int(cmsis_nn.get("requantize_shift", 0)),
        "requantize_input_zero_point": int(cmsis_nn.get("requantize_input_zero_point", 0)),
        "requantize_output_zero_point": int(cmsis_nn.get("requantize_output_zero_point", 0)),
        "output_elements": element_count_from_shape(output_shape) or 0,
    }


def _add_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    if any(field not in cmsis_nn for field in _CMSIS_ADD_REQUIRED_FIELDS):
        return None
    output_shape = _first_shape(node.output_shapes, node.outputs)
    block_size = element_count_from_shape(output_shape or [])
    if block_size is None:
        block_size = int(cmsis_nn.get("block_size", 0))
    if block_size <= 0:
        return None
    weights = graph.quantization.get("weights", {}) if graph.quantization else {}
    quant_weights = quant.get("weights", {})
    weight_name = str(quant_weights.get("weight", "")) if isinstance(quant_weights, dict) else ""
    weight_info = weights.get(weight_name, {}) if isinstance(weights, dict) else {}
    values = weight_info.get("values", []) if isinstance(weight_info, dict) else []
    symbol = _c_symbol(f"nanoc_{mapping.node_name}")
    layer: dict[str, Any] = {
        "kind": "add",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": symbol,
        "input_tensors": [node.inputs[0], node.inputs[3]]
        if mapping.onnx_op == "QLinearAdd" and len(node.inputs) > 3
        else list(node.inputs[:2]),
        "output_tensors": list(node.outputs),
        "input_1_offset": int(cmsis_nn["input_1_offset"]),
        "input_1_multiplier": int(cmsis_nn["input_1_multiplier"]),
        "input_1_shift": int(cmsis_nn["input_1_shift"]),
        "input_2_offset": int(cmsis_nn["input_2_offset"]),
        "input_2_multiplier": int(cmsis_nn["input_2_multiplier"]),
        "input_2_shift": int(cmsis_nn["input_2_shift"]),
        "left_shift": int(cmsis_nn["left_shift"]),
        "output_offset": int(cmsis_nn["output_offset"]),
        "output_multiplier": int(cmsis_nn["output_multiplier"]),
        "output_shift": int(cmsis_nn["output_shift"]),
        "activation_min": int(cmsis_nn["activation_min"]),
        "activation_max": int(cmsis_nn["activation_max"]),
        "block_size": int(block_size),
    }
    if isinstance(values, list) and values:
        layer["weight_values"] = values
        layer["constant_symbol"] = f"{symbol}_weights"
        # Mark which input position holds the constant (always the second
        # runtime input for QLinearAdd: [activation, weight]).
        layer["constant_input_index"] = 1
    return layer


def _mul_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    if any(field not in cmsis_nn for field in _CMSIS_MUL_REQUIRED_FIELDS):
        return None
    output_shape = _first_shape(node.output_shapes, node.outputs)
    block_size = element_count_from_shape(output_shape or [])
    if block_size is None:
        block_size = int(cmsis_nn.get("block_size", 0))
    if block_size <= 0:
        return None
    weights = graph.quantization.get("weights", {}) if graph.quantization else {}
    quant_weights = quant.get("weights", {})
    weight_name = str(quant_weights.get("weight", "")) if isinstance(quant_weights, dict) else ""
    weight_info = weights.get(weight_name, {}) if isinstance(weights, dict) else {}
    values = weight_info.get("values", []) if isinstance(weight_info, dict) else []
    symbol = _c_symbol(f"nanoc_{mapping.node_name}")
    layer: dict[str, Any] = {
        "kind": "mul",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": symbol,
        "input_tensors": list(node.inputs[:2]),
        "output_tensors": list(node.outputs),
        "input_1_offset": int(cmsis_nn["input_1_offset"]),
        "input_2_offset": int(cmsis_nn["input_2_offset"]),
        "output_offset": int(cmsis_nn["output_offset"]),
        "output_multiplier": int(cmsis_nn["output_multiplier"]),
        "output_shift": int(cmsis_nn["output_shift"]),
        "activation_min": int(cmsis_nn["activation_min"]),
        "activation_max": int(cmsis_nn["activation_max"]),
        "block_size": int(block_size),
    }
    if isinstance(values, list) and values:
        layer["weight_values"] = values
        layer["constant_symbol"] = f"{symbol}_weights"
        layer["constant_input_index"] = 1
    return layer


def _generated_binary_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    required = (
        "input_1_scale",
        "input_1_zero_point",
        "input_2_scale",
        "input_2_zero_point",
        "output_scale",
        "output_zero_point",
        "activation_min",
        "activation_max",
        "block_size",
    )
    if any(field not in cmsis_nn for field in required):
        return None
    output_shape = _first_shape(node.output_shapes, node.outputs)
    block_size = element_count_from_shape(output_shape or [])
    if block_size is None:
        block_size = int(cmsis_nn.get("block_size", 0))
    if block_size <= 0:
        return None
    weights = graph.quantization.get("weights", {}) if graph.quantization else {}
    quant_weights = quant.get("weights", {})
    weight_name = str(quant_weights.get("weight", "")) if isinstance(quant_weights, dict) else ""
    weight_info = weights.get(weight_name, {}) if isinstance(weights, dict) else {}
    values = weight_info.get("values", []) if isinstance(weight_info, dict) else []
    kind = str(mapping.onnx_op).lower()
    symbol = _c_symbol(f"nanoc_{mapping.node_name}")
    layer: dict[str, Any] = {
        "kind": kind,
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": symbol,
        "input_tensors": list(node.inputs[:2]),
        "output_tensors": list(node.outputs),
        "input_1_scale": float(cmsis_nn["input_1_scale"]),
        "input_1_zero_point": int(cmsis_nn["input_1_zero_point"]),
        "input_2_scale": float(cmsis_nn["input_2_scale"]),
        "input_2_zero_point": int(cmsis_nn["input_2_zero_point"]),
        "output_scale": float(cmsis_nn["output_scale"]),
        "output_zero_point": int(cmsis_nn["output_zero_point"]),
        "activation_min": int(cmsis_nn["activation_min"]),
        "activation_max": int(cmsis_nn["activation_max"]),
        "block_size": int(block_size),
    }
    if isinstance(values, list) and values:
        layer["weight_values"] = values
        layer["constant_symbol"] = f"{symbol}_weights"
        layer["constant_input_index"] = 1
    return layer


def _generated_compare_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    required = (
        "input_1_scale",
        "input_1_zero_point",
        "input_2_scale",
        "input_2_zero_point",
        "block_size",
    )
    if any(field not in cmsis_nn for field in required):
        return None
    weights = graph.quantization.get("weights", {}) if graph.quantization else {}
    quant_weights = quant.get("weights", {})
    weight_name = str(quant_weights.get("weight", "")) if isinstance(quant_weights, dict) else ""
    weight_info = weights.get(weight_name, {}) if isinstance(weights, dict) else {}
    values = weight_info.get("values", []) if isinstance(weight_info, dict) else []
    kind = str(mapping.onnx_op).lower()
    symbol = _c_symbol(f"nanoc_{mapping.node_name}")
    layer: dict[str, Any] = {
        "kind": kind,
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": symbol,
        "input_tensors": list(node.inputs[:2]),
        "output_tensors": list(node.outputs),
        "input_1_scale": float(cmsis_nn["input_1_scale"]),
        "input_1_zero_point": int(cmsis_nn["input_1_zero_point"]),
        "input_2_scale": float(cmsis_nn["input_2_scale"]),
        "input_2_zero_point": int(cmsis_nn["input_2_zero_point"]),
        "block_size": int(cmsis_nn["block_size"]),
    }
    if isinstance(values, list) and values:
        layer["weight_values"] = values
        layer["constant_symbol"] = f"{symbol}_weights"
        layer["constant_input_index"] = 1
    return layer


def _bool_logic_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    required = ("api", "block_size")
    if any(field not in cmsis_nn for field in required):
        return None
    symbol = _c_symbol(f"nanoc_{mapping.node_name}")
    return {
        "kind": str(mapping.onnx_op).lower(),
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": symbol,
        "input_tensors": [name for name in node.inputs if name],
        "output_tensors": list(node.outputs),
        "block_size": int(cmsis_nn["block_size"]),
    }


def _where_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    required = (
        "x_scale",
        "x_zero_point",
        "y_scale",
        "y_zero_point",
        "output_scale",
        "output_zero_point",
        "activation_min",
        "activation_max",
        "block_size",
        "condition_source",
    )
    if any(field not in cmsis_nn for field in required):
        return None
    weights = graph.quantization.get("weights", {}) if graph.quantization else {}
    quant_weights = quant.get("weights", {})
    weight_name = str(quant_weights.get("weight", "")) if isinstance(quant_weights, dict) else ""
    weight_info = weights.get(weight_name, {}) if isinstance(weights, dict) else {}
    values = weight_info.get("values", []) if isinstance(weight_info, dict) else []
    symbol = _c_symbol(f"nanoc_{mapping.node_name}")
    layer: dict[str, Any] = {
        "kind": "where",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": symbol,
        "input_tensors": list(node.inputs[:3]),
        "output_tensors": list(node.outputs),
        "x_scale": float(cmsis_nn["x_scale"]),
        "x_zero_point": int(cmsis_nn["x_zero_point"]),
        "y_scale": float(cmsis_nn["y_scale"]),
        "y_zero_point": int(cmsis_nn["y_zero_point"]),
        "output_scale": float(cmsis_nn["output_scale"]),
        "output_zero_point": int(cmsis_nn["output_zero_point"]),
        "activation_min": int(cmsis_nn["activation_min"]),
        "activation_max": int(cmsis_nn["activation_max"]),
        "block_size": int(cmsis_nn["block_size"]),
        "condition_source": str(cmsis_nn["condition_source"]),
    }
    condition_values = cmsis_nn.get("condition_values", [])
    if isinstance(condition_values, list) and condition_values:
        layer["condition_values"] = [1 if bool(item) else 0 for item in condition_values]
        layer["condition_symbol"] = f"{symbol}_condition"
        layer["constant_condition_index"] = 0
    if isinstance(values, list) and values:
        layer["weight_values"] = values
        layer["constant_symbol"] = f"{symbol}_weights"
        layer["constant_input_index"] = 2
    return layer


def _generated_unary_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    required = (
        "input_scale",
        "input_zero_point",
        "output_scale",
        "output_zero_point",
        "activation_min",
        "activation_max",
        "block_size",
    )
    if any(field not in cmsis_nn for field in required):
        return None
    layer = {
        "kind": str(mapping.onnx_op).lower(),
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]] if node.inputs else [],
        "output_tensors": list(node.outputs),
        "input_scale": float(cmsis_nn["input_scale"]),
        "input_zero_point": int(cmsis_nn["input_zero_point"]),
        "output_scale": float(cmsis_nn["output_scale"]),
        "output_zero_point": int(cmsis_nn["output_zero_point"]),
        "activation_min": int(cmsis_nn["activation_min"]),
        "activation_max": int(cmsis_nn["activation_max"]),
        "block_size": int(cmsis_nn["block_size"]),
    }
    if mapping.onnx_op == "LeakyRelu":
        layer["alpha"] = float(cmsis_nn.get("alpha", 0.01))
    if mapping.onnx_op in {"Elu", "ThresholdedRelu", "Celu"}:
        layer["alpha"] = float(cmsis_nn.get("alpha", 1.0))
    if mapping.onnx_op == "Selu":
        layer["alpha"] = float(cmsis_nn.get("alpha", 1.67326))
        layer["gamma"] = float(cmsis_nn.get("gamma", 1.0507))
    if mapping.onnx_op == "HardSigmoid":
        layer["alpha"] = float(cmsis_nn.get("alpha", 0.2))
        layer["beta"] = float(cmsis_nn.get("beta", 0.5))
    if mapping.onnx_op == "Clip":
        if cmsis_nn.get("clip_min") is not None:
            layer["clip_min"] = float(cmsis_nn["clip_min"])
        if cmsis_nn.get("clip_max") is not None:
            layer["clip_max"] = float(cmsis_nn["clip_max"])
    return layer


def _reduce_mean_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    required = (
        "input_scale",
        "input_zero_point",
        "output_scale",
        "output_zero_point",
        "activation_min",
        "activation_max",
        "input_shape",
        "output_shape",
        "axes",
        "keepdims",
        "block_size",
    )
    if any(field not in cmsis_nn for field in required):
        return None
    input_shape = [int(item) for item in cmsis_nn["input_shape"]]
    output_shape = [int(item) for item in cmsis_nn["output_shape"]]
    if len(input_shape) != 2 or len(output_shape) != 2:
        return None
    return {
        "kind": str(mapping.onnx_op).lower(),
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]] if node.inputs else [],
        "output_tensors": list(node.outputs),
        "input_scale": float(cmsis_nn["input_scale"]),
        "input_zero_point": int(cmsis_nn["input_zero_point"]),
        "output_scale": float(cmsis_nn["output_scale"]),
        "output_zero_point": int(cmsis_nn["output_zero_point"]),
        "activation_min": int(cmsis_nn["activation_min"]),
        "activation_max": int(cmsis_nn["activation_max"]),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "axes": [int(item) for item in cmsis_nn["axes"]],
        "keepdims": int(cmsis_nn["keepdims"]),
        "block_size": int(cmsis_nn["block_size"]),
    }


def _unsqueeze_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    required = ("input_shape", "output_shape", "axes", "block_size")
    if any(field not in cmsis_nn for field in required):
        return None
    return {
        "kind": "unsqueeze",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]] if node.inputs else [],
        "output_tensors": list(node.outputs),
        "input_shape": [int(item) for item in cmsis_nn["input_shape"]],
        "output_shape": [int(item) for item in cmsis_nn["output_shape"]],
        "axes": [int(item) for item in cmsis_nn["axes"]],
        "block_size": int(cmsis_nn["block_size"]),
    }


def _pad_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    return _static_indexing_layer(graph, mapping, "pad", ("input_shape", "output_shape", "pads", "constant_q"))


def _slice_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    return _static_indexing_layer(
        graph,
        mapping,
        "slice",
        ("input_shape", "output_shape", "starts", "ends", "axes", "steps"),
    )


def _gather_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    return _static_indexing_layer(graph, mapping, "gather", ("input_shape", "output_shape", "indices", "axis"))


def _static_indexing_layer(
    graph: ModelGraph,
    mapping: OpMapping,
    kind: str,
    required: tuple[str, ...],
) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict) or any(field not in cmsis_nn for field in required):
        return None
    layer: dict[str, Any] = {
        "kind": kind,
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]] if node.inputs else [],
        "output_tensors": list(node.outputs),
        "block_size": int(cmsis_nn.get("block_size", 0)),
    }
    for field in required:
        value = cmsis_nn[field]
        if isinstance(value, list):
            layer[field] = [int(item) for item in value]
        elif isinstance(value, float):
            layer[field] = float(value)
        else:
            layer[field] = int(value)
    if layer["block_size"] <= 0:
        layer["block_size"] = _shape_size(layer.get("output_shape", []))
    return layer


def _abs_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None or not node.inputs:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    for field in ("activation_min", "activation_max", "block_size"):
        if field not in cmsis_nn:
            return None
    output_shape = _first_shape(node.output_shapes, node.outputs)
    block_size = element_count_from_shape(output_shape or [])
    if block_size is None:
        block_size = int(cmsis_nn.get("block_size", 0))
    if block_size <= 0:
        return None
    return {
        "kind": "abs",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]],
        "output_tensors": list(node.outputs),
        "activation_min": int(cmsis_nn["activation_min"]),
        "activation_max": int(cmsis_nn["activation_max"]),
        "block_size": int(block_size),
    }


def _transpose_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None or not node.inputs:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    if any(field not in cmsis_nn for field in _CMSIS_TRANSPOSE_REQUIRED_FIELDS):
        return None
    input_dims = [int(item) for item in cmsis_nn["input_dims"]]
    output_dims = [int(item) for item in cmsis_nn["output_dims"]]
    perm = [int(item) for item in cmsis_nn["perm"]]
    if len(input_dims) != 4 or len(output_dims) != 4 or len(perm) != 4:
        return None
    return {
        "kind": "transpose",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]],
        "output_tensors": list(node.outputs),
        "input_dims": input_dims,
        "output_dims": output_dims,
        "perm": perm,
        "block_size": element_count_from_shape(output_dims) or 0,
    }


def _concat_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    node = _node_by_name(graph, mapping.node_name)
    if node is None or not node.inputs or not node.outputs:
        return None
    quant = _node_quant(graph, mapping.node_name)
    cmsis_nn = quant.get("cmsis_nn", {}) if isinstance(quant, dict) else {}
    if not isinstance(cmsis_nn, dict):
        return None
    input_shapes = [node.input_shapes.get(name) for name in node.inputs]
    output_shape = _first_shape(node.output_shapes, node.outputs)
    if (
        output_shape is None
        or len(output_shape) != 4
        or any(shape is None or len(shape) != 4 for shape in input_shapes)
    ):
        return None
    axis = int(node.normalized_attributes.get("axis", node.attributes.get("axis", 0)))
    if axis < 0:
        axis += len(output_shape)
    if axis not in {0, 1, 2, 3}:
        return None
    input_dims = [_activation_dims(shape, graph.layout) for shape in input_shapes]
    output_dims = _activation_dims(output_shape, graph.layout)
    if output_dims is None or any(dims is None for dims in input_dims):
        return None
    symbol = _c_symbol(f"nanoc_{mapping.node_name}")
    input_requantize = []
    for input_index, item in enumerate(cmsis_nn.get("input_requantize", [])):
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        entry["symbol"] = f"{symbol}_requant_{input_index}"
        input_requantize.append(entry)
    return {
        "kind": "concat",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": symbol,
        "input_tensors": list(node.inputs),
        "output_tensors": list(node.outputs),
        "axis": axis,
        "layout": graph.layout,
        "input_dims": input_dims,
        "output_dims": output_dims,
        "input_requantize": input_requantize,
    }


def _softmax_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    quant = _node_quant(graph, mapping.node_name)
    node = _node_by_name(graph, mapping.node_name)
    if node is None or quant is None:
        return None
    cmsis_nn = quant.get("cmsis_nn", {})
    if not isinstance(cmsis_nn, dict):
        return None
    if any(field not in cmsis_nn for field in _CMSIS_SOFTMAX_REQUIRED_FIELDS):
        return None
    input_shape = _first_shape(node.input_shapes, node.inputs)
    if not input_shape:
        return None
    element_count = element_count_from_shape(input_shape)
    if element_count is None:
        return None
    rank = len(input_shape)
    axis = int(node.normalized_attributes.get("axis", node.attributes.get("axis", -1)))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    row_size = 1
    for dim in input_shape[axis:]:
        if not isinstance(dim, int):
            return None
        row_size *= int(dim)
    return {
        "kind": "softmax",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]] if node.inputs else [],
        "output_tensors": list(node.outputs),
        "num_rows": max(1, element_count // int(row_size)),
        "row_size": int(row_size),
        "input_scale": float(cmsis_nn.get("input_scale", 1.0)),
        "stable_softmax": int(row_size) >= 512,
        "multiplier": int(cmsis_nn["multiplier"]),
        "shift": int(cmsis_nn["shift"]),
        "diff_min": int(cmsis_nn["diff_min"]),
    }


_CMSIS_FC_REQUIRED_FIELDS = (
    "input_offset",
    "filter_offset",
    "output_offset",
    "multiplier",
    "shift",
    "activation_min",
    "activation_max",
)

_CMSIS_CONV_REQUIRED_FIELDS = (
    "input_offset",
    "output_offset",
    "multiplier",
    "shift",
    "activation_min",
    "activation_max",
    "stride",
    "padding",
    "dilation",
    "groups",
)

_CMSIS_POOL_REQUIRED_FIELDS = (
    "api",
    "stride",
    "padding",
    "kernel_shape",
    "activation_min",
    "activation_max",
)

_CMSIS_SOFTMAX_REQUIRED_FIELDS = (
    "api",
    "multiplier",
    "shift",
    "diff_min",
)

_CMSIS_ADD_REQUIRED_FIELDS = (
    "input_1_offset",
    "input_1_multiplier",
    "input_1_shift",
    "input_2_offset",
    "input_2_multiplier",
    "input_2_shift",
    "left_shift",
    "output_offset",
    "output_multiplier",
    "output_shift",
    "activation_min",
    "activation_max",
    "block_size",
)

_CMSIS_MUL_REQUIRED_FIELDS = (
    "input_1_offset",
    "input_2_offset",
    "output_offset",
    "output_multiplier",
    "output_shift",
    "activation_min",
    "activation_max",
    "block_size",
)

_CMSIS_TRANSPOSE_REQUIRED_FIELDS = (
    "api",
    "perm",
    "input_dims",
    "output_dims",
)


def _node_by_name(graph: ModelGraph, name: str):
    for node in graph.nodes:
        if node.name == name:
            return node
    return None


def _node_quant(graph: ModelGraph, name: str) -> dict[str, Any] | None:
    quantization = graph.quantization or {}
    node_quant = quantization.get("nodes", {})
    if not isinstance(node_quant, dict):
        return None
    quant = node_quant.get(name)
    return quant if isinstance(quant, dict) else None


def _first_shape(
    shapes: dict[str, list[Any]],
    names: list[str],
) -> list[Any] | None:
    for name in names:
        shape = shapes.get(name)
        if shape:
            return shape
    return None


def _activation_dims(shape: list[Any] | None, layout: str) -> list[int] | None:
    if not shape or len(shape) not in {3, 4} or not all(isinstance(item, int) for item in shape):
        return None
    if len(shape) == 3:
        n, c, w = [int(item) for item in shape]
        if layout == "NHWC":
            return [n, 1, c, w]
        return [n, 1, w, c]
    n, d1, d2, d3 = [int(item) for item in shape]
    if layout == "NHWC":
        return [n, d1, d2, d3]
    return [n, d2, d3, d1]


def _pair(value: object, *, default: list[int]) -> list[int]:
    if isinstance(value, list) and len(value) >= 2:
        return [int(value[0]), int(value[1])]
    return default


def _reorder_conv_weight_to_ohwi(values: list[int], shape: list[Any]) -> list[int]:
    if len(shape) == 3:
        output_channels, input_channels, kernel_w = [int(item) for item in shape]
        kernel_h = 1
    else:
        output_channels, input_channels, kernel_h, kernel_w = [int(item) for item in shape]
    reordered: list[int] = []
    for output_channel in range(output_channels):
        for y in range(kernel_h):
            for x in range(kernel_w):
                for input_channel in range(input_channels):
                    if len(shape) == 3:
                        source = (
                            output_channel * input_channels * kernel_w
                            + input_channel * kernel_w
                            + x
                        )
                    else:
                        source = (
                            ((output_channel * input_channels + input_channel) * kernel_h + y)
                            * kernel_w
                            + x
                        )
                    reordered.append(values[source])
    return reordered


def _reorder_depthwise_oihw_to_1hwo(values: list[int], shape: list[Any]) -> list[int]:
    output_channels, in_per_group, kernel_h, kernel_w = [int(item) for item in shape]
    if in_per_group != 1:
        return []
    reordered: list[int] = []
    for y in range(kernel_h):
        for x in range(kernel_w):
            for output_channel in range(output_channels):
                source = (
                    ((output_channel * in_per_group) * kernel_h + y)
                    * kernel_w
                    + x
                )
                reordered.append(values[source])
    return reordered


def _is_depthwise_conv_node(node) -> bool:
    if node.op_type not in {"Conv", "QLinearConv"}:
        return False
    group = int(node.normalized_attributes.get("group", node.attributes.get("group", 1)))
    weight_shape = None
    for weight_name in node.weights:
        shape = node.input_shapes.get(weight_name)
        if shape:
            weight_shape = shape
            break
    if not weight_shape or len(weight_shape) != 4:
        return group > 1
    output_channels, in_per_group, _, _ = weight_shape
    if not isinstance(output_channels, int) or not isinstance(in_per_group, int):
        return group > 1
    return group > 1 and in_per_group == 1 and output_channels % group == 0


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


def _firmware_integration(
    options: CodegenOptions,
    memory_plan: MemoryPlan,
    status: str,
) -> str:
    return "\n".join(
        [
            "# NanoC-NN CMSIS-NN 固件接入说明",
            "",
            f"- 生成状态：`{status}`",
            f"- 目标内核：`{options.target}`",
            f"- backend：`{options.resolved_backend}`",
            f"- 估算 SRAM：`{memory_plan.total_sram_bytes}` bytes",
            f"- 估算 Flash：`{memory_plan.total_flash_bytes}` bytes",
            "",
            "## 需要加入固件工程的文件",
            "",
            "- `include/model.h`",
            "- `include/model_weights.h`",
            "- `src/model.c`",
            "",
            "如果生成目录中存在 `include/converter_weights.h`，它仅用于调试或兼容报告；"
            "真实 int8 runtime 权重在 `model_weights.h` 中。",
            "",
            "## 依赖",
            "",
            "- CMSIS-NN：需要提供 `arm_nnfunctions.h`。",
            "- CMSIS-Core：需要提供目标 Cortex-M 对应的 core 头文件。",
            "- 编译时定义 `NANOC_ENABLE_CMSIS_NN=1` 后会启用真实 CMSIS-NN 调用。",
            "",
            "## 运行约束",
            "",
            "- 生成代码为 C99。",
            "- 不调用 `malloc` 或 `free`。",
            "- 不依赖文件系统、POSIX API、线程或具体 HAL。",
            "- 输入和输出缓冲区由调用方提供，大小见 `model.h` 中的宏。",
            "- 静态激活缓冲区和 scratch buffer 在 `model.c` 中定义。",
            "",
            "## 调用方式",
            "",
            "```c",
            "#include \"model.h\"",
            "",
            "static int8_t input[NANOC_MODEL_INPUT_BYTES];",
            "static int8_t output[NANOC_MODEL_OUTPUT_BYTES];",
            "",
            "int status = nanoc_model_run(input, output);",
            "```",
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


REFERENCE_RUNTIME_OPS = {
    "Abs",
    "Add",
    "Cast",
    "Concat",
    "Conv",
    "DequantizeLinear",
    "Dropout",
    "Flatten",
    "Gemm",
    "GlobalAveragePool",
    "MatMul",
    "MaxPool",
    "Mul",
    "QuantizeLinear",
    "Relu",
    "Reshape",
    "Softmax",
    "Squeeze",
    "Transpose",
    "Unsqueeze",
}


def _reference_runtime_enabled(graph: ModelGraph) -> bool:
    reference_producers = {"pytorch", "tf2onnx", "onnx.quantize"}
    if graph.producer_name not in reference_producers:
        return False
    if not graph.has_quantization:
        return True
    quantization = graph.quantization or {}
    contract = quantization.get("int8_contract") if isinstance(quantization, dict) else None
    return isinstance(contract, dict) and contract.get("status") != "ok"


def _reference_runtime_supported(graph: ModelGraph) -> bool:
    return all(node.op_type in REFERENCE_RUNTIME_OPS for node in graph.nodes)


def _reference_model_c(graph: ModelGraph, memory_plan: MemoryPlan, status: str) -> str:
    shapes = _reference_tensor_shapes(graph)
    init_map = {item.name: item for item in graph.initializers}
    buffers = _reference_buffer_declarations(graph, shapes)
    shape_arrays = _reference_shape_arrays(shapes)
    lines = [
        "#include \"model.h\"",
        "#include \"model_weights.h\"",
        "",
        "#include <math.h>",
        "#include <stddef.h>",
        "",
        "#if defined(__GNUC__) || defined(__clang__)",
        "#define NANOC_MAYBE_UNUSED __attribute__((unused))",
        "#else",
        "#define NANOC_MAYBE_UNUSED",
        "#endif",
        "",
        "static int8_t nanoc_activation_a[NANOC_MODEL_ACTIVATION_A_BYTES];",
        "static int8_t nanoc_activation_b[NANOC_MODEL_ACTIVATION_B_BYTES];",
        "static int8_t nanoc_scratch[NANOC_MODEL_SCRATCH_BYTES];",
    ]
    lines.extend(buffers)
    lines.extend(shape_arrays)
    lines.extend(
        [
            "",
            "static void NANOC_MAYBE_UNUSED nanoc_unravel(size_t index, int rank, const int *shape, int *coords)",
            "{",
            "    for (int axis = rank - 1; axis >= 0; --axis) {",
            "        int dim = shape[axis];",
            "        coords[axis] = dim > 0 ? (int)(index % (size_t)dim) : 0;",
            "        if (dim > 0) {",
            "            index /= (size_t)dim;",
            "        }",
            "    }",
            "}",
            "",
            "static size_t NANOC_MAYBE_UNUSED nanoc_ravel(int rank, const int *shape, const int *coords)",
            "{",
            "    size_t offset = 0u;",
            "    for (int axis = 0; axis < rank; ++axis) {",
            "        offset = offset * (size_t)shape[axis] + (size_t)coords[axis];",
            "    }",
            "    return offset;",
            "}",
            "",
            "static size_t NANOC_MAYBE_UNUSED nanoc_broadcast_offset(size_t out_index, int out_rank, const int *out_shape, int in_rank, const int *in_shape)",
            "{",
            "    int out_coords[8] = {0};",
            "    int in_coords[8] = {0};",
            "    nanoc_unravel(out_index, out_rank, out_shape, out_coords);",
            "    int rank_delta = out_rank - in_rank;",
            "    for (int axis = 0; axis < in_rank; ++axis) {",
            "        int out_axis = axis + rank_delta;",
            "        int coord = out_axis >= 0 ? out_coords[out_axis] : 0;",
            "        in_coords[axis] = in_shape[axis] == 1 ? 0 : coord;",
            "    }",
            "    return nanoc_ravel(in_rank, in_shape, in_coords);",
            "}",
            "",
            "static int NANOC_MAYBE_UNUSED nanoc_axis_index(size_t index, int rank, const int *shape, int axis)",
            "{",
            "    int coords[8] = {0};",
            "    if (axis < 0) {",
            "        axis += rank;",
            "    }",
            "    nanoc_unravel(index, rank, shape, coords);",
            "    return coords[axis];",
            "}",
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
            "    return NANOC_STATUS_BLOCKED;",
            "}",
            "",
            "int nanoc_model_run_float(const float * const *inputs, float *output)",
            "{",
            "    (void)nanoc_activation_a;",
            "    (void)nanoc_activation_b;",
            "    (void)nanoc_scratch;",
        ]
    )
    for node in graph.nodes:
        lines.extend(_reference_node_lines(graph, node, shapes, init_map))
    if graph.outputs:
        out_name = graph.outputs[0].name
        out_expr = _reference_expr(graph, out_name)
        out_size = _shape_size(shapes[out_name])
        if out_expr != "output":
            lines.extend(
                [
                    f"    for (size_t i = 0u; i < {out_size}u; ++i) {{",
                    f"        output[i] = {out_expr}[i];",
                    "    }",
                ]
            )
    lines.extend(["    return NANOC_STATUS_OK;", "}", ""])
    return "\n".join(lines)


def _reference_node_lines(
    graph: ModelGraph,
    node,
    shapes: dict[str, list[int]],
    init_map: dict[str, InitializerSpec],
) -> list[str]:
    op = node.op_type
    inputs = node.inputs
    outputs = node.outputs
    if not outputs:
        return []
    out = outputs[0]
    out_expr = _reference_expr(graph, out)
    out_shape = shapes[out]
    out_size = _shape_size(out_shape)
    lines = [f"    /* {op}: {node.name} */"]
    if op in {"Flatten", "Reshape", "Dropout", "Cast", "Squeeze", "Unsqueeze"}:
        src = _reference_expr(graph, inputs[0])
        lines.extend(_copy_loop(src, out_expr, out_size))
    elif op == "Abs":
        src = _reference_expr(graph, inputs[0])
        lines.extend(
            [
                f"    for (size_t i = 0u; i < {out_size}u; ++i) {{",
                f"        float v = {src}[i];",
                f"        {out_expr}[i] = v < 0.0f ? -v : v;",
                "    }",
            ]
        )
    elif op == "Relu":
        src = _reference_expr(graph, inputs[0])
        lines.extend(
            [
                f"    for (size_t i = 0u; i < {out_size}u; ++i) {{",
                f"        float v = {src}[i];",
                f"        {out_expr}[i] = v > 0.0f ? v : 0.0f;",
                "    }",
            ]
        )
    elif op in {"Add", "Mul"}:
        a = _reference_expr(graph, inputs[0])
        b = _reference_expr(graph, inputs[1])
        a_shape = shapes[inputs[0]]
        b_shape = shapes[inputs[1]]
        operator = "+" if op == "Add" else "*"
        lines.extend(
            [
                f"    for (size_t i = 0u; i < {out_size}u; ++i) {{",
                f"        size_t ai = nanoc_broadcast_offset(i, {len(out_shape)}, {_shape_symbol(out)}, {len(a_shape)}, {_shape_symbol(inputs[0])});",
                f"        size_t bi = nanoc_broadcast_offset(i, {len(out_shape)}, {_shape_symbol(out)}, {len(b_shape)}, {_shape_symbol(inputs[1])});",
                f"        {out_expr}[i] = {a}[ai] {operator} {b}[bi];",
                "    }",
            ]
        )
    elif op == "MatMul":
        lines.extend(_reference_matmul_lines(graph, node, shapes))
    elif op == "Gemm":
        lines.extend(_reference_gemm_lines(graph, node, shapes))
    elif op == "Conv":
        lines.extend(_reference_conv_lines(graph, node, shapes))
    elif op == "MaxPool":
        lines.extend(_reference_maxpool_lines(graph, node, shapes))
    elif op == "GlobalAveragePool":
        lines.extend(_reference_gap_lines(graph, node, shapes))
    elif op == "Softmax":
        lines.extend(_reference_softmax_lines(graph, node, shapes))
    elif op == "Transpose":
        lines.extend(_reference_transpose_lines(graph, node, shapes))
    elif op == "Concat":
        lines.extend(_reference_concat_lines(graph, node, shapes))
    elif op == "QuantizeLinear":
        lines.extend(_reference_quantize_lines(graph, node, shapes, init_map))
    elif op == "DequantizeLinear":
        lines.extend(_reference_dequantize_lines(graph, node, shapes))
    else:
        lines.append(f"    /* unsupported reference op: {op} */")
        lines.append("    return NANOC_STATUS_UNSUPPORTED;")
    lines.append("")
    return lines


def _reference_matmul_lines(graph: ModelGraph, node, shapes: dict[str, list[int]]) -> list[str]:
    a, b = node.inputs[:2]
    out = node.outputs[0]
    a_expr = _reference_expr(graph, a)
    b_expr = _reference_expr(graph, b)
    out_expr = _reference_expr(graph, out)
    m, k = shapes[a]
    _, n = shapes[b]
    return [
        f"    for (int row = 0; row < {m}; ++row) {{",
        f"        for (int col = 0; col < {n}; ++col) {{",
        "            float acc = 0.0f;",
        f"            for (int kk = 0; kk < {k}; ++kk) {{",
        f"                acc += {a_expr}[row * {k} + kk] * {b_expr}[kk * {n} + col];",
        "            }",
        f"            {out_expr}[row * {n} + col] = acc;",
        "        }",
        "    }",
    ]


def _reference_gemm_lines(graph: ModelGraph, node, shapes: dict[str, list[int]]) -> list[str]:
    a, b = node.inputs[:2]
    c = node.inputs[2] if len(node.inputs) > 2 else ""
    out = node.outputs[0]
    attrs = node.normalized_attributes
    trans_a = int(attrs.get("transA", 0))
    trans_b = int(attrs.get("transB", 0))
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    a_shape = shapes[a]
    b_shape = shapes[b]
    m = a_shape[1] if trans_a else a_shape[0]
    k = a_shape[0] if trans_a else a_shape[1]
    n = b_shape[0] if trans_b else b_shape[1]
    a_expr = _reference_expr(graph, a)
    b_expr = _reference_expr(graph, b)
    c_expr = _reference_expr(graph, c) if c else ""
    out_expr = _reference_expr(graph, out)
    a_at = (
        f"{a_expr}[kk * {a_shape[1]} + row]"
        if trans_a
        else f"{a_expr}[row * {a_shape[1]} + kk]"
    )
    b_at = (
        f"{b_expr}[col * {b_shape[1]} + kk]"
        if trans_b
        else f"{b_expr}[kk * {b_shape[1]} + col]"
    )
    bias_line = f"            acc += {beta:.9g}f * {c_expr}[col];" if c else ""
    lines = [
        f"    for (int row = 0; row < {m}; ++row) {{",
        f"        for (int col = 0; col < {n}; ++col) {{",
        "            float acc = 0.0f;",
        f"            for (int kk = 0; kk < {k}; ++kk) {{",
        f"                acc += {a_at} * {b_at};",
        "            }",
        f"            acc *= {_c_float_literal(alpha)};",
    ]
    if bias_line:
        lines.append(bias_line.replace(f"{beta:.9g}f", _c_float_literal(beta)))
    lines.extend([f"            {out_expr}[row * {n} + col] = acc;", "        }", "    }"])
    return lines


def _reference_conv_lines(graph: ModelGraph, node, shapes: dict[str, list[int]]) -> list[str]:
    x, w = node.inputs[:2]
    b = node.inputs[2] if len(node.inputs) > 2 else ""
    y = node.outputs[0]
    x_expr = _reference_expr(graph, x)
    w_expr = _reference_expr(graph, w)
    b_expr = _reference_expr(graph, b) if b else ""
    y_expr = _reference_expr(graph, y)
    n, c, h, width = shapes[x]
    oc, ic_per_group, kh, kw = shapes[w]
    _, _, oh, ow = shapes[y]
    attrs = node.normalized_attributes
    strides = [int(v) for v in attrs.get("strides", [1, 1])]
    pads = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
    dilations = [int(v) for v in attrs.get("dilations", [1, 1])]
    groups = int(attrs.get("group", 1))
    oc_per_group = oc // groups
    lines = [
        f"    for (int ni = 0; ni < {n}; ++ni) {{",
        f"        for (int oc_i = 0; oc_i < {oc}; ++oc_i) {{",
        "            int group_i = oc_i / " + str(oc_per_group) + ";",
        f"            for (int oh_i = 0; oh_i < {oh}; ++oh_i) {{",
        f"                for (int ow_i = 0; ow_i < {ow}; ++ow_i) {{",
        f"                    float acc = {b_expr}[oc_i];" if b else "                    float acc = 0.0f;",
        f"                    for (int icg = 0; icg < {ic_per_group}; ++icg) {{",
        f"                        int ic_i = group_i * {ic_per_group} + icg;",
        f"                        for (int kh_i = 0; kh_i < {kh}; ++kh_i) {{",
        f"                            int ih_i = oh_i * {strides[0]} + kh_i * {dilations[0]} - {pads[0]};",
        f"                            if (ih_i < 0 || ih_i >= {h}) continue;",
        f"                            for (int kw_i = 0; kw_i < {kw}; ++kw_i) {{",
        f"                                int iw_i = ow_i * {strides[1]} + kw_i * {dilations[1]} - {pads[1]};",
        f"                                if (iw_i < 0 || iw_i >= {width}) continue;",
        f"                                float xv = {x_expr}[ni * {c * h * width} + ic_i * {h * width} + ih_i * {width} + iw_i];",
        f"                                float wv = {w_expr}[oc_i * {ic_per_group * kh * kw} + icg * {kh * kw} + kh_i * {kw} + kw_i];",
        "                                acc += xv * wv;",
        "                            }",
        "                        }",
        "                    }",
        f"                    {y_expr}[ni * {oc * oh * ow} + oc_i * {oh * ow} + oh_i * {ow} + ow_i] = acc;",
        "                }",
        "            }",
        "        }",
        "    }",
    ]
    return lines


def _reference_maxpool_lines(graph: ModelGraph, node, shapes: dict[str, list[int]]) -> list[str]:
    x = node.inputs[0]
    y = node.outputs[0]
    x_expr = _reference_expr(graph, x)
    y_expr = _reference_expr(graph, y)
    n, c, h, width = shapes[x]
    _, _, oh, ow = shapes[y]
    attrs = node.normalized_attributes
    kernel = [int(v) for v in attrs.get("kernel_shape", [1, 1])]
    strides = [int(v) for v in attrs.get("strides", kernel)]
    pads = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
    return [
        f"    for (int ni = 0; ni < {n}; ++ni) {{",
        f"        for (int ci = 0; ci < {c}; ++ci) {{",
        f"            for (int oh_i = 0; oh_i < {oh}; ++oh_i) {{",
        f"                for (int ow_i = 0; ow_i < {ow}; ++ow_i) {{",
        "                    float max_v = -3.402823466e+38f;",
        f"                    for (int kh_i = 0; kh_i < {kernel[0]}; ++kh_i) {{",
        f"                        int ih_i = oh_i * {strides[0]} + kh_i - {pads[0]};",
        f"                        if (ih_i < 0 || ih_i >= {h}) continue;",
        f"                        for (int kw_i = 0; kw_i < {kernel[1]}; ++kw_i) {{",
        f"                            int iw_i = ow_i * {strides[1]} + kw_i - {pads[1]};",
        f"                            if (iw_i < 0 || iw_i >= {width}) continue;",
        f"                            float v = {x_expr}[ni * {c * h * width} + ci * {h * width} + ih_i * {width} + iw_i];",
        "                            if (v > max_v) max_v = v;",
        "                        }",
        "                    }",
        f"                    {y_expr}[ni * {c * oh * ow} + ci * {oh * ow} + oh_i * {ow} + ow_i] = max_v;",
        "                }",
        "            }",
        "        }",
        "    }",
    ]


def _reference_gap_lines(graph: ModelGraph, node, shapes: dict[str, list[int]]) -> list[str]:
    x = node.inputs[0]
    y = node.outputs[0]
    x_expr = _reference_expr(graph, x)
    y_expr = _reference_expr(graph, y)
    shape = shapes[x]
    n, c = shape[0], shape[1]
    spatial = _shape_size(shape[2:])
    return [
        f"    for (int ni = 0; ni < {n}; ++ni) {{",
        f"        for (int ci = 0; ci < {c}; ++ci) {{",
        "            float acc = 0.0f;",
        f"            for (int si = 0; si < {spatial}; ++si) {{",
        f"                acc += {x_expr}[ni * {c * spatial} + ci * {spatial} + si];",
        "            }",
            f"            {y_expr}[ni * {c} + ci] = acc / {_c_float_literal(float(spatial))};",
        "        }",
        "    }",
    ]


def _reference_softmax_lines(graph: ModelGraph, node, shapes: dict[str, list[int]]) -> list[str]:
    x = node.inputs[0]
    y = node.outputs[0]
    x_expr = _reference_expr(graph, x)
    y_expr = _reference_expr(graph, y)
    shape = shapes[x]
    rank = len(shape)
    axis = int(node.normalized_attributes.get("axis", rank - 1))
    if axis < 0:
        axis += rank
    inner = _shape_size(shape[axis + 1 :])
    axis_dim = shape[axis]
    outer = _shape_size(shape[:axis])
    return [
        f"    for (int outer = 0; outer < {outer}; ++outer) {{",
        f"        for (int inner = 0; inner < {inner}; ++inner) {{",
        "            float max_v = -3.402823466e+38f;",
        f"            for (int ai = 0; ai < {axis_dim}; ++ai) {{",
        f"                size_t idx = (size_t)outer * {axis_dim * inner}u + (size_t)ai * {inner}u + (size_t)inner;",
        f"                float v = {x_expr}[idx];",
        "                if (v > max_v) max_v = v;",
        "            }",
        "            float sum = 0.0f;",
        f"            for (int ai = 0; ai < {axis_dim}; ++ai) {{",
        f"                size_t idx = (size_t)outer * {axis_dim * inner}u + (size_t)ai * {inner}u + (size_t)inner;",
        f"                float e = expf({x_expr}[idx] - max_v);",
        f"                {y_expr}[idx] = e;",
        "                sum += e;",
        "            }",
        f"            for (int ai = 0; ai < {axis_dim}; ++ai) {{",
        f"                size_t idx = (size_t)outer * {axis_dim * inner}u + (size_t)ai * {inner}u + (size_t)inner;",
        f"                {y_expr}[idx] = {y_expr}[idx] / sum;",
        "            }",
        "        }",
        "    }",
    ]


def _reference_transpose_lines(graph: ModelGraph, node, shapes: dict[str, list[int]]) -> list[str]:
    x = node.inputs[0]
    y = node.outputs[0]
    x_expr = _reference_expr(graph, x)
    y_expr = _reference_expr(graph, y)
    perm = [int(v) for v in node.normalized_attributes.get("perm", list(reversed(range(len(shapes[x])))))]
    out_size = _shape_size(shapes[y])
    rank = len(perm)
    lines = [
        f"    for (size_t i = 0u; i < {out_size}u; ++i) {{",
        "        int out_coords[8] = {0};",
        "        int in_coords[8] = {0};",
        f"        nanoc_unravel(i, {rank}, {_shape_symbol(y)}, out_coords);",
    ]
    for axis, source_axis in enumerate(perm):
        lines.append(f"        in_coords[{source_axis}] = out_coords[{axis}];")
    lines.extend(
        [
            f"        size_t src = nanoc_ravel({rank}, {_shape_symbol(x)}, in_coords);",
            f"        {y_expr}[i] = {x_expr}[src];",
            "    }",
        ]
    )
    return lines


def _reference_concat_lines(graph: ModelGraph, node, shapes: dict[str, list[int]]) -> list[str]:
    y = node.outputs[0]
    y_expr = _reference_expr(graph, y)
    out_shape = shapes[y]
    rank = len(out_shape)
    axis = int(node.normalized_attributes.get("axis", 0))
    if axis < 0:
        axis += rank
    lines = [
        "    {",
        f"        size_t offset_axis = 0u;",
    ]
    for input_name in node.inputs:
        x_expr = _reference_expr(graph, input_name)
        x_shape = shapes[input_name]
        x_size = _shape_size(x_shape)
        lines.extend(
            [
                f"        for (size_t i = 0u; i < {x_size}u; ++i) {{",
                "            int coords[8] = {0};",
                f"            nanoc_unravel(i, {rank}, {_shape_symbol(input_name)}, coords);",
                f"            coords[{axis}] += (int)offset_axis;",
                f"            size_t dst = nanoc_ravel({rank}, {_shape_symbol(y)}, coords);",
                f"            {y_expr}[dst] = {x_expr}[i];",
                "        }",
                f"        offset_axis += {x_shape[axis]}u;",
            ]
        )
    lines.append("    }")
    return lines


def _reference_quantize_lines(
    graph: ModelGraph,
    node,
    shapes: dict[str, list[int]],
    init_map: dict[str, InitializerSpec],
) -> list[str]:
    x, scale, zero = node.inputs[:3]
    y = node.outputs[0]
    x_expr = _reference_expr(graph, x)
    scale_expr = _reference_expr(graph, scale)
    zero_expr = _reference_expr(graph, zero)
    y_expr = _reference_expr(graph, y)
    out_size = _shape_size(shapes[y])
    axis = node.normalized_attributes.get("axis")
    zp_type = init_map.get(zero).elem_type if zero in init_map else "INT8"
    qmin, qmax = (0, 255) if zp_type == "UINT8" else (-128, 127)
    scale_size = _shape_size(shapes[scale])
    lines = [
        f"    for (size_t i = 0u; i < {out_size}u; ++i) {{",
    ]
    if scale_size == 1 or axis is None:
        lines.append("        int qi = 0;")
    else:
        lines.append(
            f"        int qi = nanoc_axis_index(i, {len(shapes[x])}, {_shape_symbol(x)}, {int(axis)});"
        )
    lines.extend(
        [
            f"        float q = nearbyintf({x_expr}[i] / {scale_expr}[qi] + (float){zero_expr}[qi]);",
            f"        if (q < {qmin}.0f) q = {qmin}.0f;",
            f"        if (q > {qmax}.0f) q = {qmax}.0f;",
            f"        {y_expr}[i] = q;",
            "    }",
        ]
    )
    return lines


def _reference_dequantize_lines(graph: ModelGraph, node, shapes: dict[str, list[int]]) -> list[str]:
    x, scale, zero = node.inputs[:3]
    y = node.outputs[0]
    x_expr = _reference_expr(graph, x)
    scale_expr = _reference_expr(graph, scale)
    zero_expr = _reference_expr(graph, zero)
    y_expr = _reference_expr(graph, y)
    out_size = _shape_size(shapes[y])
    scale_size = _shape_size(shapes[scale])
    axis = node.normalized_attributes.get("axis")
    lines = [f"    for (size_t i = 0u; i < {out_size}u; ++i) {{"]
    if scale_size == 1 or axis is None:
        lines.append("        int qi = 0;")
    else:
        lines.append(
            f"        int qi = nanoc_axis_index(i, {len(shapes[x])}, {_shape_symbol(x)}, {int(axis)});"
        )
    lines.extend(
        [
            f"        {y_expr}[i] = ((float){x_expr}[i] - (float){zero_expr}[qi]) * {scale_expr}[qi];",
            "    }",
        ]
    )
    return lines


def _copy_loop(src: str, dst: str, size: int) -> list[str]:
    return [
        f"    for (size_t i = 0u; i < {size}u; ++i) {{",
        f"        {dst}[i] = {src}[i];",
        "    }",
    ]


def _reference_buffer_declarations(graph: ModelGraph, shapes: dict[str, list[int]]) -> list[str]:
    model_outputs = {tensor.name for tensor in graph.outputs}
    model_inputs = {tensor.name for tensor in graph.inputs}
    initializers = {item.name for item in graph.initializers}
    lines: list[str] = []
    declared: set[str] = set()
    for node in graph.nodes:
        for name in node.outputs:
            if name in model_outputs or name in model_inputs or name in initializers or name in declared:
                continue
            size = _shape_size(shapes[name])
            lines.append(f"static float {_reference_buffer_symbol(name)}[{size}u];")
            declared.add(name)
    return lines


def _reference_shape_arrays(shapes: dict[str, list[int]]) -> list[str]:
    lines: list[str] = []
    for name, shape in sorted(shapes.items()):
        if not shape:
            lines.append(f"static const int {_shape_symbol(name)}[1] NANOC_MAYBE_UNUSED = {{1}};")
        else:
            values = ", ".join(str(int(dim)) for dim in shape)
            lines.append(
                f"static const int {_shape_symbol(name)}[{len(shape)}] "
                f"NANOC_MAYBE_UNUSED = {{{values}}};"
            )
    return lines


def _reference_tensor_shapes(graph: ModelGraph) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for tensor in [*graph.inputs, *graph.outputs]:
        if tensor.element_count is not None:
            shapes[tensor.name] = [int(dim) for dim in tensor.shape if isinstance(dim, int)]
    for initializer in graph.initializers:
        if initializer.element_count is not None:
            shapes[initializer.name] = [int(dim) for dim in initializer.shape if isinstance(dim, int)]
    for node in graph.nodes:
        for shape_map in (node.input_shapes, node.output_shapes):
            for name, shape in shape_map.items():
                if element_count_from_shape(shape) is not None:
                    shapes[name] = [int(dim) for dim in shape if isinstance(dim, int)]
    return shapes


def _reference_expr(graph: ModelGraph, name: str) -> str:
    for index, tensor in enumerate(graph.inputs):
        if tensor.name == name:
            return f"inputs[{index}]"
    if graph.outputs and graph.outputs[0].name == name:
        return "output"
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer.c_name
    return _reference_buffer_symbol(name)


def _reference_buffer_symbol(name: str) -> str:
    return f"nanoc_ref_{_tensor_symbol(name)}"


def _shape_symbol(name: str) -> str:
    return f"nanoc_shape_{_tensor_symbol(name)}"


def _shape_size(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _c_float_literal(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text.lower() and "." not in text:
        text = f"{text}.0"
    return f"{text}f"


def _c_int_list(values: list[int]) -> str:
    return ", ".join(str(int(item)) for item in values)

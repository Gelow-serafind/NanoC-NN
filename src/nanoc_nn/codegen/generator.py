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
            "AveragePool",
            "Concat",
            "Conv",
            "Gemm",
            "GlobalAveragePool",
            "MatMul",
            "MaxPool",
            "QLinearAdd",
            "QLinearConv",
            "QLinearGlobalAveragePool",
            "QLinearMatMul",
            "Softmax",
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
        out_dir / "include" / "model.h": _model_h(memory_plan),
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
            "    NANOC_STATUS_UNSUPPORTED = 3,",
            "    NANOC_STATUS_OVERSIZE = 4",
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
        "#include \"arm_nnfunctions.h\"",
        "#endif",
        "",
        "static int8_t nanoc_activation_a[NANOC_MODEL_ACTIVATION_A_BYTES];",
        "static int8_t nanoc_activation_b[NANOC_MODEL_ACTIVATION_B_BYTES];",
        "static int8_t nanoc_scratch[NANOC_MODEL_SCRATCH_BYTES];",
    ]
    # Declare NHWC input buffer if the model input is multi-channel 4-D
    input_shape = _graph_input_shape(graph)
    need_input_xpose = (
        input_shape is not None
        and len(input_shape) == 4
        and isinstance(input_shape[1], int)
        and input_shape[1] > 1
    )
    if need_input_xpose:
        input_count = element_count_from_shape(input_shape) or max(1, memory_plan.input_buffers[0].size_bytes)
        lines.append(f"static int8_t nanoc_input_nhwc[{input_count}u];")
    lines.extend(tensor_buffers)
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
        tensor_symbols = _tensor_symbol_map(graph, runtime_layers)
        for sym in tensor_symbols.values():
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
    body = _cmsis_tensor_runtime_run_body(graph, layers)
    if not body:
        return body

    # Determine if input/output need 4-D transposes
    input_shape = None
    output_shape = None
    if graph.inputs:
        input_shape = _tensor_shape(graph, graph.inputs[0].name)
    if graph.outputs:
        output_shape = _tensor_shape(graph, graph.outputs[0].name)

    need_input_xpose = (
        input_shape is not None
        and len(input_shape) == 4
        and isinstance(input_shape[1], int)
        and input_shape[1] > 1
    )
    need_output_xpose = (
        output_shape is not None
        and len(output_shape) == 4
        and isinstance(output_shape[1], int)
        and output_shape[1] > 1
    )

    if not need_input_xpose and not need_output_xpose:
        return body

    # Replace "input" pointer with NHWC buffer in CMSIS-NN call arguments
    if need_input_xpose:
        body = [l.replace("(void)input;", "(void)input; (void)nanoc_input_nhwc;")
                 .replace("            input,", "            nanoc_input_nhwc,")
                 for l in body]

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

    # Insert the body (skip the #if and cmsis_status lines)
    result.extend(body[2:])

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
                last_expr = "output"

        result.append(f"    /* NHWC→NCHW output transpose ({n}x{c}x{h}x{w}) */")
        result.append(f"    for (int _ni = 0; _ni < {n}; ++_ni)")
        result.append(f"        for (int _ci = 0; _ci < {c}; ++_ci)")
        result.append(f"            for (int _hi = 0; _hi < {h}; ++_hi)")
        result.append(f"                for (int _wi = 0; _wi < {w}; ++_wi)")
        result.append(
            f"                    output[_ni * {c * h * w} + _ci * {h * w} + _hi * {w} + _wi] = "
            f"{last_expr}[_ni * {h * w * c} + _hi * {w * c} + _wi * {c} + _ci];"
        )

    return result


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
) -> list[str]:
    aliases = _tensor_aliases(graph)
    flatten_transforms = _flatten_layout_transforms(graph)
    tensor_symbols = _tensor_symbol_map(graph, layers)
    model_inputs = {tensor.name for tensor in graph.inputs}
    model_outputs = {tensor.name for tensor in graph.outputs}
    last_output_expr = "output"

    def tensor_expr(name: str) -> str | None:
        resolved = _resolve_tensor_alias(name, aliases)
        if resolved in model_inputs:
            return "input"
        if resolved in model_outputs:
            return "output"
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
            if constant_idx is not None and i == constant_idx and constant_symbol:
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
        elif layer["kind"] == "concat":
            lines.extend(_cmsis_concat_call(layer, output_expr))
        elif layer["kind"] == "softmax":
            lines.extend(_cmsis_softmax_call(layer, output_expr))
        else:
            lines.extend(_cmsis_fc_call(layer, output_expr))
        last_output_expr = output_expr

    output_expr = tensor_expr(graph.outputs[0].name) if graph.outputs else None
    output_expr = output_expr or last_output_expr
    if output_expr != "output":
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
    for tensor_name, transform in _flatten_layout_transforms(graph).items():
        symbol = str(transform["symbol"])
        size = int(transform["size"])
        if tensor_name in declared or size <= 0:
            continue
        declared[tensor_name] = size
        lines.append(f"static int8_t {symbol}[{size}u];")
    return lines


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
        "Clip",
        "DequantizeLinear",
        "Dropout",
        "Flatten",
        "QuantizeLinear",
        "Relu",
        "Reshape",
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
    if layer["kind"] == "add":
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


def _graph_output_size(graph: ModelGraph) -> int:
    if not graph.outputs:
        return 1
    return graph.outputs[0].element_count or 1


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
            "    }",
            "",
        ]
    )
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
    for input_index, dims in enumerate(input_dims):
        input_n, input_h, input_w, input_c = dims
        input_expr = input_exprs[input_index] if input_index < len(input_exprs) else "current_input"
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
        if "weight_values" not in layer:
            continue
        weight_values = _int_values(layer["weight_values"])
        if layer["kind"] == "add":
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
            "Conv",
            "Concat",
            "Gemm",
            "MatMul",
            "QLinearAdd",
            "QLinearConv",
            "QLinearMatMul",
        }:
            continue
        if node.op_type in {"Conv", "QLinearConv"}:
            action = (
                "arm_depthwise_conv_wrapper_s8"
                if _is_depthwise_conv_node(node)
                else "arm_convolve_wrapper_s8"
            )
        elif node.op_type == "QLinearAdd":
            action = "arm_elementwise_add_s8"
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
        elif mapping.onnx_op in {"Gemm", "MatMul", "QLinearMatMul"}:
            layer = _fc_layer(graph, mapping)
        elif mapping.onnx_op == "QLinearAdd":
            layer = _add_layer(graph, mapping)
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
        "input_tensors": [node.inputs[0], node.inputs[3]] if len(node.inputs) > 3 else [],
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


def _concat_layer(graph: ModelGraph, mapping: OpMapping) -> dict[str, Any] | None:
    node = _node_by_name(graph, mapping.node_name)
    if node is None or not node.inputs or not node.outputs:
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
    return {
        "kind": "concat",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": list(node.inputs),
        "output_tensors": list(node.outputs),
        "axis": axis,
        "layout": graph.layout,
        "input_dims": input_dims,
        "output_dims": output_dims,
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
    row_size = input_shape[-1]
    element_count = element_count_from_shape(input_shape)
    if not isinstance(row_size, int) or element_count is None:
        return None
    return {
        "kind": "softmax",
        "index": mapping.index,
        "name": mapping.node_name,
        "symbol": _c_symbol(f"nanoc_{mapping.node_name}"),
        "input_tensors": [node.inputs[0]] if node.inputs else [],
        "output_tensors": list(node.outputs),
        "num_rows": max(1, element_count // int(row_size)),
        "row_size": int(row_size),
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

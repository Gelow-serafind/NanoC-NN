from __future__ import annotations

import re

from .model import (
    BufferPlan,
    CodegenOptions,
    MemoryPlan,
    ModelGraph,
    OpMapping,
    dtype_size_bytes,
    element_count_from_shape,
)


def plan_memory(
    graph: ModelGraph,
    mappings: list[OpMapping],
    options: CodegenOptions,
    *,
    weights_header_size: int = 0,
) -> MemoryPlan:
    input_buffers = [
        _tensor_buffer(f"input:{tensor.name}", tensor.shape, "model input int8 buffer")
        for tensor in graph.inputs
    ]
    output_buffers = [
        _tensor_buffer(f"output:{tensor.name}", tensor.shape, "model output int8 buffer")
        for tensor in graph.outputs
    ]

    activation_candidates: list[BufferPlan] = []
    for mapping in mappings:
        if mapping.status == "folded":
            continue
        for output_name, shape in mapping.output_shapes.items():
            activation_candidates.append(
                _tensor_buffer(
                    f"activation:{output_name}",
                    shape,
                    f"output of node {mapping.index}:{mapping.node_name}",
                )
            )
    activation_buffers = _two_largest_activation_buffers(activation_candidates)
    scratch_buffers = [
        _scratch_buffer(mapping, options) for mapping in mappings if mapping.needs_scratch
    ]

    weight_flash_bytes = _estimate_weight_flash(graph, weights_header_size)
    total_sram = (
        sum(item.size_bytes for item in input_buffers)
        + sum(item.size_bytes for item in output_buffers)
        + sum(item.size_bytes for item in activation_buffers)
        + max((item.size_bytes for item in scratch_buffers), default=0)
    )
    total_flash = weight_flash_bytes

    sram_budget = parse_budget(options.sram_budget)
    flash_budget = parse_budget(options.flash_budget)
    notes = [
        "activation and scratch sizes are conservative first-version estimates",
        (
            "generated FC s8 code calls CMSIS-NN buffer size getters at runtime; "
            "non-FC scratch sizes remain conservative estimates"
        ),
    ]
    if graph.layout == "NCHW":
        notes.append("converter layout is NCHW while CMSIS-NN runtime is NHWC")
    if not graph.has_quantization:
        notes.append(
            "float32 converter weights are counted as source flash, not final int8 weights"
        )

    return MemoryPlan(
        input_buffers=input_buffers,
        output_buffers=output_buffers,
        activation_buffers=activation_buffers,
        scratch_buffers=scratch_buffers,
        weight_flash_bytes=weight_flash_bytes,
        total_sram_bytes=total_sram,
        total_flash_bytes=total_flash,
        sram_budget_bytes=sram_budget,
        flash_budget_bytes=flash_budget,
        sram_budget_status=_budget_status(total_sram, sram_budget),
        flash_budget_status=_budget_status(total_flash, flash_budget),
        notes=notes,
    )


def parse_budget(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip().lower()
    match = re.fullmatch(r"(\d+)([km]?)b?", text)
    if not match:
        return None
    amount = int(match.group(1))
    suffix = match.group(2)
    if suffix == "k":
        return amount * 1024
    if suffix == "m":
        return amount * 1024 * 1024
    return amount


def _tensor_buffer(name: str, shape: list[int | str | None], reason: str) -> BufferPlan:
    elements = element_count_from_shape(shape)
    size = 0 if elements is None else elements
    return BufferPlan(name=name, size_bytes=size, reason=reason)


def _two_largest_activation_buffers(candidates: list[BufferPlan]) -> list[BufferPlan]:
    sorted_candidates = sorted(candidates, key=lambda item: item.size_bytes, reverse=True)
    selected = sorted_candidates[:2]
    if not selected:
        selected = [
            BufferPlan("activation:empty_a", 1, "minimum C99-safe placeholder"),
            BufferPlan("activation:empty_b", 1, "minimum C99-safe placeholder"),
        ]
    if len(selected) == 1:
        selected.append(BufferPlan("activation:secondary_placeholder", 1, "double-buffer reserve"))
    return [
        BufferPlan("activation:a", max(1, selected[0].size_bytes), selected[0].reason),
        BufferPlan("activation:b", max(1, selected[1].size_bytes), selected[1].reason),
    ]


def _scratch_buffer(mapping: OpMapping, options: CodegenOptions) -> BufferPlan:
    if "fully_connected" in mapping.cmsis_action:
        output_channels = _largest_last_dim(mapping.output_shapes)
        if options.resolved_backend == "mve":
            estimate = max(1, output_channels * 4)
        else:
            estimate = 1
        return BufferPlan(
            name=f"scratch:{mapping.index}:{mapping.node_name}",
            size_bytes=estimate,
            reason=(
                "reserved for arm_fully_connected_s8_get_buffer_size; generated C "
                "checks the official CMSIS-NN getter at runtime"
            ),
        )

    largest_input = 0
    for shape in mapping.input_shapes.values():
        elements = element_count_from_shape(shape)
        if elements is not None:
            largest_input = max(largest_input, elements)
    if "convolve" in mapping.cmsis_action:
        estimate = max(1, largest_input * 2)
    elif "avgpool" in mapping.cmsis_action:
        estimate = max(1, largest_input)
    else:
        estimate = max(1, largest_input // 2)
    return BufferPlan(
        name=f"scratch:{mapping.index}:{mapping.node_name}",
        size_bytes=estimate,
        reason=f"heuristic scratch estimate for {mapping.cmsis_action}",
    )


def _largest_last_dim(shape_map: dict[str, list[int | str | None]]) -> int:
    largest = 0
    for shape in shape_map.values():
        if shape and isinstance(shape[-1], int):
            largest = max(largest, shape[-1])
    return largest


def _estimate_weight_flash(graph: ModelGraph, weights_header_size: int) -> int:
    if weights_header_size > 0:
        return weights_header_size
    total = 0
    for initializer in graph.initializers:
        if not initializer.is_c_exportable:
            continue
        elements = initializer.element_count
        if elements is None:
            continue
        total += elements * dtype_size_bytes(initializer.elem_type)
    return total


def _budget_status(used: int, budget: int | None) -> str:
    if budget is None:
        return "not_configured"
    if used <= budget:
        return "ok"
    return "over_budget"

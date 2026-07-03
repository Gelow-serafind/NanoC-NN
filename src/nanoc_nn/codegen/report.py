from __future__ import annotations

from pathlib import Path

from .model import (
    CodegenOptions,
    GenerationResult,
    MemoryPlan,
    ModelGraph,
    OpMapping,
    QuantizationIssue,
    ValidationIssue,
)
from .renderer import render_text


def write_reports(result: GenerationResult, options: CodegenOptions) -> list[Path]:
    reports_dir = result.out_dir / "reports"
    files = {
        "codegen_report.txt": codegen_report(result, options),
        "op_mapping.md": op_mapping_report(result.mappings),
        "memory_plan.md": memory_plan_report(result.memory_plan, options),
        "quantization.md": quantization_report(result.quantization_issues, result.model_graph),
        "unsupported_ops.md": unsupported_ops_report(result.mappings),
        "target_report.md": target_report(result.memory_plan, options),
        "source_notice.md": source_notice_report(options),
        "firmware_integration.md": firmware_integration_report(options),
    }
    written: list[Path] = []
    for name, content in files.items():
        path = reports_dir / name
        render_text(path, content)
        written.append(path)
    return written


def codegen_report(result: GenerationResult, options: CodegenOptions) -> str:
    issue_counts = _issue_counts(result.validation_issues)
    lines = [
        "NanoC-NN CMSIS-NN Codegen Report",
        "",
        f"status: {result.status}",
        f"input_dir: {result.input_dir}",
        f"out_dir: {result.out_dir}",
        f"model_path: {result.model_graph.model_path}",
        f"graph_name: {result.model_graph.graph_name or 'unnamed'}",
        f"layout: {result.model_graph.layout}",
        f"target: {options.target}",
        f"backend: {options.resolved_backend}",
        f"cmsis_nn_root: {options.cmsis_nn_root or 'not provided'}",
        f"cmsis_path: {options.cmsis_path or 'not provided'}",
        f"cmsis_version: {options.cmsis_version or 'unknown'}",
        f"nodes: {len(result.model_graph.nodes)}",
        f"mappings_blocked_or_unsupported: {len(result.blocking_mappings)}",
        f"quantization_issues: {len(result.quantization_issues)}",
        f"schema_warnings: {issue_counts.get('warning', 0)}",
        f"schema_errors: {issue_counts.get('error', 0)}",
        f"sram_bytes_estimated: {result.memory_plan.total_sram_bytes}",
        f"flash_bytes_estimated: {result.memory_plan.total_flash_bytes}",
        "",
        "Generated files:",
    ]
    lines.extend(f"- {path.relative_to(result.out_dir)}" for path in result.generated_files)
    lines.extend(["", "Validation issues:"])
    lines.extend(_validation_issue_lines(result.validation_issues))
    lines.extend(["", "Blocking mappings:"])
    blockers = result.blocking_mappings
    if blockers:
        lines.extend(
            f"- node {item.index} {item.node_name} ({item.onnx_op}): {item.reason}"
            for item in blockers
        )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def op_mapping_report(mappings: list[OpMapping]) -> str:
    lines = [
        "# ONNX 到 CMSIS-NN 映射报告",
        "",
        "| # | 节点 | ONNX Op | Converter 状态 | Codegen 状态 | CMSIS-NN 动作 | 原因 | Layout |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in mappings:
        lines.append(
            "| "
            f"{item.index} | `{_escape(item.node_name)}` | `{item.onnx_op}` | "
            f"`{item.converter_status}` | `{item.status}` | `{_escape(item.cmsis_action)}` | "
            f"{_escape(item.reason)} | {_escape(item.layout_note)} |"
        )
    lines.append("")
    return "\n".join(lines)


def memory_plan_report(memory: MemoryPlan, options: CodegenOptions) -> str:
    lines = [
        "# 静态内存规划",
        "",
        f"- Target：`{options.target}`",
        f"- Backend：`{options.resolved_backend}`",
        f"- SRAM 估算：`{memory.total_sram_bytes}` bytes",
        f"- Flash 估算：`{memory.total_flash_bytes}` bytes",
        (
            f"- SRAM 预算：`{_budget(memory.sram_budget_bytes)}`，"
            f"状态：`{memory.sram_budget_status}`"
        ),
        (
            f"- Flash 预算：`{_budget(memory.flash_budget_bytes)}`，"
            f"状态：`{memory.flash_budget_status}`"
        ),
        "",
    ]
    lines.extend(_buffer_section("输入 buffer", memory.input_buffers))
    lines.extend(_buffer_section("输出 buffer", memory.output_buffers))
    lines.extend(_buffer_section("激活 buffer", memory.activation_buffers))
    lines.extend(_buffer_section("Scratch buffer", memory.scratch_buffers))
    lines.extend(["## 说明", ""])
    lines.extend(f"- {note}" for note in memory.notes)
    lines.append("")
    return "\n".join(lines)


def quantization_report(
    issues: list[QuantizationIssue],
    graph: ModelGraph,
) -> str:
    quantization = graph.quantization or {}
    tensors = quantization.get("tensors", {}) if isinstance(quantization, dict) else {}
    weights = quantization.get("weights", {}) if isinstance(quantization, dict) else {}
    nodes = quantization.get("nodes", {}) if isinstance(quantization, dict) else {}
    lines = [
        "# 量化参数报告",
        "",
        f"- Converter 是否提供量化 section：`{str(graph.has_quantization).lower()}`",
        f"- Tensor 量化数量：`{len(tensors) if isinstance(tensors, dict) else 0}`",
        f"- Quantized weight 数量：`{len(weights) if isinstance(weights, dict) else 0}`",
        f"- Node 量化数量：`{len(nodes) if isinstance(nodes, dict) else 0}`",
        "",
    ]
    if isinstance(nodes, dict) and nodes:
        lines.extend(
            [
                "## CMSIS-NN 节点参数",
                "",
                "| 节点 | API | multiplier | shift | input_offset | output_offset | activation |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for node_name, node_quant in nodes.items():
            if not isinstance(node_quant, dict):
                continue
            cmsis_nn = node_quant.get("cmsis_nn", {})
            if not isinstance(cmsis_nn, dict):
                continue
            lines.append(
                "| "
                f"`{_escape(node_name)}` | `{_escape(cmsis_nn.get('api', 'unknown'))}` | "
                f"{cmsis_nn.get('multiplier', '-')} | {cmsis_nn.get('shift', '-')} | "
                f"{cmsis_nn.get('input_offset', '-')} | {cmsis_nn.get('output_offset', '-')} | "
                f"`{cmsis_nn.get('activation_min', '-')}`.."
                f"`{cmsis_nn.get('activation_max', '-')}` |"
            )
        lines.append("")
    if not issues:
        lines.extend(["当前未发现量化阻塞项。", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "| 节点 | ONNX Op | 状态 | 缺口 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for issue in issues:
        lines.append(
            "| "
            f"`{_escape(issue.node_name)}` | `{issue.onnx_op}` | `{issue.status}` | "
            f"{_escape(issue.requirement)} |"
        )
    lines.append("")
    return "\n".join(lines)


def unsupported_ops_report(mappings: list[OpMapping]) -> str:
    blocking = [item for item in mappings if item.status in {"blocked", "unsupported"}]
    lines = [
        "# Unsupported / Blocked Ops",
        "",
        "| # | 节点 | ONNX Op | 状态 | 原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    if not blocking:
        lines.append("| - | - | - | - | none |")
    for item in blocking:
        lines.append(
            "| "
            f"{item.index} | `{_escape(item.node_name)}` | `{item.onnx_op}` | "
            f"`{item.status}` | {_escape(item.reason)} |"
        )
    lines.append("")
    return "\n".join(lines)


def target_report(memory: MemoryPlan, options: CodegenOptions) -> str:
    lines = [
        "# 目标平台报告",
        "",
        f"- 目标内核：`{options.target}`",
        f"- CMSIS-NN backend：`{options.resolved_backend}`",
        "- 典型平台：STM32/GD32 等 Cortex-M MCU",
        "- 运行环境：裸机 main loop 或 RTOS task",
        "- 动态内存：不使用 `malloc/free`",
        "- HAL/IDE 绑定：不绑定具体厂商 HAL、启动文件、链接脚本或 IDE 工程",
        f"- SRAM 估算：`{memory.total_sram_bytes}` bytes",
        f"- Flash 估算：`{memory.total_flash_bytes}` bytes",
        "",
        "## 接入判断",
        "",
        (
            "- 若报告中存在 `blocked`，生成物只能作为结构化工程骨架和移植准备，"
            "不代表可运行 int8 推理。"
        ),
        "- 若 SRAM/Flash 超预算，应先缩小模型或启用更严格的量化/内存复用策略。",
        "- Cortex-M0/M3 默认使用 scalar，性能风险较高；M4/M7/M33 可考虑 DSP；M55/M85 可考虑 MVE。",
        "",
    ]
    return "\n".join(lines)


def source_notice_report(options: CodegenOptions) -> str:
    lines = [
        "# Source Notice",
        "",
        "- Generated by nanoc_nn.codegen.",
        f"- CMSIS-NN root: `{options.cmsis_nn_root or 'not provided'}`",
        f"- CMSIS-Core path: `{options.cmsis_path or 'not provided'}`",
        f"- CMSIS-NN version label: `{options.cmsis_version or 'unknown'}`",
        "- Vendored CMSIS-NN source in this repository keeps its upstream Apache-2.0 license.",
        "",
    ]
    return "\n".join(lines)


def firmware_integration_report(options: CodegenOptions) -> str:
    return "\n".join(
        [
            "# STM32/GD32 固件接入说明",
            "",
            "## 生成物",
            "",
            "- `include/model.h`：推理接口和 buffer 尺寸宏。",
            "- `include/model_weights.h`：权重入口和 Flash 尺寸宏。",
            "- `src/model.c`：静态 buffer 与节点执行骨架。",
            "- `src/main.c`：仅用于 smoke test，真实固件可不使用。",
            "",
            "## 接入步骤",
            "",
            "1. 将 `include/` 和 `src/model.c` 加入 STM32/GD32 工程。",
            "2. 将 CMSIS-Core 与 CMSIS-NN include/source 加入工程配置。",
            "3. 在用户主循环或 RTOS task 中分配并填充输入 buffer。",
            "4. 调用 `nanoc_model_run(input, output)`。",
            "5. 检查返回值，`0` 表示运行成功，非 0 表示当前生成物仍存在阻塞或不支持项。",
            "",
            "## 当前限制",
            "",
            f"- 目标内核按 `{options.target}` 记录，实际编译参数仍需在用户工程中配置。",
            "- 第一版不生成完整 STM32CubeMX、Keil MDK 或 GD32 IDE 工程。",
            "- 当量化报告存在缺口时，不应把生成物用于真实推理结果验证。",
            "",
        ]
    )


def _buffer_section(title: str, buffers: list) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| 名称 | 大小 bytes | 依据 |",
        "| --- | ---: | --- |",
    ]
    if not buffers:
        lines.append("| 无 | 0 | - |")
    for item in buffers:
        lines.append(f"| `{_escape(item.name)}` | {item.size_bytes} | {_escape(item.reason)} |")
    lines.append("")
    return lines


def _validation_issue_lines(issues: list[ValidationIssue]) -> list[str]:
    if not issues:
        return ["- none"]
    return [f"- {item.level}: {item.field}: {item.message}" for item in issues]


def _issue_counts(issues: list[ValidationIssue]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue.level] = counts.get(issue.level, 0) + 1
    return counts


def _budget(value: int | None) -> str:
    return "not configured" if value is None else str(value)


def _escape(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")

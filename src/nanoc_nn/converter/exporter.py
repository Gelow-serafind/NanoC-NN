from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .c_writer import write_weights_header
from .model import ConversionOptions, ModelInfo
from .parser import parse_model
from .shape import shape_to_text


def convert_model(options: ConversionOptions) -> ModelInfo:
    model_info = parse_model(options)
    options.out_dir.mkdir(parents=True, exist_ok=True)
    write_output_readme(model_info, options.out_dir / "README.md")
    write_conversion_report(model_info, options.out_dir / "conversion_report.txt")
    write_model_summary(model_info, options.out_dir / "model_summary.md")
    write_model_graph_json(model_info, options.out_dir / "model_graph.json")
    write_weights_header(model_info, options.out_dir / "weights.h", prefix=options.prefix)
    return model_info


def write_model_graph_json(model_info: ModelInfo, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(model_info.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_model_summary(model_info: ModelInfo, output_path: Path) -> None:
    output_path.write_text(_summary_markdown(model_info), encoding="utf-8")


def write_output_readme(model_info: ModelInfo, output_path: Path) -> None:
    output_path.write_text(_output_readme_markdown(model_info), encoding="utf-8")


def write_conversion_report(model_info: ModelInfo, output_path: Path) -> None:
    output_path.write_text(_report_text(model_info), encoding="utf-8")


def _summary_markdown(model_info: ModelInfo) -> str:
    lines: list[str] = [
        "# NanoC-NN ONNX 转换摘要",
        "",
        "## 模型信息",
        "",
        f"- 输入模型：`{model_info.model_path}`",
        f"- Graph：`{model_info.graph_name or 'unnamed'}`",
        f"- IR version：`{model_info.ir_version}`",
        f"- Opset：`{json.dumps(model_info.opsets, ensure_ascii=False)}`",
        f"- Producer：`{model_info.producer_name or 'unknown'} "
        f"{model_info.producer_version or ''}`",
        f"- Layout 标注：`{model_info.layout}`",
        "",
    ]

    lines.extend(_tensor_section("输入", model_info.inputs))
    lines.extend(_tensor_section("输出", model_info.outputs))
    lines.extend(_initializer_section(model_info))
    lines.extend(_node_section(model_info))
    lines.extend(_quantization_section(model_info))
    lines.extend(_message_section("警告", model_info.warnings))
    lines.extend(_message_section("错误", model_info.errors))
    lines.extend(
        [
            "## 输出文件",
            "",
            "- `README.md`：输出目录说明和阅读顺序。",
            "- `conversion_report.txt`：纯文本转换报告。",
            "- `model_summary.md`：人类可读结构摘要。",
            "- `model_graph.json`：机器可读图结构。",
            "- `weights.h`：float32 权重 C99 头文件。",
            "",
        ]
    )
    return "\n".join(lines)


def _tensor_section(title: str, tensors: list[Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| 名称 | 数据类型 | shape | 原始 shape |",
        "| --- | --- | --- | --- |",
    ]
    if not tensors:
        lines.append("| 无 | - | - | - |")
    for tensor in tensors:
        lines.append(
            "| "
            f"`{_escape(tensor.name)}` | `{tensor.elem_type}` | "
            f"`{shape_to_text(tensor.shape)}` | `{shape_to_text(tensor.raw_shape)}` |"
        )
    lines.append("")
    return lines


def _initializer_section(model_info: ModelInfo) -> list[str]:
    lines = [
        "## Initializer 与权重",
        "",
        "| ONNX 名称 | C 名称 | 角色 | C 导出 | 数据类型 | shape | 元素数量 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if not model_info.initializers:
        lines.append("| 无 | - | - | - | - | - | - |")
    for item in model_info.initializers:
        lines.append(
            "| "
            f"`{_escape(item.name)}` | `{item.c_name}` | `{item.role}` | "
            f"`{str(item.is_c_exportable).lower()}` | `{item.elem_type}` | "
            f"`{shape_to_text(item.shape)}` | `{item.element_count}` |"
        )
    lines.append("")
    return lines


def _node_section(model_info: ModelInfo) -> list[str]:
    lines = [
        "## 计算图节点",
        "",
        "| # | 名称 | 算子 | 状态 | 输入 | 输入 shape | 输出 | 输出 shape | 权重 | 归一化属性 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if not model_info.nodes:
        lines.append("| - | 无 | - | - | - | - | - | - | - | - |")
    for node in model_info.nodes:
        lines.append(
            "| "
            f"{node.index} | `{_escape(node.name)}` | `{node.op_type}` | `{node.status}` | "
            f"{_code_list(node.inputs)} | {_shape_map_text(node.inputs, node.input_shapes)} | "
            f"{_code_list(node.outputs)} | {_shape_map_text(node.outputs, node.output_shapes)} | "
            f"{_code_list(node.weights)} | "
            f"`{_escape(json.dumps(node.normalized_attributes, ensure_ascii=False))}` |"
        )
    lines.append("")
    return lines


def _quantization_section(model_info: ModelInfo) -> list[str]:
    quantization = model_info.quantization
    lines = [
        "## 量化信息",
        "",
        f"- 是否存在量化 section：`{str(bool(quantization)).lower()}`",
    ]
    if not quantization:
        lines.extend(
            [
                "- 当前模型未提取到 Q/DQ 量化信息；CMSIS-NN int8 codegen 将保持 blocked。",
                "",
            ]
        )
        return lines
    tensors = quantization.get("tensors", {})
    weights = quantization.get("weights", {})
    nodes = quantization.get("nodes", {})
    contract = quantization.get("int8_contract", {})
    contract_status = (
        contract.get("status", "unknown") if isinstance(contract, dict) else "unknown"
    )
    lines.extend(
        [
            f"- int8 合同状态：`{contract_status}`",
            f"- Tensor 量化数量：`{len(tensors) if isinstance(tensors, dict) else 0}`",
            f"- Quantized weight 数量：`{len(weights) if isinstance(weights, dict) else 0}`",
            f"- Node 量化数量：`{len(nodes) if isinstance(nodes, dict) else 0}`",
            "",
        ]
    )
    return lines


def _message_section(title: str, messages: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not messages:
        lines.append("- 无")
    else:
        lines.extend(f"- {message}" for message in messages)
    lines.append("")
    return lines


def _code_list(values: list[str]) -> str:
    if not values:
        return "-"
    return "<br>".join(f"`{_escape(value)}`" for value in values)


def _shape_map_text(names: list[str], shape_map: dict[str, list[Any]]) -> str:
    if not names:
        return "-"
    items: list[str] = []
    for name in names:
        shape = shape_map.get(name)
        shape_text = "unknown" if shape is None else shape_to_text(shape)
        items.append(f"`{_escape(name)}`: `{shape_text}`")
    return "<br>".join(items)


def _output_readme_markdown(model_info: ModelInfo) -> str:
    warning_count = len(model_info.warnings)
    exported_weight_count = len(model_info.c_exportable_initializers)
    lines = [
        "# NanoC-NN 转换输出",
        "",
        f"- 输入模型：`{model_info.model_path}`",
        f"- 节点数量：`{len(model_info.nodes)}`",
        f"- Initializer 数量：`{len(model_info.initializers)}`",
        f"- C 权重数量：`{exported_weight_count}`",
        f"- 量化 section：`{str(bool(model_info.quantization)).lower()}`",
        f"- 警告数量：`{warning_count}`",
        "",
        "## 建议阅读顺序",
        "",
        "1. `conversion_report.txt`：先看整体转换是否有警告。",
        "2. `model_summary.md`：查看网络层顺序、算子属性、输入输出 shape 和权重对应关系。",
        "3. `weights.h`：查看可直接被 C99 工程包含的 float32 权重数组。",
        "4. `model_graph.json`：需要脚本处理时读取这个结构化文件。",
        "",
        "## 文件说明",
        "",
        "- `README.md`：当前输出目录说明。",
        "- `conversion_report.txt`：纯文本报告，适合快速查看或贴到日志里。",
        "- `model_summary.md`：面向人工走读的网络结构文档。",
        (
            "- `model_graph.json`：包含输入、输出、节点、属性、shape、权重映射和警告信息，"
            "是后续 CMSIS-NN codegen 的主要输入。"
        ),
        "- `weights.h`：由 ONNX float32 参数 initializer 导出的 C99 权重头文件。",
        "- Q/DQ 模型会在 `model_graph.json.quantization` 中携带 int8 codegen 所需量化资料。",
        "",
        "## 当前边界",
        "",
        "- 初期只导出 float32 参数权重，shape 常量等辅助 initializer 不写入 C 权重数组。",
        (
            "- 量化权重不写入 `weights.h`，由后续 CMSIS-NN codegen 从 "
            "`model_graph.json.quantization` 消费。"
        ),
        "- `--layout` 仅作为标注，不做自动 transpose。",
        (
            "- 当前输出目录不直接包含 `model.c`；完整 CMSIS-NN 推理代码由 "
            "`nanoc_nn.codegen` 生成。"
        ),
        "- 遇到未知算子会在报告中标注，`--strict` 模式下会直接失败。",
        "",
    ]
    return "\n".join(lines)


def _report_text(model_info: ModelInfo) -> str:
    unsupported_ops = sorted(
        {node.op_type for node in model_info.nodes if node.status == "unsupported"}
    )
    lines = [
        "NanoC-NN ONNX Conversion Report",
        "",
        f"model: {model_info.model_path}",
        f"graph: {model_info.graph_name or 'unnamed'}",
        f"ir_version: {model_info.ir_version}",
        f"opsets: {json.dumps(model_info.opsets, ensure_ascii=False)}",
        f"layout: {model_info.layout}",
        f"inputs: {len(model_info.inputs)}",
        f"outputs: {len(model_info.outputs)}",
        f"nodes: {len(model_info.nodes)}",
        f"initializers: {len(model_info.initializers)}",
        f"c_exportable_initializers: {len(model_info.c_exportable_initializers)}",
        f"quantization: {'present' if model_info.quantization else 'none'}",
        f"int8_contract: {_int8_contract_status(model_info.quantization)}",
        f"unsupported_ops: {', '.join(unsupported_ops) if unsupported_ops else 'none'}",
        "",
        "Output files:",
        "- README.md",
        "- conversion_report.txt",
        "- model_summary.md",
        "- model_graph.json",
        "- weights.h",
        "",
        "Nodes:",
    ]
    for node in model_info.nodes:
        lines.append(f"- #{node.index} {node.op_type} {node.name} [{node.status}]")
    lines.extend(["", "Initializers:"])
    for initializer in model_info.initializers:
        lines.append(
            f"- {initializer.name} -> {initializer.c_name}, "
            f"role={initializer.role}, c_export={initializer.is_c_exportable}, "
            f"dtype={initializer.elem_type}, shape={shape_to_text(initializer.shape)}, "
            f"size={initializer.element_count}"
        )
    lines.extend(["", "Warnings:"])
    if model_info.warnings:
        lines.extend(f"- {warning}" for warning in model_info.warnings)
    else:
        lines.append("- none")
    lines.extend(["", "Errors:"])
    if model_info.errors:
        lines.extend(f"- {error}" for error in model_info.errors)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _int8_contract_status(quantization: dict[str, Any]) -> str:
    if not quantization:
        return "none"
    contract = quantization.get("int8_contract")
    if not isinstance(contract, dict):
        return "missing"
    return str(contract.get("status", "unknown"))


def _escape(value: str) -> str:
    return value.replace("|", "\\|")

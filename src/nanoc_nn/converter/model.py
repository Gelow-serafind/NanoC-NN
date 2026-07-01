from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ShapeValue = int | str | None


@dataclass(frozen=True)
class ConversionOptions:
    model_path: Path
    out_dir: Path
    layout: str = "NCHW"
    prefix: str = "nanoc"
    batch_size: int = 1
    strict: bool = False
    verbose: bool = False


@dataclass
class TensorInfo:
    name: str
    elem_type: str
    shape: list[ShapeValue]
    raw_shape: list[ShapeValue]
    dynamic_axes: list[int] = field(default_factory=list)

    @property
    def is_dynamic(self) -> bool:
        return bool(self.dynamic_axes)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elem_type": self.elem_type,
            "shape": self.shape,
            "raw_shape": self.raw_shape,
            "dynamic_axes": self.dynamic_axes,
            "is_dynamic": self.is_dynamic,
        }


@dataclass
class InitializerInfo:
    name: str
    elem_type: str
    shape: list[int]
    element_count: int
    c_name: str
    role: str
    array: Any = field(repr=False)

    @property
    def is_float32(self) -> bool:
        return self.elem_type == "FLOAT"

    @property
    def is_c_exportable(self) -> bool:
        return self.role == "parameter" and self.is_float32

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elem_type": self.elem_type,
            "shape": self.shape,
            "element_count": self.element_count,
            "c_name": self.c_name,
            "role": self.role,
            "is_float32": self.is_float32,
            "is_c_exportable": self.is_c_exportable,
        }


@dataclass
class NodeInfo:
    index: int
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any]
    normalized_attributes: dict[str, Any]
    input_shapes: dict[str, list[ShapeValue]]
    output_shapes: dict[str, list[ShapeValue]]
    weights: list[str]
    status: str

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
            "normalized_attributes": self.normalized_attributes,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "weights": self.weights,
            "status": self.status,
        }


@dataclass
class ModelInfo:
    model_path: Path
    ir_version: int
    opsets: dict[str, int]
    producer_name: str
    producer_version: str
    graph_name: str
    layout: str
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    initializers: list[InitializerInfo]
    nodes: list[NodeInfo]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def c_exportable_initializers(self) -> list[InitializerInfo]:
        return [item for item in self.initializers if item.is_c_exportable]

    def to_json(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "ir_version": self.ir_version,
            "opsets": self.opsets,
            "producer_name": self.producer_name,
            "producer_version": self.producer_version,
            "graph_name": self.graph_name,
            "layout": self.layout,
            "inputs": [item.to_json() for item in self.inputs],
            "outputs": [item.to_json() for item in self.outputs],
            "initializers": [item.to_json() for item in self.initializers],
            "nodes": [item.to_json() for item in self.nodes],
            "warnings": self.warnings,
            "errors": self.errors,
            "unsupported_ops": sorted(
                {node.op_type for node in self.nodes if node.status == "unsupported"}
            ),
            "c_exportable_initializers": [
                item.name for item in self.c_exportable_initializers
            ],
        }


class ConversionError(RuntimeError):
    """Raised when strict conversion cannot continue."""

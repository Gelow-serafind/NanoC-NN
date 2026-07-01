from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ShapeValue = int | str | None


@dataclass(frozen=True)
class CodegenOptions:
    input_dir: Path
    out_dir: Path
    cmsis_nn_root: Path | None = None
    cmsis_path: Path | None = None
    cmsis_version: str | None = None
    target: str = "cortex-m4"
    backend: str | None = None
    sram_budget: str | None = None
    flash_budget: str | None = None
    project_style: str = "cmake"
    strict: bool = False
    verbose: bool = False

    @property
    def resolved_backend(self) -> str:
        if self.backend:
            return self.backend
        target = self.target.lower()
        if target in {"cortex-m55", "cortex-m85"}:
            return "mve"
        if target in {"cortex-m4", "cortex-m7", "cortex-m33", "cortex-m35p"}:
            return "dsp"
        return "scalar"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    elem_type: str
    shape: list[ShapeValue]
    raw_shape: list[ShapeValue] = field(default_factory=list)
    dynamic_axes: list[int] = field(default_factory=list)
    is_dynamic: bool = False

    @property
    def element_count(self) -> int | None:
        return element_count_from_shape(self.shape)


@dataclass(frozen=True)
class InitializerSpec:
    name: str
    elem_type: str
    shape: list[ShapeValue]
    element_count: int | None
    c_name: str
    role: str
    is_float32: bool = False
    is_c_exportable: bool = False


@dataclass(frozen=True)
class NodeSpec:
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
    converter_status: str


@dataclass(frozen=True)
class ModelGraph:
    model_path: str
    ir_version: int | None
    opsets: dict[str, int]
    producer_name: str
    producer_version: str
    graph_name: str
    layout: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    initializers: list[InitializerSpec]
    nodes: list[NodeSpec]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unsupported_ops: list[str] = field(default_factory=list)
    c_exportable_initializers: list[str] = field(default_factory=list)
    schema_version: str | None = None
    quantization: dict[str, Any] | None = None

    @property
    def has_quantization(self) -> bool:
        return bool(self.quantization)


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    field: str
    message: str


@dataclass
class CodegenInput:
    model_graph: ModelGraph
    raw_model_graph: dict[str, Any]
    input_dir: Path
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    conversion_report: str = ""
    weights_header: str = ""


@dataclass(frozen=True)
class OpMapping:
    index: int
    node_name: str
    onnx_op: str
    converter_status: str
    status: str
    cmsis_action: str
    reason: str
    inputs: list[str]
    outputs: list[str]
    input_shapes: dict[str, list[ShapeValue]]
    output_shapes: dict[str, list[ShapeValue]]
    needs_quantization: bool = False
    needs_scratch: bool = False
    layout_note: str = ""

    @property
    def is_blocking(self) -> bool:
        return self.status in {"blocked", "unsupported"}


@dataclass(frozen=True)
class QuantizationIssue:
    node_name: str
    onnx_op: str
    requirement: str
    status: str = "blocked"


@dataclass(frozen=True)
class BufferPlan:
    name: str
    size_bytes: int
    reason: str


@dataclass(frozen=True)
class MemoryPlan:
    input_buffers: list[BufferPlan]
    output_buffers: list[BufferPlan]
    activation_buffers: list[BufferPlan]
    scratch_buffers: list[BufferPlan]
    weight_flash_bytes: int
    total_sram_bytes: int
    total_flash_bytes: int
    sram_budget_bytes: int | None = None
    flash_budget_bytes: int | None = None
    sram_budget_status: str = "not_configured"
    flash_budget_status: str = "not_configured"
    notes: list[str] = field(default_factory=list)

    @property
    def max_scratch_bytes(self) -> int:
        return max((item.size_bytes for item in self.scratch_buffers), default=0)


@dataclass(frozen=True)
class GenerationResult:
    status: str
    input_dir: Path
    out_dir: Path
    model_graph: ModelGraph
    validation_issues: list[ValidationIssue]
    mappings: list[OpMapping]
    quantization_issues: list[QuantizationIssue]
    memory_plan: MemoryPlan
    generated_files: list[Path]

    @property
    def blocking_mappings(self) -> list[OpMapping]:
        return [item for item in self.mappings if item.is_blocking]


class CodegenError(RuntimeError):
    """Raised when CMSIS-NN code generation cannot continue."""


def element_count_from_shape(shape: list[ShapeValue]) -> int | None:
    total = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return None
        total *= dim
    return total


def dtype_size_bytes(elem_type: str, *, target_int8: bool = False) -> int:
    if target_int8:
        return 1
    sizes = {
        "BOOL": 1,
        "INT8": 1,
        "UINT8": 1,
        "INT16": 2,
        "UINT16": 2,
        "FLOAT16": 2,
        "BFLOAT16": 2,
        "INT32": 4,
        "UINT32": 4,
        "FLOAT": 4,
        "INT64": 8,
        "UINT64": 8,
        "DOUBLE": 8,
    }
    return sizes.get(elem_type, 4)

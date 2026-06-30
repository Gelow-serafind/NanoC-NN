from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass
class CodegenInput:
    model_graph: dict[str, Any]
    input_dir: Path
    warnings: list[str] = field(default_factory=list)


class CodegenError(RuntimeError):
    """Raised when CMSIS-NN code generation cannot continue."""

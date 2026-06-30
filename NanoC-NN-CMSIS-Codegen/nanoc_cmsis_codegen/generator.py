from __future__ import annotations

from .model import CodegenError, CodegenOptions


def generate_project(options: CodegenOptions) -> None:
    raise CodegenError(
        f"CMSIS-NN project generation is not implemented yet: {options.out_dir}"
    )

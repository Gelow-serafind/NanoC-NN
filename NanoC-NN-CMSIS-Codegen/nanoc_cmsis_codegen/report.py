from __future__ import annotations

from pathlib import Path


def write_placeholder_report(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "codegen_report.txt").write_text(
        "CMSIS-NN code generation is planned but not implemented yet.\n",
        encoding="utf-8",
    )

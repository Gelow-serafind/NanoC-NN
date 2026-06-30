from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoc-cmsis-codegen",
        description="Generate a CMSIS-NN C project from NanoC-NN converter outputs.",
    )
    parser.add_argument("--input", required=True, help="converter output directory")
    parser.add_argument("--out", required=True, help="generated C project directory")
    parser.add_argument("--cmsis-nn-root", help="CMSIS-NN source or installation path")
    parser.add_argument(
        "--cmsis-path",
        help="CMSIS repository path that provides CMSIS/Core/Include",
    )
    parser.add_argument("--cmsis-version", help="target CMSIS-NN version label")
    parser.add_argument("--target", default="cortex-m4", help="target Cortex-M profile")
    parser.add_argument(
        "--backend",
        choices=["scalar", "dsp", "mve"],
        help="CMSIS-NN backend used for mapping and scratch buffer planning",
    )
    parser.add_argument("--sram-budget", help="target SRAM budget for model buffers, e.g. 128K")
    parser.add_argument(
        "--flash-budget",
        help="target Flash budget for weights and code, e.g. 512K",
    )
    parser.add_argument(
        "--project-style",
        default="cmake",
        choices=["cmake"],
        help="generated project style",
    )
    parser.add_argument("--strict", action="store_true", help="fail on unsupported cases")
    parser.add_argument("--verbose", action="store_true", help="print detailed diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    if not input_dir.exists():
        parser.exit(status=2, message=f"error: input directory does not exist: {input_dir}\n")
    if not (input_dir / "model_graph.json").exists():
        parser.exit(
            status=2,
            message=f"error: model_graph.json does not exist in input directory: {input_dir}\n",
        )

    parser.exit(
        status=2,
        message="error: CMSIS-NN code generation is planned but not implemented yet.\n",
    )

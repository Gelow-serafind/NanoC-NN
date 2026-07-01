from __future__ import annotations

import argparse
from pathlib import Path

from .generator import generate_project
from .model import CodegenError, CodegenOptions


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
    parser.add_argument(
        "--sram-budget",
        help="target SRAM budget for model buffers, e.g. 128K",
    )
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

    options = CodegenOptions(
        input_dir=Path(args.input),
        out_dir=Path(args.out),
        cmsis_nn_root=Path(args.cmsis_nn_root) if args.cmsis_nn_root else None,
        cmsis_path=Path(args.cmsis_path) if args.cmsis_path else None,
        cmsis_version=args.cmsis_version,
        target=args.target,
        backend=args.backend,
        sram_budget=args.sram_budget,
        flash_budget=args.flash_budget,
        project_style=args.project_style,
        strict=args.strict,
        verbose=args.verbose,
    )

    try:
        result = generate_project(options)
    except CodegenError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")
    except Exception as exc:  # noqa: BLE001 - keep CLI concise.
        parser.exit(status=1, message=f"error: unexpected codegen failure: {exc}\n")

    print(f"generated: {result.out_dir}")
    print(f"status: {result.status}")
    print(f"nodes: {len(result.model_graph.nodes)}")
    print(f"blocked_or_unsupported: {len(result.blocking_mappings)}")
    print(f"reports: {result.out_dir / 'reports'}")
    if args.verbose:
        for mapping in result.mappings:
            print(
                f"- {mapping.index}:{mapping.node_name} {mapping.onnx_op} "
                f"=> {mapping.status} ({mapping.cmsis_action})"
            )
    return 0 if result.status == "ok" or not args.strict else 2

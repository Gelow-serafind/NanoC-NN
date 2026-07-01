from __future__ import annotations

import argparse
import sys

from .codegen.cli import main as codegen_main
from .converter.cli import main as converter_main
from .pipeline.cli import main as pipeline_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoc",
        description="NanoC-NN integrated command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "convert",
        help="run ONNX converter; remaining arguments are passed through",
    )
    subparsers.add_parser(
        "codegen",
        help="run CMSIS-NN codegen; remaining arguments are passed through",
    )
    subparsers.add_parser(
        "onnx-to-cmsis",
        help="run converter and codegen pipeline; remaining arguments are passed through",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        build_parser().print_help()
        return 2
    if argv[0] in {"-h", "--help"}:
        build_parser().parse_args(argv)
        return 0

    command, passthrough = argv[0], argv[1:]
    if command == "convert":
        return converter_main(passthrough)
    if command == "codegen":
        return codegen_main(passthrough)
    if command == "onnx-to-cmsis":
        return pipeline_main(passthrough)

    parser = build_parser()
    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

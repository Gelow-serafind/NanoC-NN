from __future__ import annotations

import argparse
from pathlib import Path

from .exporter import convert_model
from .model import ConversionError, ConversionOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoc-onnx-converter",
        description="Parse an ONNX model and export NanoC-NN C weight/reference files.",
    )
    parser.add_argument("--model", required=True, help="input ONNX model path")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--layout", default="NCHW", help="layout label, default: NCHW")
    parser.add_argument("--prefix", default="nanoc", help="C symbol prefix, default: nanoc")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="fixed batch size for dynamic batch",
    )
    parser.add_argument("--strict", action="store_true", help="fail on unsupported or unsafe cases")
    parser.add_argument("--verbose", action="store_true", help="print detailed conversion result")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    options = ConversionOptions(
        model_path=Path(args.model),
        out_dir=Path(args.out),
        layout=args.layout,
        prefix=args.prefix,
        batch_size=args.batch_size,
        strict=args.strict,
        verbose=args.verbose,
    )

    try:
        model_info = convert_model(options)
    except ConversionError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")
    except Exception as exc:  # noqa: BLE001 - CLI boundary should not expose traceback by default.
        parser.exit(status=1, message=f"error: unexpected conversion failure: {exc}\n")

    print(f"converted: {options.model_path}")
    print(f"output: {options.out_dir}")
    print(f"nodes: {len(model_info.nodes)}")
    print(f"initializers: {len(model_info.initializers)}")
    if model_info.warnings:
        print(f"warnings: {len(model_info.warnings)}")
        if args.verbose:
            for warning in model_info.warnings:
                print(f"- {warning}")
    else:
        print("warnings: 0")
    return 0

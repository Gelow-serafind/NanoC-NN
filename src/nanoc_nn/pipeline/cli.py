from __future__ import annotations

import argparse
from pathlib import Path

from .runner import PipelineOptions, run_pipeline, safe_prefix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoc-onnx-to-cmsis",
        description=(
            "Run NanoC-NN ONNX Converter and then CMSIS-NN Codegen for one ONNX model."
        ),
    )
    parser.add_argument("--model", required=True, help="input ONNX model path")
    parser.add_argument(
        "--out-root",
        help="pipeline output directory; default is <onnx-dir>/<onnx-stem>-nanoc-cmsis",
    )
    parser.add_argument("--layout", default="NCHW", help="converter layout label")
    parser.add_argument("--prefix", help="C symbol prefix; default uses ONNX file stem")
    parser.add_argument("--batch-size", type=int, default=1, help="fixed dynamic batch size")
    parser.add_argument("--cmsis-nn-root", help="CMSIS-NN source or installation path")
    parser.add_argument(
        "--cmsis-path",
        help="CMSIS repository path that provides CMSIS/Core/Include",
    )
    parser.add_argument("--cmsis-version", help="target CMSIS-NN version label")
    parser.add_argument("--target", default="cortex-m4", help="target Cortex-M profile")
    parser.add_argument("--backend", choices=["scalar", "dsp", "mve"], help="CMSIS-NN backend")
    parser.add_argument("--sram-budget", help="target SRAM budget, e.g. 128K")
    parser.add_argument("--flash-budget", help="target Flash budget, e.g. 512K")
    parser.add_argument("--strict", action="store_true", help="fail on blocked/unsupported cases")
    parser.add_argument("--verbose", action="store_true", help="print detailed diagnostics")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip host C99 smoke compile of generated sources",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        parser.exit(status=2, message=f"error: ONNX model does not exist: {model_path}\n")
    if not model_path.is_file():
        parser.exit(status=2, message=f"error: ONNX model path is not a file: {model_path}\n")

    output_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else model_path.parent / f"{model_path.stem}-nanoc-cmsis"
    )
    options = PipelineOptions(
        model_path=model_path,
        output_root=output_root,
        layout=args.layout,
        prefix=args.prefix or safe_prefix(model_path.stem),
        batch_size=args.batch_size,
        cmsis_nn_root=Path(args.cmsis_nn_root).expanduser().resolve()
        if args.cmsis_nn_root
        else _default_cmsis_nn_root(),
        cmsis_path=Path(args.cmsis_path).expanduser().resolve() if args.cmsis_path else None,
        cmsis_version=args.cmsis_version,
        target=args.target,
        backend=args.backend,
        sram_budget=args.sram_budget,
        flash_budget=args.flash_budget,
        strict=args.strict,
        verbose=args.verbose,
        compile_smoke=not args.no_compile,
    )

    try:
        result = run_pipeline(options)
    except Exception as exc:  # noqa: BLE001 - CLI boundary.
        parser.exit(status=1, message=f"error: pipeline failed: {exc}\n")

    print(f"pipeline output: {result.options.output_root}")
    print(f"converter output: {result.options.converter_dir}")
    print(f"cmsis-codegen output: {result.options.codegen_dir}")
    print(f"codegen status: {result.codegen_result.status}")
    print(f"c99 smoke compile: {result.compile_status}")
    print(f"pipeline report: {result.options.output_root / 'pipeline_report.md'}")
    if options.verbose:
        for mapping in result.codegen_result.mappings:
            print(
                f"- {mapping.index}:{mapping.node_name} {mapping.onnx_op} "
                f"=> {mapping.status} ({mapping.cmsis_action})"
            )
    return 0


def _default_cmsis_nn_root() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "third_party" / "CMSIS-NN"
    return candidate if candidate.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())

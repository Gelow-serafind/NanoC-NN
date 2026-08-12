"""
tdd/scripts/run_numeric_tests.py — ONNX Runtime vs generated C numeric checks.

This runner executes full-model numeric tests:
  1. Generate or locate the ONNX test model.
  2. Run the NanoC-NN ONNX -> CMSIS-NN pipeline.
  3. Run the original ONNX with ONNX Runtime on persistent fixtures.
  4. Compile generated C with CMSIS-NN enabled and run the same inputs.
  5. Compare top1, saturation ratio, and optional absolute error.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
TDD_ROOT = _SCRIPT_DIR.parent
REPO_ROOT = TDD_ROOT.parent
WORK_ROOT = TDD_ROOT / "work"
MODELS_ROOT = WORK_ROOT / "models"
NUMERIC_WORK_ROOT = WORK_ROOT / "numeric"
RESULTS_DIR = TDD_ROOT / "results"
FIXTURES_ROOT = TDD_ROOT / "fixtures"
CMSIS_NN_ROOT = REPO_ROOT / "third_party" / "CMSIS-NN"
WORKDIR_MARKER = ".nanoc_tdd_workdir"

sys.path.insert(0, str(_SCRIPT_DIR))
from cases_registry import CASE_MAP, NumericCheck  # noqa: E402
from run_tests import detect_status, run_pipeline  # noqa: E402


@dataclass
class NumericCaseResult:
    case_id: str
    dataset_id: str
    onnx_path: str
    output_dir: str
    passed: bool = False
    pipeline_status: str = "unknown"
    pipeline_exit_code: int = -1
    compile_ok: bool = False
    run_ok: bool = False
    sample_count: int = 0
    top1_matches: int = 0
    top1_match_ratio: float = 0.0
    labeled_count: int = 0
    onnx_label_matches: int = 0
    c_label_matches: int = 0
    onnx_label_accuracy: float | None = None
    c_label_accuracy: float | None = None
    label_accuracy_delta: float | None = None
    saturation_ratio: float = 0.0
    max_abs_error: float | None = None
    error_msg: str = ""
    samples: list[dict[str, Any]] = field(default_factory=list)
    duration_sec: float = 0.0


@dataclass
class NumericReport:
    timestamp: str = ""
    git_commit: str = ""
    platform: str = ""
    command: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ONNX-vs-C numeric TDD checks.")
    parser.add_argument("--case", help="only run one numeric case")
    parser.add_argument("--dataset", help="override the registered dataset id")
    parser.add_argument("--generate", action="store_true", help="generate ONNX test models first")
    parser.add_argument("--list", action="store_true", help="list numeric-enabled cases")
    args = parser.parse_args(argv)

    numeric_cases = _numeric_cases()
    if args.list:
        for case_id in numeric_cases:
            check = CASE_MAP[case_id].numeric
            print(f"{case_id}: dataset={check.dataset_id if check else '-'}")
        return 0

    if args.generate:
        _generate_models(args.case)

    selected = [args.case] if args.case else numeric_cases
    selected = [case_id for case_id in selected if case_id in numeric_cases]
    if not selected:
        print("未找到启用数值验收的用例。")
        return 1

    report = NumericReport(
        timestamp=datetime.now().isoformat(),
        git_commit=_git_commit(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        command=" ".join(sys.argv),
    )

    print(f"\n{'=' * 70}")
    print(f"  TDD 数值测试 — {len(selected)} 个用例")
    print(f"{'=' * 70}\n")

    for case_id in selected:
        result = run_numeric_case(case_id, dataset_override=args.dataset)
        report.total += 1
        if result.passed:
            report.passed += 1
            icon = "PASS"
        else:
            report.failed += 1
            icon = "FAIL"
        print(
            f"  [{icon}] {case_id:<15} "
            f"top1={result.top1_matches}/{result.sample_count} "
            f"label_acc={_fmt_optional_ratio(result.c_label_accuracy)} "
            f"sat={result.saturation_ratio:.2f} "
            f"max_abs={_fmt_float(result.max_abs_error)} "
            f"cc={'OK' if result.compile_ok else '-'} "
            f"run={'OK' if result.run_ok else '-'} "
            f"({result.duration_sec:.1f}s)"
        )
        if not result.passed and result.error_msg:
            print(f"         -> {result.error_msg[:240]}")
        report.results.append(asdict(result))

    print(f"\n{'=' * 70}")
    print(f"  总计: {report.total}  |  通过: {report.passed}  |  失败: {report.failed}")
    print(f"{'=' * 70}\n")

    path = _save_report(report)
    print(f"数值报告已保存: {path}")
    return 0 if report.failed == 0 else 1


def run_numeric_case(
    case_id: str,
    *,
    dataset_override: str | None = None,
) -> NumericCaseResult:
    case_def = CASE_MAP[case_id]
    check = case_def.numeric
    if check is None:
        raise ValueError(f"case has no numeric check: {case_id}")
    if dataset_override:
        check = NumericCheck(
            dataset_id=dataset_override,
            input_scale=check.input_scale,
            top1_min_match_ratio=check.top1_min_match_ratio,
            top1_tie_margin=check.top1_tie_margin,
            max_abs_error=check.max_abs_error,
            max_saturation_ratio=check.max_saturation_ratio,
            float_api=check.float_api,
        )

    onnx_path = _find_model(case_id)
    work_dir = NUMERIC_WORK_ROOT / case_id
    _reset_work_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / WORKDIR_MARKER).write_text("numeric\n", encoding="utf-8")

    result = NumericCaseResult(
        case_id=case_id,
        dataset_id=check.dataset_id,
        onnx_path=_repo_relative(onnx_path),
        output_dir=_repo_relative(work_dir),
    )

    t0 = time.monotonic()
    exit_code, stdout, stderr = run_pipeline(onnx_path, work_dir)
    result.pipeline_exit_code = exit_code
    result.pipeline_status = detect_status(work_dir, exit_code, stdout, stderr)
    if exit_code != 0 or result.pipeline_status != "ok":
        result.error_msg = f"pipeline failed: status={result.pipeline_status}, exit={exit_code}, {stderr[:300]}"
        result.duration_sec = round(time.monotonic() - t0, 2)
        return result

    codegen_dir = work_dir / "cmsis-codegen"
    graph_path = work_dir / "converter-output" / "model_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    samples = _load_dataset(check)
    onnx_outputs = _run_onnx(onnx_path, samples, check)

    c_runner = work_dir / "numeric_runner.c"
    binary = work_dir / "numeric_runner"
    if check.index_compare:
        _write_index_c_runner(c_runner, samples, graph, check)
        compile_ok, compile_error = _compile_c_runner(codegen_dir, c_runner, binary)
    elif check.float_api:
        _write_float_c_runner(c_runner, samples, graph, check)
        compile_ok, compile_error = _compile_float_c_runner(codegen_dir, c_runner, binary)
    else:
        _write_c_runner(c_runner, samples, graph, check)
        compile_ok, compile_error = _compile_c_runner(codegen_dir, c_runner, binary)
    result.compile_ok = compile_ok
    if not compile_ok:
        result.error_msg = f"C runner compile failed: {compile_error[:500]}"
        result.duration_sec = round(time.monotonic() - t0, 2)
        return result

    if check.index_compare:
        c_raw_outputs, run_error = _run_index_c_runner(binary)
        c_outputs = c_raw_outputs
    elif check.float_api:
        c_outputs, run_error = _run_float_c_runner(binary)
        c_raw_outputs = [np.asarray([], dtype=np.int8) for _ in c_outputs]
    else:
        c_raw_outputs, run_error = _run_c_runner(binary)
        c_outputs = []
    result.run_ok = run_error == ""
    if run_error:
        result.error_msg = f"C runner failed: {run_error[:500]}"
        result.duration_sec = round(time.monotonic() - t0, 2)
        return result

    if check.index_compare:
        _compare_index_outputs(result, samples, onnx_outputs, c_raw_outputs, check)
    else:
        if not check.float_api:
            output_quant = _model_output_quant(graph)
            c_outputs = [
                (raw.astype(np.float32) - float(output_quant["zero_point"])) * float(output_quant["scale"])
                for raw in c_raw_outputs
            ]
        _compare_outputs(result, samples, onnx_outputs, c_raw_outputs, c_outputs, check)
    result.duration_sec = round(time.monotonic() - t0, 2)
    return result


def _numeric_cases() -> list[str]:
    return [case.case_id for case in CASE_MAP.values() if case.numeric is not None]


def _generate_models(case_id: str | None) -> None:
    cmd = [sys.executable, str(_SCRIPT_DIR / "generate_models.py")]
    if case_id:
        cmd.extend(["--case", case_id])
    completed = subprocess.run(cmd, text=True, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _find_model(case_id: str) -> Path:
    for path in MODELS_ROOT.rglob(f"{case_id}.onnx"):
        return path
    raise FileNotFoundError(
        f"missing generated model for {case_id}; run with --generate first"
    )


def _reset_work_dir(path: Path) -> None:
    if not path.exists():
        return
    marker = path / WORKDIR_MARKER
    if not marker.exists():
        try:
            path.resolve().relative_to(NUMERIC_WORK_ROOT.resolve())
        except ValueError:
            raise RuntimeError(
                f"refuse to delete unmarked directory: {path}. "
                "Move manual assets to tdd/fixtures; tdd/work is runner-owned."
            ) from None
    for attempt in range(3):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.1 * (attempt + 1))


def _load_dataset(check: NumericCheck) -> list[dict[str, Any]]:
    path = FIXTURES_ROOT / "datasets" / check.dataset_id / "dataset.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for item in raw.get("samples", []):
        inputs = _sample_inputs(item)
        pixels = _sample_pixels(item, raw) if inputs is None else None
        samples.append(
            {
                "id": item.get("id", f"sample_{len(samples)}"),
                "label": item.get("label", item.get("label_index")),
                "pixels": pixels,
                "inputs": inputs,
            }
        )
    if not samples:
        raise ValueError(f"dataset has no samples: {path}")
    return samples


def _sample_inputs(item: dict[str, Any]) -> dict[str, np.ndarray] | None:
    raw_inputs = item.get("inputs")
    if not isinstance(raw_inputs, dict):
        return None
    return {
        str(name): np.asarray(value, dtype=np.float32)
        for name, value in raw_inputs.items()
    }


def _sample_pixels(item: dict[str, Any], dataset: dict[str, Any]) -> np.ndarray:
    if "values" in item:
        shape = dataset.get("input_shape")
        if not shape:
            raise ValueError(
                f"dataset {dataset.get('dataset_id')} uses values but has no input_shape"
            )
        arr = np.asarray(item["values"], dtype=np.float32)
        expected_size = int(np.prod(np.asarray(shape, dtype=np.int64)))
        if arr.size != expected_size:
            raise ValueError(
                f"sample {item.get('id')} must contain {expected_size} values for shape {shape}"
            )
        return arr.reshape(tuple(int(dim) for dim in shape))

    if "float_pixels" in item:
        arr = np.asarray(item["float_pixels"], dtype=np.float32)
        shape = dataset.get("input_shape", [1, 1, 28, 28])
        expected_size = int(np.prod(np.asarray(shape, dtype=np.int64)))
        if arr.size != expected_size:
            raise ValueError(
                f"sample {item.get('id')} must contain {expected_size} pixels for shape {shape}"
            )
        return arr.reshape(tuple(int(dim) for dim in shape))

    pattern = item.get("pattern")
    shape = tuple(int(dim) for dim in dataset.get("input_shape", [1, 1, 28, 28]))
    arr = np.zeros(shape, dtype=np.float32)
    if pattern == "blank":
        return arr
    if len(shape) == 4 and shape[1] == 3 and shape[2] > 28 and shape[3] > 28:
        return _image_pattern(pattern, shape, item)
    arr = np.zeros((1, 1, 28, 28), dtype=np.float32)
    if pattern == "vertical_center":
        arr[0, 0, 4:24, 13:16] = 1.0
        return arr
    if pattern == "diagonal":
        for i in range(5, 23):
            arr[0, 0, i, i] = 1.0
            if i + 1 < 28:
                arr[0, 0, i, i + 1] = 0.7
        return arr
    if pattern == "anti_diagonal":
        for i in range(5, 23):
            j = 27 - i
            arr[0, 0, i, j] = 1.0
            if j - 1 >= 0:
                arr[0, 0, i, j - 1] = 0.7
        return arr
    if pattern == "horizontal_center":
        arr[0, 0, 13:16, 4:24] = 1.0
        return arr
    if pattern == "cross":
        arr[0, 0, 4:24, 13:16] = 1.0
        arr[0, 0, 13:16, 4:24] = 1.0
        return arr
    if pattern == "box":
        arr[0, 0, 5:8, 6:22] = 1.0
        arr[0, 0, 20:23, 6:22] = 1.0
        arr[0, 0, 5:23, 6:9] = 1.0
        arr[0, 0, 5:23, 19:22] = 1.0
        return arr
    if pattern == "top_arc":
        arr[0, 0, 6:9, 8:20] = 1.0
        arr[0, 0, 9:15, 6:9] = 0.8
        arr[0, 0, 9:15, 19:22] = 0.8
        return arr
    if pattern == "bottom_arc":
        arr[0, 0, 19:22, 8:20] = 1.0
        arr[0, 0, 13:19, 6:9] = 0.8
        arr[0, 0, 13:19, 19:22] = 0.8
        return arr
    if pattern == "two_columns":
        arr[0, 0, 5:23, 8:11] = 1.0
        arr[0, 0, 5:23, 17:20] = 1.0
        return arr
    raise ValueError(f"unknown sample pattern: {pattern!r}")


def _image_pattern(
    pattern: str,
    shape: tuple[int, ...],
    item: dict[str, Any],
) -> np.ndarray:
    if len(shape) != 4:
        raise ValueError(f"image pattern requires NCHW shape, got {shape}")
    n, c, h, w = shape
    if n != 1:
        raise ValueError(f"image pattern only supports batch=1, got {shape}")
    if c not in (1, 3):
        raise ValueError(f"image pattern only supports 1 or 3 channels, got {shape}")

    value_range = item.get("value_range", [0.0, 255.0])
    low = float(value_range[0])
    high = float(value_range[1])
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(1, h, 1)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, 1, w)
    out = np.zeros(shape, dtype=np.float32)

    if pattern == "image_midgray":
        out.fill((low + high) * 0.5)
        return out

    if pattern == "image_channel_ramps":
        channels = [
            np.broadcast_to(xx, (1, h, w)),
            np.broadcast_to(yy, (1, h, w)),
            np.broadcast_to((xx + yy) * 0.5, (1, h, w)),
        ]
        for ch in range(c):
            out[0, ch] = low + (high - low) * channels[ch % len(channels)][0]
        return out

    if pattern == "image_checker":
        tile = int(item.get("tile", 16))
        grid_y = np.arange(h, dtype=np.int32).reshape(h, 1) // tile
        grid_x = np.arange(w, dtype=np.int32).reshape(1, w) // tile
        checker = ((grid_y + grid_x) % 2).astype(np.float32)
        for ch in range(c):
            phase = checker if ch % 2 == 0 else 1.0 - checker
            out[0, ch] = low + (high - low) * phase
        return out

    if pattern == "image_center_blob":
        cy = float(item.get("center_y", 0.5))
        cx = float(item.get("center_x", 0.5))
        sigma = float(item.get("sigma", 0.16))
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = np.exp(-dist2 / max(2.0 * sigma * sigma, 1e-6)).astype(np.float32)
        for ch in range(c):
            gain = 1.0 - 0.18 * ch
            out[0, ch] = low + (high - low) * np.clip(blob[0] * gain, 0.0, 1.0)
        return out

    raise ValueError(f"unknown image sample pattern: {pattern!r}")


def _run_onnx(
    onnx_path: Path,
    samples: list[dict[str, Any]],
    check: NumericCheck,
) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    input_names = [item.name for item in session.get_inputs()]
    outputs = []
    for sample in samples:
        if sample.get("inputs") is not None:
            sample_inputs = sample["inputs"]
            inp = {
                name: np.asarray(sample_inputs[name], dtype=np.float32) * float(check.input_scale)
                for name in input_names
            }
        else:
            input_name = input_names[0]
            inp = {input_name: sample["pixels"].astype(np.float32) * float(check.input_scale)}
        out = session.run(None, inp)[0]
        outputs.append(np.asarray(out, dtype=np.float32).reshape(-1))
    return outputs


def _write_float_c_runner(
    path: Path,
    samples: list[dict[str, Any]],
    graph: dict[str, Any],
    check: NumericCheck,
) -> None:
    del check
    input_names = [item["name"] for item in graph.get("inputs", [])]
    arrays: list[str] = []
    sample_input_rows: list[str] = []
    for sample_index, sample in enumerate(samples):
        sample_inputs = sample.get("inputs")
        if not isinstance(sample_inputs, dict):
            raise ValueError("float_api numeric case requires dataset samples with inputs")
        ptrs: list[str] = []
        for input_index, input_name in enumerate(input_names):
            values = np.asarray(sample_inputs[input_name], dtype=np.float32).reshape(-1)
            symbol = f"sample_{sample_index}_{input_index}"
            literal_values = ", ".join(_c_float_literal(float(value)) for value in values.tolist())
            arrays.append(f"static const float {symbol}[{values.size}] = {{{literal_values}}};")
            ptrs.append(symbol)
        row_symbol = f"sample_inputs_{sample_index}"
        sample_input_rows.append(
            f"static const float *{row_symbol}[{len(ptrs)}] = {{{', '.join(ptrs)}}};"
        )

    sample_list = ", ".join(f"sample_inputs_{index}" for index in range(len(samples)))
    code = f"""/* Auto-generated numeric TDD float runner. */
#include <stdio.h>
#include "model.h"

#define NUM_SAMPLES {len(samples)}

{chr(10).join(arrays)}
{chr(10).join(sample_input_rows)}

static const float **samples[NUM_SAMPLES] = {{{sample_list}}};
static float output_buffer[NANOC_MODEL_OUTPUT_FLOATS];

int main(void)
{{
    for (int i = 0; i < NUM_SAMPLES; ++i) {{
        int ret = nanoc_model_run_float(samples[i], output_buffer);
        if (ret != 0) {{
            printf("C_ERROR sample=%d ret=%d\\n", i, ret);
            return ret;
        }}
        printf("C_OUT sample=%d", i);
        for (unsigned j = 0; j < NANOC_MODEL_OUTPUT_FLOATS; ++j) {{
            printf(" %.9g", output_buffer[j]);
        }}
        printf("\\n");
    }}
    return 0;
}}
"""
    path.write_text(code, encoding="utf-8")


def _write_c_runner(
    path: Path,
    samples: list[dict[str, Any]],
    graph: dict[str, Any],
    check: NumericCheck,
) -> None:
    input_arrays = []
    for index, sample in enumerate(samples):
        quantized = _pack_quantized_inputs(sample, graph, check)
        values = ", ".join(str(int(value)) for value in quantized.reshape(-1).tolist())
        input_arrays.append(f"static const int8_t sample_{index}[{quantized.size}] = {{{values}}};")

    sample_list = ", ".join(f"sample_{index}" for index in range(len(samples)))
    code = f"""/* Auto-generated numeric TDD runner. */
#include <stdint.h>
#include <stdio.h>
#include "model.h"

#define NUM_SAMPLES {len(samples)}

{chr(10).join(input_arrays)}

static const int8_t *samples[NUM_SAMPLES] = {{{sample_list}}};
static int8_t output_buffer[NANOC_MODEL_OUTPUT_BYTES];

int main(void)
{{
    for (int i = 0; i < NUM_SAMPLES; ++i) {{
        int ret = nanoc_model_run(samples[i], output_buffer);
        if (ret != 0) {{
            printf("C_ERROR sample=%d ret=%d\\n", i, ret);
            return ret;
        }}
        printf("C_OUT sample=%d", i);
        for (int j = 0; j < NANOC_MODEL_OUTPUT_BYTES; ++j) {{
            printf(" %d", (int)output_buffer[j]);
        }}
        printf("\\n");
    }}
    return 0;
}}
"""
    path.write_text(code, encoding="utf-8")


def _write_index_c_runner(
    path: Path,
    samples: list[dict[str, Any]],
    graph: dict[str, Any],
    check: NumericCheck,
) -> None:
    """索引输出模型 runner：int32 index buffer + nanoc_model_run_index 入口。"""
    input_arrays = []
    for index, sample in enumerate(samples):
        quantized = _pack_quantized_inputs(sample, graph, check)
        values = ", ".join(str(int(value)) for value in quantized.reshape(-1).tolist())
        input_arrays.append(f"static const int8_t sample_{index}[{quantized.size}] = {{{values}}};")

    sample_list = ", ".join(f"sample_{index}" for index in range(len(samples)))
    code = f"""/* Auto-generated numeric TDD runner (index output). */
#include <stdint.h>
#include <stdio.h>
#include "model.h"

#define NUM_SAMPLES {len(samples)}

{chr(10).join(input_arrays)}

static const int8_t *samples[NUM_SAMPLES] = {{{sample_list}}};
static int32_t output_buffer[NANOC_MODEL_OUTPUT_INDEX_COUNT];

int main(void)
{{
    for (int i = 0; i < NUM_SAMPLES; ++i) {{
        int ret = nanoc_model_run_index(samples[i], output_buffer);
        if (ret != 0) {{
            printf("C_ERROR sample=%d ret=%d\\n", i, ret);
            return ret;
        }}
        printf("C_OUT sample=%d", i);
        for (int j = 0; j < NANOC_MODEL_OUTPUT_INDEX_COUNT; ++j) {{
            printf(" %d", (int)output_buffer[j]);
        }}
        printf("\\n");
    }}
    return 0;
}}
"""
    path.write_text(code, encoding="utf-8")


def _pack_quantized_inputs(
    sample: dict[str, Any],
    graph: dict[str, Any],
    check: NumericCheck,
) -> np.ndarray:
    input_specs = graph.get("inputs", [])
    input_quants = _model_input_quants(graph)
    if sample.get("inputs") is not None:
        sample_inputs = sample["inputs"]
        packed: list[np.ndarray] = []
        for spec in input_specs:
            name = str(spec["name"])
            if name not in sample_inputs:
                raise ValueError(f"sample {sample['id']} missing input {name!r}")
            quant = input_quants[name]
            packed.append(_quantize_input(np.asarray(sample_inputs[name]), quant, check).reshape(-1))
        return np.concatenate(packed).astype(np.int8)

    if not input_specs:
        raise ValueError("model graph has no inputs")
    first_name = str(input_specs[0]["name"])
    return _quantize_input(sample["pixels"], input_quants[first_name], check).reshape(-1)


def _quantize_input(
    pixels: np.ndarray,
    input_quant: dict[str, Any],
    check: NumericCheck,
) -> np.ndarray:
    scale = float(input_quant["scale"])
    zero_point = int(input_quant["zero_point"])
    raw = pixels.astype(np.float32) * float(check.input_scale)
    quantized = np.rint(raw / scale + zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)


def _compile_c_runner(codegen_dir: Path, runner_c: Path, binary: Path) -> tuple[bool, str]:
    cc = "cc"
    if shutil.which(cc) is None:
        return False, f"C compiler not found: {cc}"
    cmd = [
        cc,
        "-std=c99",
        "-O2",
        "-Wno-unknown-attributes",
        "-DNANOC_ENABLE_CMSIS_NN=1",
        "-I",
        str(codegen_dir / "include"),
        "-I",
        str(CMSIS_NN_ROOT / "Include"),
        str(runner_c),
        str(codegen_dir / "src" / "model.c"),
        *[str(path) for path in _collect_cmsis_sources()],
        "-lm",
        "-o",
        str(binary),
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return False, completed.stderr
    return True, ""


def _compile_float_c_runner(codegen_dir: Path, runner_c: Path, binary: Path) -> tuple[bool, str]:
    cc = "cc"
    if shutil.which(cc) is None:
        return False, f"C compiler not found: {cc}"
    cmd = [
        cc,
        "-std=c99",
        "-O2",
        "-I",
        str(codegen_dir / "include"),
        str(runner_c),
        str(codegen_dir / "src" / "model.c"),
        "-lm",
        "-o",
        str(binary),
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return False, completed.stderr
    return True, ""


def _collect_cmsis_sources() -> list[Path]:
    dirs = [
        CMSIS_NN_ROOT / "Source" / "ConvolutionFunctions",
        CMSIS_NN_ROOT / "Source" / "PoolingFunctions",
        CMSIS_NN_ROOT / "Source" / "FullyConnectedFunctions",
        CMSIS_NN_ROOT / "Source" / "BasicMathFunctions",
        CMSIS_NN_ROOT / "Source" / "NNSupportFunctions",
        CMSIS_NN_ROOT / "Source" / "ActivationFunctions",
        CMSIS_NN_ROOT / "Source" / "SoftmaxFunctions",
        CMSIS_NN_ROOT / "Source" / "ConcatenationFunctions",
        CMSIS_NN_ROOT / "Source" / "TransposeFunctions",
    ]
    sources: list[Path] = []
    for directory in dirs:
        for path in sorted(directory.glob("*.c")):
            name = path.name.lower()
            if "_f16" in name or "_f32" in name or "lstm" in name:
                continue
            sources.append(path)
    return sources


def _run_c_runner(binary: Path) -> tuple[list[np.ndarray], str]:
    completed = subprocess.run(
        [str(binary)],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        return [], (completed.stderr or completed.stdout)[:1000]
    outputs: list[np.ndarray] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("C_OUT"):
            continue
        parts = line.split()
        values = [int(value) for value in parts[2:]]
        outputs.append(np.asarray(values, dtype=np.int8))
    if not outputs:
        return [], "C runner produced no C_OUT lines"
    return outputs, ""


def _run_index_c_runner(binary: Path) -> tuple[list[np.ndarray], str]:
    completed = subprocess.run(
        [str(binary)],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        return [], (completed.stderr or completed.stdout)[:1000]
    outputs: list[np.ndarray] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("C_OUT"):
            continue
        parts = line.split()
        values = [int(value) for value in parts[2:]]
        outputs.append(np.asarray(values, dtype=np.int32))
    if not outputs:
        return [], "C runner produced no C_OUT lines"
    return outputs, ""


def _run_float_c_runner(binary: Path) -> tuple[list[np.ndarray], str]:
    completed = subprocess.run(
        [str(binary)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        return [], (completed.stderr or completed.stdout)[:1000]
    outputs: list[np.ndarray] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("C_OUT"):
            continue
        parts = line.split()
        values = [float(value) for value in parts[2:]]
        outputs.append(np.asarray(values, dtype=np.float32))
    if not outputs:
        return [], "C runner produced no C_OUT lines"
    return outputs, ""


def _compare_outputs(
    result: NumericCaseResult,
    samples: list[dict[str, Any]],
    onnx_outputs: list[np.ndarray],
    c_raw_outputs: list[np.ndarray],
    c_outputs: list[np.ndarray],
    check: NumericCheck,
) -> None:
    result.sample_count = len(samples)
    total_values = 0
    saturated_values = 0
    max_abs = 0.0
    matches = 0
    labeled = 0
    onnx_label_matches = 0
    c_label_matches = 0
    details: list[dict[str, Any]] = []

    for sample, onnx_out, raw_out, c_out in zip(
        samples,
        onnx_outputs,
        c_raw_outputs,
        c_outputs,
        strict=True,
    ):
        onnx_top1 = int(np.argmax(onnx_out))
        c_top1 = int(np.argmax(c_out))
        onnx_top5 = _topk_indices(onnx_out, 5)
        c_top5 = _topk_indices(c_out, 5)
        top1_margin = 0.0
        near_tie = False
        if onnx_out.size and c_top1 < onnx_out.size:
            top1_margin = float(
                onnx_out.reshape(-1)[onnx_top1] - onnx_out.reshape(-1)[c_top1]
            )
            near_tie = top1_margin <= check.top1_tie_margin
        exact_match = onnx_top1 == c_top1
        match = exact_match or near_tie
        if match:
            matches += 1
        label = sample.get("label")
        if label is not None:
            labeled += 1
            label_int = int(label)
            if onnx_top1 == label_int:
                onnx_label_matches += 1
            if c_top1 == label_int:
                c_label_matches += 1
        if raw_out.size:
            saturated_values += int(np.count_nonzero((raw_out == -128) | (raw_out == 127)))
            total_values += int(raw_out.size)
        diff = np.abs(onnx_out.astype(np.float32) - c_out.astype(np.float32))
        sample_max_abs = float(np.max(diff)) if diff.size else 0.0
        max_abs = max(max_abs, sample_max_abs)
        details.append(
            {
                "id": sample["id"],
                "label": sample.get("label"),
                "onnx_top1": onnx_top1,
                "c_top1": c_top1,
                "onnx_top5": onnx_top5,
                "c_top5": c_top5,
                "match": match,
                "exact_match": exact_match,
                "near_tie": near_tie and not exact_match,
                "onnx_top1_margin_to_c_top1": top1_margin,
                "c_int8": [int(value) for value in raw_out.tolist()],
                "c_float": [float(value) for value in c_out.tolist()],
                "max_abs_error": sample_max_abs,
            }
        )

    result.top1_matches = matches
    result.top1_match_ratio = matches / len(samples) if samples else 0.0
    result.labeled_count = labeled
    result.onnx_label_matches = onnx_label_matches
    result.c_label_matches = c_label_matches
    if labeled:
        result.onnx_label_accuracy = onnx_label_matches / labeled
        result.c_label_accuracy = c_label_matches / labeled
        result.label_accuracy_delta = abs(result.onnx_label_accuracy - result.c_label_accuracy)
    result.saturation_ratio = saturated_values / total_values if total_values else 0.0
    result.max_abs_error = max_abs
    result.samples = details

    issues = []
    if result.top1_match_ratio < check.top1_min_match_ratio:
        issues.append(
            f"top1 match ratio {result.top1_match_ratio:.3f} < {check.top1_min_match_ratio:.3f}"
        )
    if result.saturation_ratio > check.max_saturation_ratio:
        issues.append(
            f"saturation ratio {result.saturation_ratio:.3f} > {check.max_saturation_ratio:.3f}"
        )
    if check.max_abs_error is not None and result.max_abs_error > check.max_abs_error:
        issues.append(
            f"max abs error {result.max_abs_error:.3f} > {check.max_abs_error:.3f}"
        )

    result.passed = not issues
    result.error_msg = "; ".join(issues)


def _compare_index_outputs(
    result: NumericCaseResult,
    samples: list[dict[str, Any]],
    onnx_outputs: list[np.ndarray],
    c_index_outputs: list[np.ndarray],
    check: NumericCheck,
) -> None:
    """索引输出精确比较：ONNX int64 index 与 C int32 index 逐元素相等。

    索引输出不做 top1/saturation 统计（无量化饱和概念），只要求 index 精确一致。
    """
    result.sample_count = len(samples)
    matches = 0
    max_abs = 0.0
    details: list[dict[str, Any]] = []
    for sample, onnx_out, c_out in zip(
        samples,
        onnx_outputs,
        c_index_outputs,
        strict=True,
    ):
        onnx_idx = np.asarray(onnx_out, dtype=np.int64).reshape(-1)
        c_idx = np.asarray(c_out, dtype=np.int64).reshape(-1)
        diff = np.abs(onnx_idx - c_idx) if onnx_idx.size == c_idx.size else np.array([1.0])
        sample_max_abs = float(np.max(diff)) if diff.size else 0.0
        max_abs = max(max_abs, sample_max_abs)
        exact = bool(sample_max_abs == 0.0)
        if exact:
            matches += 1
        details.append(
            {
                "id": sample["id"],
                "onnx_index": [int(value) for value in onnx_idx.tolist()],
                "c_index": [int(value) for value in c_idx.tolist()],
                "exact_match": exact,
                "max_abs_error": sample_max_abs,
            }
        )
    result.top1_matches = matches
    result.top1_match_ratio = matches / len(samples) if samples else 0.0
    result.saturation_ratio = 0.0
    result.max_abs_error = max_abs
    result.samples = details

    issues = []
    if result.top1_match_ratio < check.top1_min_match_ratio:
        issues.append(
            f"index match ratio {result.top1_match_ratio:.3f} < {check.top1_min_match_ratio:.3f}"
        )
    if check.max_abs_error is not None and result.max_abs_error > check.max_abs_error:
        issues.append(
            f"index max abs error {result.max_abs_error:.3f} > {check.max_abs_error:.3f}"
        )
    result.passed = not issues
    result.error_msg = "; ".join(issues)


def _topk_indices(values: np.ndarray, k: int) -> list[int]:
    if values.size == 0:
        return []
    count = min(int(k), int(values.size))
    order = np.argsort(values.reshape(-1))[::-1][:count]
    return [int(index) for index in order.tolist()]


def _model_input_quant(graph: dict[str, Any]) -> dict[str, Any]:
    input_name = graph["inputs"][0]["name"]
    return graph["quantization"]["tensors"][input_name]


def _model_input_quants(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensor_quants = graph["quantization"]["tensors"]
    return {
        str(item["name"]): tensor_quants[str(item["name"])]
        for item in graph.get("inputs", [])
    }


def _model_output_quant(graph: dict[str, Any]) -> dict[str, Any]:
    output_name = graph["outputs"][0]["name"]
    return graph["quantization"]["tensors"][output_name]


def _save_report(report: NumericReport) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    content = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    latest_path = RESULTS_DIR / "numeric_latest.json"
    latest_path.write_text(content, encoding="utf-8")

    history_dir = RESULTS_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    history_path = history_dir / f"{date_str}-numeric.json"
    history_path.write_text(content, encoding="utf-8")
    return latest_path


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _c_float_literal(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text.lower() and "." not in text:
        text = f"{text}.0"
    return f"{text}f"


def _fmt_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.1f}"


def _fmt_optional_ratio(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())

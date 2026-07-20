"""
tdd/terminal/scripts/run_terminal_tests.py — optional ARM terminal TDD checks.

This runner adds the last TDD stage for hardware-in-the-loop validation:

  ONNX Runtime reference
      vs generated Host C
      vs generated ARM C running on a connected Cortex-M board.

The stage is intentionally optional. In auto mode, a missing serial port or
programmer records a skipped report and exits successfully, so ordinary full
regression stays usable without hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import select
import shutil
import struct
import subprocess
import sys
import termios
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TERMINAL_ROOT = SCRIPT_DIR.parent
TDD_ROOT = TERMINAL_ROOT.parent
REPO_ROOT = TDD_ROOT.parent
TDD_SCRIPTS = TDD_ROOT / "scripts"
REPORTS_DIR = TERMINAL_ROOT / "reports"
NUMERIC_WORK_ROOT = TDD_ROOT / "work" / "numeric"
TERMINAL_WORK_ROOT = TDD_ROOT / "work" / "terminal"
MODELS_ROOT = TDD_ROOT / "work" / "models"
WORKDIR_MARKER = ".nanoc_tdd_workdir"

sys.path.insert(0, str(TDD_SCRIPTS))
from cases_registry import CASE_MAP  # noqa: E402


REQ_MAGIC = 0x4454434E  # NCTD
RESP_MAGIC = 0x5254434E  # NCTR
PROTOCOL_VERSION = 1
CMD_RUN_CASE = 1


@dataclass
class TerminalSampleResult:
    id: str
    onnx_top1: int | None
    host_c_top1: int | None
    arm_c_top1: int | None
    host_c_int8: list[int]
    arm_c_int8: list[int]
    exact_match: bool
    top1_match: bool
    elapsed_us: int | None
    error_msg: str = ""


@dataclass
class TerminalCaseResult:
    case_id: str
    terminal_case_id: str
    dataset_id: str
    board_id: str
    passed: bool = False
    skipped: bool = False
    skip_reason: str = ""
    host_numeric_passed: bool = False
    build_ok: bool = False
    flash_ok: bool = False
    run_ok: bool = False
    sample_count: int = 0
    onnx_host_c_top1_matches: int = 0
    onnx_arm_c_top1_matches: int = 0
    host_c_arm_c_exact_matches: int = 0
    elapsed_us_min: int | None = None
    elapsed_us_max: int | None = None
    elapsed_us_avg: float | None = None
    error_msg: str = ""
    samples: list[dict[str, Any]] = field(default_factory=list)
    duration_sec: float = 0.0


@dataclass
class TerminalReport:
    timestamp: str = ""
    git_commit: str = ""
    platform: str = ""
    command: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run optional ONNX-vs-HostC-vs-ARMC terminal checks.")
    parser.add_argument("--case", help="TDD case id, for example TOPO_003")
    parser.add_argument("--board", help="board id, defaults to the first board listed by the terminal case")
    parser.add_argument("--generate", action="store_true", help="regenerate ONNX models before Host numeric run")
    parser.add_argument("--flash", dest="flash", action="store_true", default=True, help="flash firmware before serial test")
    parser.add_argument("--no-flash", dest="flash", action="store_false", help="skip flashing and only use current firmware")
    parser.add_argument("--auto", action="store_true", help="skip successfully when board resources are missing")
    parser.add_argument("--require-board", action="store_true", help="fail when board resources are missing")
    parser.add_argument("--all-capabilities", action="store_true", help="attempt every registered TDD case on terminal hardware")
    parser.add_argument("--list", action="store_true", help="list terminal-enabled cases")
    args = parser.parse_args(argv)

    cases = _load_terminal_cases(all_capabilities=args.all_capabilities)
    if args.list:
        for item in cases:
            boards = ", ".join(item.get("board_ids", []))
            print(f"{item['case_id']}: {item['terminal_case_id']} boards=[{boards}]")
        return 0

    selected = [item for item in cases if args.case in (None, item.get("case_id"), item.get("terminal_case_id"))]
    if args.case and not selected:
        print(f"未找到终端 case: {args.case}")
        return 1

    report = TerminalReport(
        timestamp=datetime.now().isoformat(),
        git_commit=_git_commit(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        command=" ".join(sys.argv),
    )

    print(f"\n{'=' * 70}")
    print(f"  TDD 终端实机测试 — {len(selected)} 个用例")
    print(f"{'=' * 70}\n")

    exit_code = 0
    for case_cfg in selected:
        board_id = args.board or case_cfg.get("board_ids", [""])[0]
        board_cfg = _load_board(board_id)
        result = run_terminal_case(
            case_cfg,
            board_cfg,
            generate=args.generate,
            flash=args.flash,
            allow_skip=args.auto or not args.require_board,
        )
        report.total += 1
        if result.skipped:
            report.skipped += 1
            icon = "SKIP"
        elif result.passed:
            report.passed += 1
            icon = "PASS"
        else:
            report.failed += 1
            icon = "FAIL"
            exit_code = 1
        print(
            f"  [{icon}] {result.case_id:<10} board={result.board_id} "
            f"host_arm_exact={result.host_c_arm_c_exact_matches}/{result.sample_count} "
            f"onnx_arm_top1={result.onnx_arm_c_top1_matches}/{result.sample_count} "
            f"elapsed={_fmt_elapsed(result)}"
        )
        if result.skipped:
            print(f"         -> skipped: {result.skip_reason}")
        elif not result.passed and result.error_msg:
            print(f"         -> {result.error_msg[:240]}")
        report.results.append(asdict(result))

    path = _save_report(report)
    print(f"\n{'=' * 70}")
    print(f"  终端实机: PASS={report.passed} FAIL={report.failed} SKIP={report.skipped}")
    print(f"  报告: {path}")
    print(f"{'=' * 70}\n")
    return exit_code


def run_terminal_case(
    case_cfg: dict[str, Any],
    board_cfg: dict[str, Any],
    *,
    generate: bool,
    flash: bool,
    allow_skip: bool,
) -> TerminalCaseResult:
    case_id = str(case_cfg["case_id"])
    board_id = str(board_cfg["board_id"])
    dataset_id = str(case_cfg["dataset_id"])
    result = TerminalCaseResult(
        case_id=case_id,
        terminal_case_id=str(case_cfg["terminal_case_id"]),
        dataset_id=dataset_id,
        board_id=board_id,
    )
    t0 = time.monotonic()

    missing_reason = _board_missing_reason(board_cfg, flash=flash)
    if missing_reason:
        result.skipped = allow_skip
        result.error_msg = missing_reason
        result.skip_reason = missing_reason
        result.duration_sec = round(time.monotonic() - t0, 2)
        return result

    if case_id not in CASE_MAP:
        result.error_msg = f"unknown case: {case_id}"
        result.duration_sec = round(time.monotonic() - t0, 2)
        return result
    case_def = CASE_MAP[case_id]

    if generate:
        generated_ok, generated_error = _generate_models(case_id)
        if not generated_ok:
            result.error_msg = f"model generation failed: {generated_error[:500]}"
            result.duration_sec = round(time.monotonic() - t0, 2)
            return result

    if case_def.expected != "ok":
        matched, actual, detail = _check_expected_rejection(case_id)
        result.skipped = True
        result.skip_reason = f"expected {case_def.expected}, actual {actual}; not flashed"
        result.passed = matched
        if not matched:
            result.skipped = False
            result.error_msg = detail
        result.duration_sec = round(time.monotonic() - t0, 2)
        return result

    if case_def.numeric is None:
        numeric = _run_zero_smoke_case(case_id)
    else:
        numeric = _run_registered_numeric_case(case_id, dataset_id)
    result.host_numeric_passed = numeric["passed"]
    if not numeric["passed"]:
        result.error_msg = f"Host numeric failed before ARM terminal run: {numeric['error_msg']}"
        result.duration_sec = round(time.monotonic() - t0, 2)
        return result

    codegen_dir = Path(numeric["codegen_dir"])
    samples = numeric["samples_bytes"]
    numeric_samples = numeric["samples"]
    model_info = _read_model_info(codegen_dir / "include" / "model.h")
    if _is_f103_oversize(case_cfg, model_info):
        result.skipped = True
        result.skip_reason = (
            "oversize for STM32F103 terminal slot: "
            f"input={model_info['input_bytes']} output={model_info['output_bytes']} "
            f"sram={model_info['estimated_sram_bytes']}"
        )
        result.duration_sec = round(time.monotonic() - t0, 2)
        return result

    _sync_codegen_if_needed(case_cfg, codegen_dir)
    _write_terminal_model_config(case_cfg, model_info)
    build_ok, build_error = _build_arm_project(case_cfg)
    result.build_ok = build_ok
    if not build_ok:
        result.error_msg = f"ARM firmware build failed: {build_error[:500]}"
        result.duration_sec = round(time.monotonic() - t0, 2)
        return result

    if flash:
        flash_ok, flash_error = _flash_firmware(case_cfg, board_cfg)
        result.flash_ok = flash_ok
        if not flash_ok:
            result.error_msg = f"ARM firmware flash failed: {flash_error[:500]}"
            result.duration_sec = round(time.monotonic() - t0, 2)
            return result
    else:
        result.flash_ok = True

    arm_outputs, serial_error = _run_uart_samples(
        board_cfg,
        arm_case_id=int(case_cfg["arm_case_id"]),
        output_bytes=int(model_info["output_bytes"]),
        samples=samples,
    )
    result.run_ok = serial_error == ""
    if serial_error:
        result.error_msg = f"ARM serial run failed: {serial_error[:500]}"
        result.duration_sec = round(time.monotonic() - t0, 2)
        return result

    sample_results = _compare_samples(case_cfg, numeric_samples, arm_outputs)
    result.samples = [asdict(item) for item in sample_results]
    result.sample_count = len(sample_results)
    result.onnx_host_c_top1_matches = sum(
        1 for item in numeric_samples if item.get("onnx_top1") == item.get("c_top1")
    )
    result.onnx_arm_c_top1_matches = sum(1 for item in sample_results if item.top1_match)
    result.host_c_arm_c_exact_matches = sum(1 for item in sample_results if item.exact_match)
    elapsed = [item.elapsed_us for item in sample_results if item.elapsed_us is not None]
    if elapsed:
        result.elapsed_us_min = min(elapsed)
        result.elapsed_us_max = max(elapsed)
        result.elapsed_us_avg = round(sum(elapsed) / len(elapsed), 2)
    result.passed = (
        result.host_numeric_passed
        and result.build_ok
        and result.flash_ok
        and result.run_ok
        and all(not item.error_msg for item in sample_results)
    )
    result.duration_sec = round(time.monotonic() - t0, 2)
    return result


def _run_registered_numeric_case(case_id: str, dataset_id: str) -> dict[str, Any]:
    try:
        from run_numeric_tests import run_numeric_case  # noqa: PLC0415
    except Exception as exc:
        return {"passed": False, "error_msg": f"cannot import Host numeric runner dependencies: {exc}"}

    numeric = run_numeric_case(case_id, dataset_override=dataset_id)
    if not numeric.passed:
        return {"passed": False, "error_msg": numeric.error_msg}
    codegen_dir = REPO_ROOT / numeric.output_dir / "cmsis-codegen"
    return {
        "passed": True,
        "error_msg": "",
        "codegen_dir": codegen_dir,
        "samples": numeric.samples,
        "samples_bytes": _parse_numeric_runner_samples(
            NUMERIC_WORK_ROOT / case_id / "numeric_runner.c",
            int(_read_model_info(codegen_dir / "include" / "model.h")["input_bytes"]),
        ),
    }


def _load_terminal_cases(*, all_capabilities: bool = False) -> list[dict[str, Any]]:
    if all_capabilities:
        return [_default_terminal_case(case_id) for case_id in CASE_MAP]
    cases = []
    for path in sorted((TERMINAL_ROOT / "cases").glob("*.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    return cases


def _default_terminal_case(case_id: str) -> dict[str, Any]:
    case_def = CASE_MAP[case_id]
    dataset_id = case_def.numeric.dataset_id if case_def.numeric else "__terminal_zero_smoke__"
    return {
        "case_id": case_id,
        "terminal_case_id": f"{case_id}_terminal_auto",
        "description": f"Auto terminal check for {case_id}",
        "board_ids": ["stm32f103vetx_uart_tim6"],
        "arm_case_id": 100,
        "dataset_id": dataset_id,
        "arm_project": "ARM-PROJ/stm32f103/nanoc-nn",
        "firmware_elf": "ARM-PROJ/stm32f103/nanoc-nn/build/Debug/nanoc-nn.elf",
        "limits": {
            "max_input_bytes": 16384,
            "max_output_bytes": 8192,
            "max_estimated_sram_bytes": 57344,
        },
        "sync_codegen": {
            "enabled": True,
            "include_dir": "ARM-PROJ/stm32f103/nanoc-nn/app/models/terminal_model/include",
            "src_dir": "ARM-PROJ/stm32f103/nanoc-nn/app/models/terminal_model/src",
            "files": [
                ["include/model.h", "include/model.h"],
                ["include/model_weights.h", "include/model_weights.h"],
                ["include/converter_weights.h", "include/converter_weights.h"],
                ["src/model.c", "src/model.c"],
            ],
        },
        "compare": {
            "require_exact_int8": True,
            "require_top1_match": case_def.numeric is not None,
        },
    }


def _load_board(board_id: str) -> dict[str, Any]:
    path = TERMINAL_ROOT / "boards" / f"{board_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing board config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _board_missing_reason(board_cfg: dict[str, Any], *, flash: bool) -> str:
    port = Path(board_cfg["uart"]["port"])
    if not port.exists():
        return f"UART port not found: {port}"
    if flash:
        cli = Path(board_cfg["programmer"]["cli"])
        if not cli.exists():
            return f"programmer CLI not found: {cli}"
    return ""


def _check_expected_rejection(case_id: str) -> tuple[bool, str, str]:
    from run_tests import detect_status, run_pipeline  # noqa: PLC0415

    case_def = CASE_MAP[case_id]
    onnx_path = _find_model(case_id)
    work_dir = TERMINAL_WORK_ROOT / case_id / "rejection"
    _reset_work_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / WORKDIR_MARKER).write_text("terminal\n", encoding="utf-8")
    exit_code, stdout, stderr = run_pipeline(onnx_path, work_dir)
    actual = detect_status(work_dir, exit_code, stdout, stderr)
    matched = actual == case_def.expected
    detail = "" if matched else f"expected {case_def.expected}, actual {actual}: {stderr[:300]}"
    return matched, actual, detail


def _run_zero_smoke_case(case_id: str) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415
    from run_tests import detect_status, run_pipeline  # noqa: PLC0415
    from run_numeric_tests import (  # noqa: PLC0415
        NumericCheck,
        _compile_c_runner,
        _model_output_quant,
        _run_c_runner,
        _run_onnx,
        _write_c_runner,
    )

    onnx_path = _find_model(case_id)
    work_dir = TERMINAL_WORK_ROOT / case_id / "zero_smoke"
    _reset_work_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / WORKDIR_MARKER).write_text("terminal\n", encoding="utf-8")

    exit_code, stdout, stderr = run_pipeline(onnx_path, work_dir)
    actual = detect_status(work_dir, exit_code, stdout, stderr)
    if exit_code != 0 or actual != "ok":
        return {"passed": False, "error_msg": f"pipeline failed: status={actual}, exit={exit_code}, {stderr[:300]}"}

    codegen_dir = work_dir / "cmsis-codegen"
    graph_path = work_dir / "converter-output" / "model_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    check = NumericCheck(dataset_id="__terminal_zero_smoke__", input_scale=1.0, top1_min_match_ratio=1.0)
    sample = _zero_sample(graph)
    samples = [sample]
    onnx_error = ""
    try:
        onnx_outputs = _run_onnx(onnx_path, samples, check)
    except Exception as exc:  # noqa: BLE001 - ORT compatibility is part of terminal reporting.
        onnx_outputs = [None for _ in samples]
        onnx_error = str(exc)

    runner_c = work_dir / "terminal_zero_runner.c"
    binary = work_dir / "terminal_zero_runner"
    _write_c_runner(runner_c, samples, graph, check)
    compile_ok, compile_error = _compile_c_runner(codegen_dir, runner_c, binary)
    if not compile_ok:
        return {"passed": False, "error_msg": f"Host C compile failed: {compile_error[:500]}"}
    c_raw_outputs, run_error = _run_c_runner(binary)
    if run_error:
        return {"passed": False, "error_msg": f"Host C run failed: {run_error[:500]}"}

    output_quant = _model_output_quant(graph)
    c_outputs = [
        (raw.astype(np.float32) - float(output_quant["zero_point"])) * float(output_quant["scale"])
        for raw in c_raw_outputs
    ]
    numeric_samples = []
    for index, (onnx_out, c_raw, c_out) in enumerate(zip(onnx_outputs, c_raw_outputs, c_outputs)):
        onnx_top1 = _array_top1(onnx_out) if onnx_out is not None else None
        c_top1 = _array_top1(c_out)
        numeric_samples.append(
            {
                "id": f"zero_smoke_{index}",
                "onnx_top1": onnx_top1,
                "c_top1": c_top1,
                "c_int8": [int(value) for value in c_raw.reshape(-1).tolist()],
                "match": (onnx_top1 == c_top1) if onnx_top1 is not None else None,
                "onnx_error": onnx_error,
            }
        )

    model_info = _read_model_info(codegen_dir / "include" / "model.h")
    return {
        "passed": True,
        "error_msg": "",
        "codegen_dir": codegen_dir,
        "samples": numeric_samples,
        "samples_bytes": _parse_numeric_runner_samples(runner_c, int(model_info["input_bytes"])),
    }


def _zero_sample(graph: dict[str, Any]) -> dict[str, Any]:
    inputs = {}
    for spec in graph.get("inputs", []):
        shape = [int(dim) for dim in spec.get("shape", [])]
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"terminal zero-smoke requires static positive input shape: {spec}")
        inputs[str(spec["name"])] = [[0.0] * int(math.prod(shape))]
        import numpy as np  # noqa: PLC0415

        inputs[str(spec["name"])] = np.zeros(tuple(shape), dtype=np.float32)
    return {"id": "zero_smoke", "label": None, "inputs": inputs}


def _array_top1(array: Any) -> int | None:
    flat = array.reshape(-1)
    if flat.size == 0:
        return None
    return int(flat.argmax())


def _find_model(case_id: str) -> Path:
    for path in MODELS_ROOT.rglob(f"{case_id}.onnx"):
        return path
    raise FileNotFoundError(f"missing generated model for {case_id}; run with --generate first")


def _reset_work_dir(path: Path) -> None:
    if not path.exists():
        return
    marker = path / WORKDIR_MARKER
    if not marker.exists():
        raise RuntimeError(f"refuse to delete unmarked terminal workdir: {path}")
    shutil.rmtree(path)


def _sync_codegen_if_needed(case_cfg: dict[str, Any], codegen_dir: Path) -> None:
    sync = case_cfg.get("sync_codegen", {})
    if not sync.get("enabled", False):
        return
    include_dir = REPO_ROOT / sync["include_dir"]
    src_dir = REPO_ROOT / sync["src_dir"]
    include_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    for src_rel, dst_rel in sync.get("files", []):
        src = codegen_dir / src_rel
        dst_root = include_dir if str(dst_rel).startswith("include/") else src_dir
        dst = dst_root / Path(dst_rel).name
        shutil.copy2(src, dst)


def _write_terminal_model_config(case_cfg: dict[str, Any], model_info: dict[str, int]) -> None:
    sync = case_cfg.get("sync_codegen", {})
    include_dir = REPO_ROOT / sync["include_dir"]
    include_dir.mkdir(parents=True, exist_ok=True)
    path = include_dir / "terminal_model_config.h"
    path.write_text(
        "\n".join(
            [
                "#ifndef NANOC_TERMINAL_MODEL_CONFIG_H",
                "#define NANOC_TERMINAL_MODEL_CONFIG_H",
                "",
                f"#define NANOC_TERMINAL_MODEL_INPUT_BYTES {model_info['input_bytes']}U",
                f"#define NANOC_TERMINAL_MODEL_OUTPUT_BYTES {model_info['output_bytes']}U",
                "",
                "#endif /* NANOC_TERMINAL_MODEL_CONFIG_H */",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _read_model_info(model_h: Path) -> dict[str, int]:
    text = model_h.read_text(encoding="utf-8")
    keys = {
        "input_bytes": "NANOC_MODEL_INPUT_BYTES",
        "output_bytes": "NANOC_MODEL_OUTPUT_BYTES",
        "estimated_sram_bytes": "NANOC_MODEL_ESTIMATED_SRAM_BYTES",
        "estimated_flash_bytes": "NANOC_MODEL_ESTIMATED_FLASH_BYTES",
    }
    result = {}
    for name, macro in keys.items():
        match = re.search(rf"#define\s+{macro}\s+([0-9]+)[uU]?", text)
        result[name] = int(match.group(1)) if match else 0
    return result


def _is_f103_oversize(case_cfg: dict[str, Any], model_info: dict[str, int]) -> bool:
    limits = case_cfg.get("limits", {})
    max_input = int(limits.get("max_input_bytes", 16384))
    max_output = int(limits.get("max_output_bytes", 8192))
    max_sram = int(limits.get("max_estimated_sram_bytes", 57344))
    return (
        int(model_info["input_bytes"]) > max_input
        or int(model_info["output_bytes"]) > max_output
        or int(model_info["estimated_sram_bytes"]) > max_sram
    )


def _generate_models(case_id: str) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, str(TDD_SCRIPTS / "generate_models.py"), "--case", case_id],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0, completed.stdout + completed.stderr


def _build_arm_project(case_cfg: dict[str, Any]) -> tuple[bool, str]:
    project = REPO_ROOT / case_cfg["arm_project"]
    completed = subprocess.run(
        ["cmake", "--build", "--preset", "Debug"],
        cwd=str(project),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0, completed.stdout + completed.stderr


def _flash_firmware(case_cfg: dict[str, Any], board_cfg: dict[str, Any]) -> tuple[bool, str]:
    elf = REPO_ROOT / case_cfg["firmware_elf"]
    cli = board_cfg["programmer"]["cli"]
    interface = board_cfg["programmer"].get("interface", "SWD")
    completed = subprocess.run(
        [cli, "-c", f"port={interface}", "-w", str(elf), "-v", "-rst"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0, completed.stdout + completed.stderr


def _parse_numeric_runner_samples(path: Path, input_bytes: int) -> list[list[int]]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"static const int8_t sample_(\d+)\[\d+\]\s*=\s*\{([^}]*)\};", re.S)
    samples: list[tuple[int, list[int]]] = []
    for match in pattern.finditer(text):
        idx = int(match.group(1))
        values = [int(item) for item in re.findall(r"-?\d+", match.group(2))]
        if len(values) != input_bytes:
            raise ValueError(f"sample_{idx} input size {len(values)} != {input_bytes}")
        samples.append((idx, values))
    samples.sort(key=lambda item: item[0])
    return [values for _, values in samples]


def _run_uart_samples(
    board_cfg: dict[str, Any],
    *,
    arm_case_id: int,
    output_bytes: int,
    samples: list[list[int]],
) -> tuple[list[dict[str, Any]], str]:
    uart = board_cfg["uart"]
    fd = os.open(str(uart["port"]), os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        _configure_serial(fd, int(uart["baudrate"]))
        termios.tcflush(fd, termios.TCIOFLUSH)
        time.sleep(0.25)
        outputs = []
        for values in samples:
            payload = bytes(value & 0xFF for value in values)
            header = struct.pack("<IBBHI", REQ_MAGIC, PROTOCOL_VERSION, CMD_RUN_CASE, arm_case_id, len(payload))
            frame = header + struct.pack("<I", _checksum(header + payload)) + payload
            os.write(fd, frame)
            response_header = _read_exact(fd, 20, timeout=8.0)
            if len(response_header) != 20:
                return outputs, f"response header timeout: {len(response_header)}/20"
            magic, version, status, case_id, elapsed_us, out_size, resp_checksum = struct.unpack(
                "<IBBHIII", response_header
            )
            payload_out = _read_exact(fd, out_size, timeout=3.0)
            checksum_ok = _checksum(response_header[:16] + payload_out) == resp_checksum
            outputs.append(
                {
                    "magic": magic,
                    "version": version,
                    "status": status,
                    "case_id": case_id,
                    "elapsed_us": elapsed_us,
                    "output_size": out_size,
                    "checksum_ok": checksum_ok,
                    "output": _signed(payload_out),
                }
            )
            if magic != RESP_MAGIC or version != PROTOCOL_VERSION or status != 0 or case_id != arm_case_id:
                return outputs, f"bad response: magic=0x{magic:08x} version={version} status={status} case={case_id}"
            if out_size != output_bytes or len(payload_out) != output_bytes:
                return outputs, f"bad output size: declared={out_size} actual={len(payload_out)} expected={output_bytes}"
            if not checksum_ok:
                return outputs, "bad response checksum"
            time.sleep(0.05)
        return outputs, ""
    finally:
        os.close(fd)


def _configure_serial(fd: int, baudrate: int) -> None:
    baud_name = f"B{baudrate}"
    if not hasattr(termios, baud_name):
        raise ValueError(f"unsupported baudrate for termios: {baudrate}")
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = getattr(termios, baud_name)
    attrs[5] = getattr(termios, baud_name)
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def _read_exact(fd: int, size: int, *, timeout: float) -> bytes:
    out = bytearray()
    deadline = time.time() + timeout
    while len(out) < size and time.time() < deadline:
        readable, _, _ = select.select([fd], [], [], max(0.0, min(0.05, deadline - time.time())))
        if not readable:
            continue
        try:
            chunk = os.read(fd, size - len(out))
        except BlockingIOError:
            chunk = b""
        if chunk:
            out.extend(chunk)
    return bytes(out)


def _compare_samples(
    case_cfg: dict[str, Any],
    numeric_samples: list[dict[str, Any]],
    arm_outputs: list[dict[str, Any]],
) -> list[TerminalSampleResult]:
    require_exact = bool(case_cfg.get("compare", {}).get("require_exact_int8", True))
    require_top1 = bool(case_cfg.get("compare", {}).get("require_top1_match", True))
    results = []
    for numeric, arm in zip(numeric_samples, arm_outputs):
        host_values = [int(value) for value in numeric.get("c_int8", [])]
        arm_values = [int(value) for value in arm.get("output", [])]
        arm_top1 = _top1(arm_values)
        onnx_top1 = numeric.get("onnx_top1")
        host_top1 = numeric.get("c_top1")
        exact = host_values == arm_values
        top1_match = (arm_top1 == onnx_top1) and (host_top1 == onnx_top1)
        errors = []
        if require_exact and not exact:
            errors.append("host C and ARM C int8 outputs differ")
        if require_top1 and not top1_match:
            errors.append("ONNX/Host C/ARM C top1 mismatch")
        results.append(
            TerminalSampleResult(
                id=str(numeric.get("id", f"sample_{len(results)}")),
                onnx_top1=int(onnx_top1) if onnx_top1 is not None else None,
                host_c_top1=int(host_top1) if host_top1 is not None else None,
                arm_c_top1=arm_top1,
                host_c_int8=host_values,
                arm_c_int8=arm_values,
                exact_match=exact,
                top1_match=top1_match,
                elapsed_us=arm.get("elapsed_us"),
                error_msg="; ".join(errors),
            )
        )
    return results


def _top1(values: list[int]) -> int | None:
    if not values:
        return None
    return max(range(len(values)), key=lambda idx: values[idx])


def _checksum(data: bytes) -> int:
    return sum(data) & 0xFFFFFFFF


def _signed(data: bytes) -> list[int]:
    return [item - 256 if item >= 128 else item for item in data]


def _save_report(report: TerminalReport) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    content = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    latest_path = REPORTS_DIR / "terminal_latest.json"
    latest_path.write_text(content, encoding="utf-8")

    history_dir = REPORTS_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}-terminal.json"
    history_path.write_text(content, encoding="utf-8")
    return latest_path


def _fmt_elapsed(result: TerminalCaseResult) -> str:
    if result.elapsed_us_avg is None:
        return "-"
    return f"{result.elapsed_us_avg:.0f}us avg"


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


if __name__ == "__main__":
    raise SystemExit(main())

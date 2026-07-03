"""
tdd/scripts/common/runner.py — 统一的测试执行器

扫描指定阶段的 ONNX 模型目录, 逐个执行 converter → codegen → compile 链路,
并汇总结果。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"


@dataclass
class CaseResult:
    """单个测试用例的执行结果。"""
    case_id: str
    onnx_path: Path
    phase: str
    expected_status: str  # "ok" / "blocked" / "unsupported"
    # --- converter ---
    converter_ok: bool = False
    converter_stderr: str = ""
    # --- codegen ---
    codegen_status: str = ""  # ok / blocked / unsupported
    codegen_stdout: str = ""
    codegen_stderr: str = ""
    pipeline_exit_code: int = -1
    # --- compile ---
    compile_ok: bool = False
    compile_stderr: str = ""
    # --- runtime ---
    runtime_ok: bool = False
    runtime_stdout: str = ""
    runtime_stderr: str = ""
    # --- meta ---
    duration_sec: float = 0.0
    passed: bool = False  # 是否符合预期


@dataclass
class PhaseReport:
    """单个阶段的测试报告。"""
    phase: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[CaseResult] = field(default_factory=list)


def discover_onnx_models(models_root: Path) -> list[tuple[str, Path]]:
    """扫描 models 目录下所有 .onnx 文件, 返回 (case_id, path) 列表。"""
    onnx_files: list[tuple[str, Path]] = []
    for onnx_path in sorted(models_root.rglob("*.onnx")):
        case_id = onnx_path.stem
        onnx_files.append((case_id, onnx_path))
    return onnx_files


def run_pipeline(
    onnx_path: Path,
    tmp_dir: Path,
    extra_args: list[str] | None = None,
) -> tuple[int, str, str]:
    """运行 nanoc onnx-to-cmsis pipeline, 返回 (exit_code, stdout, stderr)。"""
    args = [
        sys.executable,
        "-m",
        "nanoc_nn.pipeline",
        "onnx-to-cmsis",
        "--model",
        str(onnx_path),
        "--target",
        "cortex-m4",
        "--sram-budget",
        "256K",
        "--flash-budget",
        "1M",
        "--out-dir",
        str(tmp_dir),
        "--no-compile",
    ]
    if extra_args:
        args.extend(extra_args)

    completed = subprocess.run(
        args,
        text=True,
        capture_output=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    return completed.returncode, completed.stdout, completed.stderr


def read_codegen_status(codegen_dir: Path) -> str:
    """从 codegen 报告文件中读取状态。"""
    report_path = codegen_dir / "reports" / "codegen_report.txt"
    if not report_path.exists():
        return "unknown"
    content = report_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("status:"):
            return line_stripped.split(":", 1)[1].strip()
        if line_stripped.startswith("codegen_status:"):
            return line_stripped.split(":", 1)[1].strip()
    return "unknown"


def compile_c_project(
    codegen_dir: Path,
    tmp_dir: Path,
    cmsis_nn_root: Path | None = None,
) -> tuple[bool, str]:
    """对生成的 C 工程做 C99 smoke compile。"""
    cc = "gcc"
    src_dir = codegen_dir / "src"
    include_dir = codegen_dir / "include"

    model_c = src_dir / "model.c"
    main_c = src_dir / "main.c"
    if not model_c.exists() or not main_c.exists():
        return False, f"Missing source files: {model_c}, {main_c}"

    cmd = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I", str(include_dir),
        str(model_c),
        str(main_c),
        "-o", str(tmp_dir / "smoke_test"),
    ]

    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return completed.returncode == 0, completed.stderr


def run_smoke_test(
    binary: Path,
    timeout_sec: int = 5,
) -> tuple[bool, str, str]:
    """运行 smoke test binary。"""
    try:
        completed = subprocess.run(
            [str(binary)],
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        return completed.returncode == 0, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired:
        return False, "", "timeout expired"
    except Exception as exc:
        return False, "", str(exc)


def run_phase(
    phase: str,
    models_root: Path,
    tmp_base: Path,
    expected_map: dict[str, str] | None = None,
    skip_compile: bool = False,
    skip_run: bool = False,
) -> PhaseReport:
    """执行单个阶段的所有用例。

    Args:
        phase: 阶段名 (如 "phase1")。
        models_root: 模型根目录。
        tmp_base: 临时输出根目录。
        expected_map: {case_id: expected_status} 映射, 未指定则默认 "ok"。
        skip_compile: 跳过 C 编译。
        skip_run: 跳过运行测试。
    """
    if expected_map is None:
        expected_map = {}

    report = PhaseReport(phase=phase)
    cases = discover_onnx_models(models_root)

    for case_id, onnx_path in cases:
        expected = expected_map.get(case_id, "ok")
        result = CaseResult(
            case_id=case_id,
            onnx_path=onnx_path,
            phase=phase,
            expected_status=expected,
        )

        t0 = time.monotonic()
        tmp_dir = tmp_base / case_id
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Pipeline (converter + codegen)
        exit_code, stdout, stderr = run_pipeline(onnx_path, tmp_dir)
        result.pipeline_exit_code = exit_code
        result.codegen_stdout = stdout
        result.codegen_stderr = stderr

        # 找到 codegen 输出目录
        codegen_dir = tmp_dir / "cmsis-codegen"
        if codegen_dir.exists():
            result.codegen_status = read_codegen_status(codegen_dir)
        else:
            result.codegen_status = "unknown"

        # Step 2: C99 smoke compile (仅对 codegen ok 的用例)
        if result.codegen_status == "ok" and not skip_compile:
            compile_ok, compile_err = compile_c_project(codegen_dir, tmp_dir)
            result.compile_ok = compile_ok
            result.compile_stderr = compile_err

            # Step 3: Smoke run
            if compile_ok and not skip_run:
                binary = tmp_dir / "smoke_test"
                if binary.exists():
                    runtime_ok, r_stdout, r_stderr = run_smoke_test(binary)
                    result.runtime_ok = runtime_ok
                    result.runtime_stdout = r_stdout
                    result.runtime_stderr = r_stderr

        result.duration_sec = time.monotonic() - t0

        # 判断通过
        result.passed = (
            result.codegen_status == expected
            and (result.codegen_status != "ok" or result.compile_ok)
            and (not result.compile_ok or result.runtime_ok)
        )

        report.results.append(result)
        report.total += 1
        if result.passed:
            report.passed += 1
        else:
            report.failed += 1

    return report


def print_report(report: PhaseReport) -> None:
    """打印分阶段测试报告。"""
    print(f"\n{'='*60}")
    print(f"  阶段: {report.phase}")
    print(f"  总计: {report.total}  |  通过: {report.passed}  |  失败: {report.failed}")
    print(f"{'='*60}")

    for r in report.results:
        status_icon = "✅" if r.passed else "❌"
        compile_str = ""
        if r.codegen_status == "ok":
            compile_str = f" | compile={'OK' if r.compile_ok else 'FAIL'}"
            if r.compile_ok:
                compile_str += f" | run={'OK' if r.runtime_ok else 'FAIL'}"
        print(
            f"  {status_icon} {r.case_id:<30} "
            f"expected={r.expected_status:<12} "
            f"codegen={r.codegen_status:<12}"
            f"{compile_str}"
            f"  ({r.duration_sec:.1f}s)"
        )

    if report.failed > 0:
        print(f"\n  失败用例详情:")
        for r in report.results:
            if not r.passed:
                print(f"\n  --- {r.case_id} ---")
                if r.codegen_status != r.expected_status:
                    print(f"    状态不匹配: expected={r.expected_status}, got={r.codegen_status}")
                if r.codegen_stderr.strip():
                    print(f"    stderr: {r.codegen_stderr[:500]}")
                if r.compile_stderr.strip():
                    print(f"    compile: {r.compile_stderr[:500]}")

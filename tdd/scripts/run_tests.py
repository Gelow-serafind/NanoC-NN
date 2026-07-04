"""
tdd/scripts/run_tests.py — TDD 测试执行器。

扫描 tdd/work/models/ 下的 ONNX 文件，逐个执行 pipeline，并把结果写入 results/latest.json。

用法:
    python tdd/scripts/run_tests.py --generate
    python tdd/scripts/run_tests.py --mode baseline --generate
    python tdd/scripts/run_tests.py --case GEMM_001 --generate
    python tdd/scripts/run_tests.py --category core/gemm --generate
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
TDD_ROOT = _SCRIPT_DIR.parent
REPO_ROOT = TDD_ROOT.parent
WORK_ROOT = TDD_ROOT / "work"
MODELS_ROOT = WORK_ROOT / "models"
STRUCTURAL_WORK_ROOT = WORK_ROOT / "structural"
RESULTS_DIR = TDD_ROOT / "results"
CAPABILITIES_PATH = TDD_ROOT / "CAPABILITIES.md"
WORKDIR_MARKER = ".nanoc_tdd_workdir"

sys.path.insert(0, str(_SCRIPT_DIR))
from cases_registry import CASE_MAP  # noqa: E402


@dataclass
class CaseResult:
    case_id: str
    category: str
    onnx_path: str
    output_dir: str
    expected: str
    actual: str = "unknown"
    passed: bool = False
    exit_code: int = -1
    required_apis: list[str] = field(default_factory=list)
    missing_apis: list[str] = field(default_factory=list)
    api_check_ok: bool = False
    compile_ok: bool = False
    runtime_ok: bool = False
    runtime_exit_code: int | None = None
    error_msg: str = ""
    duration_sec: float = 0.0


@dataclass
class TestReport:
    timestamp: str = ""
    mode: str = "target"
    git_commit: str = ""
    platform: str = ""
    command: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[dict] = field(default_factory=list)


def discover_models(
    *,
    mode: str,
    category: str | None = None,
    case: str | None = None,
) -> list[tuple[str, Path]]:
    """扫描 models/ 目录，返回 (case_id, path) 列表。"""
    if case:
        for onnx_path in MODELS_ROOT.rglob(f"{case}.onnx"):
            return [(case, onnx_path)]
        return []

    search_root = MODELS_ROOT / category if category else MODELS_ROOT
    found: list[tuple[str, Path]] = []
    for onnx_path in sorted(search_root.rglob("*.onnx")):
        case_id = onnx_path.stem
        case_def = CASE_MAP.get(case_id)
        if case_def is None:
            continue
        if mode == "baseline" and not case_def.regression:
            continue
        found.append((case_id, onnx_path))
    return found


def run_pipeline(onnx_path: Path, tmp_dir: Path) -> tuple[int, str, str]:
    """执行 nanoc pipeline，返回 (exit_code, stdout, stderr)。"""
    args = [
        sys.executable,
        "-m",
        "nanoc_nn.cli",
        "onnx-to-cmsis",
        "--model",
        str(onnx_path),
        "--target",
        "cortex-m4",
        "--out-root",
        str(tmp_dir),
        "--no-compile",
    ]
    try:
        completed = subprocess.run(
            args,
            text=True,
            capture_output=True,
            check=False,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as exc:  # noqa: BLE001 - subprocess boundary.
        return -1, "", str(exc)


def detect_status(tmp_dir: Path, exit_code: int, stdout: str, stderr: str) -> str:
    """从 pipeline 产物和输出中推断 codegen status。"""
    report_path = tmp_dir / "cmsis-codegen" / "reports" / "codegen_report.txt"
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("status:") or stripped.startswith("codegen_status:"):
                return stripped.split(":", 1)[1].strip()

    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("codegen status:"):
            return stripped.split(":", 1)[1].strip()

    pipeline_report = tmp_dir / "pipeline_report.md"
    if pipeline_report.exists():
        content = pipeline_report.read_text(encoding="utf-8").lower()
        if "unsupported" in content:
            return "unsupported"
        if "blocked" in content:
            return "blocked"
        if "ok" in content:
            return "ok"

    stderr_lower = stderr.lower()
    if exit_code != 0:
        if "oversize" in stderr_lower:
            return "oversize"
        if "unsupported" in stderr_lower or "not supported" in stderr_lower:
            return "unsupported"
        if "blocked" in stderr_lower:
            return "blocked"
        return "unsupported"

    return "unknown"


def compile_c_project(codegen_dir: Path, tmp_dir: Path) -> tuple[bool, str]:
    """对生成的 C 工程做 C99 smoke compile。"""
    cc = os.environ.get("CC", "gcc")
    if shutil.which(cc) is None:
        return False, f"C compiler not found: {cc}"

    model_c = codegen_dir / "src" / "model.c"
    main_c = codegen_dir / "src" / "main.c"
    include_dir = codegen_dir / "include"
    if not model_c.exists() or not main_c.exists():
        return False, f"missing generated C source: {model_c} or {main_c}"

    binary = tmp_dir / "smoke_test"
    cmd = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-DNANOC_ENABLE_CMSIS_NN=0",
        "-I",
        str(include_dir),
        str(model_c),
        str(main_c),
        "-o",
        str(binary),
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return False, completed.stderr[:1000]
    return True, ""


def run_smoke_binary(tmp_dir: Path) -> tuple[bool, int | None, str]:
    """运行 C99 smoke binary。

    Host smoke 只验证生成物能启动且不崩溃/不超时。默认编译关闭真实 CMSIS-NN，
    因此生成物返回 NANOC_STATUS_BLOCKED 也属于预期的保护行为。
    """
    binary = tmp_dir / "smoke_test"
    if not binary.exists():
        return False, None, f"missing smoke binary: {binary}"
    try:
        completed = subprocess.run(
            [str(binary)],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return False, None, "smoke binary timeout"
    if completed.returncode < 0:
        return False, completed.returncode, (completed.stderr or completed.stdout)[:1000]
    return True, completed.returncode, ""


def check_required_apis(codegen_dir: Path, required_apis: list[str]) -> tuple[bool, list[str]]:
    """检查生成的 model.c 是否包含用例要求的 CMSIS-NN API。"""
    if not required_apis:
        return True, []
    model_c = codegen_dir / "src" / "model.c"
    if not model_c.exists():
        return False, required_apis
    content = model_c.read_text(encoding="utf-8")
    missing = [api for api in required_apis if api not in content]
    return not missing, missing


def run_case(case_id: str, onnx_path: Path) -> CaseResult:
    """执行单个测试用例。"""
    case_def = CASE_MAP[case_id]
    tmp_dir = STRUCTURAL_WORK_ROOT / case_id
    _reset_work_dir(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / WORKDIR_MARKER).write_text("structural\n", encoding="utf-8")

    result = CaseResult(
        case_id=case_id,
        category=case_def.category,
        onnx_path=_repo_relative(onnx_path),
        output_dir=_repo_relative(tmp_dir),
        expected=case_def.expected,
        required_apis=list(case_def.required_apis),
    )

    t0 = time.monotonic()
    exit_code, stdout, stderr = run_pipeline(onnx_path, tmp_dir)
    result.exit_code = exit_code
    result.actual = detect_status(tmp_dir, exit_code, stdout, stderr)

    codegen_dir = tmp_dir / "cmsis-codegen"
    if result.actual == "ok":
        result.api_check_ok, result.missing_apis = check_required_apis(
            codegen_dir,
            result.required_apis,
        )
        result.compile_ok, compile_error = compile_c_project(codegen_dir, tmp_dir)
        if result.compile_ok:
            result.runtime_ok, result.runtime_exit_code, runtime_error = run_smoke_binary(tmp_dir)
        else:
            runtime_error = ""
    else:
        result.api_check_ok = not result.required_apis
        compile_error = ""
        runtime_error = ""

    result.passed = _case_passed(result)
    if not result.passed:
        result.error_msg = _failure_reason(result, stderr, compile_error, runtime_error)

    result.duration_sec = round(time.monotonic() - t0, 2)
    return result


def _reset_work_dir(path: Path) -> None:
    """Remove a runner-owned work directory.

    Human datasets and manual experiments must live outside tdd/work.  This guard
    prevents the runner from deleting a directory unless it was created by a TDD
    runner in a previous invocation.
    """
    if not path.exists():
        return
    marker = path / WORKDIR_MARKER
    if not marker.exists():
        raise RuntimeError(
            f"refuse to delete unmarked directory: {path}. "
            f"Move manual assets to tdd/fixtures or add {WORKDIR_MARKER} only for runner-owned work dirs."
        )
    shutil.rmtree(path)


def _case_passed(result: CaseResult) -> bool:
    if result.actual != result.expected:
        return False
    if result.expected == "ok":
        return (
            result.exit_code == 0
            and result.api_check_ok
            and result.compile_ok
            and result.runtime_ok
        )
    return result.exit_code != 0


def _failure_reason(
    result: CaseResult,
    stderr: str,
    compile_error: str,
    runtime_error: str,
) -> str:
    if result.actual != result.expected:
        return f"预期 {result.expected}，实际 {result.actual}"
    if result.expected == "ok" and result.exit_code != 0:
        return f"pipeline exit_code={result.exit_code}: {stderr[:500]}"
    if result.missing_apis:
        return f"生成代码缺少 CMSIS-NN API: {', '.join(result.missing_apis)}"
    if result.expected == "ok" and not result.compile_ok:
        return f"C99 smoke compile 失败: {compile_error[:500]}"
    if result.expected == "ok" and not result.runtime_ok:
        return f"C99 smoke run 失败: {runtime_error[:500]}"
    return stderr[:500] if stderr else "unknown failure"


def run_all(
    *,
    mode: str,
    category: str | None = None,
    case: str | None = None,
) -> TestReport:
    """执行全部或部分测试用例。"""
    models = discover_models(mode=mode, category=category, case=case)
    if not models:
        print("未找到任何测试模型。请先运行: python tdd/scripts/generate_models.py")
        sys.exit(1)

    report = TestReport(
        timestamp=datetime.now().isoformat(),
        mode=mode,
        git_commit=_git_commit(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        command=" ".join(sys.argv),
    )
    print(f"\n{'=' * 60}")
    print(f"  TDD 测试执行 — mode={mode} — {len(models)} 个用例")
    print(f"{'=' * 60}\n")

    for case_id, onnx_path in models:
        result = run_case(case_id, onnx_path)
        report.total += 1
        if result.passed:
            report.passed += 1
            icon = "PASS"
        else:
            report.failed += 1
            icon = "FAIL"

        print(
            f"  [{icon}] {case_id:<15} "
            f"expected={result.expected:<12} "
            f"actual={result.actual:<12} "
            f"api={'OK' if result.api_check_ok else 'MISS':<4} "
            f"cc={'OK' if result.compile_ok else '-':<2} "
            f"run={'OK' if result.runtime_ok else '-':<2} "
            f"({result.duration_sec:.1f}s)"
        )
        if not result.passed and result.error_msg:
            print(f"         -> {result.error_msg[:200]}")

        report.results.append(asdict(result))

    print(f"\n{'=' * 60}")
    print(f"  总计: {report.total}  |  通过: {report.passed}  |  失败: {report.failed}")
    print(f"{'=' * 60}\n")

    return report


def save_report(report: TestReport) -> Path:
    """保存报告到 results/latest.json；baseline 通过时同步保存 baseline.json。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    content = json.dumps(asdict(report), indent=2, ensure_ascii=False)

    latest_path = RESULTS_DIR / "latest.json"
    latest_path.write_text(content, encoding="utf-8")

    history_dir = RESULTS_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    history_path = history_dir / f"{date_str}-{report.mode}.json"
    history_path.write_text(content, encoding="utf-8")

    if report.mode == "baseline" and report.failed == 0 and report.total > 0:
        baseline_path = RESULTS_DIR / "baseline.json"
        baseline_path.write_text(content, encoding="utf-8")

    return latest_path


def generate_capabilities(report: TestReport) -> Path:
    """根据 target 全量测试报告生成 CAPABILITIES.md。"""
    passed_cases = []
    failed_cases = []

    for r in report.results:
        case_id = r["case_id"]
        case_def = CASE_MAP.get(case_id)
        desc = case_def.description if case_def else case_id
        category = case_def.category if case_def else r.get("category", "")
        if r["passed"]:
            passed_cases.append((case_id, category, desc))
        else:
            reason = r.get("error_msg", "unknown")
            failed_cases.append((case_id, category, desc, reason))

    date_str = report.timestamp[:10] if report.timestamp else "unknown"
    total = report.total
    passed = report.passed

    lines = [
        "# NanoC-NN 能力集",
        "",
        "> 本文件由 `python tdd/scripts/run_tests.py --mode target` 自动生成，禁止手动编辑。",
        f"> 最后更新: {report.timestamp}",
        f"> Git commit: `{report.git_commit or 'unknown'}`",
        "",
        f"**能力集大小: {passed} / {total} ({100 * passed // total if total else 0}%)**",
        "",
        "## 已验证能力 (PASS)",
        "",
        "以下用例通过测试，代表代码生成器已确认支持的能力：",
        "",
    ]

    if passed_cases:
        lines.append("| 用例 ID | 分类 | 能力描述 | 验证日期 |")
        lines.append("|---------|------|---------|---------|")
        for case_id, category, desc in passed_cases:
            lines.append(f"| {case_id} | {category} | {desc} | {date_str} |")
    else:
        lines.append("*暂无已验证能力。*")

    lines.extend([
        "",
        "## 未通过用例 (FAIL)",
        "",
        "以下用例代表目标能力但尚未实现：",
        "",
    ])

    if failed_cases:
        lines.append("| 用例 ID | 分类 | 目标能力 | 阻塞原因 |")
        lines.append("|---------|------|---------|---------|")
        for case_id, category, desc, reason in failed_cases:
            lines.append(f"| {case_id} | {category} | {desc} | {reason} |")
    else:
        lines.append("*所有用例均已通过。*")

    lines.append("")

    content = "\n".join(lines)
    CAPABILITIES_PATH.write_text(content, encoding="utf-8")
    return CAPABILITIES_PATH


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _validate_cases() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / "validate_cases.py")],
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _generate_models(case: str | None, category: str | None) -> None:
    args = [sys.executable, str(_SCRIPT_DIR / "generate_models.py")]
    if case:
        args.extend(["--case", case])
    elif category:
        args.extend(["--category", category])
    subprocess.run(args, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="TDD 测试执行器")
    parser.add_argument("--case", type=str, default=None, help="执行单个用例")
    parser.add_argument("--category", type=str, default=None, help="执行某一类")
    parser.add_argument(
        "--mode",
        choices=["target", "baseline"],
        default="target",
        help="target 执行全部目标用例；baseline 只执行已纳入稳定回归门禁的用例",
    )
    parser.add_argument("--validate-only", action="store_true", help="只校验 TDD 用例规格")
    parser.add_argument("--skip-validation", action="store_true", help="跳过用例规格校验")
    parser.add_argument("--generate", action="store_true", help="执行前先生成模型")
    args = parser.parse_args()

    if not args.skip_validation:
        _validate_cases()
    if args.validate_only:
        return

    if args.generate:
        print("生成测试模型...")
        _generate_models(args.case, args.category)
        print()

    report = run_all(mode=args.mode, category=args.category, case=args.case)
    path = save_report(report)
    print(f"报告已保存: {path}")

    should_update_capabilities = (
        args.mode == "target" and args.case is None and args.category is None
    )
    if should_update_capabilities:
        cap_path = generate_capabilities(report)
        print(f"能力集已更新: {cap_path}")
    else:
        print("能力集未更新：仅 target 全量测试会重写 CAPABILITIES.md")

    if args.mode == "baseline" and report.failed == 0 and report.total > 0:
        print(f"基线已晋升: {RESULTS_DIR / 'baseline.json'}")

    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()

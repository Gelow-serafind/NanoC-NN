"""
tdd/scripts/run_tests.py — TDD 测试执行器

扫描 tdd/models/ 下的 ONNX 文件, 逐个执行 pipeline, 输出结果到 results/latest.json。

用法:
    python tdd/scripts/run_tests.py
    python tdd/scripts/run_tests.py --case GEMM_001
    python tdd/scripts/run_tests.py --category core/gemm
    python tdd/scripts/run_tests.py --generate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
TDD_ROOT = _SCRIPT_DIR.parent
REPO_ROOT = TDD_ROOT.parent
MODELS_ROOT = TDD_ROOT / "models"
RESULTS_DIR = TDD_ROOT / "results"
CAPABILITIES_PATH = TDD_ROOT / "CAPABILITIES.md"

sys.path.insert(0, str(_SCRIPT_DIR))
from cases_registry import CASE_MAP


@dataclass
class CaseResult:
    case_id: str
    onnx_path: str
    expected: str
    actual: str = "unknown"
    passed: bool = False
    exit_code: int = -1
    error_msg: str = ""
    duration_sec: float = 0.0


@dataclass
class TestReport:
    timestamp: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[dict] = field(default_factory=list)


def discover_models(category: str | None = None, case: str | None = None) -> list[tuple[str, Path]]:
    """扫描 models/ 目录, 返回 (case_id, path) 列表。"""
    if case:
        for onnx_path in MODELS_ROOT.rglob(f"{case}.onnx"):
            return [(case, onnx_path)]
        return []

    found: list[tuple[str, Path]] = []
    search_root = MODELS_ROOT
    if category:
        search_root = MODELS_ROOT / category

    for onnx_path in sorted(search_root.rglob("*.onnx")):
        case_id = onnx_path.stem
        found.append((case_id, onnx_path))
    return found


def run_pipeline(onnx_path: Path, tmp_dir: Path) -> tuple[int, str, str]:
    """执行 nanoc pipeline, 返回 (exit_code, stdout, stderr)。"""
    args = [
        sys.executable,
        "-m", "nanoc_nn.cli",
        "onnx-to-cmsis",
        "--model", str(onnx_path),
        "--target", "cortex-m4",
        "--sram-budget", "256K",
        "--flash-budget", "1M",
        "--out-root", str(tmp_dir),
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
    except Exception as exc:
        return -1, "", str(exc)


def detect_status(tmp_dir: Path, exit_code: int, stderr: str) -> str:
    """从 pipeline 输出中推断 codegen status。"""
    report_path = tmp_dir / "cmsis-codegen" / "reports" / "codegen_report.txt"
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("status:") or stripped.startswith("codegen_status:"):
                return stripped.split(":", 1)[1].strip()

    pipeline_report = tmp_dir / "pipeline_report.md"
    if pipeline_report.exists():
        content = pipeline_report.read_text(encoding="utf-8")
        if "unsupported" in content.lower():
            return "unsupported"
        if "blocked" in content.lower():
            return "blocked"
        if "ok" in content.lower():
            return "ok"

    if exit_code != 0:
        if "unsupported" in stderr.lower() or "not supported" in stderr.lower():
            return "unsupported"
        if "blocked" in stderr.lower():
            return "blocked"
        return "unsupported"

    return "unknown"


def run_case(case_id: str, onnx_path: Path) -> CaseResult:
    """执行单个测试用例。"""
    case_def = CASE_MAP.get(case_id)
    expected = case_def.expected if case_def else "ok"
    result = CaseResult(
        case_id=case_id,
        onnx_path=str(onnx_path),
        expected=expected,
    )

    t0 = time.monotonic()
    tmp_dir = TDD_ROOT / "results" / "tmp" / case_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    exit_code, stdout, stderr = run_pipeline(onnx_path, tmp_dir)
    result.exit_code = exit_code
    result.actual = detect_status(tmp_dir, exit_code, stderr)

    if result.actual == expected:
        result.passed = True
    else:
        result.error_msg = stderr[:500] if stderr else f"expected={expected}, got={result.actual}"

    result.duration_sec = round(time.monotonic() - t0, 2)
    return result


def run_all(
    category: str | None = None,
    case: str | None = None,
) -> TestReport:
    """执行全部或部分测试用例。"""
    models = discover_models(category=category, case=case)
    if not models:
        print("未找到任何测试模型。请先运行: python tdd/scripts/generate_models.py")
        sys.exit(1)

    report = TestReport(timestamp=datetime.now().isoformat())
    print(f"\n{'='*60}")
    print(f"  TDD 测试执行 — {len(models)} 个用例")
    print(f"{'='*60}\n")

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
            f"({result.duration_sec:.1f}s)"
        )
        if not result.passed and result.error_msg:
            print(f"         -> {result.error_msg[:200]}")

        report.results.append(asdict(result))

    print(f"\n{'='*60}")
    print(f"  总计: {report.total}  |  通过: {report.passed}  |  失败: {report.failed}")
    print(f"{'='*60}\n")

    return report


def save_report(report: TestReport) -> Path:
    """保存报告到 results/latest.json；全量通过时同步晋升 baseline.json。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = RESULTS_DIR / "latest.json"
    content = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    latest_path.write_text(content, encoding="utf-8")

    history_dir = RESULTS_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    history_path = history_dir / f"{date_str}.json"
    history_path.write_text(content, encoding="utf-8")

    if report.failed == 0 and report.total > 0:
        baseline_path = RESULTS_DIR / "baseline.json"
        baseline_path.write_text(content, encoding="utf-8")

    return latest_path


def generate_capabilities(report: TestReport) -> Path:
    """根据测试报告生成 CAPABILITIES.md — 能力集权威声明。"""
    passed_cases = []
    failed_cases = []

    for r in report.results:
        case_id = r["case_id"]
        case_def = CASE_MAP.get(case_id)
        desc = case_def.description if case_def else case_id
        category = _infer_category(case_id)
        if r["passed"]:
            passed_cases.append((case_id, category, desc))
        else:
            reason = r.get("error_msg", "unknown")
            actual = r.get("actual", "unknown")
            expected = r.get("expected", "ok")
            if actual == "blocked" and expected == "ok":
                reason = "converter 未提取节点级量化字段"
            elif actual != expected:
                reason = f"预期 {expected}，实际 {actual}"
            failed_cases.append((case_id, category, desc, reason))

    date_str = report.timestamp[:10] if report.timestamp else "unknown"
    total = report.total
    passed = report.passed

    lines = [
        "# NanoC-NN 能力集",
        "",
        "> 本文件由 `python tdd/scripts/run_tests.py` 自动生成，禁止手动编辑。",
        f"> 最后更新: {report.timestamp}",
        "",
        f"**能力集大小: {passed} / {total} ({100*passed//total if total else 0}%)**",
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


def _infer_category(case_id: str) -> str:
    """从 case_id 推断分类路径。"""
    prefix = case_id.split("_")[0].lower()
    category_map = {
        "gemm": "core/gemm",
        "conv": "core/conv",
        "maxpool": "core/maxpool",
        "softmax": "core/softmax",
        "avgpool": "core/avgpool",
        "topo": "topology",
        "neg": "negative",
    }
    return category_map.get(prefix, prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="TDD 测试执行器")
    parser.add_argument("--case", type=str, default=None, help="执行单个用例")
    parser.add_argument("--category", type=str, default=None, help="执行某一类")
    parser.add_argument(
        "--generate", action="store_true",
        help="执行前先生成模型",
    )
    args = parser.parse_args()

    if args.generate:
        print("生成测试模型...")
        subprocess.run(
            [sys.executable, str(_SCRIPT_DIR / "generate_models.py")],
            check=True,
        )
        print()

    report = run_all(category=args.category, case=args.case)
    path = save_report(report)
    print(f"报告已保存: {path}")
    if report.failed == 0 and report.total > 0:
        print(f"基线已晋升: {RESULTS_DIR / 'baseline.json'}")
    else:
        baseline_path = RESULTS_DIR / "baseline.json"
        if baseline_path.exists():
            print(f"基线未变动（{report.failed} 个失败，保留上次稳定基线）")
        else:
            print("基线未建立（存在失败用例，待全量通过后自动晋升）")

    cap_path = generate_capabilities(report)
    print(f"能力集已更新: {cap_path}")

    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()

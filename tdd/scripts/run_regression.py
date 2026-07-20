"""
tdd/scripts/run_regression.py — stable structural + numeric regression runner.

This entrypoint protects already-accepted capabilities.  It intentionally keeps
the structural and numeric reports separate, then writes a small combined
summary under tdd/results/regression_latest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
TDD_ROOT = _SCRIPT_DIR.parent
REPO_ROOT = TDD_ROOT.parent
RESULTS_DIR = TDD_ROOT / "results"


@dataclass
class StepResult:
    name: str
    command: list[str]
    exit_code: int
    report_path: str = ""


@dataclass
class RegressionReport:
    timestamp: str = ""
    git_commit: str = ""
    platform: str = ""
    command: str = ""
    passed: bool = False
    steps: list[dict] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run stable TDD regression: structural baseline + numeric checks."
    )
    parser.add_argument("--generate", action="store_true", help="generate ONNX models first")
    parser.add_argument(
        "--structural-mode",
        choices=["baseline", "target"],
        default="baseline",
        help="structural test scope; baseline protects committed capabilities",
    )
    parser.add_argument("--skip-structural", action="store_true", help="skip structural tests")
    parser.add_argument("--skip-numeric", action="store_true", help="skip numeric tests")
    parser.add_argument("--structural-python", default=sys.executable, help="python executable for structural tests")
    parser.add_argument("--numeric-python", default=sys.executable, help="python executable for numeric tests")
    parser.add_argument("--terminal-python", default=sys.executable, help="python executable for terminal tests")
    parser.add_argument(
        "--numeric-pythonpath",
        help="optional PYTHONPATH for numeric and terminal subprocesses, for example src",
    )
    parser.add_argument(
        "--terminal",
        choices=["off", "auto", "required"],
        default="off",
        help="optional ARM terminal checks after Host numeric tests",
    )
    parser.add_argument("--terminal-case", help="only run one terminal case")
    parser.add_argument("--terminal-board", help="board id for terminal checks")
    parser.add_argument("--no-terminal-flash", action="store_true", help="do not flash firmware during terminal checks")
    args = parser.parse_args(argv)

    report = RegressionReport(
        timestamp=datetime.now().isoformat(),
        git_commit=_git_commit(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        command=" ".join(sys.argv),
    )

    print(f"\n{'=' * 70}", flush=True)
    print("  TDD 稳定回归 — structural + numeric", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    if not args.skip_structural:
        structural_cmd = [
            args.structural_python,
            str(_SCRIPT_DIR / "run_tests.py"),
            "--mode",
            args.structural_mode,
        ]
        if args.generate:
            structural_cmd.append("--generate")
        report.steps.append(asdict(_run_step("structural", structural_cmd, "latest.json")))

    if not args.skip_numeric:
        numeric_cmd = [args.numeric_python, str(_SCRIPT_DIR / "run_numeric_tests.py")]
        if args.generate:
            numeric_cmd.append("--generate")
        report.steps.append(
            asdict(_run_step("numeric", numeric_cmd, "numeric_latest.json", pythonpath=args.numeric_pythonpath))
        )

    if args.terminal != "off":
        terminal_cmd = [
            args.terminal_python,
            str(TDD_ROOT / "terminal" / "scripts" / "run_terminal_tests.py"),
        ]
        if args.terminal_case:
            terminal_cmd.extend(["--case", args.terminal_case])
        if args.terminal_board:
            terminal_cmd.extend(["--board", args.terminal_board])
        if args.generate:
            terminal_cmd.append("--generate")
        if args.no_terminal_flash:
            terminal_cmd.append("--no-flash")
        if args.terminal == "auto":
            terminal_cmd.append("--auto")
        else:
            terminal_cmd.append("--require-board")
        report.steps.append(
            asdict(
                _run_step(
                    "terminal",
                    terminal_cmd,
                    "terminal/reports/terminal_latest.json",
                    pythonpath=args.numeric_pythonpath,
                )
            )
        )

    report.passed = all(step["exit_code"] == 0 for step in report.steps) and bool(report.steps)
    out_path = _save_report(report)

    print(f"\n{'=' * 70}", flush=True)
    print(f"  稳定回归: {'PASS' if report.passed else 'FAIL'}", flush=True)
    print(f"  报告: {out_path}", flush=True)
    print(f"{'=' * 70}\n", flush=True)
    return 0 if report.passed else 1


def _run_step(name: str, command: list[str], report_name: str, *, pythonpath: str | None = None) -> StepResult:
    print(f"[{name}] {' '.join(command)}", flush=True)
    env = None
    if pythonpath:
        env = dict(**os.environ)
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = pythonpath if not current else f"{pythonpath}:{current}"
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False, env=env)
    report_path = (TDD_ROOT / report_name) if "/" in report_name else (RESULTS_DIR / report_name)
    return StepResult(
        name=name,
        command=command,
        exit_code=completed.returncode,
        report_path=_repo_relative(report_path) if report_path.exists() else "",
    )


def _save_report(report: RegressionReport) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    content = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    latest_path = RESULTS_DIR / "regression_latest.json"
    latest_path.write_text(content, encoding="utf-8")

    history_dir = RESULTS_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    history_path = history_dir / f"{date_str}-regression.json"
    history_path.write_text(content, encoding="utf-8")
    return latest_path


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


if __name__ == "__main__":
    raise SystemExit(main())

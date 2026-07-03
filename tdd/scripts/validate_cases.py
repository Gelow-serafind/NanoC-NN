"""
tdd/scripts/validate_cases.py — TDD 用例规格一致性校验。

该脚本只检查测试资产本身，不生成模型，也不执行 pipeline。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
TDD_ROOT = _SCRIPT_DIR.parent
CASES_ROOT = TDD_ROOT / "cases"

sys.path.insert(0, str(_SCRIPT_DIR))
from cases_registry import CASE_MAP  # noqa: E402

REQUIRED_SECTIONS = (
    "## 验证目标",
    "## 来源",
    "## 网络结构",
    "## 输入",
    "## 算子参数",
    "## 量化设计",
    "## 预期结果",
    "## 边界/风险",
)

VALID_EXPECTED = {"ok", "blocked", "unsupported"}
VALID_SOURCES = {"内部探索", "用户反馈", "缺陷复现"}


def main() -> int:
    errors: list[str] = []
    case_files = {
        path.stem: path
        for path in CASES_ROOT.rglob("*.md")
        if path.name != "README.md"
    }
    registry_ids = set(CASE_MAP)
    generator_ids = _read_generator_ids()
    file_ids = set(case_files)

    _check_set_match("cases 文件", file_ids, "registry", registry_ids, errors)
    _check_set_match("registry", registry_ids, "generator", generator_ids, errors)

    for case_id, path in sorted(case_files.items()):
        content = path.read_text(encoding="utf-8")
        case_def = CASE_MAP.get(case_id)
        if case_def is None:
            continue

        for section in REQUIRED_SECTIONS:
            if section not in content:
                errors.append(f"{path}: 缺少章节 {section}")

        expected_marker = f"**codegen status**: `{case_def.expected}`"
        if expected_marker not in content:
            errors.append(
                f"{path}: 预期结果与 registry 不一致，期望出现 {expected_marker}"
            )

        if case_def.expected not in VALID_EXPECTED:
            errors.append(f"{case_id}: 非法 expected={case_def.expected!r}")

        actual_category = path.parent.relative_to(CASES_ROOT).as_posix()
        if actual_category != case_def.category:
            errors.append(
                f"{path}: category 不一致，registry={case_def.category}, 实际={actual_category}"
            )

        source_text = _section_text(content, "## 来源")
        if source_text and not any(source in source_text for source in VALID_SOURCES):
            errors.append(f"{path}: 来源必须包含 {sorted(VALID_SOURCES)} 之一")

    if errors:
        print("TDD 用例规格校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"TDD 用例规格校验通过：{len(case_files)} 个用例")
    return 0


def _check_set_match(
    left_name: str,
    left: set[str],
    right_name: str,
    right: set[str],
    errors: list[str],
) -> None:
    missing_right = sorted(left - right)
    missing_left = sorted(right - left)
    if missing_right:
        errors.append(f"{left_name} 中存在但 {right_name} 缺失: {missing_right}")
    if missing_left:
        errors.append(f"{right_name} 中存在但 {left_name} 缺失: {missing_left}")


def _section_text(content: str, section: str) -> str:
    if section not in content:
        return ""
    after = content.split(section, 1)[1]
    parts = after.split("\n## ", 1)
    return parts[0].strip()


def _read_generator_ids() -> set[str]:
    source = (_SCRIPT_DIR / "generate_models.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_GENERATORS":
            return _literal_dict_keys(node.value)
        if isinstance(node, ast.Assign):
            if any(getattr(target, "id", "") == "_GENERATORS" for target in node.targets):
                return _literal_dict_keys(node.value)
    return set()


def _literal_dict_keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    keys: set[str] = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


if __name__ == "__main__":
    raise SystemExit(main())

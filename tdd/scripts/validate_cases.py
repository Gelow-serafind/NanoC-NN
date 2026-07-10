"""
tdd/scripts/validate_cases.py — TDD 用例规格一致性校验。

该脚本只检查测试资产本身，不生成模型，也不执行 pipeline。
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
TDD_ROOT = _SCRIPT_DIR.parent
CASES_ROOT = TDD_ROOT / "cases"
ONNX_SCHEMA_ROOT = TDD_ROOT / "onnx_schema"

sys.path.insert(0, str(_SCRIPT_DIR))
import onnx  # noqa: E402
from cases_registry import CASE_MAP  # noqa: E402
from support_matrix import (  # noqa: E402
    SUPPORT_MAP,
    SUPPORT_MATRIX,
    VALID_BACKENDS,
    VALID_CAPABILITY_TYPES,
    VALID_SCHEMA_SOURCES,
    VALID_SUPPORT_STATUSES,
)

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

VALID_EXPECTED = {"ok", "blocked", "unsupported", "oversize"}
VALID_SOURCES = {"内部探索", "用户反馈", "缺陷复现"}
CASE_FILE_RE = re.compile(r"^[A-Z]+(?:_[A-Z]+)*_[0-9]{3}$")


def main() -> int:
    errors: list[str] = []
    case_files = {
        path.stem: path
        for path in CASES_ROOT.rglob("*.md")
        if path.name != "README.md" and CASE_FILE_RE.match(path.stem)
    }
    registry_ids = set(CASE_MAP)
    generator_ids = _read_generator_ids()
    file_ids = set(case_files)

    _check_set_match("cases 文件", file_ids, "registry", registry_ids, errors)
    _check_set_match("registry", registry_ids, "generator", generator_ids, errors)
    _check_support_matrix(errors)

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

        _check_case_support_refs(case_id, case_def, errors)

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


def _check_support_matrix(errors: list[str]) -> None:
    catalog = _load_onnx_schema_catalog(errors)
    official_ops = set()
    if catalog is not None:
        official_ops = {
            (item["domain"], item["name"])
            for item in catalog.get("operators", [])
        }

    support_ids = [entry.support_id for entry in SUPPORT_MATRIX]
    duplicates = sorted({item for item in support_ids if support_ids.count(item) > 1})
    if duplicates:
        errors.append(f"support matrix 存在重复 support_id: {duplicates}")

    referenced = {
        support_id
        for case_def in CASE_MAP.values()
        for support_id in case_def.schema_refs
    }
    for entry in SUPPORT_MATRIX:
        if entry.status not in VALID_SUPPORT_STATUSES:
            errors.append(f"{entry.support_id}: 非法 status={entry.status!r}")
        if entry.schema_source not in VALID_SCHEMA_SOURCES:
            errors.append(f"{entry.support_id}: 非法 schema_source={entry.schema_source!r}")
        if entry.backend not in VALID_BACKENDS:
            errors.append(f"{entry.support_id}: 非法 backend={entry.backend!r}")
        if entry.capability_type not in VALID_CAPABILITY_TYPES:
            errors.append(
                f"{entry.support_id}: 非法 capability_type={entry.capability_type!r}"
            )
        if entry.backend == "cmsis-nn" and entry.status == "ok" and not entry.cmsis_apis:
            errors.append(f"{entry.support_id}: cmsis-nn ok 行必须声明 cmsis_apis")
        if entry.support_id not in referenced and not entry.planned:
            errors.append(
                f"{entry.support_id}: support matrix 行没有 case 覆盖，也未标记 planned"
            )
        if entry.schema_source == "official":
            key = (entry.domain, entry.op_type)
            if key not in official_ops:
                errors.append(
                    f"{entry.support_id}: official support 行不在 ONNX {onnx.__version__} schema catalog 中: {key}"
                )


def _load_onnx_schema_catalog(errors: list[str]) -> dict | None:
    expected_path = ONNX_SCHEMA_ROOT / f"onnx_{onnx.__version__.replace('.', '_')}_schema_catalog.json"
    if not expected_path.exists():
        errors.append(
            f"缺少 ONNX schema catalog: {expected_path}. "
            "请运行 tdd/scripts/generate_onnx_schema_catalog.py"
        )
        return None
    try:
        catalog = json.loads(expected_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{expected_path}: JSON 解析失败: {exc}")
        return None
    catalog_version = catalog.get("onnx_version")
    if catalog_version != onnx.__version__:
        errors.append(
            f"{expected_path}: onnx_version={catalog_version!r} 与当前 onnx={onnx.__version__!r} 不一致"
        )
    return catalog


def _check_case_support_refs(case_id: str, case_def, errors: list[str]) -> None:
    if not case_def.schema_refs:
        errors.append(f"{case_id}: 缺少 schema_refs，必须绑定 support matrix")
        return

    entries = []
    for support_id in case_def.schema_refs:
        entry = SUPPORT_MAP.get(support_id)
        if entry is None:
            errors.append(f"{case_id}: schema_refs 引用了不存在的 support_id={support_id}")
            continue
        entries.append(entry)

    if not entries:
        return

    if case_def.expected == "ok":
        non_ok = [entry.support_id for entry in entries if entry.status != "ok"]
        if non_ok:
            errors.append(f"{case_id}: ok 用例引用了非 ok support matrix 行: {non_ok}")
    else:
        matching = [entry.support_id for entry in entries if entry.status == case_def.expected]
        if not matching:
            errors.append(
                f"{case_id}: expected={case_def.expected} 但 schema_refs 中没有同状态 support 行"
            )

    cmsis_entries = [
        entry for entry in entries if entry.backend == "cmsis-nn" and entry.status == "ok"
    ]
    if cmsis_entries and not case_def.required_apis:
        errors.append(f"{case_id}: cmsis-nn ok 用例必须声明 required_apis")

    declared_apis = set(case_def.required_apis)
    required_by_matrix = {
        api
        for entry in cmsis_entries
        for api in entry.cmsis_apis
    }
    missing = sorted(required_by_matrix - declared_apis)
    if missing:
        errors.append(
            f"{case_id}: required_apis 缺少 support matrix 要求的 CMSIS-NN API: {missing}"
        )


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

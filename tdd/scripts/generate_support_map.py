"""
Generate a draggable ONNX support mind map for human review.

The map is intentionally generated from source-of-truth TDD files:
  - tdd/onnx_schema/onnx_<version>_schema_catalog.json
  - tdd/scripts/support_matrix.py
  - tdd/scripts/cases_registry.py
  - tdd/results/latest.json

Usage:
    python tdd/scripts/generate_support_map.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
TDD_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT = TDD_ROOT / "reports" / "onnx_support_map.html"

sys.path.insert(0, str(SCRIPT_DIR))
from cases_registry import CASE_MAP  # noqa: E402
from support_matrix import SUPPORT_MATRIX  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate draggable ONNX support mind map.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output HTML path")
    args = parser.parse_args(argv)

    catalog = _load_catalog()
    latest = _load_latest_results()
    payload = _build_payload(catalog, latest)
    html = _render_html(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"ONNX support mind map written: {args.out}")
    return 0


def _load_catalog() -> dict[str, Any]:
    candidates = sorted((TDD_ROOT / "onnx_schema").glob("onnx_*_schema_catalog.json"))
    if not candidates:
        raise FileNotFoundError("missing tdd/onnx_schema/onnx_*_schema_catalog.json")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _load_latest_results() -> dict[str, Any]:
    path = TDD_ROOT / "results" / "latest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_payload(catalog: dict[str, Any], latest: dict[str, Any]) -> dict[str, Any]:
    case_results = {
        str(item.get("case_id")): item
        for item in latest.get("results", [])
        if isinstance(item, dict) and item.get("case_id")
    }
    terminal_latest = _load_terminal_results()
    terminal_results = {
        str(item.get("case_id")): item
        for item in terminal_latest.get("results", [])
        if isinstance(item, dict) and item.get("case_id")
    }
    terminal_configs = _load_terminal_case_configs()
    support_to_cases = _support_to_cases()
    support_by_op: dict[str, list] = defaultdict(list)
    extension_by_op: dict[str, list] = defaultdict(list)
    for entry in SUPPORT_MATRIX:
        if entry.schema_source == "official":
            support_by_op[entry.op_type].append(entry)
        elif entry.schema_source == "extension":
            extension_by_op[entry.op_type].append(entry)

    operators = sorted(
        catalog.get("operators", []),
        key=lambda item: (str(item.get("domain", "")), str(item.get("name", ""))),
    )
    op_nodes = [
        _official_op_node(op, support_by_op, support_to_cases, case_results, terminal_configs, terminal_results)
        for op in operators
    ]
    supported = [node for node in op_nodes if node["support_state"] == "supported"]
    planned = [node for node in op_nodes if node["support_state"] == "planned"]
    unsupported = [node for node in op_nodes if node["support_state"] == "not_started"]

    extension_nodes = [
        _extension_op_node(op_type, entries, support_to_cases, case_results)
        for op_type, entries in sorted(extension_by_op.items())
    ]
    boundary_nodes = [
        _mixed_entry_node(entry, support_to_cases, case_results, terminal_configs, terminal_results)
        for entry in SUPPORT_MATRIX
        if entry.schema_source == "mixed"
    ]

    tree = {
        "id": "root",
        "label": f"ONNX {catalog.get('onnx_version', 'unknown')} support map",
        "kind": "root",
        "state": "root",
        "summary": (
            f"default ai.onnx opset {catalog.get('default_ai_onnx_opset', 'unknown')} · "
            f"{len(operators)} official current operators"
        ),
        "children": [
            {
                "id": "official",
                "label": "官方 ONNX 算子全集",
                "kind": "group",
                "state": "official",
                "summary": f"{len(operators)} ops from ONNX catalog",
                "children": [
                    {
                        "id": "official-supported",
                        "label": "已支持",
                        "kind": "status_group",
                        "state": "supported",
                        "summary": f"{len(supported)} official ops have ok/planned support rows",
                        "children": supported,
                    },
                    {
                        "id": "official-planned",
                        "label": "已登记但未完成",
                        "kind": "status_group",
                        "state": "planned",
                        "summary": f"{len(planned)} official ops are planned/blocked",
                        "children": planned,
                    },
                    {
                        "id": "official-not-started",
                        "label": "尚未开始",
                        "kind": "status_group",
                        "state": "not_started",
                        "summary": f"{len(unsupported)} official ops have no support row yet",
                        "children": unsupported,
                    },
                ],
            },
            {
                "id": "extension",
                "label": "真实模型观察扩展",
                "kind": "group",
                "state": "extension",
                "summary": f"{len(extension_nodes)} observed non-catalog op forms",
                "children": extension_nodes,
            },
            {
                "id": "boundary",
                "label": "项目边界/组合网络",
                "kind": "group",
                "state": "boundary",
                "summary": f"{len(boundary_nodes)} negative, reference, or network boundary rows",
                "children": boundary_nodes,
            },
            {
                "id": "terminal",
                "label": "ARM 终端实机验收",
                "kind": "group",
                "state": "terminal",
                "summary": _terminal_summary(terminal_latest, terminal_configs),
                "children": _terminal_group_nodes(terminal_configs, terminal_results),
            },
        ],
    }

    counts = _count_nodes(tree)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "catalog": {
            "onnx_version": catalog.get("onnx_version", "unknown"),
            "default_ai_onnx_opset": catalog.get("default_ai_onnx_opset", "unknown"),
            "schema_count_current": catalog.get("schema_count_current", len(operators)),
            "schema_count_with_history": catalog.get("schema_count_with_history", "unknown"),
        },
        "latest": {
            "timestamp": latest.get("timestamp", "unknown"),
            "total": latest.get("total", 0),
            "passed": latest.get("passed", 0),
            "failed": latest.get("failed", 0),
        },
        "terminal": {
            "timestamp": terminal_latest.get("timestamp", "unknown"),
            "total": terminal_latest.get("total", 0),
            "passed": terminal_latest.get("passed", 0),
            "failed": terminal_latest.get("failed", 0),
            "skipped": terminal_latest.get("skipped", 0),
        },
        "summary": {
            "official_total": len(operators),
            "official_supported": len(supported),
            "official_planned": len(planned),
            "official_not_started": len(unsupported),
            "extension_total": len(extension_nodes),
            "boundary_total": len(boundary_nodes),
            "visible_node_total": counts,
        },
        "tree": tree,
    }


def _support_to_cases() -> dict[str, list]:
    result: dict[str, list] = defaultdict(list)
    for case in CASE_MAP.values():
        for support_id in case.schema_refs:
            result[support_id].append(case)
    return result


def _official_op_node(
    op: dict[str, Any],
    support_by_op: dict[str, list],
    support_to_cases: dict[str, list],
    case_results: dict[str, dict],
    terminal_configs: dict[str, dict],
    terminal_results: dict[str, dict],
) -> dict[str, Any]:
    name = str(op.get("name", "unknown"))
    entries = sorted(support_by_op.get(name, []), key=lambda item: (item.status != "ok", item.support_id))
    if entries and any(entry.status == "ok" for entry in entries):
        state = "supported"
    elif entries:
        state = "planned"
    else:
        state = "not_started"
    children = [
        _support_entry_node(entry, support_to_cases.get(entry.support_id, []), case_results, terminal_configs, terminal_results)
        for entry in entries
    ]
    if not children:
        children = [
            {
                "id": f"todo-{name}",
                "label": "未登记 support row",
                "kind": "todo",
                "state": "not_started",
                "summary": "需要按真实模型或最小 case 驱动新增 support matrix 行",
                "details": ["当前没有 TDD case 声明该 ONNX 算子的任何可交付形态。"],
                "children": [],
            }
        ]
    return {
        "id": f"op-{name}",
        "label": name,
        "kind": "operator",
        "state": state,
        "support_state": state,
        "summary": _op_summary(op),
        "details": _op_details(op),
        "children": children,
    }


def _extension_op_node(op_type: str, entries: list, support_to_cases: dict[str, list], case_results: dict[str, dict]) -> dict[str, Any]:
    state = "supported" if any(entry.status == "ok" for entry in entries) else "planned"
    return {
        "id": f"extension-{op_type}",
        "label": op_type,
        "kind": "extension_op",
        "state": state,
        "summary": "Observed in real quantized models, not present in bound official ONNX catalog",
        "details": ["这类节点不能算 ONNX 官方 schema 支持率，但需要被真实模型 TDD 跟踪。"],
        "children": [
            _support_entry_node(entry, support_to_cases.get(entry.support_id, []), case_results)
            for entry in entries
        ],
    }


def _mixed_entry_node(
    entry,
    support_to_cases: dict[str, list],
    case_results: dict[str, dict],
    terminal_configs: dict[str, dict],
    terminal_results: dict[str, dict],
) -> dict[str, Any]:
    return _support_entry_node(
        entry,
        support_to_cases.get(entry.support_id, []),
        case_results,
        terminal_configs,
        terminal_results,
    )


def _support_entry_node(
    entry,
    cases: list,
    case_results: dict[str, dict],
    terminal_configs: dict[str, dict] | None = None,
    terminal_results: dict[str, dict] | None = None,
) -> dict[str, Any]:
    state = "planned" if entry.planned else _state_from_status(entry.status)
    terminal_configs = terminal_configs or {}
    terminal_results = terminal_results or {}
    case_children = [
        _case_node(case, case_results.get(case.case_id), terminal_configs.get(case.case_id), terminal_results.get(case.case_id))
        for case in cases
    ]
    if not case_children:
        case_children = [
            {
                "id": f"case-missing-{entry.support_id}",
                "label": "尚无 case 覆盖",
                "kind": "case_missing",
                "state": "planned",
                "summary": "support row 尚未绑定 TDD case",
                "details": [],
                "children": [],
            }
        ]
    return {
        "id": f"support-{entry.support_id}",
        "label": entry.support_id,
        "kind": "support",
        "state": state,
        "summary": entry.schema_form,
        "details": [
            f"domain: {entry.domain}",
            f"opset: {entry.opset_range}",
            f"backend: {entry.backend}",
            f"lowering: {entry.lowering}",
            f"CMSIS API: {', '.join(entry.cmsis_apis) if entry.cmsis_apis else '-'}",
            f"status: {entry.status}",
            f"notes: {entry.notes}" if entry.notes else "",
        ],
        "children": case_children,
    }


def _case_node(case, result: dict | None, terminal_config: dict | None = None, terminal_result: dict | None = None) -> dict[str, Any]:
    actual = str(result.get("actual", "not-run")) if result else "not-run"
    passed = bool(result and result.get("passed"))
    state = "supported" if passed else "planned"
    details = [
        f"expected: {case.expected}",
        f"actual: {actual}",
        f"category: {case.category}",
        f"required API: {', '.join(case.required_apis) if case.required_apis else '-'}",
    ]
    if case.numeric:
        details.append(f"numeric dataset: {case.numeric.dataset_id}")
    children = []
    if terminal_config or terminal_result:
        children.append(_terminal_node(case.case_id, terminal_config, terminal_result))
    return {
        "id": f"case-{case.case_id}",
        "label": case.case_id,
        "kind": "case",
        "state": state,
        "summary": case.description,
        "details": details,
        "children": children,
    }


def _load_terminal_results() -> dict[str, Any]:
    path = TDD_ROOT / "terminal" / "reports" / "terminal_latest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_terminal_case_configs() -> dict[str, dict]:
    cases_dir = TDD_ROOT / "terminal" / "cases"
    if not cases_dir.exists():
        return {}
    configs = {}
    for path in sorted(cases_dir.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("case_id"):
            configs[str(item["case_id"])] = item
    return configs


def _terminal_summary(terminal_latest: dict[str, Any], terminal_configs: dict[str, dict]) -> str:
    if not terminal_configs:
        return "no terminal cases configured"
    if not terminal_latest:
        return f"{len(terminal_configs)} configured, no terminal run yet"
    return (
        f"{terminal_latest.get('passed', 0)}/{terminal_latest.get('total', 0)} PASS · "
        f"{terminal_latest.get('skipped', 0)} skipped"
    )


def _terminal_group_nodes(terminal_configs: dict[str, dict], terminal_results: dict[str, dict]) -> list[dict[str, Any]]:
    nodes = []
    for case_id, config in sorted(terminal_configs.items()):
        nodes.append(_terminal_node(case_id, config, terminal_results.get(case_id)))
    if not nodes:
        nodes.append(
            {
                "id": "terminal-empty",
                "label": "尚无终端 case",
                "kind": "terminal_empty",
                "state": "planned",
                "summary": "可在 tdd/terminal/cases/ 中绑定已通过 Host numeric 的 case",
                "details": [],
                "children": [],
            }
        )
    return nodes


def _terminal_node(case_id: str, config: dict | None, result: dict | None) -> dict[str, Any]:
    if result:
        if result.get("passed"):
            state = "supported"
        elif result.get("skipped"):
            state = "planned"
        else:
            state = "unsupported"
        summary = (
            f"board {result.get('board_id', '-')} · "
            f"host/arm exact {result.get('host_c_arm_c_exact_matches', 0)}/{result.get('sample_count', 0)} · "
            f"onnx/arm top1 {result.get('onnx_arm_c_top1_matches', 0)}/{result.get('sample_count', 0)}"
        )
        details = [
            f"terminal case: {result.get('terminal_case_id', '-')}",
            f"dataset: {result.get('dataset_id', '-')}",
            f"board: {result.get('board_id', '-')}",
            f"passed: {result.get('passed')}",
            f"skipped: {result.get('skipped')}",
            f"elapsed avg us: {result.get('elapsed_us_avg', '-')}",
            f"error: {result.get('error_msg', '')}" if result.get("error_msg") else "",
        ]
    else:
        state = "planned"
        boards = ", ".join(config.get("board_ids", [])) if config else "-"
        summary = f"configured for {boards}, not run yet"
        details = [
            f"terminal case: {config.get('terminal_case_id', '-') if config else '-'}",
            f"dataset: {config.get('dataset_id', '-') if config else '-'}",
            f"arm case id: {config.get('arm_case_id', '-') if config else '-'}",
        ]
    return {
        "id": f"terminal-{case_id}",
        "label": f"{case_id} ARM C",
        "kind": "terminal",
        "state": state,
        "summary": summary,
        "details": details,
        "children": [],
    }


def _op_summary(op: dict[str, Any]) -> str:
    inputs = len(op.get("inputs", []))
    outputs = len(op.get("outputs", []))
    return (
        f"since opset {op.get('latest_since_version', '?')} · "
        f"{inputs} input spec · {outputs} output spec"
    )


def _op_details(op: dict[str, Any]) -> list[str]:
    attrs = op.get("attributes", [])
    type_constraints = op.get("type_constraints", [])
    versions = op.get("versions", [])
    return [
        f"domain: {op.get('domain', '-')}",
        f"versions: {', '.join(str(item) for item in versions)}",
        f"attributes: {', '.join(str(item.get('name')) for item in attrs) if attrs else '-'}",
        f"type constraints: {len(type_constraints)}",
    ]


def _state_from_status(status: str) -> str:
    if status == "ok":
        return "supported"
    if status == "unsupported":
        return "unsupported"
    if status == "blocked":
        return "planned"
    return status or "planned"


def _count_nodes(node: dict[str, Any]) -> int:
    return 1 + sum(_count_nodes(child) for child in node.get("children", []))


def _render_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    title = f"NanoC-NN ONNX {payload['catalog']['onnx_version']} Support Mind Map"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #edf2f6;
      --panel: #ffffff;
      --ink: #162231;
      --muted: #65758a;
      --line: #bac8d8;
      --supported: #138461;
      --planned: #bd7100;
      --unsupported: #aa283b;
      --not-started: #758295;
      --root: #1d4f91;
      --extension: #5f4b99;
      --boundary: #56616f;
      --terminal: #0b7794;
      --shadow: 0 12px 28px rgba(22, 34, 49, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      overflow: hidden;
      color: var(--ink);
      background: var(--bg);
      font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #app {{
      display: grid;
      grid-template-rows: auto 1fr;
      height: 100vh;
    }}
    header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(12px);
      z-index: 2;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .subtitle {{
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 3px 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      white-space: nowrap;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    button, input {{
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    button {{
      padding: 0 10px;
      cursor: pointer;
    }}
    input {{
      width: 260px;
      padding: 0 10px;
    }}
    #stage {{
      position: relative;
      min-height: 0;
    }}
    svg {{
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
      background:
        linear-gradient(90deg, rgba(186, 200, 216, 0.26) 1px, transparent 1px),
        linear-gradient(rgba(186, 200, 216, 0.26) 1px, transparent 1px);
      background-size: 40px 40px;
    }}
    svg.dragging {{ cursor: grabbing; }}
    .link {{
      fill: none;
      stroke: rgba(100, 115, 133, 0.52);
      stroke-width: 1.4;
    }}
    .node rect {{
      fill: #fff;
      stroke: var(--line);
      stroke-width: 1.2;
      rx: 8;
      filter: drop-shadow(0 5px 10px rgba(22, 34, 49, 0.10));
    }}
    .node text {{
      pointer-events: none;
      fill: var(--ink);
    }}
    .node .label {{
      font-weight: 700;
      font-size: 13px;
    }}
    .node .summary {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .node .badge {{
      font-size: 10px;
      font-weight: 700;
      fill: #fff;
    }}
    .state-root rect {{ stroke: var(--root); stroke-width: 2; }}
    .state-supported rect {{ stroke: var(--supported); }}
    .state-planned rect {{ stroke: var(--planned); }}
    .state-unsupported rect {{ stroke: var(--unsupported); }}
    .state-not_started rect {{ stroke: var(--not-started); }}
    .state-extension rect {{ stroke: var(--extension); }}
    .state-boundary rect {{ stroke: var(--boundary); }}
    .state-terminal rect {{ stroke: var(--terminal); }}
    .state-official rect {{ stroke: var(--root); }}
    .node.selected rect {{ stroke-width: 2.8; }}
    .badge-bg.root {{ fill: var(--root); }}
    .badge-bg.supported {{ fill: var(--supported); }}
    .badge-bg.planned {{ fill: var(--planned); }}
    .badge-bg.unsupported {{ fill: var(--unsupported); }}
    .badge-bg.not_started {{ fill: var(--not-started); }}
    .badge-bg.extension {{ fill: var(--extension); }}
    .badge-bg.boundary {{ fill: var(--boundary); }}
    .badge-bg.terminal {{ fill: var(--terminal); }}
    .badge-bg.official {{ fill: var(--root); }}
    #detail {{
      position: absolute;
      right: 16px;
      top: 16px;
      width: min(420px, calc(100vw - 32px));
      max-height: calc(100% - 32px);
      overflow: auto;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px;
    }}
    #detail h2 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    #detail .kind {{
      color: var(--muted);
      margin-bottom: 12px;
    }}
    #detail ul {{
      padding-left: 18px;
      margin: 8px 0 0;
    }}
    #detail code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
    #legend {{
      position: absolute;
      left: 16px;
      bottom: 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      max-width: calc(100% - 32px);
      padding: 8px;
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--not-started);
    }}
    .dot.supported {{ background: var(--supported); }}
    .dot.planned {{ background: var(--planned); }}
    .dot.unsupported {{ background: var(--unsupported); }}
    .dot.not_started {{ background: var(--not-started); }}
    .dot.extension {{ background: var(--extension); }}
    .hint {{
      position: absolute;
      left: 16px;
      top: 16px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.80);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 9px;
      pointer-events: none;
    }}
  </style>
</head>
<body>
<div id="app">
  <header>
    <div>
      <h1>ONNX 支持思维导图</h1>
      <div class="subtitle">
        <span class="pill">ONNX {escape(str(payload["catalog"]["onnx_version"]))}</span>
        <span class="pill">ai.onnx opset {escape(str(payload["catalog"]["default_ai_onnx_opset"]))}</span>
        <span class="pill">官方 current ops {payload["summary"]["official_total"]}</span>
        <span class="pill">已支持 {payload["summary"]["official_supported"]}</span>
        <span class="pill">已登记未完成 {payload["summary"]["official_planned"]}</span>
        <span class="pill">尚未开始 {payload["summary"]["official_not_started"]}</span>
        <span class="pill">target {payload["latest"]["passed"]}/{payload["latest"]["total"]} PASS</span>
        <span class="pill">ARM terminal {payload["terminal"]["passed"]}/{payload["terminal"]["total"]} PASS · skip {payload["terminal"]["skipped"]}</span>
      </div>
    </div>
    <div class="toolbar">
      <input id="search" type="search" placeholder="搜索算子 / case / CMSIS API">
      <button type="button" id="zoom-in">放大</button>
      <button type="button" id="zoom-out">缩小</button>
      <button type="button" id="fit">适配</button>
      <button type="button" id="expand-supported">展开已支持</button>
      <button type="button" id="collapse">收起</button>
    </div>
  </header>
  <section id="stage">
    <svg id="map" aria-label="ONNX support mind map">
      <g id="viewport">
        <g id="links"></g>
        <g id="nodes"></g>
      </g>
    </svg>
    <aside id="detail">
      <h2>点击节点查看详情</h2>
      <div class="kind">拖动画布移动，滚轮缩放，点击节点展开/收起。</div>
    </aside>
    <div class="hint">鼠标拖动画布 · 滚轮缩放 · 点击节点展开</div>
    <div id="legend">
      <span class="legend-item"><span class="dot supported"></span>已支持</span>
      <span class="legend-item"><span class="dot planned"></span>已登记/待完成</span>
      <span class="legend-item"><span class="dot unsupported"></span>明确不支持</span>
      <span class="legend-item"><span class="dot not_started"></span>尚未开始</span>
      <span class="legend-item"><span class="dot extension"></span>观察扩展</span>
      <span class="legend-item"><span class="dot" style="background: var(--terminal);"></span>ARM 终端</span>
    </div>
  </section>
</div>
<script>
const PAYLOAD = {payload_json};
const tree = PAYLOAD.tree;
const expanded = new Set(["root", "official", "official-supported", "official-planned", "extension", "boundary", "terminal"]);
let selectedId = "root";
let transform = {{ x: 70, y: 90, k: 1 }};
const nodeW = 250;
const nodeH = 58;
const levelGap = 340;
const rowGap = 88;
const svg = document.getElementById("map");
const viewport = document.getElementById("viewport");
const linksLayer = document.getElementById("links");
const nodesLayer = document.getElementById("nodes");
const detail = document.getElementById("detail");
let visibleNodes = [];
let visibleLinks = [];
let isPanning = false;
let lastPoint = null;

function stateLabel(state) {{
  return {{
    root: "root",
    official: "official",
    supported: "supported",
    planned: "planned",
    unsupported: "unsupported",
    not_started: "not started",
    extension: "extension",
    boundary: "boundary",
    terminal: "terminal"
  }}[state] || state || "node";
}}

function childrenOf(node) {{
  return node.children || [];
}}

function computeVisible() {{
  visibleNodes = [];
  visibleLinks = [];
  const rows = new Map();
  function walk(node, depth, parent) {{
    const index = rows.get(depth) || 0;
    rows.set(depth, index + 1);
    const item = {{ node, depth, x: depth * levelGap, y: index * rowGap }};
    visibleNodes.push(item);
    if (parent) visibleLinks.push({{ source: parent, target: item }});
    if (expanded.has(node.id)) {{
      for (const child of childrenOf(node)) walk(child, depth + 1, item);
    }}
  }}
  walk(tree, 0, null);
}}

function applyTransform() {{
  viewport.setAttribute("transform", `translate(${{transform.x}},${{transform.y}}) scale(${{transform.k}})`);
}}

function render() {{
  computeVisible();
  linksLayer.innerHTML = "";
  nodesLayer.innerHTML = "";
  for (const link of visibleLinks) {{
    const sx = link.source.x + nodeW;
    const sy = link.source.y + nodeH / 2;
    const tx = link.target.x;
    const ty = link.target.y + nodeH / 2;
    const mx = (sx + tx) / 2;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "link");
    path.setAttribute("d", `M ${{sx}} ${{sy}} C ${{mx}} ${{sy}}, ${{mx}} ${{ty}}, ${{tx}} ${{ty}}`);
    linksLayer.appendChild(path);
  }}
  for (const item of visibleNodes) {{
    nodesLayer.appendChild(renderNode(item));
  }}
  applyTransform();
  showDetail(findNodeById(selectedId) || tree);
}}

function renderNode(item) {{
  const node = item.node;
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const state = node.state || "not_started";
  g.setAttribute("class", `node state-${{state}}${{node.id === selectedId ? " selected" : ""}}`);
  g.setAttribute("transform", `translate(${{item.x}},${{item.y}})`);
  g.dataset.id = node.id;
  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  rect.setAttribute("width", nodeW);
  rect.setAttribute("height", nodeH);
  g.appendChild(rect);

  const badgeBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  badgeBg.setAttribute("class", `badge-bg ${{state}}`);
  badgeBg.setAttribute("x", 10);
  badgeBg.setAttribute("y", 9);
  badgeBg.setAttribute("width", 74);
  badgeBg.setAttribute("height", 18);
  badgeBg.setAttribute("rx", 9);
  g.appendChild(badgeBg);

  const badge = textEl(stateLabel(state), 17, 22, "badge");
  g.appendChild(badge);
  g.appendChild(textEl(truncate(node.label, 24), 94, 22, "label"));
  g.appendChild(textEl(truncate(node.summary || "", 36), 14, 44, "summary"));

  if (childrenOf(node).length) {{
    const marker = textEl(expanded.has(node.id) ? "−" : "+", nodeW - 22, 23, "label");
    marker.setAttribute("text-anchor", "middle");
    g.appendChild(marker);
  }}

  g.addEventListener("click", (event) => {{
    event.stopPropagation();
    selectedId = node.id;
    if (childrenOf(node).length) {{
      if (expanded.has(node.id)) expanded.delete(node.id);
      else expanded.add(node.id);
    }}
    render();
  }});
  return g;
}}

function textEl(value, x, y, cls) {{
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", x);
  text.setAttribute("y", y);
  text.setAttribute("class", cls);
  text.textContent = value;
  return text;
}}

function truncate(value, max) {{
  value = String(value || "");
  return value.length > max ? value.slice(0, max - 1) + "…" : value;
}}

function findNodeById(id, node = tree) {{
  if (node.id === id) return node;
  for (const child of childrenOf(node)) {{
    const found = findNodeById(id, child);
    if (found) return found;
  }}
  return null;
}}

function showDetail(node) {{
  const details = (node.details || []).filter(Boolean).map((item) => `<li><code>${{escapeHtml(item)}}</code></li>`).join("");
  const childCount = childrenOf(node).length;
  detail.innerHTML = `
    <h2>${{escapeHtml(node.label)}}</h2>
    <div class="kind">${{escapeHtml(node.kind || "node")}} · ${{escapeHtml(stateLabel(node.state))}} · children ${{childCount}}</div>
    <p>${{escapeHtml(node.summary || "")}}</p>
    ${{details ? `<ul>${{details}}</ul>` : ""}}
  `;
}}

function escapeHtml(value) {{
  return String(value || "").replace(/[&<>"']/g, (ch) => ({{
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }}[ch]));
}}

function setTransform(next) {{
  transform = next;
  applyTransform();
}}

svg.addEventListener("mousedown", (event) => {{
  isPanning = true;
  lastPoint = {{ x: event.clientX, y: event.clientY }};
  svg.classList.add("dragging");
}});
window.addEventListener("mousemove", (event) => {{
  if (!isPanning || !lastPoint) return;
  transform.x += event.clientX - lastPoint.x;
  transform.y += event.clientY - lastPoint.y;
  lastPoint = {{ x: event.clientX, y: event.clientY }};
  applyTransform();
}});
window.addEventListener("mouseup", () => {{
  isPanning = false;
  lastPoint = null;
  svg.classList.remove("dragging");
}});
svg.addEventListener("wheel", (event) => {{
  event.preventDefault();
  const scale = event.deltaY < 0 ? 1.08 : 0.92;
  const rect = svg.getBoundingClientRect();
  const cx = event.clientX - rect.left;
  const cy = event.clientY - rect.top;
  const nextK = Math.min(2.5, Math.max(0.18, transform.k * scale));
  const ratio = nextK / transform.k;
  transform.x = cx - (cx - transform.x) * ratio;
  transform.y = cy - (cy - transform.y) * ratio;
  transform.k = nextK;
  applyTransform();
}}, {{ passive: false }});

document.getElementById("zoom-in").onclick = () => setTransform({{ ...transform, k: Math.min(2.5, transform.k * 1.18) }});
document.getElementById("zoom-out").onclick = () => setTransform({{ ...transform, k: Math.max(0.18, transform.k / 1.18) }});
document.getElementById("fit").onclick = () => fit();
document.getElementById("collapse").onclick = () => {{
  expanded.clear();
  expanded.add("root");
  selectedId = "root";
  fit();
  render();
}};
document.getElementById("expand-supported").onclick = () => {{
  expanded.clear();
  expandPath(tree, (node) => ["root", "official", "official-supported", "extension", "boundary"].includes(node.id) || node.state === "supported");
  fit();
  render();
}};
document.getElementById("search").addEventListener("input", (event) => {{
  const q = event.target.value.trim().toLowerCase();
  if (!q) {{
    render();
    return;
  }}
  expanded.clear();
  const found = expandMatches(tree, q);
  if (found) selectedId = found.id;
  fit();
  render();
}});

function expandPath(node, predicate) {{
  if (predicate(node)) expanded.add(node.id);
  for (const child of childrenOf(node)) expandPath(child, predicate);
}}

function expandMatches(node, q) {{
  let first = null;
  function walk(current, ancestors) {{
    const haystack = [current.label, current.summary, current.kind, ...(current.details || [])].join(" ").toLowerCase();
    let matched = haystack.includes(q);
    for (const child of childrenOf(current)) {{
      const childMatch = walk(child, [...ancestors, current]);
      matched = matched || Boolean(childMatch);
      if (!first && childMatch && childMatch !== true) first = childMatch;
    }}
    if (matched) {{
      for (const item of ancestors) expanded.add(item.id);
      expanded.add(current.id);
      return first || current;
    }}
    return null;
  }}
  return walk(node, []);
}}

function fit() {{
  computeVisible();
  if (!visibleNodes.length) return;
  const maxX = Math.max(...visibleNodes.map((item) => item.x + nodeW));
  const maxY = Math.max(...visibleNodes.map((item) => item.y + nodeH));
  const rect = svg.getBoundingClientRect();
  const kx = (rect.width - 80) / Math.max(maxX, 1);
  const ky = (rect.height - 80) / Math.max(maxY, 1);
  transform.k = Math.min(1.2, Math.max(0.18, Math.min(kx, ky)));
  transform.x = 40;
  transform.y = 40;
  applyTransform();
}}

render();
fit();
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

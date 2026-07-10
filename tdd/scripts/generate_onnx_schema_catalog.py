"""
Generate the version-bound ONNX operator schema catalog used by TDD.

The generated catalog is the exhaustive official ONNX schema baseline for the
installed `onnx` package version.  NanoC-NN support entries are validated
against this catalog before they can be treated as official ONNX coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
from onnx import defs


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = REPO_ROOT / "tdd" / "onnx_schema"


def main() -> int:
    catalog = build_catalog()
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    path = CATALOG_DIR / f"onnx_{onnx.__version__.replace('.', '_')}_schema_catalog.json"
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(path)
    return 0


def build_catalog() -> dict[str, Any]:
    history = defs.get_all_schemas_with_history()
    latest = defs.get_all_schemas()
    latest_by_key = {
        (_domain(schema), schema.name): schema
        for schema in latest
    }

    version_history: dict[tuple[str, str], list[int]] = {}
    for schema in history:
        key = (_domain(schema), schema.name)
        version_history.setdefault(key, []).append(schema.since_version)

    operators = []
    for (domain, name), schema in sorted(latest_by_key.items()):
        versions = sorted(set(version_history.get((domain, name), [schema.since_version])))
        operators.append(
            {
                "domain": domain,
                "name": name,
                "latest_since_version": schema.since_version,
                "versions": versions,
                "inputs": [_formal_parameter(item) for item in schema.inputs],
                "outputs": [_formal_parameter(item) for item in schema.outputs],
                "attributes": [
                    {
                        "name": attr.name,
                        "type": str(attr.type),
                        "required": bool(attr.required),
                    }
                    for attr in sorted(schema.attributes.values(), key=lambda item: item.name)
                ],
                "type_constraints": [
                    {
                        "type_param": item.type_param_str,
                        "allowed_types": list(item.allowed_type_strs),
                    }
                    for item in schema.type_constraints
                ],
            }
        )

    return {
        "onnx_version": onnx.__version__,
        "default_ai_onnx_opset": defs.onnx_opset_version(),
        "schema_count_current": len(latest),
        "schema_count_with_history": len(history),
        "operators": operators,
    }


def _domain(schema) -> str:
    return schema.domain or "ai.onnx"


def _formal_parameter(param) -> dict[str, Any]:
    return {
        "name": param.name,
        "type_str": param.type_str,
        "option": str(param.option),
        "is_homogeneous": bool(param.is_homogeneous),
        "min_arity": int(param.min_arity),
    }


if __name__ == "__main__":
    raise SystemExit(main())


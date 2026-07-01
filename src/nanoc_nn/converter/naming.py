from __future__ import annotations

import re
from collections.abc import Iterable


def _clean_token(value: str, fallback: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]", "_", value.strip())
    token = re.sub(r"_+", "_", token).strip("_").lower()
    return token or fallback


def sanitize_c_identifier(name: str, prefix: str = "nanoc") -> str:
    """Return a valid, prefixed C identifier for an ONNX name."""
    base = _clean_token(name, "tensor")
    clean_prefix = _clean_token(prefix, "")
    identifier = f"{clean_prefix}_{base}" if clean_prefix else base
    if identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier


def make_unique_c_names(names: Iterable[str], prefix: str = "nanoc") -> dict[str, str]:
    """Build a stable original-name to unique-C-name mapping."""
    used: set[str] = set()
    result: dict[str, str] = {}

    for name in names:
        candidate = sanitize_c_identifier(name, prefix)
        unique = candidate
        suffix = 2
        while unique in used:
            unique = f"{candidate}_{suffix}"
            suffix += 1
        used.add(unique)
        result[name] = unique

    return result


def macro_name(identifier: str) -> str:
    clean = sanitize_c_identifier(identifier, prefix="")
    return clean.upper()


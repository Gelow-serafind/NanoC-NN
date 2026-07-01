from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanoc_nn.pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

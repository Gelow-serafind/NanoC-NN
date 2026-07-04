#!/usr/bin/env python3
"""Local MNIST fixture capture UI.

The server writes hand-drawn samples to:
  tdd/fixtures/datasets/mnist_hand_drawn/dataset.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

TOOL_DIR = Path(__file__).resolve().parent
TDD_ROOT = TOOL_DIR.parents[1]
DATASET_PATH = TDD_ROOT / "fixtures" / "datasets" / "mnist_hand_drawn" / "dataset.json"
INDEX_PATH = TOOL_DIR / "index.html"


class CaptureHandler(BaseHTTPRequestHandler):
    server_version = "NanoCTDDCapture/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_bytes(INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/dataset":
            self._send_json(_read_dataset())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        if parsed.path != "/api/samples":
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            sample = _validate_sample(payload)
            dataset = _read_dataset()
            dataset.setdefault("samples", []).append(sample)
            _write_dataset(dataset)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(
            {
                "ok": True,
                "sample_count": len(dataset.get("samples", [])),
                "sample": {"id": sample["id"], "label": sample.get("label")},
            }
        )

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[mnist-capture] {self.address_string()} {fmt % args}")

    def _send_json(self, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(encoded, "application/json; charset=utf-8", status)

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _read_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _write_dataset(dataset: dict) -> None:
    DATASET_PATH.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_sample(payload: dict) -> dict:
    label = payload.get("label")
    if label is not None:
        try:
            label = int(label)
        except (TypeError, ValueError) as exc:
            raise ValueError("label must be an integer from 0 to 9") from exc
        if label < 0 or label > 9:
            raise ValueError("label must be an integer from 0 to 9")

    pixels = payload.get("float_pixels")
    if not isinstance(pixels, list) or len(pixels) != 784:
        raise ValueError("float_pixels must contain 784 values")
    normalized = []
    for value in pixels:
        try:
            item = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("float_pixels must be numeric") from exc
        normalized.append(max(0.0, min(1.0, item)))

    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sample_id = str(payload.get("id") or f"hand_{now}")
    return {
        "id": sample_id,
        "label": label,
        "source": "mnist_capture_ui",
        "created_at": datetime.now().isoformat(),
        "float_pixels": normalized,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local MNIST capture UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), CaptureHandler)
    print(f"MNIST capture UI: http://{args.host}:{args.port}")
    print(f"Writing dataset: {DATASET_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MNIST capture UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

try:
    import onnx
except Exception:  # noqa: BLE001 - UI reports the import error through /api/health.
    onnx = None


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
WORK_DIR = REPO_ROOT / "ui" / "work"
MODELS_DIR = WORK_DIR / "models"
JOBS_DIR = WORK_DIR / "jobs"
SRC_DIR = REPO_ROOT / "src"

JOBS: dict[str, dict] = {}
MODELS: dict[str, dict] = {}


class UiHandler(BaseHTTPRequestHandler):
    server_version = "NanoC-NN-UI/0.1"

    def do_GET(self) -> None:  # noqa: N802 - http.server API.
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._json({"ok": True, "onnx": onnx is not None, "repo": str(REPO_ROOT)})
            return
        if path == "/api/sample-models":
            self._json({"models": _sample_models()})
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            self._json(JOBS.get(job_id, {"error": "job not found"}), status=HTTPStatus.OK)
            return
        if path == "/api/file":
            self._serve_generated_file(parsed.query)
            return
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802 - http.server API.
        parsed = urlparse(self.path)
        if parsed.path == "/api/models/upload":
            self._upload_model()
            return
        if parsed.path == "/api/models/sample":
            self._use_sample_model(parsed.query)
            return
        if parsed.path == "/api/convert":
            self._convert()
            return
        if parsed.path == "/api/tdd/regression":
            self._run_regression()
            return
        self._json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[ui] {self.address_string()} - {fmt % args}")

    def _upload_model(self) -> None:
        raw_name = self.headers.get("X-Filename") or "model.onnx"
        filename = _safe_filename(unquote(raw_name))
        if not filename.endswith(".onnx"):
            self._json({"error": "only .onnx files are supported"}, status=HTTPStatus.BAD_REQUEST)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(content_length)
        if not data:
            self._json({"error": "empty upload"}, status=HTTPStatus.BAD_REQUEST)
            return
        model_id = _new_id("model")
        model_dir = MODELS_DIR / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        model_path.write_bytes(data)
        self._register_model(model_id, model_path)

    def _use_sample_model(self, query: str) -> None:
        params = parse_qs(query)
        name = params.get("name", ["mnist"])[0]
        samples = {item["id"]: item for item in _sample_models()}
        if name not in samples:
            self._json({"error": f"unknown sample model: {name}"}, status=HTTPStatus.BAD_REQUEST)
            return
        source = Path(samples[name]["path"])
        if not source.exists():
            self._json({"error": f"sample file missing: {source}"}, status=HTTPStatus.NOT_FOUND)
            return
        model_id = _new_id("model")
        model_dir = MODELS_DIR / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / source.name
        shutil.copyfile(source, model_path)
        self._register_model(model_id, model_path)

    def _register_model(self, model_id: str, model_path: Path) -> None:
        try:
            info = inspect_model(model_path)
        except Exception as exc:  # noqa: BLE001 - show parse error in UI.
            self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        info["id"] = model_id
        info["path"] = str(model_path)
        MODELS[model_id] = info
        self._json({"model": info})

    def _convert(self) -> None:
        payload = self._read_json()
        model_id = str(payload.get("modelId", ""))
        model = MODELS.get(model_id)
        if not model:
            self._json({"error": "select or upload a model first"}, status=HTTPStatus.BAD_REQUEST)
            return

        job_id = _new_id("job")
        job_dir = JOBS_DIR / job_id
        out_root = Path(str(payload.get("outRoot") or job_dir / "output"))
        if not out_root.is_absolute():
            out_root = (REPO_ROOT / out_root).resolve()
        out_root.parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "nanoc_nn.cli",
            "onnx-to-cmsis",
            "--model",
            str(model["path"]),
            "--target",
            str(payload.get("target") or "cortex-m4"),
            "--out-root",
            str(out_root),
            "--no-compile",
        ]
        backend = str(payload.get("backend") or "auto")
        if backend and backend != "auto":
            command.extend(["--backend", backend])
        prefix = str(payload.get("prefix") or "").strip()
        if prefix:
            command.extend(["--prefix", prefix])
        sram_budget = str(payload.get("sramBudget") or "").strip()
        flash_budget = str(payload.get("flashBudget") or "").strip()
        if sram_budget:
            command.extend(["--sram-budget", sram_budget])
        if flash_budget:
            command.extend(["--flash-budget", flash_budget])

        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=_subprocess_env(),
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )
        duration = round(time.monotonic() - started, 2)
        status = _detect_codegen_status(out_root, completed.stdout, completed.stderr)
        job = {
            "id": job_id,
            "modelId": model_id,
            "status": status,
            "exitCode": completed.returncode,
            "durationSec": duration,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "outRoot": str(out_root),
            "reports": _collect_reports(out_root),
            "files": _collect_files(out_root),
        }
        JOBS[job_id] = job
        self._json({"job": job})

    def _run_regression(self) -> None:
        command = [sys.executable, str(REPO_ROOT / "tdd" / "scripts" / "run_regression.py"), "--generate"]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=_subprocess_env(),
            text=True,
            capture_output=True,
            check=False,
            timeout=240,
        )
        self._json(
            {
                "exitCode": completed.returncode,
                "durationSec": round(time.monotonic() - started, 2),
                "command": command,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "report": str(REPO_ROOT / "tdd" / "results" / "regression_latest.json"),
            }
        )

    def _serve_generated_file(self, query: str) -> None:
        params = parse_qs(query)
        job_id = params.get("jobId", [""])[0]
        rel = params.get("path", [""])[0]
        job = JOBS.get(job_id)
        if not job:
            self._json({"error": "job not found"}, status=HTTPStatus.NOT_FOUND)
            return
        root = Path(job["outRoot"]).resolve()
        target = (root / rel).resolve()
        if root not in target.parents and target != root:
            self._json({"error": "invalid path"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not target.exists() or not target.is_file():
            self._json({"error": "file not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._json({"path": rel, "content": target.read_text(encoding="utf-8", errors="replace")})

    def _serve_static(self, path: str) -> None:
        rel = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        if not target.exists() or target.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = _content_type(target)
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def inspect_model(path: Path) -> dict:
    if onnx is None:
        raise RuntimeError("Python package 'onnx' is not available in this environment")
    model = onnx.load(str(path))
    ops = Counter(node.op_type for node in model.graph.node)
    return {
        "name": path.name,
        "sizeBytes": path.stat().st_size,
        "irVersion": model.ir_version,
        "producer": model.producer_name or "-",
        "opsets": [{"domain": item.domain or "onnx", "version": item.version} for item in model.opset_import],
        "inputs": [_tensor_info(item) for item in model.graph.input],
        "outputs": [_tensor_info(item) for item in model.graph.output],
        "nodeCount": len(model.graph.node),
        "opCounts": [{"op": op, "count": count} for op, count in sorted(ops.items())],
        "quantizedHint": any(op in ops for op in ("QLinearConv", "QuantizeLinear", "DequantizeLinear")),
        "nodes": [
            {
                "name": node.name or f"node_{index}",
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
            for index, node in enumerate(model.graph.node[:80])
        ],
    }


def _tensor_info(value_info) -> dict:
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.dim_value else dim.dim_param or "?")
    return {"name": value_info.name, "elemType": tensor_type.elem_type, "shape": dims}


def _sample_models() -> list[dict]:
    return [
        {
            "id": "mnist",
            "name": "MNIST int8",
            "path": str(REPO_ROOT / "tdd" / "fixtures" / "onnx" / "mnist-12-int8.onnx"),
        },
        {
            "id": "squeezenet",
            "name": "SqueezeNet 1.0 int8",
            "path": str(REPO_ROOT / "tdd" / "fixtures" / "onnx" / "squeezenet1.0-12-int8.onnx"),
        },
    ]


def _collect_reports(out_root: Path) -> list[dict]:
    candidates = [
        out_root / "pipeline_report.md",
        out_root / "converter-output" / "model_summary.md",
        out_root / "converter-output" / "conversion_report.txt",
        out_root / "cmsis-codegen" / "reports" / "codegen_report.txt",
        out_root / "cmsis-codegen" / "reports" / "op_mapping.md",
        out_root / "cmsis-codegen" / "reports" / "quantization.md",
        out_root / "cmsis-codegen" / "reports" / "memory_plan.md",
        out_root / "cmsis-codegen" / "reports" / "target_report.md",
        out_root / "cmsis-codegen" / "reports" / "unsupported_ops.md",
    ]
    return [
        {"name": path.name, "path": path.relative_to(out_root).as_posix()}
        for path in candidates
        if path.exists()
    ]


def _collect_files(out_root: Path) -> list[dict]:
    roots = [out_root / "cmsis-codegen" / "include", out_root / "cmsis-codegen" / "src"]
    files: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "name": path.name,
                        "path": path.relative_to(out_root).as_posix(),
                        "sizeBytes": path.stat().st_size,
                    }
                )
    return files


def _detect_codegen_status(out_root: Path, stdout: str, stderr: str) -> str:
    report = out_root / "cmsis-codegen" / "reports" / "codegen_report.txt"
    if report.exists():
        for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("status:"):
                return line.split(":", 1)[1].strip()
    for text in (stdout, stderr):
        match = re.search(r"codegen status:\s*([A-Za-z0-9_-]+)", text)
        if match:
            return match.group(1)
    return "unknown"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)
    return cleaned or "model.onnx"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}"


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
    }.get(suffix, "application/octet-stream")


def main() -> int:
    parser = argparse.ArgumentParser(description="NanoC-NN local UI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), UiHandler)
    print(f"NanoC-NN UI running at http://{args.host}:{args.port}/")
    print(f"Runtime work dir: {WORK_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

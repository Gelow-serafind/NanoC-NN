from __future__ import annotations

import re
import sys
import traceback
from collections import Counter
from pathlib import Path

try:
    import onnx
except Exception:  # noqa: BLE001 - displayed in the UI.
    onnx = None

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parents[1] / "Resources"
    return Path(__file__).resolve().parents[2]


REPO_ROOT = app_root()
WORK_ROOT = Path.home() / "Library" / "Application Support" / "NanoC-NN" / "work"


class PipelineWorker(QThread):
    finished = Signal(dict)

    def __init__(self, command: list[str], options: dict[str, object]) -> None:
        super().__init__()
        self.command = command
        self.options = options

    def run(self) -> None:
        out_root = Path(str(self.options["output_root"]))
        stdout_lines: list[str] = []
        stderr = ""
        exit_code = 1
        status = "error"
        try:
            from nanoc_nn.pipeline.runner import PipelineOptions, run_pipeline

            result = run_pipeline(
                PipelineOptions(
                    model_path=Path(str(self.options["model_path"])),
                    output_root=out_root,
                    layout=str(self.options["layout"]),
                    prefix=str(self.options["prefix"]),
                    batch_size=int(self.options["batch_size"]),
                    cmsis_nn_root=self.options["cmsis_nn_root"],
                    target=str(self.options["target"]),
                    backend=self.options["backend"],
                    sram_budget=self.options["sram_budget"],
                    flash_budget=self.options["flash_budget"],
                    compile_smoke=False,
                )
            )
            status = result.codegen_result.status
            stdout_lines = [
                f"pipeline output: {result.options.output_root}",
                f"converter output: {result.options.converter_dir}",
                f"cmsis-codegen output: {result.options.codegen_dir}",
                f"codegen status: {result.codegen_result.status}",
                f"c99 smoke compile: {result.compile_status}",
                f"pipeline report: {result.options.output_root / 'pipeline_report.md'}",
            ]
            exit_code = 0 if status == "ok" else 2
        except Exception:  # noqa: BLE001 - UI boundary.
            stderr = traceback.format_exc()
        self.finished.emit(
            {
                "exit_code": exit_code,
                "stdout": "\n".join(stdout_lines),
                "stderr": stderr,
                "status": detect_codegen_status(out_root, "\n".join(stdout_lines), stderr)
                if status == "error"
                else status,
                "out_root": str(out_root),
                "command": self.command,
            }
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NanoC-NN Desktop")
        self.resize(1320, 840)
        self.model_path: Path | None = None
        self.current_out_root: Path | None = None
        self.worker: PipelineWorker | None = None
        self.log_collapsed = False

        self._build_actions()
        self._build_ui()
        self._apply_style()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _build_actions(self) -> None:
        open_action = QAction("Open ONNX...", self)
        open_action.triggered.connect(self.choose_model)
        self.menuBar().addMenu("File").addAction(open_action)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._left_panel())
        splitter.addWidget(self._center_panel())
        splitter.addWidget(self._right_panel())
        splitter.setSizes([260, 680, 420])
        root_layout.addWidget(splitter, 1)

        log_header = QHBoxLayout()
        log_title = QLabel("Pipeline Log")
        log_title.setObjectName("sectionTitle")
        self.log_toggle_button = QPushButton("Hide Log")
        self.log_toggle_button.clicked.connect(self.toggle_log)
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        log_header.addWidget(self.log_toggle_button)
        root_layout.addLayout(log_header)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(110)
        self.log_view.setPlaceholderText("Pipeline logs will appear here.")
        root_layout.addWidget(self.log_view)

        self.setCentralWidget(root)

    def _left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)

        title = QLabel("Project")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderHidden(True)
        self.project_tree.itemClicked.connect(self.open_tree_item)
        layout.addWidget(self.project_tree, 1)

        fixture_box = QGroupBox("Fixtures")
        fixture_layout = QVBoxLayout(fixture_box)
        mnist_button = QPushButton("Load MNIST int8")
        mnist_button.clicked.connect(lambda: self.load_fixture("mnist-12-int8.onnx"))
        squeeze_button = QPushButton("Load SqueezeNet int8")
        squeeze_button.clicked.connect(lambda: self.load_fixture("squeezenet1.0-12-int8.onnx"))
        fixture_layout.addWidget(mnist_button)
        fixture_layout.addWidget(squeeze_button)
        layout.addWidget(fixture_box)

        return panel

    def _center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)

        header = QHBoxLayout()
        self.model_label = QLabel("No ONNX selected")
        self.model_label.setObjectName("modelTitle")
        choose_button = QPushButton("Select ONNX")
        choose_button.clicked.connect(self.choose_model)
        header.addWidget(self.model_label, 1)
        header.addWidget(choose_button)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._overview_tab(), "Model")
        self.tabs.addTab(self._operators_tab(), "Operators")
        self.tabs.addTab(self._reports_tab(), "Reports / Code")
        layout.addWidget(self.tabs, 1)
        return panel

    def _overview_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.summary_table = QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.summary_table)

        return tab

    def _operators_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.ops_table = QTableWidget(0, 2)
        self.ops_table.setHorizontalHeaderLabels(["ONNX Op", "Count"])
        self.ops_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.ops_table)

        return tab

    def _reports_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.file_preview_title = QLabel("No file selected")
        self.file_preview_title.setObjectName("sectionTitle")
        self.file_preview = QPlainTextEdit()
        self.file_preview.setReadOnly(True)
        self.file_preview.setFont(QFont("Menlo", 11))
        layout.addWidget(self.file_preview_title)
        layout.addWidget(self.file_preview, 1)

        return tab

    def _right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(380)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Target & Export")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        config_box = QGroupBox("Target")
        form = QFormLayout(config_box)
        self.target_combo = QComboBox()
        self.target_combo.addItems(["cortex-m4", "cortex-m3", "cortex-m7", "cortex-m33", "cortex-m55"])
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["auto", "scalar", "dsp", "mve"])
        self.sram_edit = QLineEdit()
        self.sram_edit.setPlaceholderText("optional, e.g. 128K")
        self.flash_edit = QLineEdit()
        self.flash_edit.setPlaceholderText("optional, e.g. 512K")
        self.prefix_edit = QLineEdit("nanoc_model")
        self.out_root_edit = QLineEdit(str(WORK_ROOT / "last-output"))
        self.out_root_edit.setToolTip(self.out_root_edit.text())
        self.out_root_edit.setMinimumWidth(220)
        browse_output_button = QPushButton("Browse")
        browse_output_button.clicked.connect(self.choose_output_dir)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.out_root_edit, 1)
        output_layout.addWidget(browse_output_button)
        form.addRow("Core", self.target_combo)
        form.addRow("Backend", self.backend_combo)
        form.addRow("SRAM", self.sram_edit)
        form.addRow("Flash", self.flash_edit)
        form.addRow("Prefix", self.prefix_edit)
        form.addRow("Output", output_row)
        layout.addWidget(config_box)

        self.status_frame = QFrame()
        self.status_frame.setObjectName("statusFrame")
        status_layout = QVBoxLayout(self.status_frame)
        self.status_label = QLabel("idle")
        self.status_label.setObjectName("statusLabel")
        self.status_hint = QLabel("Select a model to start.")
        self.status_hint.setWordWrap(True)
        status_layout.addWidget(QLabel("Codegen Status"))
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_hint)
        layout.addWidget(self.status_frame)

        self.run_button = QPushButton("Generate C Project")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_pipeline)
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)

        refresh_button = QPushButton("Refresh Output Tree")
        refresh_button.clicked.connect(self.refresh_output_tree)
        layout.addWidget(refresh_button)
        layout.addStretch(1)

        return panel

    def choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ONNX model",
            str(REPO_ROOT),
            "ONNX Models (*.onnx)",
        )
        if path:
            self.load_model(Path(path))

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            self.out_root_edit.text() or str(WORK_ROOT),
        )
        if path:
            self.out_root_edit.setText(path)
            self.out_root_edit.setToolTip(path)

    def load_fixture(self, name: str) -> None:
        path = REPO_ROOT / "tdd" / "fixtures" / "onnx" / name
        if not path.exists():
            QMessageBox.warning(self, "Missing fixture", f"Fixture not found:\n{path}")
            return
        self.load_model(path)

    def load_model(self, path: Path) -> None:
        self.model_path = path
        self.model_label.setText(path.name)
        self.out_root_edit.setText(str(WORK_ROOT / f"{path.stem}-output"))
        self.out_root_edit.setToolTip(self.out_root_edit.text())
        self.append_log(f"Loaded model: {path}")
        if onnx is None:
            QMessageBox.warning(self, "Missing dependency", "Python package 'onnx' is not installed.")
            return
        try:
            info = inspect_model(path)
        except Exception as exc:  # noqa: BLE001 - UI boundary.
            QMessageBox.critical(self, "ONNX parse failed", str(exc))
            return
        self.populate_model_info(info)
        self.populate_tree(model_path=path, out_root=None)
        self.run_button.setEnabled(True)
        self.set_codegen_status("ready", "Model loaded. Ready to generate C project.")
        self.statusBar().showMessage("Model loaded")

    def populate_model_info(self, info: dict[str, object]) -> None:
        rows = [
            ("File", str(info["name"])),
            ("Size", format_bytes(int(info["size_bytes"]))),
            ("IR version", str(info["ir_version"])),
            ("Producer", str(info["producer"])),
            ("Opsets", str(info["opsets"])),
            ("Inputs", str(info["inputs"])),
            ("Outputs", str(info["outputs"])),
            ("Nodes", str(info["node_count"])),
            ("Quantized", "yes" if info["quantized_hint"] else "unknown"),
        ]
        fill_table(self.summary_table, rows)
        fill_table(self.ops_table, [(op, str(count)) for op, count in info["op_counts"]])

    def run_pipeline(self) -> None:
        if self.model_path is None:
            QMessageBox.information(self, "No model", "Select an ONNX model first.")
            return

        out_root = Path(self.out_root_edit.text()).expanduser()
        if not out_root.is_absolute():
            out_root = (REPO_ROOT / out_root).resolve()
        out_root.parent.mkdir(parents=True, exist_ok=True)
        self.current_out_root = out_root

        command = [
            "nanoc_nn",
            "onnx-to-cmsis",
            "--model",
            str(self.model_path),
            "--target",
            self.target_combo.currentText(),
            "--out-root",
            str(out_root),
            "--no-compile",
        ]
        backend = self.backend_combo.currentText()
        if backend != "auto":
            command.extend(["--backend", backend])
        prefix = self.prefix_edit.text().strip()
        if prefix:
            command.extend(["--prefix", prefix])
        if self.sram_edit.text().strip():
            command.extend(["--sram-budget", self.sram_edit.text().strip()])
        if self.flash_edit.text().strip():
            command.extend(["--flash-budget", self.flash_edit.text().strip()])

        cmsis_nn_root = REPO_ROOT / "third_party" / "CMSIS-NN"
        options = {
            "model_path": self.model_path,
            "output_root": out_root,
            "layout": "NCHW",
            "prefix": prefix or safe_prefix(self.model_path.stem),
            "batch_size": 1,
            "cmsis_nn_root": cmsis_nn_root if cmsis_nn_root.exists() else None,
            "target": self.target_combo.currentText(),
            "backend": backend if backend != "auto" else None,
            "sram_budget": self.sram_edit.text().strip() or None,
            "flash_budget": self.flash_edit.text().strip() or None,
        }

        self.run_button.setEnabled(False)
        self.set_codegen_status("running", "Pipeline is running.")
        self.append_log("$ " + " ".join(command))
        self.worker = PipelineWorker(command, options)
        self.worker.finished.connect(self.on_pipeline_finished)
        self.worker.start()

    def on_pipeline_finished(self, result: dict[str, object]) -> None:
        status = str(result["status"])
        self.set_codegen_status(status, status_hint(status))
        self.append_log(f"exit={result['exit_code']} status={status}")
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        if stdout:
            self.append_log(stdout)
        if stderr:
            self.append_log("STDERR:\n" + stderr)
        self.current_out_root = Path(str(result["out_root"]))
        self.populate_tree(model_path=self.model_path, out_root=self.current_out_root)
        self.statusBar().showMessage(f"Pipeline finished: {status}")
        self.run_button.setEnabled(self.model_path is not None)

    def refresh_output_tree(self) -> None:
        self.populate_tree(model_path=self.model_path, out_root=self.current_out_root)

    def populate_tree(self, *, model_path: Path | None, out_root: Path | None) -> None:
        self.project_tree.clear()
        if model_path:
            model_item = QTreeWidgetItem(["Model"])
            model_item.addChild(QTreeWidgetItem([model_path.name]))
            self.project_tree.addTopLevelItem(model_item)
            model_item.setExpanded(True)
        if out_root and out_root.exists():
            reports = QTreeWidgetItem(["Reports"])
            files = QTreeWidgetItem(["Generated Files"])
            for path in collect_report_files(out_root):
                item = QTreeWidgetItem([path.name])
                item.setData(0, Qt.UserRole, str(path))
                reports.addChild(item)
            for path in collect_generated_files(out_root):
                item = QTreeWidgetItem([path.relative_to(out_root).as_posix()])
                item.setData(0, Qt.UserRole, str(path))
                files.addChild(item)
            self.project_tree.addTopLevelItem(reports)
            self.project_tree.addTopLevelItem(files)
            reports.setExpanded(True)
            files.setExpanded(True)

    def open_tree_item(self, item: QTreeWidgetItem) -> None:
        path_text = item.data(0, Qt.UserRole)
        if not path_text:
            return
        path = Path(str(path_text))
        if path.exists() and path.is_file():
            self.file_preview_title.setText(str(path))
            self.file_preview.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
            self.tabs.setCurrentIndex(2)

    def set_codegen_status(self, status: str, hint: str) -> None:
        self.status_label.setText(status)
        self.status_hint.setText(hint)
        self.status_frame.setProperty("state", status)
        self.status_frame.style().unpolish(self.status_frame)
        self.status_frame.style().polish(self.status_frame)

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text.rstrip())

    def toggle_log(self) -> None:
        self.log_collapsed = not self.log_collapsed
        self.log_view.setVisible(not self.log_collapsed)
        self.log_toggle_button.setText("Show Log" if self.log_collapsed else "Hide Log")

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #eef2f6; }
            QWidget { font-size: 13px; color: #17202a; }
            QGroupBox {
                border: 1px solid #d7e0ea;
                border-radius: 6px;
                margin-top: 12px;
                padding: 12px;
                background: #ffffff;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #667487; }
            QTreeWidget, QTableWidget, QPlainTextEdit, QLineEdit, QComboBox {
                border: 1px solid #d7e0ea;
                border-radius: 6px;
                background: #ffffff;
            }
            QPlainTextEdit { font-family: Menlo, Consolas, monospace; }
            QPushButton {
                min-height: 30px;
                border: 1px solid #c7d2df;
                border-radius: 6px;
                padding: 4px 10px;
                background: #f8fafc;
            }
            QPushButton:disabled {
                color: #8a97a8;
                background: #eef2f6;
                border-color: #d7e0ea;
            }
            QPushButton#primaryButton {
                background: #2563eb;
                color: #ffffff;
                border-color: #2563eb;
                font-weight: 700;
            }
            QPushButton#primaryButton:disabled {
                background: #dbe3ec;
                color: #7d8998;
                border-color: #c8d2de;
            }
            QLabel#sectionTitle, QLabel#modelTitle { font-size: 17px; font-weight: 700; }
            QLabel#statusLabel { font-size: 26px; font-weight: 800; }
            QFrame#statusFrame {
                border: 1px solid #d7e0ea;
                border-radius: 6px;
                background: #f8fafc;
            }
            QFrame#statusFrame[state="ok"] { background: #e7f8ee; border-color: #9ed4b6; }
            QFrame#statusFrame[state="ready"] { background: #eef6ff; border-color: #9cc4ee; }
            QFrame#statusFrame[state="blocked"] { background: #fff6df; border-color: #e2c577; }
            QFrame#statusFrame[state="oversize"] { background: #fff1ef; border-color: #dfaaa6; }
            QFrame#statusFrame[state="unsupported"] { background: #fff1ef; border-color: #dfaaa6; }
            """
        )


def inspect_model(path: Path) -> dict[str, object]:
    if onnx is None:
        raise RuntimeError("onnx is not installed")
    model = onnx.load(str(path))
    op_counts = Counter(node.op_type for node in model.graph.node)
    opsets = ", ".join(
        f"{item.domain or 'onnx'}:{item.version}" for item in model.opset_import
    )
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "ir_version": model.ir_version,
        "producer": model.producer_name or "-",
        "opsets": opsets,
        "inputs": "; ".join(tensor_info(item) for item in model.graph.input),
        "outputs": "; ".join(tensor_info(item) for item in model.graph.output),
        "node_count": len(model.graph.node),
        "op_counts": sorted(op_counts.items()),
        "quantized_hint": any(
            op in op_counts for op in ("QLinearConv", "QuantizeLinear", "DequantizeLinear")
        ),
    }


def tensor_info(value_info) -> str:
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        dims.append(str(dim.dim_value if dim.dim_value else dim.dim_param or "?"))
    return f"{value_info.name} [{', '.join(dims)}]"


def fill_table(table: QTableWidget, rows: list[tuple[str, str]]) -> None:
    table.setRowCount(len(rows))
    for row_index, (left, right) in enumerate(rows):
        table.setItem(row_index, 0, QTableWidgetItem(left))
        table.setItem(row_index, 1, QTableWidgetItem(right))
    table.resizeColumnsToContents()


def detect_codegen_status(out_root: Path, stdout: str, stderr: str) -> str:
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


def collect_report_files(out_root: Path) -> list[Path]:
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
    return [path for path in candidates if path.exists()]


def collect_generated_files(out_root: Path) -> list[Path]:
    roots = [out_root / "cmsis-codegen" / "include", out_root / "cmsis-codegen" / "src"]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(path for path in sorted(root.rglob("*")) if path.is_file())
    return files


def format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value / 1024 / 1024:.1f} MB"


def status_hint(status: str) -> str:
    return {
        "ok": "Generated project is deliverable for the current supported path.",
        "blocked": "Operator, quantization, layout, or renderer support is incomplete.",
        "unsupported": "This ONNX shape is explicitly unsupported.",
        "oversize": "Generation semantics passed, but the target budget is too small.",
        "running": "Converter and codegen are running.",
        "ready": "Model loaded. Ready to generate C project.",
        "error": "Pipeline failed before reports were completed; check logs.",
    }.get(status, "Check reports and logs.")


def safe_prefix(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    prefix = "".join(chars).strip("_")
    if not prefix:
        prefix = "nanoc"
    if prefix[0].isdigit():
        prefix = f"m_{prefix}"
    return prefix


def main() -> int:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

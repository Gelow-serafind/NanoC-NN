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

from PySide6.QtCore import QEvent, Qt, QThread, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
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


NODE_WIDTH = 190
NODE_HEADER_HEIGHT = 32
NODE_LINE_HEIGHT = 18
NODE_X_GAP = 240
NODE_Y_GAP = 118

TARGET_MEMORY_PRESETS = {
    "cortex-m3": ("64K", "256K"),
    "cortex-m4": ("128K", "512K"),
    "cortex-m7": ("512K", "2M"),
    "cortex-m33": ("256K", "1M"),
    "cortex-m55": ("512K", "2M"),
}


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


class GraphNodeItem(QGraphicsRectItem):
    def __init__(self, node: dict[str, object], x: float, y: float) -> None:
        self.node = node
        self.params = list(node.get("params", []))
        height = self.node_height()
        super().__init__(0, 0, NODE_WIDTH, height)
        self.default_pen = QPen(QColor("#2f3337"), 1.2)
        self.hover_pen = QPen(QColor("#2563eb"), 1.8)
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(self.default_pen)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setToolTip(str(node["tooltip"]))

        header = QGraphicsRectItem(0, 0, NODE_WIDTH, NODE_HEADER_HEIGHT, self)
        header.setBrush(QBrush(QColor(str(node["header_color"]))))
        header.setPen(QPen(QColor(str(node["header_color"])), 1.0))
        header.setAcceptedMouseButtons(Qt.NoButton)

        title = QGraphicsSimpleTextItem(str(node["title"]), self)
        title.setBrush(QBrush(QColor(str(node["title_color"]))))
        title.setFont(QFont("Arial", 11))
        title.setPos(9, 6)
        title.setAcceptedMouseButtons(Qt.NoButton)

        if self.params:
            for row, param in enumerate(self.params[:8]):
                label = QGraphicsSimpleTextItem(str(param), self)
                label.setBrush(QBrush(QColor("#111827")))
                label.setFont(QFont("Arial", 9))
                label.setPos(9, NODE_HEADER_HEIGHT + 8 + row * NODE_LINE_HEIGHT)
                label.setAcceptedMouseButtons(Qt.NoButton)

    def node_height(self) -> int:
        return graph_node_height(self.node)

    def hoverEnterEvent(self, event) -> None:  # noqa: ANN001 - Qt event type.
        self.setPen(self.hover_pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: ANN001 - Qt event type.
        if self.isSelected():
            self.setPen(QPen(QColor("#2563eb"), 2.0))
        else:
            self.setPen(self.default_pen)
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):  # noqa: ANN001 - Qt item hook.
        if change == QGraphicsRectItem.ItemSelectedHasChanged:
            if bool(value):
                self.setPen(QPen(QColor("#2563eb"), 2.0))
            else:
                self.setPen(self.default_pen)
        return super().itemChange(change, value)


class NetworkGraphView(QGraphicsView):
    selection_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.graph_scene = QGraphicsScene(self)
        self.setScene(self.graph_scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self._scale = 1.0
        self.node_items: dict[str, GraphNodeItem] = {}
        self.graph_scene.selectionChanged.connect(self._emit_selection)

    def draw_graph(self, graph: dict[str, object]) -> None:
        self.graph_scene.clear()
        self.node_items = {}
        nodes = list(graph.get("nodes", []))
        edges = list(graph.get("edges", []))
        if not nodes:
            self.graph_scene.addText("No graph loaded.")
            return

        ordered_nodes = order_graph_nodes(nodes, edges)
        depth_counts = Counter(int(node["depth"]) for node in ordered_nodes)
        depth_heights: dict[int, int] = {}
        for node in ordered_nodes:
            depth = int(node["depth"])
            depth_heights[depth] = max(depth_heights.get(depth, 0), graph_node_height(node))
        y_by_depth: dict[int, float] = {}
        cursor_y = 30.0
        for depth in sorted(depth_heights):
            y_by_depth[depth] = cursor_y
            cursor_y += depth_heights[depth] + 74
        lanes: dict[int, int] = {}
        for node in ordered_nodes:
            depth = int(node["depth"])
            lane = lanes.get(depth, 0)
            lanes[depth] = lane + 1
            count = depth_counts[depth]
            x = 360 + (lane - (count - 1) / 2) * NODE_X_GAP
            y = y_by_depth[depth]
            item = GraphNodeItem(node, x, y)
            self.graph_scene.addItem(item)
            self.node_items[str(node["id"])] = item

        edge_pen = QPen(QColor("#20252b"), 1.1)
        for edge in edges:
            source = self.node_items.get(str(edge["source"]))
            target = self.node_items.get(str(edge["target"]))
            if source is None or target is None:
                continue
            source_rect = source.sceneBoundingRect()
            target_rect = target.sceneBoundingRect()
            start = source_rect.center()
            start.setY(source_rect.bottom())
            end = target_rect.center()
            end.setY(target_rect.top())
            dy = max(34.0, abs(end.y() - start.y()) * 0.42)
            path = QPainterPath(start)
            path.cubicTo(start.x(), start.y() + dy, end.x(), end.y() - dy, end.x(), end.y())
            item = QGraphicsPathItem(path)
            item.setPen(edge_pen)
            item.setZValue(-1)
            item.setToolTip(str(edge["tensor"]))
            self.graph_scene.addItem(item)
            label = str(edge.get("label") or "")
            if label:
                text = QGraphicsSimpleTextItem(label)
                text.setBrush(QBrush(QColor("#111827")))
                text.setFont(QFont("Arial", 9))
                center_x = (start.x() + end.x()) / 2
                center_y = (start.y() + end.y()) / 2
                text.setPos(center_x + 8, center_y - 10)
                text.setToolTip(str(edge["tensor"]))
                self.graph_scene.addItem(text)

        self.graph_scene.setSceneRect(self.graph_scene.itemsBoundingRect().adjusted(-60, -60, 80, 80))
        self.reset_zoom()
        first_item = self.node_items.get(str(nodes[0]["id"]))
        if first_item is not None:
            self.centerOn(first_item)

    def fit_to_view(self) -> None:
        rect = self.graph_scene.itemsBoundingRect()
        if rect.isValid() and not rect.isNull():
            self.fitInView(rect.adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)
            self._scale = self.transform().m11()

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._scale = 1.0

    def viewportEvent(self, event) -> bool:  # noqa: ANN001 - Qt event type.
        if event.type() == QEvent.Type.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture:
            self.apply_zoom(1.0 + event.value())
            event.accept()
            return True
        return super().viewportEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: ANN001 - Qt event type.
        if not event.modifiers() & Qt.ControlModifier:
            super().wheelEvent(event)
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.apply_zoom(factor)
        event.accept()

    def apply_zoom(self, factor: float) -> None:
        next_scale = self._scale * factor
        if 0.15 <= next_scale <= 4.0:
            self.scale(factor, factor)
            self._scale = next_scale

    def _emit_selection(self) -> None:
        for item in self.graph_scene.selectedItems():
            if isinstance(item, GraphNodeItem):
                self.selection_changed.emit(item.node)
                return
        self.selection_changed.emit({})


class WideComboBox(QComboBox):
    def showPopup(self) -> None:
        width = max(self.width(), self.minimumSizeHint().width(), self.sizeHint().width())
        self.view().setMinimumWidth(width + 36)
        super().showPopup()


class ClickablePathLabel(QLabel):
    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)
        self.setWordWrap(True)
        self.setObjectName("pathLink")

    def enterEvent(self, event) -> None:  # noqa: ANN001 - Qt event type.
        font = self.font()
        font.setUnderline(True)
        self.setFont(font)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: ANN001 - Qt event type.
        font = self.font()
        font.setUnderline(False)
        self.setFont(font)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001 - Qt event type.
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class HelpIcon(QLabel):
    def __init__(self, tooltip: str) -> None:
        super().__init__("?")
        self.setObjectName("helpIcon")
        self.setToolTip(tooltip)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(12, 12)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NanoC-NN Desktop")
        self.resize(1320, 840)
        self.model_path: Path | None = None
        self.current_out_root: Path | None = None
        self.output_root_path = WORK_ROOT / "last-output"
        self.worker: PipelineWorker | None = None
        self.log_collapsed = True
        self._last_sram_preset = ""
        self._last_flash_preset = ""

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
        splitter.setSizes([300, 740, 320])
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
        self.log_view.setVisible(False)
        self.log_toggle_button.setText("Show Log")
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
        self.project_tree.setMaximumHeight(86)
        layout.addWidget(self.project_tree)

        summary_box = QGroupBox("Model Summary")
        summary_box.setMinimumHeight(168)
        summary_layout = QVBoxLayout(summary_box)
        self.left_summary_label = QLabel("No model loaded.")
        self.left_summary_label.setWordWrap(True)
        self.left_summary_label.setObjectName("sideInfo")
        summary_layout.addWidget(self.left_summary_label)
        layout.addWidget(summary_box)

        ops_box = QGroupBox("Operator Mix")
        ops_box.setMinimumHeight(176)
        ops_layout = QVBoxLayout(ops_box)
        self.left_ops_label = QLabel("No model loaded.")
        self.left_ops_label.setWordWrap(True)
        self.left_ops_label.setObjectName("sideInfo")
        ops_layout.addWidget(self.left_ops_label)
        layout.addWidget(ops_box)

        layout.addStretch(1)

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
        self.tabs.addTab(self._graph_tab(), "Graph")
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

    def _graph_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        fit_button = QPushButton("Fit")
        reset_button = QPushButton("100%")
        fit_button.clicked.connect(lambda: self.graph_view.fit_to_view())
        reset_button.clicked.connect(lambda: self.graph_view.reset_zoom())
        self.graph_detail_label = QLabel("Load a model to inspect its node graph.")
        self.graph_detail_label.setWordWrap(True)
        toolbar.addWidget(self.graph_detail_label, 1)
        toolbar.addWidget(fit_button)
        toolbar.addWidget(reset_button)
        layout.addLayout(toolbar)

        self.graph_view = NetworkGraphView()
        self.graph_view.selection_changed.connect(self.on_graph_selection_changed)
        layout.addWidget(self.graph_view, 1)
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
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Target & Export")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        config_box = QGroupBox("Target")
        form = QFormLayout(config_box)
        self.target_combo = WideComboBox()
        self.target_combo.setMinimumWidth(136)
        self.target_combo.addItems(["cortex-m4", "cortex-m3", "cortex-m7", "cortex-m33", "cortex-m55"])
        self.target_combo.currentTextChanged.connect(self.apply_target_memory_preset)
        self.backend_combo = WideComboBox()
        self.backend_combo.setMinimumWidth(104)
        self.backend_combo.addItems(["auto", "scalar", "dsp", "mve"])
        self.sram_edit = QLineEdit()
        self.sram_edit.setPlaceholderText("optional, e.g. 128K")
        self.flash_edit = QLineEdit()
        self.flash_edit.setPlaceholderText("optional, e.g. 512K")
        self.prefix_edit = QLineEdit("nanoc_model")
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_path_label = ClickablePathLabel()
        self.output_path_label.clicked.connect(self.choose_output_dir)
        output_layout.addWidget(self.output_path_label, 1)
        self.set_output_root(self.output_root_path)
        form.addRow(
            target_field_label(
                "Core",
                "Target Cortex-M core profile. It controls memory presets and backend suitability.",
            ),
            self.target_combo,
        )
        form.addRow(
            target_field_label(
                "Backend",
                "CMSIS-NN backend preference. 'auto' lets NanoC-NN pick scalar, DSP, or MVE based on the selected core.",
            ),
            self.backend_combo,
        )
        form.addRow(
            target_field_label(
                "SRAM",
                "Available runtime SRAM budget for activation buffers and scratch buffers. Preset by core, editable for your exact MCU.",
            ),
            self.sram_edit,
        )
        form.addRow(
            target_field_label(
                "Flash",
                "Available program Flash budget for generated code, weights, and constant data. Preset by core, editable.",
            ),
            self.flash_edit,
        )
        form.addRow(
            target_field_label(
                "Prefix",
                "C symbol prefix used for generated functions, arrays, and headers. Keep it unique when integrating multiple models.",
            ),
            self.prefix_edit,
        )
        form.addRow(
            target_field_label(
                "Output",
                "Output directory for converter reports and the generated CMSIS-NN C project. Click the path text to change it.",
            ),
            output_row,
        )
        layout.addWidget(config_box)
        self.apply_target_memory_preset(self.target_combo.currentText())

        self.status_frame = QFrame()
        self.status_frame.setObjectName("statusFrame")
        self.status_frame.setMinimumHeight(128)
        status_layout = QVBoxLayout(self.status_frame)
        self.status_label = QLabel("idle")
        self.status_label.setObjectName("statusLabel")
        self.status_hint = QLabel("Select a model to start.")
        self.status_hint.setWordWrap(True)
        self.status_hint.setObjectName("statusHint")
        status_layout.addWidget(QLabel("Codegen Status"))
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_hint)
        layout.addWidget(self.status_frame)

        self.run_button = QPushButton("Generate C Project")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_pipeline)
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)

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
            str(self.output_root_path),
        )
        if path:
            self.set_output_root(Path(path))

    def apply_target_memory_preset(self, target: str) -> None:
        preset = TARGET_MEMORY_PRESETS.get(target)
        if preset is None:
            return
        sram, flash = preset
        sram_text = self.sram_edit.text().strip()
        flash_text = self.flash_edit.text().strip()
        if not sram_text or sram_text == self._last_sram_preset:
            self.sram_edit.setText(sram)
        if not flash_text or flash_text == self._last_flash_preset:
            self.flash_edit.setText(flash)
        self._last_sram_preset = sram
        self._last_flash_preset = flash
        self.sram_edit.setToolTip(f"Preset for {target}; editable")
        self.flash_edit.setToolTip(f"Preset for {target}; editable")

    def load_model(self, path: Path) -> None:
        self.model_path = path
        self.model_label.setText(path.name)
        self.set_output_root(WORK_ROOT / f"{path.stem}-output")
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
        self.left_summary_label.setText(
            "\n".join(
                [
                    f"File: {info['name']}",
                    f"Input: {info['inputs']}",
                    f"Output: {info['outputs']}",
                    f"Nodes: {info['node_count']}",
                    f"Quantized: {'yes' if info['quantized_hint'] else 'unknown'}",
                ]
            )
        )
        self.left_ops_label.setText(
            "\n".join(f"{op}: {count}" for op, count in list(info["op_counts"])[:8])
            or "No operators."
        )
        self.graph_view.draw_graph(dict(info["graph"]))
        self.graph_detail_label.setText(
            "Drag or two-finger scroll to pan. Pinch to zoom. Click a node to inspect it."
        )
        self.tabs.setCurrentIndex(1)

    def on_graph_selection_changed(self, node: dict[str, object]) -> None:
        if not node:
            self.graph_detail_label.setText(
                "Drag or two-finger scroll to pan. Pinch to zoom. Click a node to inspect it."
            )
            return
        self.graph_detail_label.setText(
            f"{node['title']} | {node['subtitle']} | "
            f"inputs: {node['input_count']} | outputs: {node['output_count']}"
        )

    def run_pipeline(self) -> None:
        if self.model_path is None:
            QMessageBox.information(self, "No model", "Select an ONNX model first.")
            return

        out_root = self.output_root_path.expanduser()
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

    def set_output_root(self, path: Path) -> None:
        self.output_root_path = path.expanduser()
        self.output_path_label.setText(str(self.output_root_path))
        self.output_path_label.setToolTip(str(self.output_root_path))

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
        reports = collect_report_files(self.current_out_root)
        if reports:
            self.preview_file(reports[0])
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
            self.preview_file(path)
            self.tabs.setCurrentIndex(3)

    def preview_file(self, path: Path) -> None:
        self.file_preview_title.setText(str(path))
        self.file_preview.setPlainText(path.read_text(encoding="utf-8", errors="replace"))

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
            QLabel#sideInfo { font-size: 12px; line-height: 1.25; }
            QLabel#pathLink {
                font-family: Menlo, Consolas, monospace;
                font-size: 11px;
                color: #2563eb;
                padding: 2px 0;
            }
            QLabel#helpIcon {
                border: 1px solid #94a3b8;
                border-radius: 6px;
                color: #475569;
                background: #f8fafc;
                font-size: 8px;
                font-weight: 700;
            }
            QLabel#statusLabel { font-size: 26px; font-weight: 800; }
            QLabel#statusHint { font-size: 12px; }
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
        "graph": build_graph_view_model(model),
    }


def build_graph_view_model(model) -> dict[str, object]:  # noqa: ANN001 - ONNX model type.
    initializer_names = {item.name for item in model.graph.initializer}
    initializer_shapes = {item.name: "x".join(str(dim) for dim in item.dims) for item in model.graph.initializer}
    shape_map = collect_tensor_shapes(model)
    graph_inputs = [item for item in model.graph.input if item.name not in initializer_names]
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    tensor_owner: dict[str, str] = {}
    tensor_depth: dict[str, int] = {}

    for value in graph_inputs:
        node_id = f"input:{value.name}"
        tensor_owner[value.name] = node_id
        tensor_depth[value.name] = 0
        nodes.append(
            {
                "id": node_id,
                "title": shorten(value.name, 24),
                "subtitle": "Input",
                "depth": 0,
                "input_count": 0,
                "output_count": 1,
                "params": [],
                "header_color": "#f4f5f7",
                "title_color": "#111827",
                "tooltip": f"Input\n{tensor_info(value)}",
            }
        )

    for index, node in enumerate(model.graph.node):
        node_id = f"node:{index}"
        input_names = [name for name in node.input if name]
        output_names = [name for name in node.output if name]
        input_depths = [
            tensor_depth[input_name]
            for input_name in node.input
            if input_name in tensor_depth and input_name not in initializer_names
        ]
        if not input_depths and any(name in initializer_names for name in input_names):
            source_shape = next(
                (initializer_shapes[name] for name in input_names if name in initializer_shapes),
                "scalar",
            )
            for output_name in output_names:
                initializer_names.add(output_name)
                initializer_shapes[output_name] = shape_map.get(output_name, source_shape)
            continue
        depth = (max(input_depths) + 1) if input_depths else 1
        title = node.name or f"{node.op_type}_{index}"
        params = node_params(node, input_names, initializer_names, initializer_shapes)
        nodes.append(
            {
                "id": node_id,
                "title": shorten(node.op_type, 22),
                "subtitle": node.op_type,
                "depth": depth,
                "input_count": len(input_names),
                "output_count": len(output_names),
                "params": params,
                "header_color": op_color(node.op_type),
                "title_color": "#ffffff",
                "tooltip": "\n".join(
                    [
                        f"{title}",
                        f"op: {node.op_type}",
                        f"inputs: {', '.join(input_names) or '-'}",
                        f"outputs: {', '.join(output_names) or '-'}",
                    ]
                ),
            }
        )
        for input_name in input_names:
            source = tensor_owner.get(input_name)
            if source:
                edges.append(
                    {
                        "source": source,
                        "target": node_id,
                        "tensor": input_name,
                        "label": shape_map.get(input_name, ""),
                    }
                )
        for output_name in output_names:
            tensor_owner[output_name] = node_id
            tensor_depth[output_name] = depth

    max_depth = max((int(node["depth"]) for node in nodes), default=0)
    for value in model.graph.output:
        node_id = f"output:{value.name}"
        source = tensor_owner.get(value.name)
        depth = tensor_depth.get(value.name, max_depth) + 1
        nodes.append(
            {
                "id": node_id,
                "title": shorten(value.name, 24),
                "subtitle": "Output",
                "depth": depth,
                "input_count": 1 if source else 0,
                "output_count": 0,
                "params": [],
                "header_color": "#f4f5f7",
                "title_color": "#111827",
                "tooltip": f"Output\n{tensor_info(value)}",
            }
        )
        if source:
            edges.append(
                {
                    "source": source,
                    "target": node_id,
                    "tensor": value.name,
                    "label": shape_map.get(value.name, ""),
                }
            )

    return {"nodes": nodes, "edges": edges}


def collect_tensor_shapes(model) -> dict[str, str]:  # noqa: ANN001 - ONNX model type.
    shapes: dict[str, str] = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        shape = tensor_shape(value)
        if shape:
            shapes[value.name] = shape
    for initializer in model.graph.initializer:
        if initializer.dims:
            shapes[initializer.name] = "x".join(str(dim) for dim in initializer.dims)
    return shapes


def tensor_shape(value_info) -> str:  # noqa: ANN001 - ONNX value info type.
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return ""
    dims = []
    for dim in tensor_type.shape.dim:
        dims.append(str(dim.dim_value if dim.dim_value else dim.dim_param or "?"))
    return "x".join(dims)


def node_params(node, input_names: list[str], initializer_names: set[str], initializer_shapes: dict[str, str]) -> list[str]:  # noqa: ANN001 - ONNX node type.
    params: list[str] = []
    for input_name in input_names[1:]:
        short = simplify_tensor_name(input_name)
        if input_name in initializer_names:
            shape = initializer_shapes.get(input_name, "")
            params.append(f"{short} ({shape or 'scalar'})")
        elif len(params) < 5:
            params.append(short)
    for attr in node.attribute[:3]:
        params.append(f"{attr.name} = {attribute_value(attr)}")
    if len(params) > 8:
        return params[:7] + ["..."]
    return params


def simplify_tensor_name(value: str) -> str:
    tail = value.rsplit("/", 1)[-1]
    return shorten(tail, 24)


def attribute_value(attr) -> str:  # noqa: ANN001 - ONNX attribute type.
    if attr.type == 1:
        return f"{attr.f:.5g}"
    if attr.type == 2:
        return str(attr.i)
    if attr.type == 3:
        return shorten(attr.s.decode("utf-8", errors="replace"), 18)
    if attr.type == 7:
        return "x".join(str(item) for item in attr.ints[:4])
    if attr.type == 6:
        return ", ".join(f"{item:.5g}" for item in attr.floats[:3])
    return "..."


def op_color(op_type: str) -> str:
    if op_type.startswith("QLinear") or op_type in {"QuantizeLinear", "DequantizeLinear"}:
        return "#303030"
    if op_type in {"Conv", "Gemm", "MatMul"}:
        return "#303030"
    if op_type in {"MaxPool", "AveragePool", "GlobalAveragePool"}:
        return "#37623f"
    if op_type in {"Relu", "Clip", "Sigmoid", "Tanh"}:
        return "#37623f"
    if op_type in {"Flatten", "Reshape", "Transpose", "Squeeze", "Unsqueeze"}:
        return "#7a554a"
    return "#303030"


def shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def order_graph_nodes(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> list[dict[str, object]]:
    original_index = {str(node["id"]): index for index, node in enumerate(nodes)}
    by_depth: dict[int, list[dict[str, object]]] = {}
    incoming: dict[str, list[str]] = {}
    for node in nodes:
        by_depth.setdefault(int(node["depth"]), []).append(node)
    for edge in edges:
        incoming.setdefault(str(edge["target"]), []).append(str(edge["source"]))

    ordered: list[dict[str, object]] = []
    position: dict[str, float] = {}
    for depth in sorted(by_depth):
        group = by_depth[depth]

        def sort_key(node: dict[str, object]) -> tuple[float, int]:
            parents = [position[parent] for parent in incoming.get(str(node["id"]), []) if parent in position]
            if parents:
                return (sum(parents) / len(parents), original_index[str(node["id"])])
            return (float(original_index[str(node["id"])]), original_index[str(node["id"])])

        group = sorted(group, key=sort_key)
        for index, node in enumerate(group):
            position[str(node["id"])] = float(index)
            ordered.append(node)
    return ordered


def graph_node_height(node: dict[str, object]) -> int:
    params = list(node.get("params", []))
    if not params:
        return NODE_HEADER_HEIGHT
    return NODE_HEADER_HEIGHT + 14 + min(len(params), 8) * NODE_LINE_HEIGHT


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


def target_field_label(text: str, tooltip: str) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    layout.addWidget(HelpIcon(tooltip), 0, Qt.AlignTop)
    label = QLabel(text)
    layout.addWidget(label)
    layout.addStretch(1)
    return widget


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

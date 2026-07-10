"""
Build small fixed datasets and ONNX inference summaries for time-series models.

These fixtures are not accuracy benchmarks yet.  They are deterministic smoke
datasets that make the real-world time-series candidates runnable and provide a
stable starting point for later ONNX-vs-C numeric tests.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:  # pragma: no cover - environment guidance
    raise SystemExit(
        "onnxruntime is required to build time-series inference summaries. "
        "Install it in the active environment, or run with the temporary "
        "PYTHONPATH used by this iteration."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
TDD_ROOT = SCRIPT_DIR.parent
REPO_ROOT = TDD_ROOT.parent
ONNX_DIR = TDD_ROOT / "fixtures" / "onnx"
DATASET_DIR = TDD_ROOT / "fixtures" / "datasets"
EXTERNAL_DIR = TDD_ROOT / "fixtures" / "external" / "timeseries_sources"


@dataclass(frozen=True)
class RunSpec:
    dataset_id: str
    model_files: list[str]
    task: str
    labels: list[str]
    source_kind: str
    reasonability_rule: str


def main() -> int:
    specs = [
        _build_cwru(),
        _build_kws(),
        _build_stwin(),
        _build_met(),
    ]
    _write_index(specs)
    print(f"Built {len(specs)} time-series dataset fixtures under {DATASET_DIR}")
    return 0


def _build_cwru() -> RunSpec:
    labels = ["healthy", "faulty"]
    samples: list[dict[str, Any]] = []
    npz_path = EXTERNAL_DIR / "edge_infer" / "cwru_test_samples.npz"
    if npz_path.exists():
        npz = np.load(npz_path)
        xs = npz["X_test"].astype(np.float32)
        ys = npz["y_test"].astype(np.int64)
        selected: list[int] = []
        for wanted_label in [0, 1]:
            selected.extend(np.where(ys == wanted_label)[0][:3].tolist())
        if not selected:
            selected = list(range(min(6, len(xs))))
        for rank, idx in enumerate(selected):
            label_index = int(ys[idx])
            samples.append(
                {
                    "id": f"cwru_real_{rank:02d}_{labels[label_index]}",
                    "label_index": label_index,
                    "label_name": labels[label_index],
                    "inputs": {"input": xs[idx].reshape(1, 1, 1, 32)},
                    "source": "edge-infer CWRU held-out test_samples.npz",
                }
            )
    else:
        rng = np.random.default_rng(20260708)
        healthy = rng.normal(0.0, 0.2, size=(3, 32)).astype(np.float32)
        faulty = rng.normal(1.0, 0.35, size=(3, 32)).astype(np.float32)
        for label_index, arrs in [(0, healthy), (1, faulty)]:
            for i, arr in enumerate(arrs):
                samples.append(
                    {
                        "id": f"cwru_synthetic_{labels[label_index]}_{i:02d}",
                        "label_index": label_index,
                        "label_name": labels[label_index],
                        "inputs": {"input": arr.reshape(1, 1, 1, 32)},
                        "source": "deterministic synthetic fallback",
                    }
                )

    dataset_id = "timeseries_cwru_bearing_smoke"
    model_files = ["timeseries_cwru_bearing_mlp.onnx"]
    _write_dataset(
        dataset_id,
        {
            "dataset_id": dataset_id,
            "task": "CWRU bearing vibration healthy/faulty classification",
            "model_files": model_files,
            "labels": labels,
            "input_shapes": {"input": [1, 1, 1, 32]},
            "sample_role": "real held-out samples when source npz is present; otherwise synthetic smoke probes",
            "samples": samples,
        },
    )
    _write_inference(dataset_id, model_files, samples, labels)
    _write_readme(
        dataset_id,
        "CWRU Bearing Vibration Smoke Dataset",
        [
            "用于振动轴承健康/故障二分类模型的固定推理样本。",
            "优先使用 edge-infer 原仓库的 `test_samples.npz`，包含 held-out 特征窗和标签。",
            "合理性观察重点：真实样本的 ONNX top1 应尽量匹配 `healthy/faulty` 标签；若后续 C 端结果偏离该分布，需要回查 Gemm/Flatten 数值链路。",
        ],
    )
    return RunSpec(
        dataset_id,
        model_files,
        "bearing vibration binary classification",
        labels,
        "real source samples",
        "top1 should match provided healthy/faulty labels for most fixed samples",
    )


def _build_kws() -> RunSpec:
    label_map_path = EXTERNAL_DIR / "nano_kws" / "ds_cnn_small_int8.label_map.json"
    if label_map_path.exists():
        labels = json.loads(label_map_path.read_text(encoding="utf-8"))["labels"]
    else:
        labels = [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "_silence_",
            "_unknown_",
        ]
    rng = np.random.default_rng(20260708)
    silence = np.zeros((1, 1, 40, 97), dtype=np.float32)
    quiet_noise = rng.normal(0.0, 0.025, size=(1, 1, 40, 97)).astype(np.float32)
    vertical_band = np.zeros((1, 1, 40, 97), dtype=np.float32)
    vertical_band[:, :, 8:28, 35:56] = 1.2
    diagonal = np.zeros((1, 1, 40, 97), dtype=np.float32)
    for t in range(97):
        center = int(8 + 24 * t / 96)
        diagonal[:, :, max(0, center - 2) : min(40, center + 3), t] = 0.9
    impulse = rng.normal(0.0, 0.02, size=(1, 1, 40, 97)).astype(np.float32)
    impulse[:, :, :, 45:50] += 1.0
    samples = [
        {"id": "kws_silence_zero", "inputs": {"input": silence}, "source": "deterministic log-mel probe"},
        {"id": "kws_quiet_noise", "inputs": {"input": quiet_noise}, "source": "deterministic log-mel probe"},
        {"id": "kws_vertical_energy_band", "inputs": {"input": vertical_band}, "source": "deterministic log-mel probe"},
        {"id": "kws_rising_diagonal_band", "inputs": {"input": diagonal}, "source": "deterministic log-mel probe"},
        {"id": "kws_broadband_impulse", "inputs": {"input": impulse}, "source": "deterministic log-mel probe"},
    ]
    dataset_id = "timeseries_kws_dscnn_smoke"
    model_files = [
        "timeseries_kws_dscnn_small_int8.onnx",
        "timeseries_kws_dscnn_small_qat_int8.onnx",
    ]
    _write_dataset(
        dataset_id,
        {
            "dataset_id": dataset_id,
            "task": "keyword spotting DS-CNN log-mel classification",
            "model_files": model_files,
            "labels": labels,
            "input_shapes": {"input": [1, 1, 40, 97]},
            "sample_role": "deterministic log-mel smoke probes; not a speech-commands accuracy set",
            "samples": samples,
        },
    )
    _write_inference(dataset_id, model_files, samples, labels)
    _write_readme(
        dataset_id,
        "Keyword Spotting DS-CNN Smoke Dataset",
        [
            "用于两个 nano-kws INT8/QAT INT8 ONNX 的固定 log-mel 探针样本。",
            "这些样本不是 Speech Commands 真实音频，因此不声称准确率，只用于确认模型可运行、输出有限、类别分布稳定。",
            "合理性观察重点：silence/noise 类探针不应产生 NaN/Inf；两个同架构模型应输出可比较但不必完全相同的 top1。",
        ],
    )
    return RunSpec(
        dataset_id,
        model_files,
        "keyword spotting log-mel classification",
        labels,
        "deterministic probes",
        "outputs must be finite and stable across reruns; no accuracy claim yet",
    )


def _build_stwin() -> RunSpec:
    labels = ["A", "E", "I", "O", "U"]
    x = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    neutral = np.full((1, 6, 20, 20), 0.5, dtype=np.float32)
    ramp = np.zeros((1, 6, 20, 20), dtype=np.float32)
    ramp[:, 0] = x.reshape(1, 20)
    sine = np.full((1, 6, 20, 20), 0.5, dtype=np.float32)
    grid = np.sin(np.linspace(0, 2 * math.pi, 20, dtype=np.float32)).reshape(20, 1)
    sine[:, 3] += 0.45 * grid
    checker = np.indices((20, 20)).sum(axis=0) % 2
    checker = checker.astype(np.float32)
    checker = checker.reshape(1, 1, 20, 20).repeat(6, axis=1)
    impulse = np.zeros((1, 6, 20, 20), dtype=np.float32)
    impulse[:, :, 9:12, 9:12] = 1.0
    samples = [
        {"id": "stwin_neutral_midpoint", "inputs": {"input": neutral}, "source": "deterministic normalized IMU image probe"},
        {"id": "stwin_axis0_horizontal_ramp", "inputs": {"input": ramp}, "source": "deterministic normalized IMU image probe"},
        {"id": "stwin_axis3_sine_motion", "inputs": {"input": sine}, "source": "deterministic normalized IMU image probe"},
        {"id": "stwin_six_axis_checker", "inputs": {"input": checker}, "source": "deterministic normalized IMU image probe"},
        {"id": "stwin_center_impulse", "inputs": {"input": impulse}, "source": "deterministic normalized IMU image probe"},
    ]
    dataset_id = "timeseries_stwin_vowel_smoke"
    model_files = ["timeseries_stwin_vowel.onnx"]
    _write_dataset(
        dataset_id,
        {
            "dataset_id": dataset_id,
            "task": "STWIN IMU vowel gesture classification",
            "model_files": model_files,
            "labels": labels,
            "input_shapes": {"input": [1, 6, 20, 20]},
            "sample_role": "deterministic normalized IMU-image probes; not labeled vowel recordings",
            "samples": samples,
        },
    )
    _write_inference(dataset_id, model_files, samples, labels)
    _write_readme(
        dataset_id,
        "STWIN Vowel IMU Smoke Dataset",
        [
            "用于 6 通道 20x20 IMU vowel 模型的固定归一化输入探针。",
            "源项目真实数据由 DVC 管理，本轮未固化真实 vowel 采集样本；因此这里只验证推理稳定性和输出分布。",
            "合理性观察重点：输出应为 5 类概率分布；不同运动纹理探针应能触发不同或至少稳定的置信度变化。",
        ],
    )
    return RunSpec(
        dataset_id,
        model_files,
        "IMU vowel gesture classification",
        labels,
        "deterministic probes",
        "outputs should be finite 5-class distributions",
    )


def _build_met() -> RunSpec:
    labels = ["sedentary", "light", "moderate", "vigorous"]
    t = np.linspace(0.0, 2 * math.pi, 150, dtype=np.float32)

    def make_sample(sample_id: str, amplitude: float, feat_level: float) -> dict[str, Any]:
        raw = np.zeros((1, 150, 8), dtype=np.float32)
        raw[0, :, 0] = amplitude * np.sin(t)
        raw[0, :, 1] = amplitude * np.cos(t)
        raw[0, :, 2] = 0.5 * amplitude * np.sin(2 * t)
        raw[0, :, 3] = 0.25 * amplitude * np.cos(2 * t)
        raw[0, :, 4] = 0.1 * amplitude
        raw[0, :, 5] = -0.1 * amplitude
        feat = np.zeros((1, 36), dtype=np.float32)
        feat[0, 0:6] = feat_level
        feat[0, 6:12] = amplitude
        feat[0, 12:18] = -amplitude
        feat[0, 18:24] = 0.5 * amplitude
        feat[0, 24:30] = -0.5 * amplitude
        feat[0, 30:36] = feat_level + amplitude
        return {
            "id": sample_id,
            "inputs": {"feat_input": feat, "raw_input": raw},
            "source": "deterministic standardized wearable-motion probe",
        }

    samples = [
        make_sample("met_sedentary_baseline", 0.0, 0.0),
        make_sample("met_light_periodic_motion", 0.25, 0.2),
        make_sample("met_moderate_periodic_motion", 0.8, 0.6),
        make_sample("met_vigorous_periodic_motion", 1.6, 1.2),
    ]
    dataset_id = "timeseries_met_hybrid_smoke"
    model_files = ["timeseries_met_hybrid.onnx"]
    _write_dataset(
        dataset_id,
        {
            "dataset_id": dataset_id,
            "task": "wearable/smartphone MET activity-intensity classification",
            "model_files": model_files,
            "labels": labels,
            "input_shapes": {"feat_input": [1, 36], "raw_input": [1, 150, 8]},
            "sample_role": "deterministic standardized motion probes; not WISDM/MotionSense/UCI-HAR rows",
            "samples": samples,
        },
    )
    _write_inference(dataset_id, model_files, samples, labels)
    _write_readme(
        dataset_id,
        "MET Hybrid Activity Smoke Dataset",
        [
            "用于双输入 MET 活动强度模型的固定推理探针，覆盖静止、轻微、中等、强运动幅值。",
            "当前样本在模型输入空间中构造，尚未绑定 WISDM/MotionSense/UCI-HAR 原始行和 scaler；因此只作为可运行性/趋势观察。",
            "合理性观察重点：活动幅值变化时 4 类输出应保持有限且可复现；如果所有探针完全同类且置信度饱和，后续需要补真实 scaler+数据源。",
        ],
    )
    return RunSpec(
        dataset_id,
        model_files,
        "MET activity-intensity classification",
        labels,
        "deterministic probes",
        "outputs should be finite 4-class distributions over increasing motion probes",
    )


def _write_dataset(dataset_id: str, content: dict[str, Any]) -> None:
    out_dir = DATASET_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(content)
    (out_dir / "dataset.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_inference(
    dataset_id: str,
    model_files: list[str],
    samples: list[dict[str, Any]],
    labels: list[str],
) -> None:
    out_dir = DATASET_DIR / dataset_id
    summaries = []
    for model_file in model_files:
        model_path = ONNX_DIR / model_file
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        model_summary = {
            "model_file": model_file,
            "input_names": input_names,
            "output_names": output_names,
            "samples": [],
        }
        for sample in samples:
            feeds = {
                name: np.asarray(sample["inputs"][name], dtype=np.float32)
                for name in input_names
            }
            outputs = session.run(None, feeds)
            vector = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            probs = _as_probabilities(vector)
            top_index = int(np.argmax(probs))
            entry = {
                "sample_id": sample["id"],
                "raw_output": _round_list(vector.tolist(), 6),
                "probabilities": _round_list(probs.tolist(), 6),
                "top1_index": top_index,
                "top1_label": labels[top_index] if top_index < len(labels) else str(top_index),
                "top1_probability": round(float(probs[top_index]), 6),
            }
            if "label_index" in sample:
                entry["expected_label_index"] = sample["label_index"]
                entry["expected_label_name"] = sample.get("label_name", "")
                entry["top1_matches_expected"] = top_index == sample["label_index"]
            model_summary["samples"].append(entry)
        summaries.append(model_summary)
    (out_dir / "onnx_inference_summary.json").write_text(
        json.dumps({"dataset_id": dataset_id, "models": summaries}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _write_readme(dataset_id: str, title: str, paragraphs: list[str]) -> None:
    out_dir = DATASET_DIR / dataset_id
    body = [f"# {title}", ""]
    body.extend(paragraphs)
    body.extend(
        [
            "",
            "## Files",
            "",
            "- `dataset.json`: 固定输入样本。",
            "- `onnx_inference_summary.json`: ONNX Runtime 推理结果和 top1 摘要。",
        ]
    )
    (out_dir / "README.md").write_text("\n\n".join(body) + "\n", encoding="utf-8")


def _write_index(specs: list[RunSpec]) -> None:
    lines = [
        "# Time-Series Dataset Fixtures",
        "",
        "本目录下的 `timeseries_*` 数据集用于把已下载的时域/传感器 ONNX 先运行起来，作为后续 ONNX-vs-C 数值回归的入口。",
        "",
        "| Dataset | Models | Task | Source kind | Reasonability rule |",
        "|---|---|---|---|---|",
    ]
    for spec in specs:
        lines.append(
            f"| `{spec.dataset_id}` | {', '.join(f'`{m}`' for m in spec.model_files)} | "
            f"{spec.task} | {spec.source_kind} | {spec.reasonability_rule} |"
        )
    lines.extend(
        [
            "",
            "说明：除 CWRU 使用原仓库 held-out 样本外，本轮其他数据集主要是固定探针，不宣称真实准确率。",
            "这些探针的价值是确保模型可加载、输入格式明确、输出可复现，并为下一步 C 端数值对比建立样本契约。",
        ]
    )
    (DATASET_DIR / "TIMESERIES_DATASETS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _as_probabilities(vector: np.ndarray) -> np.ndarray:
    if vector.size > 1 and np.all(vector >= -1e-4) and abs(float(vector.sum()) - 1.0) < 1e-3:
        return vector
    shifted = vector - np.max(vector)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _round_nested(value.tolist(), 6)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def _round_nested(value: Any, digits: int) -> Any:
    if isinstance(value, list):
        return [_round_nested(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def _round_list(values: list[float], digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort


WINDOW_SIZE = 10
CLASS_NAMES = ["no_jump", "up_jump", "down_jump"]


def example_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exported ONNX model.")
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help="Comma-separated 10-point signal window, for example: 9,9,9,1,1,1,1,1,1,1.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=example_root() / "data" / "inference" / "input_windows.csv",
        help="CSV file containing x0..x9 columns for batch inference.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=example_root() / "outputs" / "onnx" / "signal_jump.onnx",
        help="Path to the exported ONNX model.",
    )
    return parser.parse_args()


def parse_window(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != WINDOW_SIZE:
        raise ValueError(f"--window must contain exactly {WINDOW_SIZE} values.")
    return values


def load_windows(args: argparse.Namespace) -> list[list[float]]:
    if args.window is not None:
        return [parse_window(args.window)]
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    with args.input_csv.open("r", newline="", encoding="utf-8") as csv_file:
        rows = csv.DictReader(csv_file)
        return [[float(row[f"x{i}"]) for i in range(WINDOW_SIZE)] for row in rows]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}. Run train_and_export.py first.")

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    windows = load_windows(args)

    for window in windows:
        input_array = np.array(window, dtype=np.float32).reshape(1, 1, WINDOW_SIZE)
        logits = session.run(None, {input_name: input_array})[0]
        probabilities = softmax(logits)
        class_id = int(np.argmax(probabilities, axis=1)[0])

        print(f"window={window}")
        print(f"logits={logits[0].tolist()}")
        print(f"probabilities={probabilities[0].tolist()}")
        print(f"class_id={class_id}")
        print(f"class_name={CLASS_NAMES[class_id]}")
        print(f"has_jump={class_id != 0}")


if __name__ == "__main__":
    main()


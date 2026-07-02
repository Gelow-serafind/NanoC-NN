from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort


def example_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exported ONNX model.")
    parser.add_argument(
        "--value",
        type=float,
        default=None,
        help=(
            "Single input number to classify. If omitted, read values from "
            "data/inference/input_values.csv."
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=example_root() / "data" / "inference" / "input_values.csv",
        help="CSV file containing a 'value' column for batch inference.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=example_root() / "outputs" / "onnx" / "is_over_10.int8.onnx",
        help="Path to the exported ONNX model.",
    )
    return parser.parse_args()


def load_values(args: argparse.Namespace) -> list[float]:
    if args.value is not None:
        return [args.value]
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    with args.input_csv.open("r", newline="", encoding="utf-8") as csv_file:
        rows = csv.DictReader(csv_file)
        return [float(row["value"]) for row in rows]


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
    values = load_values(args)

    for value in values:
        input_array = np.array([[value]], dtype=np.float32)
        logits = session.run(None, {input_name: input_array})[0]
        probabilities = softmax(logits)
        predicted_class = int(np.argmax(probabilities, axis=1)[0])

        print(f"value={value}")
        print(f"logits={logits[0].tolist()}")
        print(f"probabilities={probabilities[0].tolist()}")
        print(f"predicted_class={predicted_class}")
        print(f"is_over_10={bool(predicted_class)}")


if __name__ == "__main__":
    main()

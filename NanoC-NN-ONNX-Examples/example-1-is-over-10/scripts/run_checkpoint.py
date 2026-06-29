from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from network import IsOver10Net


def example_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained PyTorch checkpoint.")
    parser.add_argument(
        "--value",
        type=float,
        default=None,
        help="Single input number to classify. If omitted, read values from data/inference/input_values.csv.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=example_root() / "data" / "inference" / "input_values.csv",
        help="CSV file containing a 'value' column for batch inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=example_root() / "outputs" / "checkpoints" / "is_over_10.pt",
        help="Path to the trained checkpoint.",
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


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run train_and_export.py first."
        )

    model = IsOver10Net()
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    values = load_values(args)
    input_tensor = torch.tensor([[value] for value in values], dtype=torch.float32)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1).tolist()

    for index, value in enumerate(values):
        predicted_class = int(predicted_classes[index])
        print(f"value={value}")
        print(f"logits={logits.tolist()[index]}")
        print(f"probabilities={probabilities.tolist()[index]}")
        print(f"predicted_class={predicted_class}")
        print(f"is_over_10={bool(predicted_class)}")


if __name__ == "__main__":
    main()

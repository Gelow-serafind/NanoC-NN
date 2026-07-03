from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from network import SignalJumpNet

WINDOW_SIZE = 10
CLASS_NAMES = ["no_jump", "up_jump", "down_jump"]


def example_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained PyTorch checkpoint.")
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help="Comma-separated 10-point signal window, for example: 5,5,5,5,5,11,11,11,11,11.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=example_root() / "data" / "inference" / "input_windows.csv",
        help="CSV file containing x0..x9 columns for batch inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=example_root() / "outputs" / "checkpoints" / "signal_jump.pt",
        help="Path to the trained checkpoint.",
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


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run train_and_export.py first."
        )

    model = SignalJumpNet()
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    windows = load_windows(args)
    input_tensor = torch.tensor(windows, dtype=torch.float32).view(-1, 1, WINDOW_SIZE)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1).tolist()

    for index, window in enumerate(windows):
        class_id = int(predictions[index])
        print(f"window={window}")
        print(f"logits={logits.tolist()[index]}")
        print(f"probabilities={probabilities.tolist()[index]}")
        print(f"class_id={class_id}")
        print(f"class_name={CLASS_NAMES[class_id]}")
        print(f"has_jump={class_id != 0}")


if __name__ == "__main__":
    main()

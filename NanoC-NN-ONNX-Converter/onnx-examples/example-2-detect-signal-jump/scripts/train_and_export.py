from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from network import SignalJumpNet


SEED = 20260629
WINDOW_SIZE = 10
JUMP_THRESHOLD = 5.0
SAMPLES_PER_CLASS = 512
TEST_SAMPLES_PER_CLASS = 128
EPOCHS = 900
LEARNING_RATE = 0.01


CLASS_NAMES = ["no_jump", "up_jump", "down_jump"]


def example_root() -> Path:
    return Path(__file__).resolve().parents[1]


def no_jump_window(generator: torch.Generator) -> torch.Tensor:
    start = torch.randint(-10, 11, (1,), generator=generator, dtype=torch.int64).float()
    steps = torch.randint(-2, 3, (WINDOW_SIZE - 1,), generator=generator, dtype=torch.int64).float()
    values = [start.item()]
    for step in steps.tolist():
        values.append(values[-1] + step)
    return torch.tensor(values, dtype=torch.float32)


def jump_window(generator: torch.Generator, direction: int) -> torch.Tensor:
    jump_index = int(torch.randint(1, WINDOW_SIZE, (1,), generator=generator).item())
    base = float(torch.randint(-8, 9, (1,), generator=generator).item())
    jump_size = float(torch.randint(int(JUMP_THRESHOLD), 11, (1,), generator=generator).item())
    before_noise = torch.randint(-1, 2, (jump_index,), generator=generator, dtype=torch.int64).float()
    after_noise = torch.randint(
        -1, 2, (WINDOW_SIZE - jump_index,), generator=generator, dtype=torch.int64
    ).float()
    before = base + before_noise
    after = base + direction * jump_size + after_noise
    return torch.cat([before, after]).to(torch.float32)


def make_dataset(samples_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    windows: list[torch.Tensor] = []
    labels: list[int] = []

    for _ in range(samples_per_class):
        windows.append(no_jump_window(generator))
        labels.append(0)
        windows.append(jump_window(generator, direction=1))
        labels.append(1)
        windows.append(jump_window(generator, direction=-1))
        labels.append(2)

    values = torch.stack(windows).view(-1, 1, WINDOW_SIZE)
    targets = torch.tensor(labels, dtype=torch.long)
    order = torch.randperm(values.shape[0], generator=generator)
    return values[order], targets[order]


def write_csv(path: Path, values: torch.Tensor, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([f"x{i}" for i in range(WINDOW_SIZE)] + ["class_id", "class_name"])
        for window, label in zip(values[:, 0, :].tolist(), labels.tolist(), strict=True):
            writer.writerow([f"{item:.3f}" for item in window] + [label, CLASS_NAMES[label]])


def accuracy(model: nn.Module, values: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(values)
        predictions = torch.argmax(logits, dim=1)
        return float((predictions == labels).float().mean().item())


def train_model(train_values: torch.Tensor, train_labels: torch.Tensor) -> SignalJumpNet:
    torch.manual_seed(SEED)
    model = SignalJumpNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(train_values)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

    return model


def export_onnx(model: nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.zeros((1, 1, WINDOW_SIZE), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["signal_window"],
        output_names=["logits"],
    )


def main() -> None:
    root = example_root()
    train_values, train_labels = make_dataset(SAMPLES_PER_CLASS, SEED)
    test_values, test_labels = make_dataset(TEST_SAMPLES_PER_CLASS, SEED + 1)

    write_csv(root / "data" / "train" / "signal_jump_train.csv", train_values, train_labels)
    write_csv(root / "data" / "test" / "signal_jump_test.csv", test_values, test_labels)

    model = train_model(train_values, train_labels)
    train_acc = accuracy(model, train_values, train_labels)
    test_acc = accuracy(model, test_values, test_labels)

    checkpoint_path = root / "outputs" / "checkpoints" / "signal_jump.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    onnx_path = root / "outputs" / "onnx" / "signal_jump.onnx"
    export_onnx(model, onnx_path)

    print(f"train_accuracy={train_acc:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"onnx={onnx_path}")


if __name__ == "__main__":
    main()


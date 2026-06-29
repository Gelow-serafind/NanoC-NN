from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from network import IsOver10Net

SEED = 20260629
EPOCHS = 800
LEARNING_RATE = 0.03


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_dataset(start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(start, stop + 1, dtype=torch.float32).view(-1, 1)
    labels = (values.view(-1) > 10.0).long()
    return values, labels


def write_csv(path: Path, values: torch.Tensor, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["value", "is_over_10"])
        for value, label in zip(values.view(-1).tolist(), labels.tolist(), strict=True):
            writer.writerow([f"{value:.1f}", int(label)])


def accuracy(model: nn.Module, values: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(values)
        predictions = torch.argmax(logits, dim=1)
        return float((predictions == labels).float().mean().item())


def train_model(train_values: torch.Tensor, train_labels: torch.Tensor) -> IsOver10Net:
    torch.manual_seed(SEED)
    model = IsOver10Net()
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
    dummy_input = torch.tensor([[11.0]], dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )


def main() -> None:
    root = project_root()
    train_values, train_labels = make_dataset(-40, 60)
    test_values, test_labels = make_dataset(-20, 40)

    write_csv(root / "data" / "train" / "is_over_10_train.csv", train_values, train_labels)
    write_csv(root / "data" / "test" / "is_over_10_test.csv", test_values, test_labels)

    model = train_model(train_values, train_labels)
    train_acc = accuracy(model, train_values, train_labels)
    test_acc = accuracy(model, test_values, test_labels)

    checkpoint_path = root / "outputs" / "checkpoints" / "is_over_10.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    onnx_path = root / "outputs" / "onnx" / "is_over_10.onnx"
    export_onnx(model, onnx_path)

    print(f"train_accuracy={train_acc:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"onnx={onnx_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import onnx
import torch
from network import IsOver10Net
from onnx import TensorProto, helper, numpy_helper
from torch import nn

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
    weight = model.classifier.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.classifier.bias.detach().cpu().numpy().astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gemm",
                ["input", "fc.weight", "fc.bias"],
                ["logits"],
                name="fc",
                transB=1,
            )
        ],
        "is_over_10_float",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])],
        [
            numpy_helper.from_array(weight, name="fc.weight"),
            numpy_helper.from_array(bias, name="fc.bias"),
        ],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="nanoc-nn-example",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx_model.ir_version = 10
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path)


def quantize_symmetric(array: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(array)))
    scale = max(max_abs / 127.0, 1.0e-8)
    quantized = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return quantized, scale


def activation_scale(values: np.ndarray) -> float:
    max_abs = float(np.max(np.abs(values)))
    return max(max_abs / 127.0, 1.0e-8)


def export_quantized_onnx(
    model: IsOver10Net,
    calibration_values: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    weight = model.classifier.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.classifier.bias.detach().cpu().numpy().astype(np.float32)
    weight_q, weight_scale = quantize_symmetric(weight)
    input_scale = activation_scale(calibration_values.detach().cpu().numpy())
    with torch.no_grad():
        logits = model(calibration_values).detach().cpu().numpy()
    output_scale = activation_scale(logits)

    initializers = [
        numpy_helper.from_array(np.array(input_scale, dtype=np.float32), name="input.scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="input.zero_point"),
        numpy_helper.from_array(np.array(weight_scale, dtype=np.float32), name="fc.weight.scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="fc.weight.zero_point"),
        numpy_helper.from_array(np.array(output_scale, dtype=np.float32), name="logits.scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="logits.zero_point"),
        numpy_helper.from_array(weight_q, name="fc.weight.q"),
        numpy_helper.from_array(bias, name="fc.bias"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "input.scale", "input.zero_point"],
            ["input.q"],
            name="input_quant",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input.q", "input.scale", "input.zero_point"],
            ["input.dq"],
            name="input_dequant",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["fc.weight.q", "fc.weight.scale", "fc.weight.zero_point"],
            ["fc.weight.dq"],
            name="weight_dequant",
        ),
        helper.make_node(
            "Gemm",
            ["input.dq", "fc.weight.dq", "fc.bias"],
            ["logits.raw"],
            name="fc",
            transB=1,
        ),
        helper.make_node(
            "QuantizeLinear",
            ["logits.raw", "logits.scale", "logits.zero_point"],
            ["logits.q"],
            name="output_quant",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["logits.q", "logits.scale", "logits.zero_point"],
            ["logits"],
            name="output_dequant",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "is_over_10_int8_qdq",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])],
        initializers,
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="nanoc-nn-example",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx_model.ir_version = 10
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path)


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
    int8_onnx_path = root / "outputs" / "onnx" / "is_over_10.int8.onnx"
    export_quantized_onnx(model, train_values, int8_onnx_path)

    print(f"train_accuracy={train_acc:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"onnx={onnx_path}")
    print(f"int8_onnx={int8_onnx_path}")


if __name__ == "__main__":
    main()

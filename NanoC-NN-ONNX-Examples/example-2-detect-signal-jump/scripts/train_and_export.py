from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import onnx
import torch
from network import SignalJumpNet
from onnx import TensorProto, helper, numpy_helper
from torch import nn

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
    before_noise = torch.randint(
        -1, 2, (jump_index,), generator=generator, dtype=torch.int64
    ).float()
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
    conv1 = model.features[0]
    conv2 = model.features[2]
    classifier = model.classifier
    initializers = [
        numpy_helper.from_array(
            conv1.weight.detach().cpu().numpy().astype(np.float32),
            name="conv1.weight",
        ),
        numpy_helper.from_array(
            conv1.bias.detach().cpu().numpy().astype(np.float32),
            name="conv1.bias",
        ),
        numpy_helper.from_array(
            conv2.weight.detach().cpu().numpy().astype(np.float32),
            name="conv2.weight",
        ),
        numpy_helper.from_array(
            conv2.bias.detach().cpu().numpy().astype(np.float32),
            name="conv2.bias",
        ),
        numpy_helper.from_array(
            classifier.weight.detach().cpu().numpy().astype(np.float32),
            name="fc.weight",
        ),
        numpy_helper.from_array(
            classifier.bias.detach().cpu().numpy().astype(np.float32),
            name="fc.bias",
        ),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["signal_window", "conv1.weight", "conv1.bias"],
            ["conv1.out"],
            name="conv1",
            kernel_shape=[2],
            strides=[1],
            pads=[0, 0],
        ),
        helper.make_node("Relu", ["conv1.out"], ["relu1.out"], name="relu1"),
        helper.make_node(
            "Conv",
            ["relu1.out", "conv2.weight", "conv2.bias"],
            ["conv2.out"],
            name="conv2",
            kernel_shape=[2],
            strides=[1],
            pads=[0, 0],
        ),
        helper.make_node("Relu", ["conv2.out"], ["relu2.out"], name="relu2"),
        helper.make_node("Flatten", ["relu2.out"], ["flat"], name="flatten", axis=1),
        helper.make_node(
            "Gemm",
            ["flat", "fc.weight", "fc.bias"],
            ["logits"],
            name="fc",
            transB=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "signal_jump_float",
        [helper.make_tensor_value_info("signal_window", TensorProto.FLOAT, [1, 1, WINDOW_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
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


def quantize_symmetric(array: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(array)))
    scale = max(max_abs / 127.0, 1.0e-8)
    quantized = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return quantized, scale


def activation_scale(array: np.ndarray) -> float:
    max_abs = float(np.max(np.abs(array)))
    return max(max_abs / 127.0, 1.0e-8)


def add_qdq(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    *,
    tensor_name: str,
    output_name: str,
    scale: float,
    prefix: str,
) -> str:
    scale_name = f"{prefix}.scale"
    zero_name = f"{prefix}.zero_point"
    quantized_name = f"{prefix}.q"
    initializers.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), name=scale_name))
    initializers.append(numpy_helper.from_array(np.array(0, dtype=np.int8), name=zero_name))
    nodes.append(
        helper.make_node(
            "QuantizeLinear",
            [tensor_name, scale_name, zero_name],
            [quantized_name],
            name=f"{prefix}_quant",
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            [quantized_name, scale_name, zero_name],
            [output_name],
            name=f"{prefix}_dequant",
        )
    )
    return output_name


def add_weight_dq(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    *,
    weight: np.ndarray,
    prefix: str,
) -> str:
    weight_q, scale = quantize_symmetric(weight.astype(np.float32))
    weight_q_name = f"{prefix}.q"
    scale_name = f"{prefix}.scale"
    zero_name = f"{prefix}.zero_point"
    output_name = f"{prefix}.dq"
    initializers.extend(
        [
            numpy_helper.from_array(weight_q, name=weight_q_name),
            numpy_helper.from_array(np.array(scale, dtype=np.float32), name=scale_name),
            numpy_helper.from_array(np.array(0, dtype=np.int8), name=zero_name),
        ]
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            [weight_q_name, scale_name, zero_name],
            [output_name],
            name=f"{prefix}_dequant",
        )
    )
    return output_name


def calibration_tensors(model: SignalJumpNet, values: torch.Tensor) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        conv1 = model.features[0](values)
        relu1 = model.features[1](conv1)
        conv2 = model.features[2](relu1)
        relu2 = model.features[3](conv2)
        flat = model.features[4](relu2)
        logits = model.classifier(flat)
    return {
        "input": values.detach().cpu().numpy(),
        "conv1": conv1.detach().cpu().numpy(),
        "relu1": relu1.detach().cpu().numpy(),
        "conv2": conv2.detach().cpu().numpy(),
        "relu2": relu2.detach().cpu().numpy(),
        "flat": flat.detach().cpu().numpy(),
        "logits": logits.detach().cpu().numpy(),
    }


def export_quantized_onnx(
    model: SignalJumpNet,
    calibration_values: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    calib = calibration_tensors(model, calibration_values)

    conv1 = model.features[0]
    conv2 = model.features[2]
    classifier = model.classifier
    initializers: list[onnx.TensorProto] = [
        numpy_helper.from_array(
            conv1.bias.detach().cpu().numpy().astype(np.float32),
            name="conv1.bias",
        ),
        numpy_helper.from_array(
            conv2.bias.detach().cpu().numpy().astype(np.float32),
            name="conv2.bias",
        ),
        numpy_helper.from_array(
            classifier.bias.detach().cpu().numpy().astype(np.float32),
            name="fc.bias",
        ),
    ]
    nodes: list[onnx.NodeProto] = []

    input_dq = add_qdq(
        nodes,
        initializers,
        tensor_name="signal_window",
        output_name="signal_window.dq",
        scale=activation_scale(calib["input"]),
        prefix="input",
    )
    conv1_weight = add_weight_dq(
        nodes,
        initializers,
        weight=conv1.weight.detach().cpu().numpy(),
        prefix="conv1.weight",
    )
    nodes.append(
        helper.make_node(
            "Conv",
            [input_dq, conv1_weight, "conv1.bias"],
            ["conv1.raw"],
            name="conv1",
            kernel_shape=[2],
            strides=[1],
            pads=[0, 0],
            dilations=[1],
            group=1,
        )
    )
    relu1_in = add_qdq(
        nodes,
        initializers,
        tensor_name="conv1.raw",
        output_name="conv1.dq",
        scale=activation_scale(calib["conv1"]),
        prefix="conv1.output",
    )
    nodes.append(helper.make_node("Relu", [relu1_in], ["relu1.raw"], name="relu1"))
    conv2_in = add_qdq(
        nodes,
        initializers,
        tensor_name="relu1.raw",
        output_name="relu1.dq",
        scale=activation_scale(calib["relu1"]),
        prefix="relu1.output",
    )
    conv2_weight = add_weight_dq(
        nodes,
        initializers,
        weight=conv2.weight.detach().cpu().numpy(),
        prefix="conv2.weight",
    )
    nodes.append(
        helper.make_node(
            "Conv",
            [conv2_in, conv2_weight, "conv2.bias"],
            ["conv2.raw"],
            name="conv2",
            kernel_shape=[2],
            strides=[1],
            pads=[0, 0],
            dilations=[1],
            group=1,
        )
    )
    relu2_in = add_qdq(
        nodes,
        initializers,
        tensor_name="conv2.raw",
        output_name="conv2.dq",
        scale=activation_scale(calib["conv2"]),
        prefix="conv2.output",
    )
    nodes.append(helper.make_node("Relu", [relu2_in], ["relu2.raw"], name="relu2"))
    flatten_in = add_qdq(
        nodes,
        initializers,
        tensor_name="relu2.raw",
        output_name="relu2.dq",
        scale=activation_scale(calib["relu2"]),
        prefix="relu2.output",
    )
    nodes.append(helper.make_node("Flatten", [flatten_in], ["flat.raw"], name="flatten", axis=1))
    fc_in = add_qdq(
        nodes,
        initializers,
        tensor_name="flat.raw",
        output_name="flat.dq",
        scale=activation_scale(calib["flat"]),
        prefix="flat.output",
    )
    fc_weight = add_weight_dq(
        nodes,
        initializers,
        weight=classifier.weight.detach().cpu().numpy(),
        prefix="fc.weight",
    )
    nodes.append(
        helper.make_node(
            "Gemm",
            [fc_in, fc_weight, "fc.bias"],
            ["logits.raw"],
            name="fc",
            transB=1,
        )
    )
    add_qdq(
        nodes,
        initializers,
        tensor_name="logits.raw",
        output_name="logits",
        scale=activation_scale(calib["logits"]),
        prefix="logits",
    )

    graph = helper.make_graph(
        nodes,
        "signal_jump_int8_qdq",
        [helper.make_tensor_value_info("signal_window", TensorProto.FLOAT, [1, 1, WINDOW_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
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
    int8_onnx_path = root / "outputs" / "onnx" / "signal_jump.int8.onnx"
    export_quantized_onnx(model, train_values, int8_onnx_path)

    print(f"train_accuracy={train_acc:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"onnx={onnx_path}")
    print(f"int8_onnx={int8_onnx_path}")


if __name__ == "__main__":
    main()

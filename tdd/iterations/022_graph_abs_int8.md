# 022: ONNX Abs QDQ/int8 图谱驱动覆盖

## 背景

本轮按新的 TDD 范式推进 `Abs`：先从 ONNX 官方算子图谱确认 `Abs` 是需要覆盖的基础逐元素算子，再新增 support matrix 行、最小用例、数据集和预期，最后根据失败面补齐 converter/codegen。

这次不从完整网络盲修，而是把需求落到一个可复现、可回归的最小能力点：静态 QDQ/int8 tensor，输入输出使用相同 scale/zero_point，验证 ONNX Runtime 与生成 C 的逐元素绝对值结果一致。

## 新增能力

- support id: `ONNX_ABS_QDQ_INT8`
- case: `ABS_001`
- dataset: `abs_qdq_smoke`
- ONNX op: official `Abs`
- backend: `cmsis-nn`
- lowering: `generated_c_abs_s8`

CMSIS-NN 当前没有专用 `s8 Abs` kernel，因此 codegen 采用生成 C99 逐元素循环的方式实现。该路径仍属于 CMSIS-NN 工程生成链路的一部分，但不会虚假声明调用了不存在的 CMSIS API。

## 实现要点

- converter 将 `Abs` 纳入 supported op 和 int8 contract。
- converter 在 QDQ 场景下提取输入/输出量化信息，当前要求输入输出 `scale` 和 `zero_point` 完全一致。
- converter 为 `Abs` 记录 `activation_min`、`activation_max` 和静态 `block_size`。
- codegen registry 将 `Abs` 映射到 `generated_c_abs_s8`。
- codegen runtime layer 新增 `abs` 层，生成 C99 逐元素绝对值循环。
- `int8` 最小值 `-128` 的绝对值无法在 int8 正区间表达，当前显式 clamp 到 `127`，避免溢出回绕。
- 数值测试使用正负混合输入和 negative top1 样本，验证输出逐元素值与 ONNX 一致。

## 验证结果

```text
python tdd/scripts/run_tests.py --case ABS_001 --generate
PASS: ABS_001, actual=ok, api=OK, cc=OK, run=OK

PYTHONPATH=src python tdd/scripts/run_numeric_tests.py --case ABS_001 --generate
PASS: ABS_001, top1=2/2, sat=0.00, max_abs=0.0, cc=OK, run=OK

python tdd/scripts/run_tests.py --mode target --generate
PASS: 37/37

python tdd/scripts/run_regression.py --generate
PASS: baseline 19/19 + numeric 15/15
```

## 当前边界

本轮只声明以下白名单能力：

- 静态 shape。
- QDQ/int8 tensor。
- 输入和输出量化参数相同。
- 输出仍为 int8。
- `-128` 绝对值按 int8 可表示范围 clamp 到 `127`。

以下情况尚未声明支持，需要后续继续由 TDD 补 case：

- 输入输出 scale/zero_point 不同的 `Abs`。
- 动态 shape。
- float `Abs` 到 CMSIS-NN 生成链路。
- 与 `Abs` 相邻的广播、layout transform 或多分支拓扑组合。

## 对新范式的评价

这轮验证了“官方 ONNX 算子图谱 + support matrix + case registry + 数值数据集”的推进方式是可用的。`Abs` 的实现路径很短，但它暴露了一个重要规则：并不是每个官方 ONNX 算子都有对应 CMSIS-NN API，图谱中需要允许 `generated_c_*` 这种生成期 C 路径，同时必须在 support matrix 中如实标注，避免报告层面把它误写成 CMSIS kernel 调用。

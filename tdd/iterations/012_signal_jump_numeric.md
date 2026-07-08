# 迭代 012: Signal Jump 数值闭环

## 基本信息

- **日期**: 2026-07-06
- **前置迭代**: 011
- **触发来源**: 人类需求 / 将内部构造的 signal jump 网络推进到 ONNX-vs-C 数值一致

## 目标

将 `NET_005` 从结构 `ok` 推进到完整数值回归：同一组 signal jump 输入样本分别运行原始 ONNX 和生成 C 代码，要求 top1 和标签准确率接近一致，并将通过结果沉淀进能力集。

## 新增/修改的测试用例

- `NET_005` 注册 `NumericCheck`，数据集为 `net_005_signal_jump_smoke`。
- `NET_005` 加入稳定回归结构 baseline。
- `run_numeric_tests.py` 支持非 MNIST 数据集的 `values` 输入和 `input_shape` reshape。

## 执行结果

首次执行 `NET_005` 数值测试失败：

```text
NET_005: top1=1/3, label_acc=0.33, sat=0.00, max_abs=110.8
```

失败说明生成 C 可以编译运行，但 Conv1d -> Flatten -> FC 的数值语义与 ONNX 不一致。

进一步定位后发现两个问题：

1. `Flatten` 只支持 4D NCHW 场景，未覆盖 Conv1d 的 3D NCW 场景。CMSIS runtime buffer 是 NHWC-like 顺序，直接喂给 FC 会破坏 ONNX flatten 顺序。
2. `Relu` 被作为 alias 折叠，但 Conv/Depthwise 的 CMSIS activation range 没有同步 clamp 到输出 zero point，导致负激活继续流入后续层。

## 代码修改

- `src/nanoc_nn/codegen/generator.py`
  - 扩展 flatten layout transform，支持 3D `NCW -> NWC runtime buffer -> NCW flatten`。
  - 当 Conv/Depthwise 输出经 Q/DQ 流入 `Relu` 时，将 `Relu` 折叠为 CMSIS activation min clamp。
- `tdd/scripts/run_numeric_tests.py`
  - 支持 dataset 中的 `values` 和 `input_shape`，让非 MNIST 数值用例可复用同一 runner。
- `tdd/scripts/cases_registry.py`
  - 将 `NET_005` 注册为 numeric regression。
- `tdd/cases/networks/NET_005.md`
  - 更新数值验收结论和本轮修复保护点。

## 最终结果

```text
NET_005 numeric: top1=3/3, label_acc=1.00, sat=0.00, max_abs=0.0
run_numeric_tests.py --generate: 2/2 PASS
run_tests.py --mode target --generate: 23/23 PASS
run_regression.py --generate: PASS
  structural baseline: 4/4 PASS
  numeric: 2/2 PASS
python -m pytest: 19/19 PASS
```

## 发现与后续

- `NET_005` 是 MNIST 之后第二个完整 ONNX-vs-C 数值闭环。
- 本轮证明 Conv1d + Relu + Flatten + FC 的内部构造小型时序网络可以正确生成并数值一致。
- 后续应把本轮暴露的两个问题拆成独立最小数值用例：
  - Conv1d NCW flatten layout。
  - Conv/Relu Q/DQ activation range folding。
- 下一轮可优先推进 `NET_004` KWS-style 的数值数据集，继续覆盖 depthwise separable conv。

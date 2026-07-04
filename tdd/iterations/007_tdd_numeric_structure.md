# 迭代 007: TDD 目录重构与数值验收入口

## 基本信息

- **日期**: 2026-07-04
- **前置迭代**: 006
- **触发来源**: 人类需求 / `results/tmp` 误删人工 demo，以及 MNIST 完整网络需要 ONNX-vs-C 精度对比

## 目标

重构 TDD 测试资产边界，支持两类新需求：

1. 测试覆盖“原始 ONNX 推理”与“生成 C 代码推理”的数值对比。
2. 测试覆盖完整神经网络，而不是只覆盖单算子或小拓扑。

## 目录规则调整

新增 `tdd/STRUCTURE.md`，明确：

- `tdd/cases/`: 永久需求规格。
- `tdd/fixtures/`: 永久测试资产，包括真实 ONNX、数据集和期望输出。
- `tdd/results/`: 测试报告，不放人工数据集。
- `tdd/work/`: runner 临时工作区，可删除，不进入版本管理。

`run_tests.py` 与 `generate_models.py` 已迁移到：

```text
tdd/work/models/
tdd/work/structural/<case_id>/
```

runner 清理目录前必须看到 `.nanoc_tdd_workdir` marker，避免误删人工资产。

## 数值测试入口

新增：

```bash
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --generate
```

数值测试流程：

1. 生成或定位 ONNX 测试模型。
2. 执行 NanoC-NN pipeline。
3. 使用 ONNX Runtime 跑原始 ONNX。
4. 编译生成的 C 代码，并启用真实 CMSIS-NN。
5. 对同一组输入比较 C 输出和 ONNX 输出。

当前比较指标：

- top1 一致率。
- C int8 输出饱和值比例。
- 最大绝对误差。

## 新增 fixture

新增 `tdd/fixtures/datasets/mnist_synthetic_smoke/`。

这是一个最小持久数据集，用来验证完整 MNIST 网络的 ONNX-vs-C 一致性。它不宣称真实 MNIST 准确率，只用于稳定触发完整网络数值路径。

## 当前验证结果

结构验收：

```bash
python tdd/scripts/run_tests.py --mode target --generate
```

结果：`17/17 PASS`。

数值验收：

```bash
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --generate
```

结果：`FAIL`。

```text
top1=1/3
saturation_ratio=0.97
max_abs_error=9768.7
```

这说明新的数值测试已经能抓住 MNIST 生成 C 后的饱和问题。当前 `TOPO_003` 只能称为结构通过，不能称为完整网络数值正确。

## 下一步

下一轮开发应以 `TOPO_003` 数值失败为任务单，逐层 dump 对比：

- Conv1 output
- Pool1 output
- Conv2 output
- Pool2 output
- Reshape/FC input
- QLinearMatMul output
- QLinearAdd output

优先排查 NHWC 中间内存进入 NCHW 语义 Flatten/FC 时的布局错位。

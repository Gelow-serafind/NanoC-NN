# 迭代 003: MNIST QLinear 能力固化

## 基本信息

- **日期**: 2026-07-04
- **前置迭代**: 002
- **触发来源**: 人类需求 / 真实 MNIST int8 模型能力固化

## 目标

把真实 MNIST int8 ONNX 暴露出的能力要求纳入 TDD，而不是停留在一次手工转换结论中。该能力代表当前低算力 MCU 场景中最小但完整的图像分类量化网络链路。

## 新增/修改的测试用例

- 新增 `TOPO_003`: 真实 MNIST QLinear int8 分类链路。
- 新增 fixture: `tdd/fixtures/onnx/mnist-12-int8.onnx`。
- `TOPO_003` 纳入 regression，作为稳定能力回归门禁。

该用例覆盖：

- `QLinearConv` -> `arm_convolve_wrapper_s8`
- `MaxPool` -> `arm_max_pool_s8`
- `QLinearMatMul` -> `arm_fully_connected_s8`
- `QLinearAdd` -> `arm_elementwise_add_s8`
- ONNX uint8 量化域到 int8 runtime 的转换
- QLinearMatMul 到 CMSIS-NN FC 的权重方向处理

## 执行结果

- 规格校验：通过，当前 TDD 用例总数为 14。
- 单用例执行：`python tdd/scripts/run_tests.py --case TOPO_003 --generate`。
- 结果：FAIL。
- 报告层状态：`actual=ok`。
- 生成物验收：失败，`model.c` 缺少必需的 CMSIS-NN API：
  - `arm_convolve_wrapper_s8`
  - `arm_max_pool_s8`
  - `arm_fully_connected_s8`
  - `arm_elementwise_add_s8`

该失败符合迭代 002 的“生成物优先”规则：报告 `ok` 但生成物没有真实推理路径，应判定为失败。

## 代码修改

- `tdd/scripts/cases_registry.py`: 注册 `TOPO_003`，声明预期状态和必须出现的 CMSIS-NN API。
- `tdd/scripts/generate_models.py`: 新增 `gen_topo_003()`，从 fixture 生成测试模型。
- `tdd/cases/topology/TOPO_003.md`: 新增用例规格。
- `tdd/fixtures/onnx/mnist-12-int8.onnx`: 固化真实模型输入。

## 最终结果

MNIST QLinear 能力已经纳入 TDD 能力集，但当前实现尚未通过该用例。下一轮修复应以 `TOPO_003` 为入口，直到生成物包含真实 CMSIS-NN 调用并通过 C99 smoke compile。

## 发现与后续

- 真实模型用例应优先沉淀为 fixture 或可复现生成脚本，避免依赖 `tmp/` 目录。
- 小型真实模型比人为合成图更容易暴露 ONNX exporter 的实际图结构差异。
- 该用例同时继承迭代 002 的生成物优先规则：报告 `ok` 后仍必须检查真实调用和 C99 smoke compile。

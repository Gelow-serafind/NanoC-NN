# 迭代 011: 真实嵌入式网络覆盖面扩展

## 基本信息

- **日期**: 2026-07-06
- **前置迭代**: 010
- **触发来源**: 人类需求 / UI 合并后继续扩大低算力嵌入式网络测试覆盖面

## 目标

在 MNIST 数值闭环稳定后，引入更多用户可能实际输入的 int8 ONNX 网络。目标不是一次性修复所有网络，而是先把真实网络固定成 TDD 资产，让生成器稳定分类为 `ok`、`blocked`、`unsupported` 或 `oversize`，防止假阳性。

## 执行步骤

1. 先执行合并后的全量 TDD 门禁。
2. 获取或生成 5 个新增完整网络 fixture。
3. 为每个网络新增 case、dataset 位置、manifest 记录和 registry 条目。
4. 执行网络分类测试，根据真实 pipeline 结果调整 expected。
5. 执行全量 target、稳定 regression 和 pytest。

## 新增/修改的测试用例

| 用例 | 网络 | 来源 | 当前结论 |
|------|------|------|----------|
| `NET_002` | MobileNetV2 int8/QLinear | ONNX Model Zoo / Hugging Face | `ok` 结构生成通过 |
| `NET_003` | SSD-MobileNet int8 | ONNX Model Zoo / Hugging Face | `unsupported` 正确拒绝 |
| `NET_004` | KWS DS-CNN-style int8 | 内部探索生成 fixture | `ok` 结构生成通过 |
| `NET_005` | Tiny signal jump int8 | `NanoC-NN-ONNX-Examples` 示例导出 | `ok` 结构生成通过 |
| `NET_006` | EfficientNet-Lite4 int8 | ONNX Model Zoo / Hugging Face | `unsupported` 正确拒绝 |

## 执行结果

`NET_002` 首次预期设为 `blocked`，实际 pipeline 返回 `ok`。按生成物优先原则检查后，`model.c` 中存在真实 CMSIS-NN 运行路径，包括：

- `arm_convolve_wrapper_s8`
- `arm_depthwise_conv_wrapper_s8`
- `arm_elementwise_add_s8`
- `arm_avgpool_s8`
- `arm_fully_connected_per_channel_s8`

因此本轮将 `NET_002` 提升为结构 `ok`。但它尚未做 ONNX-vs-C 数值对比，不能宣称 ImageNet 分类准确率已经验证。

`NET_004` 和 `NET_005` 也通过了生成物检查，分别覆盖 KWS-style depthwise separable conv 和时序 Conv1d 分类链路。

`NET_003` 与 `NET_006` 返回 `unsupported` 是正确行为：

- `NET_003` 包含检测后处理、多输出、动态 shape、`Loop`、`Split`、`Squeeze`、`Sigmoid` 等当前不支持能力。
- `NET_006` 暴露 `Transpose`、`QLinearAveragePool` 和 `Squeeze` 缺口。

## 代码修改

- `tdd/scripts/cases_registry.py`: 注册 `NET_002~NET_006`，并为结构 `ok` 网络增加 required CMSIS-NN API。
- `tdd/scripts/generate_models.py`: 增加真实网络 fixture 复制函数；修复 `--category networks` 只能按前缀匹配的问题，改为按 registry category 匹配。
- `tdd/cases/networks/NET_002.md` ~ `NET_006.md`: 新增完整网络规格。
- `tdd/fixtures/onnx/`: 新增 5 个 ONNX fixture。
- `tdd/fixtures/datasets/`: 为新增网络建立 smoke dataset 位置。
- `tdd/fixtures/onnx/MANIFEST.md`: 记录来源、SHA256 和首次导入结论。
- `tdd/STATUS.md`: 更新当前能力集、真实网络状态和下一步方向。

## 最终结果

```text
run_tests.py --mode target --generate: 23/23 PASS
run_regression.py --generate: PASS
  baseline structural: 3/3 PASS
  numeric: TOPO_003 top1=10/10, sat=0.00
python -m pytest: 19/19 PASS
```

## 发现与后续

1. 将 `NET_005` 接入 ONNX-vs-C 数值验收，验证 Conv1d 小网络准确率。
2. 为 `NET_004` 训练或引入真实 KWS 权重，再接入频谱数据集数值验收。
3. 从 `NET_003` 提炼多输出、动态 shape、`Loop` 正确拒绝最小用例。
4. 从 `NET_006` 提炼 `QLinearAveragePool`、`Squeeze`、NHWC `Transpose` 最小用例。
5. 为 `NET_002` 增加显式平台预算 probe，验证具体 Cortex-M 目标上按 SRAM/Flash 返回 `oversize`。

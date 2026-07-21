# 021 - 图谱驱动支持官方 MatMul FC 形态

日期：2026-07-21

## 背景

本轮继续按 ONNX 官方算子图谱推进 TDD。上一轮后 `ONNX_MATMUL_QDQ_INT8`
仍在 support matrix 中标记为 planned/blocked：mapper 已经有 FC lowering 意图，
但 converter 对量化 `MatMul` 明确拒绝，缺少独立最小用例保护。

## 新增用例

- `MATMUL_001`：官方 `MatMul`，QDQ/int8，静态 FC-compatible 形态。
- ONNX 形态：`A[1,4] x B[4,3] -> Y[1,3]`。
- 数据集：`tdd/fixtures/datasets/matmul_qdq_smoke/dataset.json`。
- 数值阈值：top1 一致率 1.0，最大绝对误差 0.08，饱和率不超过 0.25。

## 修复内容

- support matrix 将 `ONNX_MATMUL_QDQ_INT8` 从 planned/blocked 晋升为 `ok`。
- converter 支持 `MatMul` 的 `[1,I] x [I,O]` 静态常量权重形态。
- converter 在量化信息提取阶段把 ONNX `MatMul` 权重 `B[I,O]` 转为 CMSIS-NN FC 所需的 `[O,I]` 权重布局，并记录 `layout_transform`。
- codegen 复用现有 FC renderer，生成 `arm_fully_connected_s8`。

## 验证结果

- `python tdd/scripts/run_tests.py --case MATMUL_001 --generate`：PASS
- `python tdd/scripts/run_numeric_tests.py --case MATMUL_001 --generate`：PASS，`top1=2/2`，`max_abs=0.0`
- `python tdd/scripts/run_tests.py --mode target --generate`：36/36 PASS
- `python tdd/scripts/run_numeric_tests.py --generate`：14/14 PASS
- `python tdd/scripts/run_regression.py --generate --numeric-python /Users/tiedan/anaconda3/envs/tiedanPython/bin/python --numeric-pythonpath src`：PASS，baseline 18/18，numeric 14/14
- `python tdd/scripts/generate_support_map.py`：已刷新 `tdd/reports/onnx_support_map.html`

## 当前结论

官方 `MatMul` 的第一个可交付子形态已经进入能力集。当前确认范围限定为单 batch 2-D activation、
rank=2 静态常量权重、QDQ/int8 的分类头/FC 形态。batched MatMul、双 runtime 输入矩阵乘、
高维 broadcast MatMul、动态 shape 和 per-channel MatMul 权重 scale 仍需要后续独立 case。

## 对新范式的评价

这一轮新范式工作顺畅：图谱先暴露 `MatMul` 仍 planned，TDD case 先把它变红，
失败报告定位到 converter 的显式拒绝，再通过最小改动将权重布局转换补入前端量化信息，
最后由全量结构、数值和稳定回归确认能力增长。

# TDD 状态总览

> 本文件是 agent 进入 `tdd/` 目录后的第一个入口。阅读此文件即可了解当前能力边界、最近验证结果和下一步工作方向。

## 当前结论

NanoC-NN 当前已经进入“真实完整网络驱动”的 TDD 阶段。

截至 2026-07-05：

- target 结构回归：`18/18 PASS`
- 稳定回归入口：`python tdd/scripts/run_regression.py --generate` 已通过
- 数值回归：`TOPO_003` MNIST ONNX-vs-C `top1=10/10`，饱和率 `0.00`
- 新增真实网络导入测试：`NET_001` SqueezeNet 1.0 int8，当前正确 `blocked`
- 平台预算拦截已从 `blocked` 拆出为 `oversize`

## 状态语义

| 状态 | 含义 |
|------|------|
| `ok` | converter/codegen 语义通过，生成物包含当前支持范围内的真实 CMSIS-NN 调用 |
| `blocked` | 算子、量化字段、布局、renderer 或生成语义尚不完整 |
| `unsupported` | converter 或 codegen 明确不支持该 ONNX 形态 |
| `oversize` | 代码生成语义已通过，但显式传入的目标 SRAM/Flash 预算不满足 |

常规 ONNX 能力测试以代码生成完整性和数值正确性为主，不默认使用某个 MCU 的 SRAM/Flash 预算作为能力门槛。目标平台预算只在显式传入 `--sram-budget` / `--flash-budget` 时拦截，并返回 `oversize`。

## 能力集摘要

| 指标 | 当前值 |
|------|--------|
| 测试用例总数 | 18 |
| target 通过 | 18 |
| target 失败 | 0 |
| 稳定结构回归 | 3/3 PASS |
| 数值回归 | 1/1 PASS |
| 当前能力集文件 | `tdd/CAPABILITIES.md` |
| 最新迭代记录 | `tdd/iterations/010_post_mnist_regression_plan.md` |

## 已确认主线能力

| 能力 | 代表用例 | 当前结论 |
|------|----------|----------|
| Conv / QLinearConv CMSIS-NN 生成 | `CONV_001~004`, `QLINEAR_NUM_001~003` | PASS |
| Gemm / MatMul / FC 生成 | `GEMM_001~004` | PASS |
| MaxPool int8 生成 | `MAXPOOL_001` | PASS |
| Softmax int8 生成 | `SOFTMAX_001` | PASS |
| 组合拓扑生成 | `TOPO_001`, `TOPO_002` | PASS |
| 真实 MNIST QLinear int8 完整网络 | `TOPO_003` | 结构 PASS + 数值 PASS |
| float32 无 Q/DQ 模型拒绝 | `NEG_001` | PASS |
| 真实 SqueezeNet int8 导入边界 | `NET_001` | 正确 blocked |

## 当前真实网络

### TOPO_003: MNIST int8

`TOPO_003` 是当前第一个完整打通的真实网络：

- ONNX fixture: `tdd/fixtures/onnx/mnist-12-int8.onnx`
- 数据集：
  - `tdd/fixtures/datasets/mnist_synthetic_smoke/`
  - `tdd/fixtures/datasets/mnist_hand_drawn/`
- 数值验收：ONNX Runtime 与生成 C 并行推理一致
- 关键修复：`auto_pad`、NHWC 到 NCHW flatten、per-channel FC、`QLinearAdd left_shift=0`
- 复盘文档：`tdd/cases/topology/TOPO_003_mnist/CODEGEN_CASE_STUDY.md`

### NET_001: SqueezeNet 1.0 int8

`NET_001` 是 MNIST 之后第一个真实外部 ONNX 导入测试：

- ONNX fixture: `tdd/fixtures/onnx/squeezenet1.0-12-int8.onnx`
- 来源记录：`tdd/fixtures/onnx/MANIFEST.md`
- 当前预期：`blocked`
- 当前意义：确认真实网络能被导入、解析、生成报告，并且不会被误判为可交付 `ok`

`NET_001` 暴露出的下一批真实缺口：

- Q/DQ 与 float-domain `Concat` 交错时的布局和量化处理
- `QLinearGlobalAveragePool` 量化字段提取与生成闭环
- float output 侧 `Softmax` 的折叠、拒绝或 int8 生成策略
- 平台预算与代码生成能力的分层判断

## 推荐命令

```bash
# 校验 TDD 用例规格
python tdd/scripts/run_tests.py --validate-only

# 稳定回归：baseline 结构验收 + 已登记 numeric 验收
python tdd/scripts/run_regression.py --generate

# 全量 target：更新 CAPABILITIES.md + latest.json
python tdd/scripts/run_tests.py --mode target --generate

# MNIST 数值验收
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_synthetic_smoke --generate

# 手写 MNIST 数据集采集 UI
python tdd/tools/mnist_capture/server.py
```

## 下一步方向

优先按真实模型暴露的问题继续 TDD 收敛：

1. 从 `NET_001` 提炼最小用例，而不是直接盲修完整 SqueezeNet。
2. 为 Q/DQ + `Concat` 增加独立最小结构/数值用例。
3. 为 `QLinearGlobalAveragePool` 增加独立最小用例。
4. 明确 float 侧 `Softmax` 的处理策略。
5. 增加逐层 dump 工具，用于 ONNX 节点输出与 C 中间 buffer 对比。
6. 继续导入 keyword spotting、tiny anomaly detection、简单 IMU 分类等 MCU 常见小模型。

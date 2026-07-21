# 023: 图谱驱动批量覆盖 AveragePool / Reshape / Squeeze / Add / Mul / Transpose

## 背景

本轮按 ONNX 官方支持图谱一次性推进 6 个 MCU 常见算子，而不是一个个孤立迭代。目标是验证新的图谱驱动 TDD 能否支撑批量闭环：先补 support matrix 和 case，再运行失败测试，最后按失败面修 converter/codegen，并用 ONNX-vs-C 数值测试验收。

## 新增能力

| 用例 | ONNX op | 第一批白名单形态 | lowering |
|------|---------|------------------|----------|
| `AVGPOOL_003` | `AveragePool` | rank=4 NCHW, QDQ/int8, 2x2 stride=2, no padding | `arm_avgpool_s8` |
| `RESHAPE_001` | `Reshape` | static shape, same element count, linear storage order | `generation_time_shape_alias_or_layout_copy` |
| `SQUEEZE_001` | `Squeeze` | static axes, remove size=1 dim, linear storage order | `generation_time_shape_alias_or_layout_copy` |
| `ADD_001` | `Add` | same-shape QDQ/int8, constant second input | `arm_elementwise_add_s8` |
| `MUL_001` | `Mul` | same-shape QDQ/int8, constant second input | `arm_elementwise_mul_s8` |
| `TRANSPOSE_001` | `Transpose` | rank=4 QDQ/int8, explicit `perm=[0,2,3,1]` | `arm_transpose_s8` |

## 失败面与修复

- `Squeeze` 初始被 converter 标记为 `unsupported`：补入 `SUPPORTED_OPS`、initializer role、folded mapping 和 tensor alias。
- `Add/Mul` 初始为 `blocked`：补 QDQ elementwise quant extraction、常量分支权重声明、CMSIS elementwise renderer。
- `Transpose` 初始为 `blocked`：补 QDQ transpose quant extraction、CMSIS transpose layer/renderer。
- 纯 `Transpose` 被 NCHW/NHWC 边界转置策略误伤的风险：将边界转置限制到 conv/depthwise/pool/concat 这类 NHWC runtime 图。
- `AveragePool/Add` 数值测试被 ONNX Runtime QLinear 融合要求 scalar scale/zp 卡住：将 TDD 通用 QDQ wrapper 改为 scalar scale/zp，并为 `AVGPOOL_003` 使用 scalar QDQ 参数。
- `Transpose` 数值 runner 缺少 CMSIS `TransposeFunctions`：补入 numeric runner 的 CMSIS 源集合。

## 验证结果

```text
python tdd/scripts/run_tests.py --mode target --generate
PASS: 43/43

PYTHONPATH=src python tdd/scripts/run_numeric_tests.py --generate
PASS: 21/21

python tdd/scripts/run_regression.py --generate
PASS: baseline 25/25 + numeric 21/21
```

## 当前边界

本轮只声明第一批白名单能力：

- `AveragePool`: rank=4 NCHW、2x2 stride=2、无 padding、QDQ/int8。
- `Reshape`: 静态 shape、元素数量不变、线性顺序不变。
- `Squeeze`: 显式 axes、仅去除 size=1 维。
- `Add/Mul`: 同形状、QDQ/int8、第二输入为量化常量。
- `Transpose`: rank=4、显式 perm、输入输出同量化参数。

未覆盖形态需要继续由 TDD 扩展：

- elementwise broadcast 和双动态输入。
- `AveragePool` padding、`ceil_mode`、`count_include_pad`、异量化。
- `Transpose` 缺省 reverse perm、rank=3/rank>4、与 conv 内部 NHWC buffer 相邻的复杂 layout。
- `Reshape/Squeeze` 动态 shape、`allowzero=1`、axes 缺省推断。

## 新范式评价

批量推进是可行的，但前提是每个算子仍然有独立最小 case 和独立数值数据集。本轮 6 个算子的失败面没有混在一起：测试报告清楚地区分了 converter unsupported、codegen renderer blocked、测试模型 QDQ 参数问题和 numeric runner 缺源问题。这说明“图谱作地图，TDD 作推进器”的范式已经可以支撑小批量扩张。

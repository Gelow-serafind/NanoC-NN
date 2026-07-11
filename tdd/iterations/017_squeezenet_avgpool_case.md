# 017: SqueezeNet QLinearGlobalAveragePool 最小用例

## 背景

本轮继续沿用 ONNX schema/support-map 驱动的 TDD 范式，从 `NET_001` SqueezeNet 的 blocked 边界中提炼最小可验证用例。

SqueezeNet 尾部包含：

```text
QLinearConv -> QLinearGlobalAveragePool -> DequantizeLinear -> Softmax
```

其中 `QLinearGlobalAveragePool` 过去只被 `NET_002` MobileNetV2 结构 PASS 间接覆盖，没有独立 core case。为了让能力图谱可增长、可追踪，本轮将该节点提炼为 `AVGPOOL_001`。

## 关键发现

`QLinearGlobalAveragePool` 不属于当前绑定的 ONNX 1.22.0 官方 schema catalog。真实 SqueezeNet 与 MobileNetV2 fixture 中该节点均位于 Microsoft 扩展域：

```text
domain = com.microsoft
opset = 1
op_type = QLinearGlobalAveragePool
```

因此它不能计入官方 ONNX 算子支持率，只能作为真实模型观察扩展进入 support map。这一点对后续图谱很重要：官方 schema 进度和真实模型扩展能力必须分层展示。

## 新增测试资产

- `tdd/cases/core/avgpool/AVGPOOL_001.md`
- `tdd/scripts/cases_registry.py::AVGPOOL_001`
- `tdd/scripts/generate_models.py::gen_avgpool_001`
- `tdd/reports/onnx_support_map.html`

`AVGPOOL_001` 使用最小 rank=4 NCHW 输入 `[1,4,3,3]`，通过 `QuantizeLinear -> com.microsoft::QLinearGlobalAveragePool -> DequantizeLinear` 输出 `[1,4,1,1]`。

## 实现结论

这轮没有修改 `src/` 生成器源码。现有 converter/codegen 已经具备该形态的结构生成能力，但缺少独立用例保护。

本轮真正修正的是 TDD 能力表达：

- 将 `QLinearGlobalAveragePool` 从完整网络间接覆盖提升为 core 最小 case。
- 生成模型显式引入 `com.microsoft` opset，避免误放到默认 `ai.onnx` 域。
- support map 中该能力挂在“真实模型观察扩展”分支，而不是官方 ONNX 算子全集。

## 验证结果

单项结构测试：

```text
python tdd/scripts/run_tests.py --case AVGPOOL_001 --generate
AVGPOOL_001 expected=ok actual=ok api=OK cc=OK run=OK
```

全量 target：

```text
python tdd/scripts/run_tests.py --mode target --generate
30/30 PASS
```

稳定回归：

```text
python tdd/scripts/run_regression.py --generate
baseline structural 11/11 PASS
numeric 8/8 PASS
```

## 对 SqueezeNet 的影响

`NET_001` 当前仍然保持 `blocked`，这是正确状态。本轮只收敛了其中一个子能力：

- 已独立保护：`QLinearGlobalAveragePool`
- 已独立保护：SqueezeNet-style channel `Concat`
- 尚待处理：末端 float-domain `Softmax` 策略
- 尚待处理：完整 SqueezeNet 的 ONNX-vs-C 数值验收和逐层误差定位

## 当前边界

已确认：

- Microsoft 扩展 `QLinearGlobalAveragePool`
- static rank=4 NCHW
- 全局空间池化，kernel 覆盖完整 H/W
- CMSIS-NN `arm_avgpool_s8`
- 结构生成、C99 compile、host smoke run

尚未确认：

- ONNX-vs-C 数值验收
- 不同输入/输出量化参数下的误差边界
- 普通官方 `GlobalAveragePool` / `AveragePool` 的 int8 数值 case
- NHWC 原生输入
- 完整 `NET_001` SqueezeNet 交付级 `ok`

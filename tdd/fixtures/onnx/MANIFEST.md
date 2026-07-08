# ONNX Fixture Manifest

本文件记录 `tdd/fixtures/onnx/` 下真实 ONNX fixture 的来源。真实模型是永久测试资产，不能放入 `tdd/work/` 或 `tdd/results/`。

| 文件 | 来源 | License | SHA256 | 用途 |
|------|------|---------|--------|------|
| `mnist-12-int8.onnx` | ONNX Model Zoo MNIST int8 镜像 | 未在本仓库单独固化 | 未记录 | `TOPO_003` 真实 MNIST QLinear int8 分类链路 |
| `squeezenet1.0-12-int8.onnx` | ONNX Model Zoo / Hugging Face `onnxmodelzoo/squeezenet1.0-12-int8` | Apache-2.0 | `3da17dfad1b7ba23c93fac6dbf49f6db78cd42f7519e915a2e27d37c5c0a972b` | `NET_001` 真实 SqueezeNet int8 导入评估 |
| `mobilenetv2-12-int8.onnx` | ONNX Model Zoo / Hugging Face `onnxmodelzoo/mobilenetv2-12-int8` | 未在本仓库单独固化 | `cc028fe6cae7bc11a4ff53cfc9b79c920e8be65ce33a904ec3e2a8f66d77f95f` | `NET_002` MobileNetV2 int8/QLinear 结构生成评估 |
| `ssd_mobilenet_v1_12-int8.onnx` | ONNX Model Zoo / Hugging Face `onnxmodelzoo/ssd_mobilenet_v1_12-int8` | 未在本仓库单独固化 | `2b79e6a7fb1ec6a33f332b9b10d82d9de4b7b49dcd26b5946921bb356895c954` | `NET_003` SSD-MobileNet int8 检测网络导入评估 |
| `keyword_spotting_dscnn.int8.onnx` | 内部探索生成 fixture，DS-CNN-style KWS 结构 | 本仓库测试资产 | `f1336641cf46eeac04639f21781e8d284d2bf55453c1e6e9995d63fb19ed46d0` | `NET_004` KWS DS-CNN-style int8 结构生成评估 |
| `signal_jump.int8.onnx` | `NanoC-NN-ONNX-Examples/example-2-detect-signal-jump` 训练导出 | 本仓库示例资产 | `13b3c77c82e7a3a5a3f81c653ad87e2d364feaf239b02150bcf22e43d59959a8` | `NET_005` tiny signal jump int8 时序分类结构生成与数值回归 |
| `efficientnet-lite4-11-int8.onnx` | ONNX Model Zoo / Hugging Face `onnxmodelzoo/efficientnet-lite4-11-int8` | 未在本仓库单独固化 | `2b3cbb5077262b20df565dacddecb3724c0976c35029a87e512d13aa4eff04a2` | `NET_006` EfficientNet-Lite4 int8 导入评估 |

## NET_001 首次导入结论

`squeezenet1.0-12-int8.onnx` 模型信息：

- 输入：`float[1,3,224,224]`
- 输出：`float[1,1000,1,1]`
- 节点统计：`QLinearConv x26`、`DequantizeLinear x17`、`QuantizeLinear x9`、`Concat x8`、`MaxPool x3`、`QLinearGlobalAveragePool x1`、`Softmax x1`
- 首次 pipeline probe：`blocked`
- 主要边界：
  - `MaxPool` / `Softmax` 存在缺失量化字段的 codegen 阻塞。
  - 若传入 256K SRAM 预算，估算约 `2351272` bytes，会触发目标平台 `oversize` 门禁。

因此 `NET_001` 当前定义为导入边界测试：正确行为是明确 `blocked`，不是生成可交付 C 推理代码。

## NET_002~NET_006 首次导入结论

截至 2026-07-06，本轮真实网络扩展新增 5 个 fixture：

| 用例 | 模型 | 首次结论 | 主要意义 |
|------|------|----------|----------|
| `NET_002` | MobileNetV2 int8/QLinear | `ok` 结构生成通过 | 覆盖 depthwise conv、QLinearAdd、QLinearGlobalAveragePool、QLinearMatMul |
| `NET_003` | SSD-MobileNet int8 | `unsupported` 正确拒绝 | 覆盖多输出检测、动态 shape、Loop、后处理 |
| `NET_004` | KWS DS-CNN-style int8 | `ok` 结构生成通过 | 覆盖音频特征输入、depthwise separable conv、pool、FC |
| `NET_005` | Tiny signal jump int8 | `ok` 结构 + 数值回归通过 | 覆盖 Conv1d 时序分类、Flatten、FC |
| `NET_006` | EfficientNet-Lite4 int8 | `unsupported` 正确拒绝 | 覆盖 NHWC、QLinearAveragePool、Squeeze、QLinearMatMul 边界 |

`NET_002/004` 的 `ok` 结论仅代表结构验收：生成物包含真实 CMSIS-NN 调用、C99 smoke compile/run 通过。`NET_005` 已进一步接入 ONNX-vs-C 数值回归，当前 `top1=3/3`、饱和率 `0.00`、最大绝对误差 `0.0`。

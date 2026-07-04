# ONNX Fixture Manifest

本文件记录 `tdd/fixtures/onnx/` 下真实 ONNX fixture 的来源。真实模型是永久测试资产，不能放入 `tdd/work/` 或 `tdd/results/`。

| 文件 | 来源 | License | SHA256 | 用途 |
|------|------|---------|--------|------|
| `mnist-12-int8.onnx` | ONNX Model Zoo MNIST int8 镜像 | 未在本仓库单独固化 | 未记录 | `TOPO_003` 真实 MNIST QLinear int8 分类链路 |
| `squeezenet1.0-12-int8.onnx` | ONNX Model Zoo / Hugging Face `onnxmodelzoo/squeezenet1.0-12-int8` | Apache-2.0 | `3da17dfad1b7ba23c93fac6dbf49f6db78cd42f7519e915a2e27d37c5c0a972b` | `NET_001` 真实 SqueezeNet int8 导入评估 |

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

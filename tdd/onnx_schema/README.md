# ONNX Schema Catalog

本目录保存 NanoC-NN TDD 使用的 ONNX 官方算子全集快照。

当前快照：

- `onnx_1_22_0_schema_catalog.json`
- 绑定 Python package: `onnx==1.22.0`
- 默认 `ai.onnx` opset: `27`
- 当前 schema 数量: `226`
- history schema 数量: `639`

生成命令：

```bash
PYTHONPATH=tdd/work/python_deps:src python tdd/scripts/generate_onnx_schema_catalog.py
```

## 使用原则

- 官方 ONNX 算子全集以本目录 catalog 为准。
- `tdd/scripts/support_matrix.py` 中 `schema_source="official"` 的行必须能在当前 catalog 中找到。
- 真实模型中出现但当前 ONNX 官方 catalog 没有的 op，必须显式标记为 `schema_source="extension"`，不能冒充官方支持。
- 更换 `onnx` 版本后，必须重新生成 catalog，并通过 `python tdd/scripts/run_tests.py --validate-only`。

## TDD 推进方式

下一阶段开发从 ONNX 官方全集出发：

1. 在 catalog 中选择一个官方 op。
2. 在 support matrix 中新增或细化一个 schema 子形态。
3. 在 `tdd/cases/` 中新增最小 TDD case。
4. 先确认失败或 blocked。
5. 修改 converter/codegen。
6. 通过 case 后纳入回归。

## 新模型 blocked 后的分流

遇到真实模型 blocked/unsupported 时，不直接修改 codegen。必须先分流：

1. 查模型 `opset_import` 和 node 列表。
2. 对照本目录官方 catalog，确认每个官方 op 是否存在。
3. 对照 `tdd/scripts/support_matrix.py`：
   - op 没有 support 行：新增 planned/blocked support 行，再新增最小 TDD case。
   - op 已有 support 行但形态不匹配：细化该 op 的 schema 子形态，再新增最小 TDD case。
   - op 不在官方 catalog 但真实模型出现：登记为 `schema_source="extension"`，不得冒充官方支持。
4. 最小 case 先复现失败，再修改 converter/codegen。
5. 最小 case 与完整网络都通过后，能力才进入 `CAPABILITIES.md`。

这保证开发不是围绕偶然模型做特判，而是沿着 ONNX 官方全集和 NanoC-NN 支持矩阵逐步扩张。

# 020 - 图谱驱动补齐 GlobalAveragePool 与 Flatten

日期：2026-07-21

## 背景

本轮继续采用 ONNX 官方算子图谱 + TDD 的开发范式。目标不是从代码猜能力，而是从图谱中选择真实网络已经暴露、且能独立沉淀的官方算子子形态，构造最小用例并纳入结构与数值回归。

本轮补齐两个能力：

- `AVGPOOL_002`：官方 `GlobalAveragePool`，QDQ/int8，rank=4 NCHW，全局 H/W 池化。
- `FLATTEN_001`：官方 `Flatten`，QDQ/int8，`axis=1`，保持 ONNX row-major flatten 顺序。

## 修复内容

- support matrix 新增 `ONNX_GLOBALAVGPOOL_QDQ_INT8` 与 `ONNX_FLATTEN_QDQ_INT8`。
- case registry 新增 `AVGPOOL_002` 与 `FLATTEN_001`，均加入稳定回归与数值回归。
- 新增对应 ONNX 生成器与 smoke dataset。
- converter 修正 int8 contract：当模型没有算术 runtime 节点、但仅包含 Q/DQ 与布局折叠节点时，不再误判为 `blocked`。
- codegen 新增 layout-only copy 路径：当 `Flatten/Reshape/Q/DQ` 等折叠链路最终只是输入到输出的字节重排别名，生成 `memcpy` 路径，而不是空 trace。
- memory planner 修正无 activation candidate 时的双缓冲占位，避免无 runtime layer 模型触发 `activation_buffers[1]` 越界。

## 验证结果

- `python tdd/scripts/run_tests.py --case FLATTEN_001 --generate`：PASS
- `python tdd/scripts/run_numeric_tests.py --case FLATTEN_001 --generate`：PASS，`max_abs=0.0`
- `python tdd/scripts/run_tests.py --mode target --generate`：35/35 PASS
- `python tdd/scripts/run_numeric_tests.py --generate`：13/13 PASS
- `python tdd/scripts/run_regression.py --generate --numeric-python /Users/tiedan/anaconda3/envs/tiedanPython/bin/python --numeric-pythonpath src`：PASS，baseline 17/17，numeric 13/13
- `python tdd/scripts/generate_support_map.py`：已刷新 `tdd/reports/onnx_support_map.html`

## 当前结论

图谱驱动范式继续有效：从官方 ONNX schema 选择目标，先新增最小 case，再根据失败修 generator/converter，最后把 PASS 结果沉淀回能力集与可视化图谱。

`Flatten` 这类布局算子没有 CMSIS-NN 算术 kernel，但它仍然是代码生成器必须正确处理的 ONNX 官方算子。它的验收标准不是“调用某个 CMSIS 函数”，而是生成可运行 C 路径并保证 ONNX-vs-C 数值一致。

## 后续建议

下一轮优先推进 `MatMul` 独立 QDQ/int8 用例。当前完整网络里已经存在由 MatMul/FC lowering 受益的路径，但 support matrix 仍有 `ONNX_MATMUL_QDQ_INT8` 处于 `blocked`，适合作为下一次图谱驱动 TDD 的入口。

# CMSIS-NN int8 Codegen 单链路开发计划

## 1. 目标链路

`nanoc_nn.codegen` 当前只服务一条链路：

```text
converter-output/model_graph.json
    ↓
检查 int8 Q/DQ schema、shape、layout、量化参数和支持算子
    ↓
缺字段或缺 renderer 时终止并报告 blocked / unsupported
    ↓
通过后生成 CMSIS-NN int8 C 代码
```

codegen 不直接读取原始 ONNX。原始 ONNX 只能由 `nanoc_nn.converter` 解析；
codegen 的唯一输入是 converter 标准输出目录。

## 2. 当前代码状态

已完成：

- `generator.generate_project()` 能读取 converter 输出目录并生成 C 工程目录。
- `mapper.map_model()` 能判断 ONNX 节点到 CMSIS-NN s8 API 的映射状态。
- `quantization.analyze_quantization()` 能检查节点是否缺量化字段。
- 对 Q/DQ、per-tensor、`Gemm(transB=1)` 的 Fully Connected 路径，已经可以生成：
  - `arm_fully_connected_s8()` 调用。
  - `arm_fully_connected_s8_get_buffer_size()` scratch 检查。
  - int8 权重数组。
  - int32 bias 数组。
- 对 Q/DQ、per-tensor、普通 `Conv(group=1, dilation=1)` 路径，已经可以生成：
  - `arm_convolve_wrapper_s8()` 调用。
  - `arm_convolve_wrapper_s8_get_buffer_size()` scratch 检查。
  - OIHW 到 OHWI 的 int8 权重重排。
  - int32 bias 数组。
  - per-channel 形式的 multiplier / shift 数组。
- C99 smoke compile 已经接入 pipeline。

未完成：

- DepthwiseConv 尚无真实 renderer。
- Add / Pool / Softmax / Transpose 尚无真实 renderer。
- Conv 真实 per-channel 量化、DepthwiseConv 权重 layout 重排、复杂 bias 规则尚未完成；
  当前普通 Conv 第一版使用 per-tensor Q/DQ 扩展为 CMSIS-NN per-channel 数组。
- pipeline 默认行为还偏调试：blocked 时仍可能生成骨架工程。
- `weights.h` 仍保留 float32 converter 权重导出，后续在 int8 codegen 主链路中应降级为调试产物。

## 3. 支持边界

当前主链路只接受：

- ONNX Q/DQ int8 模型。
- 固定 shape；动态 batch 可固定为 1，其它动态维度不允许静默通过。
- batch size 1。
- Cortex-M 目标，生成 CMSIS-NN s8 C 代码。
- converter 已明确 layout。
- codegen 已实现 renderer 的算子。

当前主链路不接受：

- float32/f16 模型直接生成 C 推理代码。
- 自动量化。
- 未实现 renderer 的算子。
- 静默 fallback 到空实现或注释骨架。
- codegen 重新解析 ONNX。

## 4. 成功与失败语义

`ok`：

- 所有运行期节点都有真实 CMSIS-NN int8 调用或明确的生成期折叠。
- 生成物可以作为当前支持范围内的可推理 C 代码。

`blocked`：

- 模型理论上属于 CMSIS-NN int8 可支持方向，但当前缺 schema、量化字段、
  layout 处理、buffer 规划或 renderer。

`unsupported`：

- 模型包含当前不计划支持或无法映射到 CMSIS-NN s8 主路径的算子/形态。

交付模式下，`blocked` 和 `unsupported` 都应让 CLI 返回非零。只有显式调试模式才允许
继续输出骨架和报告。

## 5. 开发里程碑

### M0：计划与工程清理

任务：

- 清理 Python 缓存、本地工具缓存和系统文件。
- 保留 `NanoC-NN-ONNX-Examples`，但不把样例输出纳入主工程。
- 重写计划文档，去掉宽泛路线，聚焦 int8 单链路。

验收：

- 工作区只剩源码/文档层面的真实改动。
- plan 中只出现当前链路需要的里程碑。

### M1：严格失败语义

状态：已完成第一版。

任务：

- 为 pipeline 增加交付模式默认失败策略。
- 当 codegen status 不是 `ok` 时，CLI 返回非零。
- 增加 `--allow-blocked-output` 或等价参数，供调试时保留报告。
- 报告中明确写出失败节点、原因和下一步缺口。

验收：

- float ONNX 默认失败。
- 缺 Q/DQ 的模型默认失败。
- 含未实现 renderer 的 int8 模型默认失败。
- 调试模式仍可生成骨架和报告。

### M2：int8 schema 校验

状态：进行中，converter 已输出 `quantization.int8_contract`；codegen 已读取该合同并
在非 `ok` 时记录量化阻塞原因。后续还需要补齐按 op type 的规则表。

任务：

- 固化 `model_graph.json.quantization` schema。
- 对运行期节点按 op type 校验所需字段。
- 把 FC、Conv、Pool、Add、Softmax 的字段需求写成代码中的规则表。
- 缺字段时给出 `node_name.field` 级别错误。

验收：

- FC 当前成功路径仍通过。
- 手工删除任意关键量化字段后，codegen 能报出具体字段缺失。
- 未实现的算子不会因为字段存在就被误判为 ok。

### M3：Fully Connected renderer 稳定化

任务：

- 清理 FC 生成代码中的临时假设。
- 明确只支持 `Gemm(transB=1)` 或经过 converter 规范化后的等价 FC。
- 多层 FC 使用 activation ping-pong buffer。
- ReLU/Clip 若能确认相邻关系，折叠到 activation min/max；否则 blocked。

验收：

- 单层 FC 和多层 FC Q/DQ int8 模型均可生成真实 C 调用。
- 生成的 `model_weights.h` 不依赖 float32 权重作为运行期数据。
- C99 smoke compile 通过。

### M4：Conv / DepthwiseConv renderer

状态：普通 `Conv(group=1, dilation=1)` 第一版已完成；DepthwiseConv、grouped Conv
和 dilation 非 1 的 Conv 仍 blocked。

任务：

- converter 输出 Conv 所需量化信息：
  - input/output scale 和 zero_point。
  - per-channel 或 per-tensor weight scale。
  - int8 权重值。
  - int32 bias。
  - multiplier/shift 数组。
- codegen 处理 ONNX `O,I,H,W` 权重到 CMSIS-NN 所需布局。
- 生成 `arm_convolve_wrapper_s8()`。
- 识别 depthwise 并生成 `arm_depthwise_conv_wrapper_s8()`。
- 生成并报告 scratch buffer 来源。

验收：

- Conv -> FC Q/DQ int8 模型生成真实 CMSIS-NN C 调用。
- depthwise 用例能独立通过或明确 blocked 到具体原因。
- layout 转换出现在报告中。

### M5：Pool / Add / Softmax renderer

状态：MaxPool / AveragePool / GlobalAveragePool 和 Softmax 第一版已完成；
Add 仍 blocked，等待双输入 requant 参数规则固化。

任务：

- 生成 MaxPool / AveragePool / GlobalAveragePool s8 调用。
- 生成 Add s8 调用，并校验双输入量化参数。
- 生成 Softmax s8 调用，并计算或读取 mult/shift/diff_min。
- 对需要临时 buffer 的算子记录 buffer size。

验收：

- 小型 CNN 分类链路完整生成：

```text
Q/DQ input -> Conv -> Relu -> Pool -> FC -> Softmax -> Q/DQ output
```

- 任一参数缺失时失败，不生成伪代码。

### M6：统一 CLI 产品化

任务：

- 固化主入口：

```bash
nanoc onnx-to-cmsis --model /path/to/model.int8.onnx
```

- 默认输出到 ONNX 同级目录。
- 成功时打印输出目录和关键 C 文件。
- 失败时打印失败报告路径和首个阻塞原因。
- README 中只把 int8 Q/DQ 作为可推理代码生成输入。

验收：

- 用户无需手动调用 converter 和 codegen。
- 成功模型返回 0。
- blocked / unsupported 模型返回非 0。

### M7：嵌入式侧最小交付

状态：已生成 `firmware_integration.md` 第一版，并在生成工程中记录 C99 文件、
CMSIS-NN/CMSIS-Core 依赖、`NANOC_ENABLE_CMSIS_NN` 宏、SRAM/Flash 估算和无堆内存约束。
后续仍需要 Arm GNU Toolchain / Arm Compiler / FVP 或开发板侧验证。

任务：

- 生成纯 C99 文件：
  - `model.c`
  - `model.h`
  - `model_weights.h`
- 生成 `firmware_integration.md`。
- 明确 CMSIS-NN/CMSIS-Core include 路径、宏和链接要求。
- 不绑定 HAL、启动文件、链接脚本或 IDE。

验收：

- 用户可以把生成文件复制进 STM32/GD32 类固件工程。
- 生成报告记录 SRAM/Flash 估算。
- 至少完成一次 Arm 工具链或等价交叉编译 smoke test。

## 6. 删除与保留原则

应删除：

- Python `__pycache__`、`.pyc`、`.pytest_cache`、`.ruff_cache`。
- 生成输出、临时模型、build 目录。
- 只服务历史宽路线的临时脚本或报告。

应保留：

- `src/nanoc_nn/converter`。
- `src/nanoc_nn/codegen`。
- `src/nanoc_nn/pipeline`。
- 与 int8 单链路直接相关的测试。
- `NanoC-NN-ONNX-Examples`，暂时作为样例来源保留。
- `third_party/CMSIS-NN` 官方源码快照，不做手工修改。

## 7. 下一步执行项

1. 固化 quantization schema 校验表。
2. 继续补 Conv / DepthwiseConv renderer。
3. 扩展 Add / Pool / Softmax renderer。
4. 用更多 Q/DQ int8 ONNX 验证默认失败和成功边界。

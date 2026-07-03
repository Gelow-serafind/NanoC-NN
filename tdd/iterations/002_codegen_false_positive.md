# 迭代 002: codegen false positive

## 基本信息
- **日期**: 2026-07-04
- **前置迭代**: 001
- **触发来源**: 人类需求 / 真实模型回归误判

## 目标

记录一次严重判断失误：MobileNetV2 int8 的 `codegen_report.txt` 和
`pipeline_report.md` 显示 `status: ok`，但实际生成的 `cmsis-codegen/src/model.c`
没有真实 CMSIS-NN 推理路径，且出现孤立的 `#else/#endif`，生成物不可作为成功交付。

## 新增/修改的测试用例

本次先沉淀流程规则，后续应新增专门的生成物完整性测试：

- 对 `ok` 用例检查 `model.c` 是否包含真实 CMSIS-NN 调用。
- 对 `ok` 用例强制执行 C99 smoke compile。
- 对 `ok` 用例拒绝只有 `Generated execution trace`、fallback stub 或空运行路径的生成物。
- 对多分支模型检查生成代码是否按 tensor name 读取历史中间张量，而不是错误退化为线性
  `current_input/current_output`。

## 执行结果

真实模型 `mobilenetv2-12-int8.onnx` 的报告层状态为：

- `codegen status: ok`
- `mappings_blocked_or_unsupported: 0`
- `quantization_issues: 0`

但生成物检查发现：

- `nanoc_model_status()` 返回 `"ok"`。
- `nanoc_model_run()` 中没有 `arm_convolve_wrapper_s8`、
  `arm_depthwise_conv_wrapper_s8`、`arm_elementwise_add_s8` 等真实调用。
- `nanoc_model_run()` 函数体内直接出现孤立 `#else/#endif` fallback。

结论：这是严重假阳性。报告状态不能代表生成物成功。

## 代码修改

本次未继续修复 runtime generator；先停止能力扩展，将误判沉淀为流程约束。

已更新：

- `tdd/README.md`：新增“生成物优先原则”。
- `AGENTS.md`：新增“生成物优先于报告状态”的验证要求。

## 最终结果

当前五个真实 ONNX 的实际状态应按生成物重新判定：

- `mnist_int8`：生成有效。
- `mobilenetv2_int8`：报告为 `ok`，但生成物无效，应判定为失败。
- `squeezenet_q`：仍 blocked，剩余量化/重 requantize 缺口。
- `momo_wakeword`：unsupported，模型不在当前 int8 Q/DQ 主链路内。
- `tinyyolov2`：unsupported，模型不在当前 int8 Q/DQ 主链路内，且资源规模超出当前 MCU 目标。

## 发现与后续

- `status: ok` 必须降级为中间信号，不能作为最终验收。
- TDD runner 需要新增生成物完整性验证，至少覆盖 C 预处理结构、真实 CMSIS-NN 调用、
  C99 compile 和空 runtime body 检测。
- codegen 的 `_renderer_is_complete` 不能只比较 layer 数量；还必须确认 runtime body
  实际生成成功。
- 真实模型回归报告应区分“分析通过”“生成候选”“可编译生成物”“可推理生成物”。

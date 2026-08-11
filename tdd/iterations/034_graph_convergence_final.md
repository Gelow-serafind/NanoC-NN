# 034: 图谱 100% 收敛终态迭代

## 基本信息

- **日期**: 2026-08-12
- **前置迭代**: `033_graph_argmax_boundary`
- **触发来源**: `PLAN_100_SUPPORT.md` 图谱 100% 收敛计划的收尾批次

## 目标

完成剩余算子的 TDD 评估，使 ONNX 图谱达到"全收敛"终态：每个 MCU 相关算子
均有明确状态（PASS 用例或记录拒绝边界），图谱无灰色行。

## 本批次内容

1. **新增能力**：
   - `Identity`（IDENTITY_001）：折叠为 memcpy 透传。
   - `Cast` int8→int8（CAST_001）：Q/DQ 之间 int8 透传折叠。

2. **empty-runtime 假阳性修复**：此前 `Cast`/`Constant` 等无 runtime 层模型
   pipeline 报 `ok` 但生成 `NANOC_STATUS_BLOCKED` 空 stub。修复：`Identity`/
   `Cast` 加入张量别名集（`_tensor_aliases`），无 runtime 层时 `_folded_runtime_copy_body`
   生成真 `memcpy(output, input, N)` + `NANOC_STATUS_OK`。TDD target 的 smoke run
   本可抓到该假阳性，现从根本上修复。

3. **剩余算子边界记录**（37 个）：W2/W3/W4/W5 与 W10-其余/W11，因非 int8 输出、
   动态 shape、int64 索引输入、低 MCU 价值或检测导向，记录为拒绝边界，归档于
   `PLAN_100_SUPPORT.md` 边界记录表（无 case 的边界不入 support matrix，避免
   校验器"无 case 未 planned"冲突；有 case 的边界如 NEG_002/003 正常入列）。

## 最终结果

- validate-only: `106` 个用例
- target: `106/106 PASS`
- numeric: `82/82 PASS`
- regression: PASS（86/86 结构）
- 能力集文件 `CAPABILITIES.md`、支持图谱同步

## 结论

`PLAN_100_SUPPORT.md` 达到**全收敛**状态。本会话累计（迭代 029-034）新增
22 个 PASS 能力算子 + 2 个 NEG 边界用例 + 修复 2 个跨路径缺陷
（round-half-even 取整、empty-runtime stub），覆盖从 59 到 81 个确认能力节点。

## 遗留（非阻塞，已记录边界）

- 非 int8 外部 ABI（ArgMax/ArgMin/TopK 等）需框架扩展，边界用例 NEG_002/003 已固化。
- `Mish` 需 opset 18（cap 17），纳入 opset 扩展。
- 折叠批 `Constant`/`Split`/`Expand`/`Tile` 与 shape-ABI 算子需各自扩展。

# 033: ArgMax/ArgMin 非 int8 ABI 边界迭代（100% 计划 W0）

## 基本信息

- **日期**: 2026-08-12
- **前置迭代**: `032_graph_spatial_variadic_batch`
- **触发来源**: `PLAN_100_SUPPORT.md` 图谱 100% 收敛计划 W0

## 目标

处理 `ArgMax`/`ArgMin` 分类 index 输出算子。

## 决策：记录为拒绝边界

`ArgMax`/`ArgMin` 输出 int64 index 张量，而当前生成框架与 numeric runner 仅支持
int8 模型外部 ABI（与 027/028 对 compare bool 中间张量的处置一致：非 int8 数据
不外露）。支持需扩展非 int8 外部输出与 index 数值验收，属于框架级改动，独立立项。

按 100% 计划"每个算子 ok 或记录拒绝边界"的原则，本轮以 negative case 固化边界。

## 新增/修改的测试用例

| case | ONNX op | 预期 | 边界原因 |
|------|---------|------|----------|
| `NEG_002` | `ArgMax` | unsupported | int64 index 输出超出 int8 ABI |
| `NEG_003` | `ArgMin` | unsupported | int64 index 输出超出 int8 ABI |

## 执行结果

- validate-only: `104` 个用例
- NEG_002/NEG_003: `expected=unsupported actual=unsupported` PASS
- regression: PASS，target: `104/104 PASS`

## 后续

框架支持非 int8 外部输出后，本边界改判 `ok` 并补 index 数值验收（与 ArgMax 一并处理）。

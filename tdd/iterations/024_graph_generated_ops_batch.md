# 024: 图谱驱动 generated C 算子批量闭环

## 背景

上一轮已经完成 `AveragePool`、`Reshape`、`Squeeze`、`Add`、`Mul`、`Transpose`，本轮继续按 ONNX 官方图谱和 TDD 规则推进一批 MCU 小模型常见基础算子：`Sub`、`Div`、`Sigmoid`、`Pad`、`Slice`、`Gather`。

## 新增 case

| case | ONNX op | 支持形态 | lowering |
|------|---------|----------|----------|
| `SUB_001` | `Sub` | 同形状 QDQ/int8，第二输入可为常量 | `generated_c_sub_s8` |
| `DIV_001` | `Div` | 同形状 QDQ/int8，第二输入为非零常量 | `generated_c_div_s8` |
| `SIGMOID_001` | `Sigmoid` | 静态 QDQ/int8 tensor | `generated_c_sigmoid_s8` |
| `PAD_001` | `Pad` | rank4、constant mode、静态 pads | `generated_c_pad_s8` |
| `SLICE_001` | `Slice` | 静态 starts/ends/axes/steps 数据路径 | `generated_c_slice_s8` |
| `GATHER_001` | `Gather` | 静态 indices、单 axis 数据路径 | `generated_c_gather_s8` |

## 实现要点

- converter 白名单新增 `Sub`、`Div`、`Sigmoid`、`Pad`，并将 `Slice/Gather` 从纯 shape-helper 语义扩展到可区分数据路径。
- converter 为六个算子抽取 QDQ/int8 quant contract，包括输入输出量化、常量分支、静态索引参数和输出 block size。
- codegen mapper 将 `Slice/Gather` 的 shape-helper 形态继续 folded，数据路径进入 runtime renderer，避免破坏已有真实网络中的 shape 构造。
- codegen 为 CMSIS-NN 无专用 s8 kernel 的算子生成 C99 路径：`Sub/Div/Sigmoid` 采用反量化、计算、再量化；`Pad/Slice/Gather` 采用静态索引 copy。
- `Sigmoid` 当前用 `expf` 作为正确性基线，后续可按 MCU 性能要求替换为 LUT 或定点近似。
- numeric runner 的 `tdd/work/numeric` 清理增加重试，并允许清理半删除残留目录，仍保留 marker 和目录归属保护，避免误删 fixtures 或用户数据集。

## 验收结果

- 新增六个单点结构测试：全部 PASS。
- 新增六个单点数值测试：全部 PASS，均为 `top1=2/2`、饱和率 `0.00`、最大绝对误差 `0.0`。
- target 全量：`49/49 PASS`。
- numeric 全量：`27/27 PASS`。
- 稳定回归：`31/31` baseline 结构 PASS，`27/27` numeric PASS。
- ONNX 支持图谱：官方支持从 `17/226` 扩展到 `23/226`。

## 边界和后续

- `Sub/Div` 当前只声明同形状 QDQ/int8；broadcast、双动态输入、不同 rank 输入尚未声明。
- `Div` 的通过形态要求分母非零；生成代码对近零分母做防御性输出 0，但这不是完整除零语义声明。
- `Pad` 当前只声明 constant mode、静态 pads、同量化；reflect/edge、动态 pads、负 padding 尚未声明。
- `Slice` 当前只声明静态正步长数据路径。
- `Gather` 当前只声明 1-D 静态 indices、非负且范围内。
- 下一批建议继续从图谱选择 `Tanh`、`LeakyRelu`、`ReduceMean`、`Where`，或从 `NET_004` 数值升级中反向提炼 case。

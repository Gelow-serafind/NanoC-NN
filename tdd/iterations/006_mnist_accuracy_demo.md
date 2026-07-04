# 迭代 006: MNIST 准确率 demo 驱动修复

## 基本信息

- **日期**: 2026-07-04
- **前置迭代**: 005
- **触发来源**: 人工手写 MNIST demo 对比显示 ONNX Runtime 与 CMSIS-NN C 输出完全不一致

## 目标

把 `mnist-12-int8.onnx` 的验收标准从“能生成、能编译、能启动”推进到“生成的 C 推理结果与 ONNX Runtime 在手写样本上的分类结果一致”。

## 复现结果

执行临时 demo：

```bash
cd tdd/results/tmp/TOPO_003/demo
python compare.py
```

观察结果：

- ONNX Runtime 对 10 个手写样本基本能给出有效分类。
- CMSIS-NN C 输出全部落在 `-128/127` 两端，呈现严重饱和。
- 预测一致率为 `0/10`。

这说明 `TOPO_003` 当前结构层 PASS 仍不足以证明 MNIST 转换正确，必须补端到端数值验收。

## 本轮发现

MNIST 两个 `QLinearConv` 节点使用：

```text
auto_pad=SAME_UPPER
kernel_shape=[5,5]
strides=[1,1]
```

ONNX 输出 shape 分别保持为 `28x28` 和 `14x14`，因此 CMSIS-NN 卷积需要 padding=2。

修复前生成物错误地写成：

```c
conv_params.padding.h = 0;
conv_params.padding.w = 0;
```

这会让 CMSIS-NN 以无 padding 的参数执行 5x5 卷积，却仍声明 SAME 输出尺寸，是导致 MNIST 输出严重错误的第一优先缺陷。

## 代码修改

- `src/nanoc_nn/converter/parser.py`
  - `normalize_attributes()` 保留 `Conv/QLinearConv` 的 `auto_pad` 属性。
  - 新增 `_effective_conv_pads()`，根据 input/output shape、kernel、stride、dilation 将 `SAME_UPPER/SAME_LOWER/VALID` 转换为显式 pads。
  - `Conv` 与 `QLinearConv` 的 CMSIS-NN quant info 生成统一使用推导后的有效 padding。

## 验证

- `python tdd/scripts/run_tests.py --case TOPO_003 --generate`: PASS
- 生成的 `model.c` 中两个 MNIST `QLinearConv` 均已变为：

```c
conv_params.padding.h = 2;
conv_params.padding.w = 2;
```

- `python tdd/scripts/run_tests.py --mode target --generate`: 17/17 PASS

## 未完成

人工 demo 放在 `tdd/results/tmp/TOPO_003/demo`，该目录会被 TDD runner 清理，因此本轮修复后无法用同一批手写样本复测准确率。

下一轮必须先将准确率 demo 迁移为持久 fixture：

- `tdd/fixtures/mnist/` 保存手写样本与期望标签。
- TDD runner 增加 MNIST numeric/accuracy 验收。
- `TOPO_003` 必须检查 ONNX Runtime 与生成 C 的 top1 一致率，而不是只检查 API、C99 compile 和 smoke run。

## 下一步任务单

1. 将手写 MNIST 样本和 compare harness 从临时目录迁移到持久 `tdd/fixtures` / `tdd/scripts`。
2. 给 `TOPO_003` 增加数值验收字段，至少要求 C top1 与 ONNX top1 一致率达到当前人工 demo 的可接受阈值。
3. 用修复后的 `auto_pad` 版本重跑准确率 demo。
4. 如果仍不一致，按节点插桩 dump：
   - Conv1 output
   - Pool1 output
   - Conv2 output
   - Pool2 output
   - Reshape/FC input
   - QLinearMatMul output
   - QLinearAdd output
5. 优先排查 NHWC 中间内存进入 NCHW 语义 Flatten/FC 时的布局错位。

# mnist_synthetic_smoke

这是 `TOPO_003` 的最小持久数值验收数据集。

它不用于评估真实 MNIST 准确率，而用于验证同一批输入经过：

1. 原始 `mnist-12-int8.onnx` + ONNX Runtime
2. NanoC-NN 生成的 CMSIS-NN C 代码

之后，二者输出是否保持一致。

样本以可复现 pattern 描述，而不是手写 784 个像素值。数值 runner 会在运行时展开为 `[1,1,28,28]` float32 输入，像素范围为 `[0,1]`，再按用例 registry 中的 `input_scale=255.0` 喂给 ONNX。

后续如果需要使用人工手写样本，应新增独立数据集目录，例如 `mnist_hand_drawn/`，不要放入 `tdd/work/` 或 `tdd/results/`。

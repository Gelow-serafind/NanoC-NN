# concat_channel_smoke

`CONCAT_001` 的最小数值验收数据集。两个 `[1,1,2,2]` 输入按 channel 维拼接为 `[1,2,2,2]`，用于对比 ONNX Runtime 与生成 C 的输出。

数据刻意覆盖正值、负值和零值，量化 scale 为 `0.1`，因此允许 `0.11` 以内的最大绝对误差。

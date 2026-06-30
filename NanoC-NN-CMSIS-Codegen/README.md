# NanoC-NN-CMSIS-Codegen

CMSIS-NN 代码生成子项目。

该模块后续负责读取 `NanoC-NN-ONNX-Converter` 生成的 `model_graph.json`、权重资料和量化信息，生成面向 Arm CMSIS-NN 的 C 推理工程。

## 子项目定位

本目录不再自研完整神经网络算子库，而是以 CMSIS-NN 为底层内核依赖，重点实现以下能力：

- 将 ONNX 算子映射到 CMSIS-NN 可调用函数。
- 生成 `model.c`、`model.h`、权重文件和静态激活缓冲区规划。
- 生成最小 `main.c` 或验证入口，用于 PC/交叉编译 smoke test。
- 输出算子支持矩阵、量化参数报告和内存占用报告。
- 对不适合直接映射到 CMSIS-NN 的算子给出明确失败原因或 fallback 策略。

## 初期技术路线

1. 固定 converter 与 codegen 的中间表示字段，优先使用 `model_graph.json`。
2. 建立 ONNX 算子到 CMSIS-NN API 的映射表。
3. 优先生成静态 shape、单输入单输出 CNN 分类网络。
4. 优先处理 int8 量化模型；float32 模型先作为结构解析和参考验证输入，不作为 CMSIS-NN 加速主路径。
5. 生成代码不在运行期调用 `malloc` 或 `free`，所有工作区大小由生成阶段计算。

## 计划输出

```text
generated/
├── include/
│   ├── model.h
│   └── model_weights.h
├── src/
│   ├── model.c
│   └── main.c
├── reports/
│   ├── op_mapping.md
│   ├── memory_plan.md
│   └── quantization.md
└── CMakeLists.txt
```

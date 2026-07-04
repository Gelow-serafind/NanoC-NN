# CMSIS-NN 编译器 TDD 测试用例生成计划

> **版本**：v1.0  
> **日期**：2026-07-03  
> **适用范围**：CMSIS-NN Q/DQ ONNX 代码生成器（NanoC-NN）  
> **当前 MVP 阶段目标**：Level 2 — 生成代码编译成功、运行不崩溃

---

## 目录

1. [第一部分：测试策略与阶段划分](#第一部分测试策略与阶段划分)
2. [第二部分：测试用例分类与详细设计要点](#第二部分测试用例分类与详细设计要点)
3. [第三部分：测试用例规格说明书](#第三部分测试用例规格说明书)
4. [第四部分：执行、评估与迭代机制](#第四部分执行评估与迭代机制)

---

## 第一部分：测试策略与阶段划分

### 1.1 整体测试金字塔

```text
                    ┌──────────────┐
                    │  阶段四       │
                    │  实战验证     │  MobileNetV1-α=0.25 / 简化的关键词识别
                    │  2 个模型    │
                    ├──────────────┤
                    │  阶段三       │
                    │  复杂结构     │  残差 Add / Concat 多分支 / 多输出
                    │  5 个模型    │  驱动白名单扩展与架构升级
                    ├──────────────┤
                    │  阶段二       │
                    │  经典拓扑     │  Conv→Pool→FC→Softmax 全链路 / 多层 FC
                    │  8 个模型    │  验证算子间数据流与内存布局衔接
                    ├──────────────┤
                    │  阶段一       │
                    │  核心算子     │  Gemm / Conv / MaxPool / AvgPool / Softmax
                    │  18 个模型   │  原子级孤立验证 + 边界/异常参数组合
                    └──────────────┘
```

### 1.2 阶段定义

#### 阶段一：核心算子原子测试（P0 — 基础保障）

| 项目 | 内容 |
|------|------|
| **目标** | 验证每个白名单算子（Gemm/FC、Conv、MaxPool、AveragePool、Softmax）在孤立环境下的参数传递、权重映射、量化参数处理、C 代码生成与 smoke compile 正确性 |
| **入口标准** | converter 已支持该算子的 Q/DQ 解析；codegen 已具备该算子 renderer 第一版 |
| **完成标准** | 全部 P0 用例 codegen status = `ok`、C99 编译通过、可执行文件运行不崩溃 |
| **模型数量** | 约 18 个 |
| **关键设计变量** | 输入/输出尺寸、weight shape、bias 有无、transB、kernel_size、stride、padding、输入/输出量化参数（对称/非对称）、边界尺寸（1×1、对齐/不对齐 4 的通道数） |

#### 阶段二：经典拓扑组合测试（P0 — 回归防线）

| 项目 | 内容 |
|------|------|
| **目标** | 将白名单算子组装成业界公认的经典微型网络拓扑，验证算子间的数据流衔接、NCHW→NHWC 布局转换、activation ping-pong buffer 切换、量化参数透传、形状推导链的正确性 |
| **入口标准** | 阶段一全部通过 |
| **完成标准** | 全部 P0 组合拓扑 codegen = `ok`、C99 编译通过、运行不崩溃；至少 1 个模型完成与 ONNX Runtime int8 推理的 argmax 精度对齐（可选，非阻塞） |
| **模型数量** | 约 8 个 |
| **关键设计变量** | 拓扑深度（2~6 层）、Conv+Relu 融合、Pool 前后量化参数一致性（满足/故意不满足）、多层 FC 串接、Softmax 位置（中间层/末端） |

#### 阶段三：复杂结构与白名单扩展驱动测试（P1 — 架构演进）

| 项目 | 内容 |
|------|------|
| **目标** | 引入多输入算子和分支拓扑，暴露当前线性拓扑架构的局限性，驱动 codegen 在数据流图表示、多输入 requant 和内存规划方面的架构升级 |
| **入口标准** | 阶段二全部通过 |
| **完成标准** | 所有 P1 用例至少达到 `blocked` 且原因明确；已实现扩展的算子在扩展后重新测试至 `ok` |
| **模型数量** | 约 7 个 |
| **关键设计变量** | Add 残差连接（双输入不同 scale）、Concat 多分支合并、多输出头、DepthwiseConv、**多 Opset 兼容性（opset 11/13/17）、动态 Shape（动态 H/W 维度）** |

> ⚠️ **TDD 架构驱动声明**：当前阶段一和阶段二固定使用 Opset 17 和固定 Shape，
> 是为了在 MVP 阶段缩小变量空间、快速验证核心算子链路的正确性。
> **未来阶段三必须引入多 Opset（如 opset 11、13、17）和动态 Shape（如动态 H/W、
> 动态 batch > 1）的测试用例**，以驱动编译器前端在 ONNX 版本兼容性和
> 动态维度校验方面的架构升级。这符合 TDD"用测试驱动架构演进"的核心理念：
> 当前这些用例预期 `blocked` 或 `unsupported`，它们的存在本身就是对编译器
> 能力缺口的精确文档化，并直接定义了下一轮架构迭代的需求规格。

#### 阶段四：实战验证（P2 — 场景覆盖）

| 项目 | 内容 |
|------|------|
| **目标** | 使用贴近真实应用的简化模型进行端到端测试，验证编译器在"真实模型缩小版"上的行为 |
| **入口标准** | 阶段三全部通过（或阻塞原因已记录且确认非编译器缺陷） |
| **完成标准** | 模型 codegen = `ok` 或 `blocked` 且原因文档化；编译通过（若 ok）；生成报告给出明确的 SRAM/Flash 估算 |
| **模型数量** | 约 2 个 |
| **关键设计变量** | MobileNetV1 的 depthwise separable conv 结构、RNN/TC-ResNet 类关键词识别模型 |

### 1.3 优先级汇总

| 优先级 | 阶段 | 模型数 | 目标 | 阻塞发布？ |
|--------|------|--------|------|-----------|
| P0 | 阶段一 + 阶段二 | 26 | 原子算子 + 线性组合拓扑全覆盖 | **是** |
| P1 | 阶段三 | 7 | 驱动白名单扩展、多 Opset 兼容性、动态 Shape 架构升级 | 否（需记录阻塞原因） |
| P2 | 阶段四 | 2 | 实战场景 smoke test | 否 |

---

## 第二部分：测试用例分类与详细设计要点

### 2.1 分类一：原子算子测试

#### 2.1.1 Gemm / Fully Connected

**目标**：验证 FC 算子的参数传递、权重映射（transB=1 → CMSIS-NN 所需布局）、量化参数（multiplier/shift/offset）计算、bias int32 量化。

**设计变量表**：

| 变量 | 取值集合 | 说明 |
|------|---------|------|
| in_features | 1, 2, 4, 8, 13, 64 | 极小、对齐 4、非对齐 4、中等 |
| out_features | 1, 2, 4, 8, 13, 64 | 同上 |
| transB | 1（必须） | 当前仅支持 transB=1 |
| with_bias | True, False | 有/无 bias |
| input quantization | symmetric(zp=0), asymmetric(zp=10), asymmetric(zp=-5) | 对称/非对称输入 |
| weight quantization | symmetric(zp=0) | 权重通常对称量化 |
| output quantization | symmetric(zp=0), asymmetric(zp≠0) | 对称/非对称输出 |

**边界/异常设计**：

- `in_features=1, out_features=1`：极小 FC，单元素矩阵乘法
- `in_features=13, out_features=7`：非 4 对齐维度，测试 CMSIS-NN 内部 padding 处理
- 无 bias 时，验证 codegen 不生成 bias 数组引用
- 输入 zero_point ≠ 0 时，验证 `input_offset` 计算 = `-input_zp`

**陷阱注入**：

- 在 FC 后紧跟非对称量化的下一层，验证 output quantization 参数透传

---

#### 2.1.2 Conv (Conv2D, group=1, dilation=1)

**目标**：验证 Conv 算子的权重布局转换（OIHW → OHWI）、per-channel multiplier/shift 数组生成、padding/stride 参数映射、scratch buffer 计算。

**设计变量表**：

| 变量 | 取值集合 | 说明 |
|------|---------|------|
| input H×W | 4×4, 8×8, 7×7, 16×16 | 正方形、非正方形特征图 |
| in_channels | 1, 3, 4, 7 | 单通道、RGB、对齐 4、非对齐 4 |
| out_channels | 1, 2, 4, 8, 11 | 对齐/非对齐 4 |
| kernel_size | 1×1, 3×3, 5×5 | 不同卷积核 |
| stride | 1, 2 | 步长 1 和步长 2 |
| padding | VALID(no pad), SAME_UPPER, 自定义(1,1,1,1) | 不同 padding 模式 |
| with_bias | True, False | 有/无 bias |
| input quantization | zp=0, zp≠0 | 对称/非对称 |
| weight quantization | per-tensor, per-channel | 当前 converter 仅支持 per-tensor; per-channel 作为 P1 驱动测试 |

**边界/异常设计**：

- `kernel_size=1×1, in=1, out=1`：1×1 pointwise 卷积，退化为逐通道缩放
- `H=3, W=3, kernel=3, stride=2`：输出尺寸为 1×1 的边界情况
- `in_channels=7, out_channels=11`：两个维度均不对齐 4
- 无 bias 时 bias 数组应为全零 int32 或不存在

**陷阱注入**：

- Conv 输入 Q/DQ scale 与 weight Q/DQ scale 差异极大（如 input_scale=0.01, weight_scale=0.5），测试 multiplier/shift 计算不溢出

---

#### 2.1.3 MaxPool

**目标**：验证 MaxPool 算子的 CMSIS-NN 参数映射。**关键约束**：输入/输出量化参数必须一致。

**设计变量表**：

| 变量 | 取值集合 | 说明 |
|------|---------|------|
| input H×W | 8×8, 16×16, 6×6 | 不同尺寸 |
| channels | 1, 4, 8 | 池化不改变通道数 |
| kernel_size | 2×2, 3×3 | 池化窗口 |
| stride | 2, 1 | 下采样/保持尺寸 |
| padding | VALID, SAME_UPPER | padding 模式 |
| input/output zp 一致性 | 一致, 不一致 | **不一致时必须 blocked** |

**边界/异常设计**：

- `H=3, W=3, kernel=3, stride=1`：输出 1×1
- 输入 zp ≠ 输出 zp：**预期 codegen 必须 blocked**（Pool 要求量化参数一致）
- 输入 scale ≠ 输出 scale：**预期 blocked**

---

#### 2.1.4 AveragePool / GlobalAveragePool

**目标**：同 MaxPool，额外关注 avg pool 的 requant 行为。

**设计变量表**：同 MaxPool，额外增加：

| 变量 | 取值集合 | 说明 |
|------|---------|------|
| GlobalAveragePool | 自适应到 1×1 输出 | 无 kernel_size 属性 |

**边界/异常设计**：与 MaxPool 相同的量化一致性约束。

---

#### 2.1.5 Softmax

**目标**：验证 Softmax 的 multiplier/shift/diff_min 参数计算和 CMSIS-NN 调用。

**设计变量表**：

| 变量 | 取值集合 | 说明 |
|------|---------|------|
| num_classes | 2, 10, 100 | 不同分类数 |
| input scale | 0.01, 0.05, 0.1 | 不同量化粒度 |
| Softmax 位置 | 末端（典型）、中间层 | 位置不影响参数但影响拓扑 |

**边界/异常设计**：

- `num_classes=2`：二分类边界
- `num_classes=1`：退化为常量输出，CMSIS-NN 行为待验证
- Softmax 前一层输出 scale 极小（如 0.001），测试 diff_min 计算是否溢出

---

### 2.2 分类二：组合拓扑测试

**总体目标**：验证以下关键路径的端到端数据流：

1. **布局转换链**：NCHW 输入 → Conv(NHWC 内部) → 各层 → 输出
2. **Activation ping-pong**：多层之间 buffer_A ↔ buffer_B 正确切换
3. **量化参数透传**：Q/DQ → 运行期算子 → Q/DQ 的 scale/zp 链
4. **形状推导链**：Conv → Pool → Flatten → FC 逐步 shape 变化

**测试拓扑模板**：

| 编号 | 拓扑 | 层数 | 测试重点 |
|------|------|------|---------|
| T1 | `Q/DQ input → Conv → Relu → Q/DQ output` | 1 RT | Conv+Relu 融合、最简 CNN |
| T2 | `Q/DQ input → Conv → MaxPool → Q/DQ output` | 2 RT | Conv→Pool 数据流、Pool 量化一致性 |
| T3 | `Q/DQ input → Conv → Relu → MaxPool → Flatten → FC → Q/DQ output` | 3 RT | 全链路经典 CNN 分类头 |
| T4 | `Q/DQ input → FC → Relu → FC → Relu → FC → Q/DQ output` | 3 RT | 多层 FC、buffer ping-pong |
| T5 | `Q/DQ input → Conv → Conv → MaxPool → Flatten → FC → Q/DQ output` | 4 RT | 双 Conv 串接 |
| T6 | `Q/DQ input → Conv → Relu → AvgPool → Flatten → FC → Softmax → Q/DQ output` | 4 RT | 含 Softmax 末端的完整分类链路 |
| T7 | `Q/DQ input → Conv → GlobalAveragePool → Flatten → FC → Q/DQ output` | 3 RT | GlobalAveragePool 作为分类头 |
| T8 | `Q/DQ input → FC → Softmax → Q/DQ output` | 2 RT | 最简单的分类器（逻辑回归） |

**陷阱注入**：

- 在 T3/T6 中，让 Conv 输入 zp=0 但 FC 输入 zp≠0，验证量化参数逐层正确传递而不混淆
- 在 T5 中，第一个 Conv 和第二个 Conv 使用不同的 stride，验证 shape 推导逐层正确

---

### 2.3 分类三：驱动扩展的测试（P1 白名单突破）

这些测试在当前架构下**预期 blocked**，用于驱动后续开发：

| 编号 | 测试点 | 当前缺口 | 期望驱动 |
|------|--------|---------|---------|
| E1 | `Q/DQ input → Conv → Add(residual) → Q/DQ output` | Add 无独立 renderer；双输入 requant 规则未固化 | Add renderer + 多输入数据流 |
| E2 | `Q/DQ input → Conv → Conv(分支1) → Concat ← Conv(分支2) → FC → Q/DQ output` | Concat 无 renderer；线性拓扑假设被打破 | 图拓扑支持 + Concat renderer |
| E3 | `Q/DQ input → DepthwiseConv → Conv(pointwise) → Q/DQ output` | DepthwiseConv 无 renderer | DepthwiseConv renderer |
| E4 | `Q/DQ input → split → headA(FC→Softmax), headB(FC→Softmax) → Q/DQ output` | 多输出不支持 | 多输出头支持 |
| E5 | `Q/DQ input → Conv → Transpose(runtime) → FC → Q/DQ output` | Transpose 被标记为折叠但运行期 transpose 无 renderer | 运行期 Transpose + 布局感知 |
| E6 | 同一模型分别以 Opset 11 / 13 / 17 导出，验证 converter 均能正确解析 | converter 仅验证 opset 11-17 范围，未对不同 opset 的算子属性默认值差异做兼容测试 | 多 Opset 兼容性 + 算子属性归一化 |
| E7 | `Q/DQ Input(batch=1, H=?, W=?) → Conv → FC → Q/DQ Output` | converter 仅支持全固定 shape；动态 H/W 维度应被拒绝但不崩溃 | 动态 Shape 校验 + 优雅错误报告 + 未来动态维度支持 |

---

### 2.4 分类四：负向测试（Robustness）

这些测试验证编译器在遇到非法输入时的失败行为是否正确（不崩溃、给出明确错误信息）：

| 编号 | 负向场景 | 预期行为 |
|------|---------|---------|
| N1 | 输入 float32 ONNX（无 Q/DQ 节点） | pipeline 返回非 0；报告"非 int8 Q/DQ 模型" |
| N2 | Gemm 使用 transB=0 | converter 标记 unsupported 或 codegen blocked |
| N3 | Conv group > 1（非 depthwise） | blocked，原因明确 |
| N4 | 动态 batch 之外的动态维度 | converter 报错，不静默通过 |
| N5 | Pool 输入/输出量化参数不一致 | codegen blocked，原因明确 |
| N6 | 存在未知 ONNX 算子 | converter 标记 unsupported，pipeline 返回非零 |
| N7 | 空模型（无节点） | 优雅失败，不崩溃 |
| N8 | 模型文件不存在/损坏 | 给出明确错误信息 |

---

## 第三部分：测试用例规格说明书

### 3.1 规格书模板

每个测试用例按以下模板定义：

```markdown
### 测试用例_ID: [阶段缩写]_[分类]_[编号]
- **名称**: [简明扼要的名称]
- **归属阶段**: [阶段一/二/三/四]
- **优先级**: [P0/P1/P2]
- **主要验证目标**: [一句话描述]
- **网络结构**:
  ```
  Q/DQ Input → [算子序列] → Q/DQ Output
  ```
- **输入张量形状**: [如: [1, 3, 8, 8]]
- **关键参数**:
  - [算子名]: {参数列表}
- **量化参数设计**:
  | 张量 | scale | zero_point | 对称性 |
  |------|-------|-----------|--------|
  | input | ... | ... | symmetric/asymmetric |
  | weight | ... | ... | symmetric |
  | output | ... | ... | symmetric/asymmetric |
- **边界/异常条件**: [描述]
- **预期暴露的风险**: [如: 权重布局错误 / zp 未参与计算 / buffer 越界]
- **当前预期结果**: [ok / blocked / unsupported]
- **模型脚本路径**: `tdd/scripts/phase{N}/gen_{case_id}.py`
- **模型输出路径**: `tdd/models/phase{N}/{case_id}.onnx`
```

### 3.2 阶段一 — 核心算子原子测试用例清单

---

#### 3.2.1 Gemm / Fully Connected（6 个用例）

---

##### P0_CORE_GEMM_001
- **名称**: FC 最小单元 — 2×2 对称量化无 bias
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 最小 FC 的 transB=1 权重映射与量化参数传递
- **网络结构**:
  ```
  Q/DQ Input → Gemm(transB=1) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 2]`
- **关键参数**:
  - Gemm: `{transB: 1, with_bias: False}`, weight shape `[3, 2]`
- **量化参数设计**:
  | 张量 | scale | zero_point | 对称性 |
  |------|-------|-----------|--------|
  | input | 0.01 | 0 | symmetric |
  | weight | 0.02 | 0 | symmetric |
  | output | 0.05 | 0 | symmetric |
- **边界/异常条件**: in_features=2, out_features=3, 极小尺寸；无 bias
- **预期暴露的风险**: 无 bias 时代码是否引用不存在的 bias 数组
- **当前预期结果**: `ok`

---

##### P0_CORE_GEMM_002
- **名称**: FC 非对称输入 — zp=10 含 bias
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 非对称输入时 `input_offset = -zp` 是否正确传入 CMSIS-NN
- **网络结构**:
  ```
  Q/DQ Input(zp=10) → Gemm(transB=1, with_bias) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 4]`
- **关键参数**:
  - Gemm: `{transB: 1, with_bias: True}`, weight shape `[8, 4]`
- **量化参数设计**:
  | 张量 | scale | zero_point | 对称性 |
  |------|-------|-----------|--------|
  | input | 0.01 | 10 | asymmetric |
  | weight | 0.02 | 0 | symmetric |
  | bias | (int32) | — | — |
  | output | 0.05 | 0 | symmetric |
- **边界/异常条件**: 输入 zp ≠ 0
- **预期暴露的风险**: `input_offset` 计算错误（应为 -10 而非 +10）；bias 量化系数使用错误 scale
- **当前预期结果**: `ok`

---

##### P0_CORE_GEMM_003
- **名称**: FC 非对称输出 — 输出 zp=-5
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 非对称输出时 `output_offset` 是否正确传入
- **网络结构**:
  ```
  Q/DQ Input → Gemm(transB=1, with_bias) → Q/DQ Output(zp=-5)
  ```
- **输入张量形状**: `[1, 4]`
- **关键参数**:
  - Gemm: `{transB: 1, with_bias: True}`, weight shape `[4, 4]`
- **量化参数设计**:
  | 张量 | scale | zero_point | 对称性 |
  |------|-------|-----------|--------|
  | input | 0.01 | 0 | symmetric |
  | weight | 0.02 | 0 | symmetric |
  | output | 0.03 | -5 | asymmetric |
- **边界/异常条件**: 输出 zp 为负数
- **预期暴露的风险**: `output_offset` 符号处理错误
- **当前预期结果**: `ok`

---

##### P0_CORE_GEMM_004
- **名称**: FC 非 4 对齐维度 — 13→7
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: CMSIS-NN `arm_fully_connected_s8` 对非 4 对齐维度的处理
- **网络结构**:
  ```
  Q/DQ Input → Gemm(transB=1) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 13]`
- **关键参数**:
  - Gemm: `{transB: 1, with_bias: True}`, weight shape `[7, 13]`
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: in_features=13, out_features=7，均不被 4 整除
- **预期暴露的风险**: CMSIS-NN 内部 padding 导致越界读写
- **当前预期结果**: `ok`

---

##### P0_CORE_GEMM_005
- **名称**: FC 单元素 — 1×1
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 退化为标量乘法的极限边界
- **网络结构**:
  ```
  Q/DQ Input → Gemm(transB=1, with_bias) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 1]`
- **关键参数**:
  - Gemm: `{transB: 1, with_bias: True}`, weight shape `[1, 1]`
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: 所有维度为 1
- **预期暴露的风险**: 维度为 1 时循环边界/scratch buffer 计算错误
- **当前预期结果**: `ok`

---

##### P0_CORE_GEMM_006
- **名称**: FC 大维度 — 64→128 对称量化
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 中等规模 FC 的权重数组正确性和 buffer 预算
- **网络结构**:
  ```
  Q/DQ Input → Gemm(transB=1) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 64]`
- **关键参数**:
  - Gemm: `{transB: 1, with_bias: True}`, weight shape `[128, 64]`
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: 权重 8192 个元素
- **预期暴露的风险**: 权重数组越界、Flash 预算超限
- **当前预期结果**: `ok`

---

#### 3.2.2 Conv（6 个用例）

---

##### P0_CORE_CONV_001
- **名称**: Conv 最小单元 — 1×1 kernel, 1→1 通道, VALID
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 退化为逐通道缩放的 pointwise 卷积
- **网络结构**:
  ```
  Q/DQ Input → Conv(kernel=1×1, group=1) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 1, 4, 4]`
- **关键参数**:
  - Conv: `{kernel: [1,1], stride: [1,1], pads: [0,0,0,0], group: 1, with_bias: True}`, weight shape `[1,1,1,1]`
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: kernel=1×1, in_ch=1, out_ch=1 — 最简 Conv
- **预期暴露的风险**: OIHW→OHWI 权重重排错误
- **当前预期结果**: `ok`

---

##### P0_CORE_CONV_002
- **名称**: Conv 标准 3×3 — 3→8 通道 SAME_UPPER
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 标准 Conv 的完整参数映射与 SAME padding
- **网络结构**:
  ```
  Q/DQ Input → Conv(kernel=3×3, padding=SAME_UPPER) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 3, 8, 8]`
- **关键参数**:
  - Conv: `{kernel: [3,3], stride: [1,1], pads: [1,1,1,1], group: 1, with_bias: True}`, weight shape `[8,3,3,3]`
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: 多通道输入、SAME padding
- **预期暴露的风险**: padding 参数映射到 CMSIS-NN 格式错误
- **当前预期结果**: `ok`

---

##### P0_CORE_CONV_003
- **名称**: Conv stride=2 — 下采样
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: stride > 1 时的输出 shape 推导和 CMSIS-NN 参数
- **网络结构**:
  ```
  Q/DQ Input → Conv(kernel=3×3, stride=2) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 4, 8, 8]`
- **关键参数**:
  - Conv: `{kernel: [3,3], stride: [2,2], pads: [1,1,1,1], group: 1}`, weight shape `[8,4,3,3]`
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: stride=2, 输出 H=W=4
- **预期暴露的风险**: stride 参数映射错误、输出 buffer 尺寸不足
- **当前预期结果**: `ok`

---

##### P0_CORE_CONV_004
- **名称**: Conv 非对称输入 — 输入 zp≠0
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 非对称输入时的 input_offset 是否正确
- **网络结构**:
  ```
  Q/DQ Input(zp=12) → Conv(kernel=3×3) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 3, 8, 8]`
- **关键参数**:
  - Conv: `{kernel: [3,3], stride: [1,1], pads: [1,1,1,1], group: 1}`, weight shape `[8,3,3,3]`
- **量化参数设计**:
  | 张量 | scale | zero_point | 对称性 |
  |------|-------|-----------|--------|
  | input | 0.01 | 12 | asymmetric |
  | weight | 0.02 | 0 | symmetric |
  | output | 0.05 | 0 | symmetric |
- **边界/异常条件**: 输入 zp ≠ 0
- **预期暴露的风险**: `input_offset` 值错误
- **当前预期结果**: `ok`

---

##### P0_CORE_CONV_005
- **名称**: Conv 非 4 对齐通道 — 7→11 无 bias
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: CMSIS-NN 对非对齐通道数的处理
- **网络结构**:
  ```
  Q/DQ Input → Conv(kernel=3×3, SAME_UPPER) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 7, 8, 8]`
- **关键参数**:
  - Conv: `{kernel: [3,3], stride: [1,1], pads: [1,1,1,1], group: 1, with_bias: False}`, weight shape `[11,7,3,3]`
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: in_ch=7, out_ch=11，均不对齐 4；无 bias
- **预期暴露的风险**: 非对齐通道时的 buffer padding 不足；无 bias 时 bias 数组处理
- **当前预期结果**: `ok`

---

##### P0_CORE_CONV_006
- **名称**: Conv 5×5 kernel — 大卷积核
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: kernel_size > 3 时的权重布局和 im2col buffer
- **网络结构**:
  ```
  Q/DQ Input → Conv(kernel=5×5, SAME_UPPER) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 3, 16, 16]`
- **关键参数**:
  - Conv: `{kernel: [5,5], stride: [1,1], pads: [2,2,2,2], group: 1}`, weight shape `[8,3,5,5]`
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: kernel=5×5，权重 600 个元素
- **预期暴露的风险**: scratch buffer 尺寸计算（大 kernel 需要更多 im2col buffer）
- **当前预期结果**: `ok`

---

#### 3.2.3 MaxPool（3 个用例）

---

##### P0_CORE_MAXPOOL_001
- **名称**: MaxPool 标准 2×2 stride=2
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: MaxPool 的参数映射，量化参数一致的正常路径
- **网络结构**:
  ```
  Q/DQ Input → MaxPool(kernel=2×2, stride=2) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 4, 8, 8]`
- **关键参数**:
  - MaxPool: `{kernel: [2,2], stride: [2,2], pads: [0,0,0,0]}`
- **量化参数设计**:
  | 张量 | scale | zero_point | 对称性 |
  |------|-------|-----------|--------|
  | input | 0.05 | 0 | symmetric |
  | output | 0.05 | 0 | symmetric |
- **边界/异常条件**: 输入/输出 scale 和 zp 完全一致（满足约束）
- **预期暴露的风险**: 正常路径无风险
- **当前预期结果**: `ok`

---

##### P0_CORE_MAXPOOL_002
- **名称**: MaxPool 量化不一致 — 应 blocked
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: Pool 输入/输出量化参数不一致时的正确拒绝行为
- **网络结构**:
  ```
  Q/DQ Input(scale=0.05) → MaxPool → Q/DQ Output(scale=0.03)
  ```
- **输入张量形状**: `[1, 4, 8, 8]`
- **关键参数**:
  - MaxPool: `{kernel: [2,2], stride: [2,2]}`
- **量化参数设计**:
  | 张量 | scale | zero_point | 对称性 |
  |------|-------|-----------|--------|
  | input | 0.05 | 0 | symmetric |
  | output | 0.03 | 0 | symmetric |
- **边界/异常条件**: 输入 scale ≠ 输出 scale — 违反 Pool 约束
- **预期暴露的风险**: N/A（期望失败）
- **当前预期结果**: `blocked` — 量化参数不一致

---

##### P0_CORE_MAXPOOL_003
- **名称**: MaxPool 输出 1×1 边界
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 池化到 1×1 的边界情况
- **网络结构**:
  ```
  Q/DQ Input → MaxPool(kernel=3×3, stride=1) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 4, 3, 3]`
- **关键参数**:
  - MaxPool: `{kernel: [3,3], stride: [1,1], pads: [0,0,0,0]}`
- **量化参数设计**: 输入/输出 scale 一致，均为 0.05, zp=0
- **边界/异常条件**: 输出 H=W=1
- **预期暴露的风险**: 单元素输出的 buffer 管理
- **当前预期结果**: `ok`

---

#### 3.2.4 Softmax（3 个用例）

---

##### P0_CORE_SOFTMAX_001
- **名称**: Softmax 10 分类标准
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 标准 Softmax 的 multiplier/shift/diff_min 计算
- **网络结构**:
  ```
  Q/DQ Input → Softmax → Q/DQ Output
  ```
- **输入张量形状**: `[1, 10]`
- **关键参数**: Softmax 无属性参数
- **量化参数设计**: 全部对称，zp=0, scale=0.05
- **边界/异常条件**: 10 分类标准场景
- **预期暴露的风险**: multiplier/shift 计算错误
- **当前预期结果**: `ok`

---

##### P0_CORE_SOFTMAX_002
- **名称**: Softmax 二分类边界
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 最小类别数的 Softmax
- **网络结构**:
  ```
  Q/DQ Input → Softmax → Q/DQ Output
  ```
- **输入张量形状**: `[1, 2]`
- **关键参数**: 无
- **量化参数设计**: 对称，zp=0
- **边界/异常条件**: num_classes=2
- **预期暴露的风险**: diff_min 计算在二分类时是否合理
- **当前预期结果**: `ok`

---

##### P0_CORE_SOFTMAX_003
- **名称**: Softmax 极小 scale
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 输入 scale 极小时 multiplier/shift 不溢出
- **网络结构**:
  ```
  Q/DQ Input(scale=0.001) → Softmax → Q/DQ Output
  ```
- **输入张量形状**: `[1, 10]`
- **关键参数**: 无
- **量化参数设计**: input/output scale=0.001, zp=0
- **边界/异常条件**: scale 极小（0.001）
- **预期暴露的风险**: diff_min 计算溢出或结果为 0
- **当前预期结果**: `ok`

---

### 3.3 阶段二 — 经典拓扑组合测试用例清单

---

##### P0_TOPO_T1
- **名称**: 最简 CNN — Conv+Relu
- **归属阶段**: 阶段二
- **优先级**: P0
- **主要验证目标**: Conv+Relu 融合（activation min/max 参数）、NCHW→NHWC 布局转换
- **网络结构**:
  ```
  Q/DQ Input(NCHW) → Conv → Relu → Q/DQ Output
  ```
- **输入张量形状**: `[1, 3, 8, 8]`
- **关键参数**:
  - Conv: `{kernel: [3,3], stride: [1,1], pads: [1,1,1,1], group: 1, with_bias: True}`, weight shape `[8,3,3,3]`
  - Relu: 融合到 Conv activation_min=0
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: 单 Conv，Relu 融合
- **预期暴露的风险**: Relu 融合时 activation_min 未正确设置
- **当前预期结果**: `ok`

---

##### P0_TOPO_T2
- **名称**: Conv→MaxPool 下采样链路
- **归属阶段**: 阶段二
- **优先级**: P0
- **主要验证目标**: Conv 输出 → MaxPool 输入的数据流衔接、Pool 量化一致性
- **网络结构**:
  ```
  Q/DQ Input → Conv → MaxPool → Q/DQ Output
  ```
- **输入张量形状**: `[1, 3, 8, 8]`
- **关键参数**:
  - Conv: `{kernel: [3,3], stride: [1,1], pads: [1,1,1,1]}`, weight `[8,3,3,3]`
  - MaxPool: `{kernel: [2,2], stride: [2,2]}`
- **量化参数设计**:
  - Conv 输出 scale 必须与 MaxPool 输入 scale 一致: 均设为 0.05
- **边界/异常条件**: Conv 输出 zp 必须 = MaxPool 输入 zp
- **预期暴露的风险**: 中间层量化参数不一致导致 Pool blocked；Conv 输出 shape 推导错误
- **当前预期结果**: `ok`

---

##### P0_TOPO_T3
- **名称**: 经典 CNN 分类头 — Conv→Relu→Pool→Flatten→FC
- **归属阶段**: 阶段二
- **优先级**: P0
- **主要验证目标**: 全链路数据流、buffer ping-pong、NCHW→NHWC→1D 展开
- **网络结构**:
  ```
  Q/DQ Input(1,3,8,8) → Conv → Relu → MaxPool → Flatten → FC(10) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 3, 8, 8]`
- **关键参数**:
  - Conv: `{kernel: [3,3], pads: [1,1,1,1]}`, weight `[8,3,3,3]`
  - MaxPool: `{kernel: [2,2], stride: [2,2]}`
  - Flatten: `{axis: 1}`, 展开后 8×4×4=128
  - FC: `{transB: 1}`, weight `[10, 128]`
- **量化参数设计**: 各层 zp 均设为 0（对称），确保 Pool 一致性
- **边界/异常条件**: 三层运行期算子 + Flatten 折叠
- **预期暴露的风险**: Flatten 后的维度映射到 FC 输入；buffer 切换错误
- **当前预期结果**: `ok`

---

##### P0_TOPO_T4
- **名称**: 三层 FC 串接 — buffer ping-pong 压力测试
- **归属阶段**: 阶段二
- **优先级**: P0
- **主要验证目标**: 多层 FC 的 activation buffer A/B 交替切换正确性
- **网络结构**:
  ```
  Q/DQ Input(1,64) → FC(64→32) → Relu → FC(32→16) → Relu → FC(16→8) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 64]`
- **关键参数**:
  - FC1: weight `[32, 64]`, transB=1
  - FC2: weight `[16, 32]`, transB=1
  - FC3: weight `[8, 16]`, transB=1
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: 三层 FC + 两层 Relu，奇数层 FC
- **预期暴露的风险**: buffer A↔B 切换在第 3 层时写回错误 buffer
- **当前预期结果**: `ok`

---

##### P0_TOPO_T5
- **名称**: 双 Conv 串接 — Conv→Conv→Pool→FC
- **归属阶段**: 阶段二
- **优先级**: P0
- **主要验证目标**: 两个连续 Conv 的 NHWC 数据流衔接
- **网络结构**:
  ```
  Q/DQ Input(1,3,16,16) → Conv1(3→8,k3) → Conv2(8→16,k3) → MaxPool(k2,s2) → Flatten → FC(16→10) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 3, 16, 16]`
- **关键参数**:
  - Conv1: `{kernel: [3,3], pads: [1,1,1,1]}`, weight `[8,3,3,3]`
  - Conv2: `{kernel: [3,3], pads: [1,1,1,1]}`, weight `[16,8,3,3]`
  - MaxPool: `{kernel: [2,2], stride: [2,2]}`
  - FC: weight `[10, 1024]` (16×8×8=1024)
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: 双 Conv → Pool
- **预期暴露的风险**: 第二个 Conv 的 NHWC 输入是否正确来自第一个 Conv 的 NHWC 输出；Pool 量化一致性
- **当前预期结果**: `ok`

---

##### P0_TOPO_T6
- **名称**: 完整 CNN 分类链路 — Conv→Pool→FC→Softmax
- **归属阶段**: 阶段二
- **优先级**: P0
- **主要验证目标**: 含 Softmax 末端的完整分类链路，验证端到端数据流
- **网络结构**:
  ```
  Q/DQ Input(1,1,8,8) → Conv(1→4,k3) → Relu → MaxPool(k2,s2) → Flatten → FC(64→10) → Softmax → Q/DQ Output
  ```
- **输入张量形状**: `[1, 1, 8, 8]`
- **关键参数**:
  - Conv: weight `[4,1,3,3]`
  - MaxPool: `{kernel: [2,2], stride: [2,2]}` → 输出 `[4,3,3]` → Flatten 36
  - FC: weight `[10, 36]`
  - Softmax: 10 类
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: 含 Softmax 末端 + Flatten 折叠
- **预期暴露的风险**: Softmax 的 multiplier 基于 FC 输出 scale 计算是否正确
- **当前预期结果**: `ok`

---

##### P0_TOPO_T7
- **名称**: GlobalAveragePool 分类头
- **归属阶段**: 阶段二
- **优先级**: P0
- **主要验证目标**: GlobalAveragePool 替代 Flatten+FC 的现代分类头
- **网络结构**:
  ```
  Q/DQ Input(1,8,4,4) → Conv(8→16,k3) → Relu → GlobalAveragePool → FC(16→10) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 8, 4, 4]`
- **关键参数**:
  - Conv: weight `[16,8,3,3]`, padding SAME
  - GlobalAveragePool: 自适应到 `[16,1,1]`
  - FC: weight `[10, 16]`
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: GlobalAveragePool 输出 H=W=1
- **预期暴露的风险**: GlobalAveragePool 参数解析、1×1 输出的 FC 衔接
- **当前预期结果**: `ok`

---

##### P0_TOPO_T8
- **名称**: 逻辑回归 — FC→Softmax 最简分类器
- **归属阶段**: 阶段二
- **优先级**: P0
- **主要验证目标**: 最小拓扑（单 FC + Softmax）
- **网络结构**:
  ```
  Q/DQ Input(1,28×28) → FC(784→10) → Softmax → Q/DQ Output
  ```
- **输入张量形状**: `[1, 784]`
- **关键参数**:
  - FC: weight `[10, 784]`, transB=1
  - Softmax: 10 类
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: 一维输入，模仿 MNIST 输入
- **预期暴露的风险**: 大权重数组（7840 个 int8 元素）的正确性
- **当前预期结果**: `ok`

---

### 3.4 阶段三 — 复杂结构测试用例清单

---

##### P1_EXT_E1
- **名称**: 残差连接 — Conv→Add(residual)
- **归属阶段**: 阶段三
- **优先级**: P1
- **主要验证目标**: Add 算子的双输入 requant 和多输入数据流
- **网络结构**:
  ```
  Q/DQ Input → Conv(k3,s1,p1) ──→ Add ←── Q/DQ Input (skip connection)
                                    ↓
                               Q/DQ Output
  ```
- **输入张量形状**: `[1, 4, 8, 8]`
- **关键参数**:
  - Conv: weight `[4,4,3,3]`, padding SAME（保持 shape）
  - Add: 双输入逐元素加法
- **量化参数设计**: 两路输入 scale 不同（0.05 vs 0.08），测试 requant
- **边界/异常条件**: 双输入不同 scale
- **预期暴露的风险**: 当前架构不支持多输入节点；Add renderer 缺失
- **当前预期结果**: `blocked` — Add 无独立 renderer

---

##### P1_EXT_E2
- **名称**: 多分支 Concat 合并
- **归属阶段**: 阶段三
- **优先级**: P1
- **主要验证目标**: 非线性的 DAG 拓扑 + Concat renderer
- **网络结构**:
  ```
                           ┌─ Conv(k1,s1) ─┐
  Q/DQ Input → Split(隐式) ─┤               ├─ Concat → FC → Q/DQ Output
                           └─ Conv(k3,s1) ─┘
  ```
- **输入张量形状**: `[1, 4, 8, 8]`
- **关键参数**:
  - Conv1: weight `[8,4,1,1]`
  - Conv2: weight `[8,4,3,3]`, padding SAME
  - Concat: axis=1（通道维），输出 `[1, 16, 8, 8]`
  - FC: weight `[10, 1024]`
- **量化参数设计**: 两分支量化参数不同
- **边界/异常条件**: DAG 拓扑打破线性假设
- **预期暴露的风险**: 线性拓扑假设导致 codegen 无法处理分支；Concat renderer 缺失
- **当前预期结果**: `blocked` — 非线性和 Concat

---

##### P1_EXT_E3
- **名称**: Depthwise Separable Conv 块
- **归属阶段**: 阶段三
- **优先级**: P1
- **主要验证目标**: DepthwiseConv renderer + pointwise 组合
- **网络结构**:
  ```
  Q/DQ Input → DepthwiseConv(k3,s1,group=in_ch) → Conv(k1,s1,group=1) → Q/DQ Output
  ```
- **输入张量形状**: `[1, 8, 8, 8]`
- **关键参数**:
  - DepthwiseConv: weight `[8,1,3,3]`, group=8
  - Conv(pointwise): weight `[16,8,1,1]`, group=1
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: DepthwiseConv 的 group ≠ 1
- **预期暴露的风险**: DepthwiseConv renderer 缺失；per-channel 量化参数
- **当前预期结果**: `blocked` — DepthwiseConv renderer

---

##### P1_EXT_E4
- **名称**: 双输出头 — 多任务
- **归属阶段**: 阶段三
- **优先级**: P1
- **主要验证目标**: 多输出模型的处理
- **网络结构**:
  ```
                            ┌─ FC(128→10) → Softmax → Q/DQ Output A
  Q/DQ Input → Conv → FC ──┤
                            └─ FC(128→5) → Softmax → Q/DQ Output B
  ```
- **输入张量形状**: `[1, 1, 8, 8]`
- **量化参数设计**: 全部对称
- **边界/异常条件**: 两个输出头
- **预期暴露的风险**: 多输出模型拓扑支持
- **当前预期结果**: `blocked` — 多输出

---

##### P1_EXT_E5
- **名称**: 运行期 Transpose
- **归属阶段**: 阶段三
- **优先级**: P1
- **主要验证目标**: 需要运行期执行的 Transpose（非折叠）
- **网络结构**:
  ```
  Q/DQ Input → Conv → Transpose(perm=[0,2,1,3]) → FC → Q/DQ Output
  ```
- **输入张量形状**: `[1, 4, 4, 4]`
- **量化参数设计**: 全部对称
- **边界/异常条件**: Transpose 在运行期节点之间
- **预期暴露的风险**: Transpose 被标记为折叠但实际需要运行期执行
- **当前预期结果**: `blocked` — 运行期 Transpose renderer

---

##### P1_EXT_E6
- **名称**: 多 Opset 兼容性 — Opset 11 / 13 / 17 同一模型
- **归属阶段**: 阶段三
- **优先级**: P1
- **主要验证目标**: converter 对不同 ONNX opset 版本的算子属性默认值差异能否正确归一化
- **网络结构**:
  ```
  Q/DQ Input → Conv(kernel=3×3) → Relu → MaxPool → Flatten → FC → Q/DQ Output
  ```
- **输入张量形状**: `[1, 3, 8, 8]`
- **关键参数**:
  - 同一拓扑分别以 opset 11、opset 13、opset 17 导出三个 ONNX 文件
  - 关键差异点：opset 11 中 Conv `pads` 默认行为、opset 13 中 Softmax axis 默认值变化
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: opset 11 模型可能缺少某些默认属性值
- **预期暴露的风险**: converter 对不同 opset 的属性归一化不完整；`SUPPORTED_OPSET_MIN`/`MAX` 边界处理错误；不同 opset 下同一算子语义差异被忽略
- **当前预期结果**: `blocked` 或 `unsupported`（若 opset 超出范围）— 当前 converter 仅验证 opset 范围，未对不同 opset 的算子默认值差异做系统测试
- **模型脚本路径**: `tdd/scripts/phase3/gen_opset_compat.py`
- **模型输出路径**: `tdd/models/phase3/ext/P1_EXT_E6_opset11.onnx` / `..._opset13.onnx` / `..._opset17.onnx`

---

##### P1_EXT_E7
- **名称**: 动态 Shape — 动态 H/W 维度
- **归属阶段**: 阶段三
- **优先级**: P1
- **主要验证目标**: converter 对动态非 batch 维度的校验与拒绝行为；为未来动态 Shape 支持埋点
- **网络结构**:
  ```
  Q/DQ Input(batch=1, C=3, H=?, W=?) → Conv(kernel=3×3, padding=SAME) →
      GlobalAveragePool → Flatten → FC(→10) → Softmax → Q/DQ Output
  ```
- **输入张量形状**: `[1, 3, "height", "width"]`（动态 H/W）
- **关键参数**:
  - Conv: weight `[8,3,3,3]`, padding SAME（需要知道输入尺寸才能计算输出）
  - FC: weight `[10, 8]`（GlobalAveragePool 后固定为 `[1,8,1,1]`，不受 H/W 影响）
- **量化参数设计**: 全部对称，zp=0
- **边界/异常条件**: 动态 H/W 维度导致 Conv 输出 shape 无法静态确定；GlobalAveragePool 通道数固定但空间维度动态
- **预期暴露的风险**: converter 未正确拒绝动态非 batch 维度而静默通过；C 端数组尺寸宏中出现动态值导致编译失败；shape inference 在动态维度时行为不确定
- **当前预期结果**: `unsupported` — 转换阶段应明确报错"动态非 batch 维度不支持"，不得静默传递
- **模型脚本路径**: `tdd/scripts/phase3/gen_dynamic_shape.py`
- **模型输出路径**: `tdd/models/phase3/ext/P1_EXT_E7.onnx`

---

### 3.5 阶段四 — 实战验证测试用例清单

---

##### P2_REAL_MOBILENET
- **名称**: MobileNetV1-α0.25 简化版
- **归属阶段**: 阶段四
- **优先级**: P2
- **主要验证目标**: 真实 mobile-scale 模型在编译器上的表现
- **网络结构**: MobileNetV1 的 depthwise separable conv block 堆叠
- **输入张量形状**: `[1, 3, 32, 32]`
- **当前预期结果**: `blocked` — DepthwiseConv 未实现（记录阻塞算子清单和预算估算）

---

##### P2_REAL_KWS
- **名称**: 简化关键词识别（TC-ResNet8 风格）
- **归属阶段**: 阶段四
- **优先级**: P2
- **主要验证目标**: 语音识别类模型的编译器行为
- **网络结构**: Conv → 残差块 × N → FC → Softmax
- **输入张量形状**: `[1, 1, 40, 32]`（MFCC 特征）
- **当前预期结果**: `blocked` — Add 残差未实现（记录阻塞算子）

---

### 3.6 负向测试用例清单

---

##### P0_NEG_N1
- **名称**: float32 ONNX 拒绝
- **归属阶段**: 阶段一（可与阶段一并行）
- **优先级**: P0
- **主要验证目标**: 无 Q/DQ 节点的 float32 模型被正确拒绝
- **网络结构**: `Input(FLOAT) → Conv → Relu → Output(FLOAT)`
- **当前预期结果**: `unsupported` — 返回非零退出码

---

##### P0_NEG_N2
- **名称**: transB=0 的 Gemm 拒绝
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 不支持的 Gemm 参数形态被拒绝
- **网络结构**: `Q/DQ Input → Gemm(transB=0) → Q/DQ Output`
- **当前预期结果**: `unsupported` 或 `blocked`

---

##### P0_NEG_N3
- **名称**: Grouped Conv (group>1, 非 depthwise) 拒绝
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: group > 1 但非 depthwise 的 Conv 被拒绝
- **当前预期结果**: `unsupported` 或 `blocked`

---

##### P0_NEG_N4
- **名称**: 动态非 batch 维度拒绝
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 动态 H/W 维度不被静默通过
- **当前预期结果**: `unsupported` — 转换阶段报错

---

##### P0_NEG_N5
- **名称**: Pool 量化不一致 blocked
- **归属阶段**: 阶段一
- **优先级**: P0
- **主要验证目标**: 同 P0_CORE_MAXPOOL_002，确保 codegen 正确拒绝
- **当前预期结果**: `blocked`

---

##### P0_NEG_N6
- **名称**: 未知算子处理
- **归属阶段**: 阶段一
- **优先级**: P1
- **主要验证目标**: converter 遇到未知算子时不崩溃
- **当前预期结果**: `unsupported`

---

##### P0_NEG_N7
- **名称**: 空模型
- **归属阶段**: 阶段一
- **优先级**: P1
- **主要验证目标**: 无节点的 ONNX 模型的优雅处理
- **当前预期结果**: 优雅失败，不崩溃

---

##### P0_NEG_N8
- **名称**: 损坏的 ONNX 文件
- **归属阶段**: 阶段一
- **优先级**: P1
- **主要验证目标**: 文件损坏时的错误处理
- **当前预期结果**: 明确错误信息

---

## 第四部分：执行、评估与迭代机制

### 4.1 工程目录规划

在项目根目录 `tdd/` 下按以下结构组织测试资产：

```text
tdd/
├── plans/                              # 测试计划与提示词
│   └── tdd_test_case_generation_plan.md   ← 本文件
│
├── models/                             # 生成的 ONNX 测试模型
│   ├── phase1/                         # 阶段一：核心算子
│   │   ├── core_gemm/                  # Gemm 用例
│   │   │   ├── P0_CORE_GEMM_001.onnx
│   │   │   ├── P0_CORE_GEMM_002.onnx
│   │   │   └── ...
│   │   ├── core_conv/                  # Conv 用例
│   │   ├── core_maxpool/               # MaxPool 用例
│   │   └── core_softmax/               # Softmax 用例
│   ├── phase2/                         # 阶段二：经典拓扑
│   │   └── topo/
│   │       ├── P0_TOPO_T1.onnx
│   │       └── ...
│   ├── phase3/                         # 阶段三：复杂结构
│   │   └── ext/
│   ├── phase4/                         # 阶段四：实战验证
│   │   └── real/
│   └── negative/                       # 负向测试
│       └── neg/
│
├── scripts/                            # 生成与执行脚本
│   ├── phase1/
│   │   ├── gen_core_gemm.py            # 生成全部 Gemm 用例
│   │   ├── gen_core_conv.py            # 生成全部 Conv 用例
│   │   ├── gen_core_maxpool.py
│   │   ├── gen_core_softmax.py
│   │   └── gen_negative.py
│   ├── phase2/
│   │   └── gen_topology.py
│   ├── phase3/
│   │   └── gen_extension.py
│   ├── phase4/
│   │   └── gen_real_world.py
│   ├── common/                         # 共享工具
│   │   ├── __init__.py
│   │   ├── model_builder.py            # 通用 ONNX 模型构建工具
│   │   ├── quant_helpers.py            # Q/DQ 量化参数辅助函数
│   │   └── runner.py                   # 统一的测试执行器
│   └── run_all.py                      # 一键运行全部测试
│
└── reports/                            # 测试结果与日志
    ├── phase1_report.md
    ├── phase2_report.md
    ├── ...
    └── logs/
        └── ...
```

### 4.2 测试模型生成流程

每个测试用例的 ONNX 模型通过以下标准流程生成：

```text
1. 编写 PyTorch 网络定义
   ↓  (使用 torch.nn.Module 构建精确的算子序列)
2. FP32 模型前向 + 权重初始化（确定性随机种子）
   ↓
3. PTQ 量化：在 FP32 模型上插入 Q/DQ 节点
   ↓  (使用 per-tensor int8 量化，按用例设计设置 scale/zp)
4. 导出 ONNX（opset 17，固定 shape）
   ↓
5. ONNX 合法性检查（onnx.checker.check_model）
   ↓
6. 保存到 tdd/models/ 对应目录
```

**关键约束**：

- 所有权重使用确定性随机种子（`torch.manual_seed(42)`），保证可复现
- Q/DQ 量化使用 `per-tensor`、`int8`，axis 为空
- 输入/输出 Q/DQ 节点必须成对出现，包围每个运行期算子
- ONNX 导出后使用 `onnx.shape_inference.infer_shapes()` 补全 shape
- 每个脚本生成一组同类别用例，共享公共工具函数

**通用模型构建工具** (`tdd/scripts/common/model_builder.py`) 应提供：

- `make_qdq_input(name, shape, scale, zp)` — 创建 Q/DQ 输入包装
- `make_qdq_output(name, shape, scale, zp)` — 创建 Q/DQ 输出包装
- `make_gemm_node(name, inputs, outputs, transB, weight_shape, with_bias)` — 创建 Gemm 节点
- `make_conv_node(name, inputs, outputs, kernel, stride, pads, group, weight_shape, with_bias)` — 创建 Conv 节点
- `make_maxpool_node(...)`, `make_avgpool_node(...)`, `make_softmax_node(...)` — 其他算子
- `make_quantize_linear(name, inputs, outputs, scale, zp)` — 创建 Q 节点
- `make_dequantize_linear(name, inputs, outputs, scale, zp)` — 创建 DQ 节点
- `build_and_save(nodes, inputs, outputs, initializers, save_path)` — 组装并保存

### 4.3 测试执行流程

```text
1. 扫描 tdd/models/ 中目标阶段的全部 ONNX 文件
   ↓
2. 对每个 ONNX 模型执行：
   a. python -m nanoc_nn.converter --model {onnx} --out {tmp}/converter-output
   b. python -m nanoc_nn.codegen --input {tmp}/converter-output --out {tmp}/cmsis-codegen
      (或使用 nanoc onnx-to-cmsis --model {onnx} --no-compile)
   ↓
3. 收集结果：
   - codegen status: ok / blocked / unsupported
   - pipeline 退出码
   - 错误/警告信息
   ↓
4. 对 status=ok 的模型，执行 C99 smoke compile：
   - gcc -std=c99 -Wall -Wextra -I{include} {model.c} {main.c} -o {smoke_bin}
   ↓
5. 对编译通过的模型，运行 smoke test：
   - 喂入全零/随机 int8 输入，检查返回值 == NANOC_STATUS_OK
   - 不检查数值精度（Level 2 目标）
   ↓
6. 生成分阶段报告
```

**统一执行器** `tdd/scripts/run_all.py` 应支持：

- `--phase 1|2|3|4|all` — 指定执行阶段
- `--compile` — 是否执行 C99 smoke compile
- `--run` — 是否执行 smoke run
- `--report-dir` — 报告输出目录

### 4.4 通过标准（Level 2 — 运行不崩溃）

| 级别 | 标准 | 适用阶段 |
|------|------|---------|
| **L1: 解析不崩溃** | converter 不抛未捕获异常；`model_graph.json` 合法 | 全部 |
| **L2: 生成不崩溃** | codegen status 符合预期（ok/blocked/unsupported）；产物目录完整 | 全部 |
| **L3: 编译通过** | C99 `-Wall -Wextra` 零错误零警告（status=ok 的用例） | P0 |
| **L4: 运行不崩溃** | 可执行文件正常退出，返回 `NANOC_STATUS_OK`，无 segfault | P0 |
| **L5: 精度对齐** | 与 ONNX Runtime int8 推理 argmax 一致（**暂不作为阻塞条件**） | 未来 |

**当前阶段（MVP）通过标准 = L1 + L2 + L3 + L4**。L5 精度对齐不在当前范围内。

### 4.5 失败处理流程

当测试用例未达到预期标准时，按以下流程记录和处理：

```text
1. 记录失败类型
   ├── CRASH:    converter/codegen 抛出未捕获异常
   ├── WRONG_STATUS: codegen status 与预期不符（预期 ok 但 blocked）
   ├── COMPILE_ERR: C99 编译失败
   └── RUNTIME_ERR: 运行崩溃/非零退出/挂起

2. 收集诊断信息
   ├── 完整 stdout/stderr
   ├── model_graph.json（若 converter 成功）
   ├── pipeline_report.md（若 pipeline 成功）
   ├── codegen_report.txt（若 codegen 成功）
   └── 编译器错误信息（若编译失败）

3. 生成失败报告
   - 存入 tdd/reports/logs/{case_id}_{timestamp}.log
   - 汇总到分阶段报告的失败部分

4. 关联到代码缺陷
   - 确定为 converter 缺陷 → 提 converter issue
   - 确定为 codegen 缺陷 → 提 codegen issue
   - 确定为测试用例设计问题 → 修正测试用例
```

### 4.6 计划迭代机制

本计划是一个**活文档**，随编译器能力演进持续更新：

| 触发条件 | 迭代动作 |
|---------|---------|
| 某阶段全部 P0 用例通过 | 解锁下一阶段入口；更新阶段状态标记 |
| 新增算子 renderer（如 DepthwiseConv） | 将对应的 P1 用例提升至 P0；新增该算子的原子测试用例 |
| 发现新风险点（如某参数组合导致崩溃） | 补充到对应分类的"陷阱注入"和"边界条件" |
| 编译器架构升级（如支持 DAG 拓扑） | 将阶段三相关用例从 `blocked` 预期改为 `ok` 预期 |
| 多 Opset 兼容性成为需求 | 激活 P1_EXT_E6：对不同 opset 的模型逐一验证 converter 解析正确性；扩展 `SUPPORTED_OPSET_MIN`/`MAX` 边界测试 |
| 动态 Shape 支持成为需求 | 激活 P1_EXT_E7：从"预期拒绝"转为"预期支持"；新增动态 batch > 1、动态 H/W 组合的端到端测试用例 |
| CMSIS-NN 版本升级 | 检查 API 兼容性；必要时新增适配层测试用例 |
| 精度对齐（L5）成为目标 | 在通过标准中激活 L5；新增参考推理脚本 |

### 4.7 与 CI/CD 的集成建议

- **阶段一 P0 用例**应纳入 pre-commit 或 PR gate，确保每次改动不破坏基础算子
- **阶段二 P0 用例**应纳入每日构建（nightly），作为回归防线
- **阶段三/四用例**按需触发，作为 release gate 的一部分

---

## 附录 A：测试用例总览表

| 阶段 | 分类 | 用例数 | P0 数 | P1 数 | P2 数 |
|------|------|--------|-------|-------|-------|
| 一 | Gemm/FC | 6 | 6 | 0 | 0 |
| 一 | Conv | 6 | 6 | 0 | 0 |
| 一 | MaxPool | 3 | 3 | 0 | 0 |
| 一 | Softmax | 3 | 3 | 0 | 0 |
| 二 | 组合拓扑 | 8 | 8 | 0 | 0 |
| 三 | 扩展驱动 | 7 | 0 | 7 | 0 |
| 四 | 实战验证 | 2 | 0 | 0 | 2 |
| 负向 | 异常处理 | 8 | 5 | 3 | 0 |
| **合计** | | **43** | **31** | **10** | **2** |

## 附录 B：当前编译器能力矩阵（2026-07-03 基线）

| ONNX 算子 | Converter 解析 | Codegen Renderer | 备注 |
|-----------|---------------|-----------------|------|
| Conv (group=1, dilation=1) | ✅ supported | ✅ `arm_convolve_wrapper_s8` | per-tensor→per-channel 扩展 |
| Gemm/MatMul (transB=1) | ✅ supported | ✅ `arm_fully_connected_s8` | |
| MaxPool | ✅ supported | ✅ `arm_max_pool_s8` | 输入/输出量化参数需一致 |
| AveragePool | ✅ supported | ✅ `arm_avgpool_s8` | 同上 |
| GlobalAveragePool | ✅ supported | ✅ `arm_avgpool_s8` | 同上 |
| Softmax | ✅ supported | ✅ `arm_softmax_s8` | |
| Relu/Clip | ✅ supported | ✅ fused (activation min/max) | 相邻算子融合 |
| QuantizeLinear/DQ | ✅ supported | ✅ folded | 量化边界 |
| Flatten/Reshape | ✅ supported | ✅ folded | 形状折叠 |
| Transpose | ✅ supported | ⚠️ folded only | 运行期 Transpose 无 renderer |
| DepthwiseConv | ✅ supported | ❌ blocked | renderer 缺失 |
| Add | ✅ supported | ❌ blocked | 双输入 requant 未固化 |
| Mul | ✅ supported | ❌ blocked | 同上 |
| Concat | ✅ supported | ❌ blocked | renderer 缺失 |

---

> **文档维护**：本计划由 CMSIS-NN 编译器测试团队维护。每次编译器能力变更后，应同步更新附录 B 的能力矩阵和受影响测试用例的预期结果。

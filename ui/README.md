# NanoC-NN UI 开发入口

本目录用于承载 NanoC-NN 图形化界面的设计、原型和后续实现。当前阶段先定义 UI 应该服务的用户流程与功能边界，不急于选框架或写界面代码。

当前正式桌面路线选择 **PySide6 / Qt for Python**，目标是做成 Windows 与 macOS 风格一致的工程工具界面，接近 STM32CubeMX 一类软件。`ui/app/` 保留为早期网页验证版，`ui/desktop_pyside6/` 是后续主线。

## UI 的定位

NanoC-NN 的核心能力是：

```text
用户输入 int8 量化 ONNX
  -> converter 解析 ONNX
  -> codegen 生成 CMSIS-NN C 推理工程
  -> 测试生成物与原始 ONNX 的一致性
  -> 面向具体 Cortex-M 平台做预算拦截
```

UI 应该把这条链路变成一个可观察、可复现、可解释的工作台。它不是一个营销首页，也不是普通文件转换器；它更像嵌入式工程师使用的模型移植控制台。

## 目标用户

- 嵌入式工程师：希望把已有 int8 ONNX 转成 STM32/GD32 可集成的 C 工程。
- 算法工程师：希望确认自己的量化模型是否符合端侧部署约束。
- 项目维护者：希望通过 UI 快速复现 TDD 中的模型转换、报告和数值测试。

## 核心工作流

### 1. 导入 ONNX

用户可以选择一个本地 `.onnx` 文件，UI 需要展示：

- 文件名、大小、路径。
- 输入/输出张量名称、shape、类型。
- ONNX opset、producer、节点数量。
- 算子统计，例如 `QLinearConv x26`、`Concat x8`。
- 是否疑似 int8/QDQ/QLinear 模型。

导入阶段只做解析预览，不应直接等同于支持。

### 2. 选择目标配置

用户可以选择或填写：

- Target core：`cortex-m3`、`cortex-m4`、`cortex-m7`、`cortex-m33`、`cortex-m55` 等。
- CMSIS-NN backend：自动 / scalar / DSP / MVE。
- SRAM budget：可选。
- Flash budget：可选。
- 输出目录。
- 是否生成 debug reports。

重要规则：

- 常规转换能力判断不应被某个默认 SRAM/Flash 预算提前拦截。
- 只有用户显式配置预算时，UI 才展示平台预算结论。
- 预算不满足时应显示 `oversize`，而不是把它混成 `blocked`。

### 3. 执行转换

UI 触发现有 CLI / pipeline：

```bash
nanoc onnx-to-cmsis --model <model.onnx> --target <core> --out-root <dir>
```

如果用户配置预算，则追加：

```bash
--sram-budget <value> --flash-budget <value>
```

执行过程中 UI 应展示：

- 当前阶段：converter / codegen / compile / numeric test。
- 命令行参数。
- 实时日志。
- 输出目录。
- 最终状态：`ok` / `blocked` / `unsupported` / `oversize`。

### 4. 查看转换报告

UI 应把生成报告结构化展示，而不是只给一堆文件路径：

- `pipeline_report.md`
- `conversion_report.txt`
- `model_summary.md`
- `codegen_report.txt`
- `op_mapping.md`
- `quantization.md`
- `memory_plan.md`
- `target_report.md`
- `unsupported_ops.md`

推荐视图：

- 总览：状态、节点数量、生成物路径、是否可交付。
- 算子映射：ONNX op -> CMSIS-NN API。
- 量化参数：scale、zero point、multiplier、shift、缺失字段。
- 内存规划：input/output/activation/scratch/weight flash。
- 不支持原因：blocked/unsupported 节点列表。
- 平台预算：SRAM/Flash 是否超限。

### 5. 生成物浏览

当状态为 `ok` 或用户允许查看 debug output 时，UI 应能打开：

- `cmsis-codegen/include/model.h`
- `cmsis-codegen/include/model_weights.h`
- `cmsis-codegen/src/model.c`
- `cmsis-codegen/src/main.c`
- `cmsis-codegen/CMakeLists.txt`
- `firmware_integration.md`

重点展示：

- `nanoc_model_run()` 是否包含真实 CMSIS-NN 调用。
- 生成的静态 buffer 尺寸。
- 是否存在 `arm_convolve_wrapper_s8`、`arm_fully_connected_per_channel_s8` 等 API。
- `NANOC_MODEL_ESTIMATED_SRAM_BYTES` 与 `NANOC_MODEL_ESTIMATED_FLASH_BYTES`。

### 6. 数值测试

UI 应支持两种数值测试：

#### TDD 固定用例

对仓库内用例运行：

```bash
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_synthetic_smoke --generate
```

展示：

- ONNX top1。
- C top1。
- top1 一致率。
- label accuracy，如果数据集有标签。
- saturation ratio。
- max abs error。
- 每个样本的对比明细。

#### 用户模型自定义数据集

后续可支持用户上传或录入输入样本：

- `.json` 数据集。
- `.npy` / `.npz` 输入。
- 图片输入和预处理配置。

这部分不应在第一版做复杂，但 UI 需要为它留位置。

### 7. TDD 工作台

UI 不只是转换工具，也应该能服务当前 TDD 开发模式：

- 查看当前能力集 `tdd/CAPABILITIES.md`。
- 查看当前状态 `tdd/STATUS.md`。
- 查看迭代记录 `tdd/iterations/`。
- 运行：
  - `run_tests.py --validate-only`
  - `run_regression.py --generate`
  - `run_tests.py --mode target --generate`
  - 单个 case。
- 展示 PASS / FAIL / blocked / oversize 的历史变化。

## 状态解释

UI 必须把状态解释清楚：

| 状态 | UI 文案 |
|------|---------|
| `ok` | 可生成当前支持范围内的 CMSIS-NN C 推理工程 |
| `blocked` | 模型语义尚未完整支持，需要查看算子、量化或 renderer 缺口 |
| `unsupported` | 当前明确不支持该 ONNX 形态 |
| `oversize` | 代码生成语义通过，但目标平台 SRAM/Flash 预算不足 |

特别注意：`oversize` 是平台适配结果，不是 ONNX 解析或 codegen 能力失败。

## 第一版功能建议

第一版 UI 应该小而完整，优先实现：

1. ONNX 文件选择。
2. 模型结构预览。
3. target / backend / budget 表单。
4. 一键运行 `onnx-to-cmsis`。
5. 状态与日志展示。
6. 报告浏览。
7. 生成物文件浏览。
8. TDD 稳定回归按钮。

暂不优先：

- 用户账号。
- 云端任务队列。
- 复杂数据集管理。
- 多模型项目管理。
- 在线训练或量化。
- 图形化编辑 ONNX。

## UI 信息架构草案

```text
ui/
├── README.md                 # 当前 UI 需求与设计入口
├── product/                  # 后续产品文档
├── prototypes/               # 低保真原型或静态页面
├── app/                      # 后续前端应用源码
└── fixtures/                 # UI 专用示例输入，不放大型模型
```

## 第一版页面草案

### Convert

主工作台页面。

- 左侧：模型导入与目标配置。
- 中间：执行进度、状态、日志。
- 右侧：状态解释、输出目录、主要报告入口。

### Reports

报告浏览页面。

- Summary
- Op Mapping
- Quantization
- Memory
- Unsupported / Blocked
- Target

### Generated Code

生成物浏览页面。

- 文件树。
- `model.c` 预览。
- 关键 CMSIS-NN API 搜索。
- buffer 宏摘要。

### TDD

测试驱动工作台。

- 能力集摘要。
- 稳定回归按钮。
- target 全量测试按钮。
- numeric 测试入口。
- 最近结果。

## 后端接口草案

第一版可以先用本地 Python server 包装现有 CLI：

| 方法 | 路径 | 用途 |
|------|------|------|
| `POST` | `/api/models/inspect` | 解析 ONNX 基本信息 |
| `POST` | `/api/convert` | 执行 converter + codegen |
| `GET` | `/api/jobs/<id>` | 查询任务状态和日志 |
| `GET` | `/api/reports/<job_id>` | 获取结构化报告列表 |
| `GET` | `/api/files/<job_id>` | 浏览生成物文件 |
| `POST` | `/api/tdd/regression` | 运行稳定回归 |
| `POST` | `/api/tdd/numeric` | 运行数值测试 |

约束：

- 后端只调用现有 pipeline 和 TDD 脚本，不复制 converter/codegen 逻辑。
- 后端必须保留原始命令和输出目录，保证结果可复现。
- UI 不能把报告状态直接当作成功，必须按 `ok`、`blocked`、`unsupported`、`oversize` 解释。

## 风险与待定问题

- 大型 ONNX 文件上传和复制策略需要谨慎，避免误把用户模型提交进 git。
- 长任务需要可取消、可查看实时日志。
- 数值测试的数据集格式尚需继续标准化。
- 生成物预览不能替代 C 编译和数值测试。
- 后续如果接入浏览器 UI，要明确本地文件权限边界。

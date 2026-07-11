"""
tdd/scripts/support_matrix.py — ONNX schema driven support matrix.

This module is the machine-readable bridge between ONNX operator schema forms
and NanoC-NN TDD cases.  Cases must reference one or more support IDs from this
file; no case should stand outside the ONNX-schema-driven tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SupportEntry:
    support_id: str
    schema_source: str
    domain: str
    op_type: str
    opset_range: str
    schema_form: str
    lowering: str
    backend: str
    capability_type: str
    status: str
    cmsis_apis: tuple[str, ...] = field(default_factory=tuple)
    planned: bool = False
    notes: str = ""


SUPPORT_MATRIX: tuple[SupportEntry, ...] = (
    SupportEntry(
        support_id="ONNX_CONV_RANK4_QDQ_INT8",
        schema_source="official",
        domain="ai.onnx",
        op_type="Conv",
        opset_range="11+",
        schema_form="rank=4 NCHW, group=1, static shape, QDQ/int8 parameters",
        lowering="cmsis_conv2d_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_convolve_wrapper_s8",),
    ),
    SupportEntry(
        support_id="ONNX_CONV_RANK3_QDQ_INT8",
        schema_source="official",
        domain="ai.onnx",
        op_type="Conv",
        opset_range="11+",
        schema_form="rank=3 NCW, group=1, static shape, QDQ/int8 parameters",
        lowering="cmsis_conv1d_as_2d_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_convolve_wrapper_s8",),
        notes="Business-facing Conv1d cases are ONNX Conv rank=3 forms.",
    ),
    SupportEntry(
        support_id="ONNX_CONV_DEPTHWISE_QDQ_INT8",
        schema_source="official",
        domain="ai.onnx",
        op_type="Conv",
        opset_range="11+",
        schema_form="rank=4, group=input_channels, depthwise multiplier, QDQ/int8 parameters",
        lowering="cmsis_depthwise_conv2d_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_depthwise_conv_wrapper_s8",),
    ),
    SupportEntry(
        support_id="ONNX_GEMM_QDQ_INT8",
        schema_source="official",
        domain="ai.onnx",
        op_type="Gemm",
        opset_range="11+",
        schema_form="static 2-D fully connected form, optional bias, QDQ/int8 parameters",
        lowering="cmsis_fully_connected_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_fully_connected_s8",),
    ),
    SupportEntry(
        support_id="ONNX_MATMUL_QDQ_INT8",
        schema_source="official",
        domain="ai.onnx",
        op_type="MatMul",
        opset_range="11+",
        schema_form="static fully connected compatible form, QDQ/int8 parameters",
        lowering="planned_cmsis_fully_connected_s8",
        backend="reject",
        capability_type="import_boundary",
        status="blocked",
        planned=True,
        notes="Present in mapper, but not yet protected by an independent TDD case.",
    ),
    SupportEntry(
        support_id="ONNX_MAXPOOL_QDQ_INT8",
        schema_source="official",
        domain="ai.onnx",
        op_type="MaxPool",
        opset_range="11+",
        schema_form="rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible pool window",
        lowering="cmsis_maxpool_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_max_pool_s8",),
    ),
    SupportEntry(
        support_id="ONNX_SOFTMAX_QDQ_INT8",
        schema_source="official",
        domain="ai.onnx",
        op_type="Softmax",
        opset_range="11+",
        schema_form="static classification vector, int8 output path",
        lowering="cmsis_softmax_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_softmax_s8",),
    ),
    SupportEntry(
        support_id="ONNX_QLINEARCONV_UINT8",
        schema_source="official",
        domain="ai.onnx",
        op_type="QLinearConv",
        opset_range="10+",
        schema_form="uint8 activation, static rank=4, CMSIS-compatible convolution",
        lowering="cmsis_qlinearconv_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_convolve_wrapper_s8",),
    ),
    SupportEntry(
        support_id="ONNX_QLINEARCONV_PER_CHANNEL",
        schema_source="official",
        domain="ai.onnx",
        op_type="QLinearConv",
        opset_range="10+",
        schema_form="static rank=4 convolution with per-channel weight scale",
        lowering="cmsis_qlinearconv_per_channel_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_convolve_wrapper_s8",),
    ),
    SupportEntry(
        support_id="ONNX_QLINEARADD_INT8",
        schema_source="extension",
        domain="nanoc.observed",
        op_type="QLinearAdd",
        opset_range="10+",
        schema_form="two static int8 tensors or activation plus bias-like tensor",
        lowering="cmsis_elementwise_add_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_elementwise_add_s8",),
        notes="Observed in quantized fixtures, but not present in official ONNX 1.22.0 schema catalog.",
    ),
    SupportEntry(
        support_id="ONNX_QGLOBALAVGPOOL_INT8",
        schema_source="extension",
        domain="nanoc.observed",
        op_type="QLinearGlobalAveragePool",
        opset_range="10+",
        schema_form="static rank=4 quantized global average pool",
        lowering="cmsis_avgpool_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_avgpool_s8",),
        notes="Observed in quantized fixtures, but not present in official ONNX 1.22.0 schema catalog.",
    ),
    SupportEntry(
        support_id="ONNX_QLINEARMATMUL_INT8",
        schema_source="official",
        domain="ai.onnx",
        op_type="QLinearMatMul",
        opset_range="21+",
        schema_form="static fully connected compatible quantized matmul",
        lowering="cmsis_fully_connected_per_channel_s8",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_fully_connected_per_channel_s8",),
    ),
    SupportEntry(
        support_id="ONNX_CONCAT_QDQ_INT8",
        schema_source="official",
        domain="ai.onnx",
        op_type="Concat",
        opset_range="11+",
        schema_form="rank=4 NCHW, axis=1 channel concat, same-quantization QDQ/int8 tensors",
        lowering="cmsis_concatenation_s8_z",
        backend="cmsis-nn",
        capability_type="cmsis_int8",
        status="ok",
        cmsis_apis=("arm_concatenation_s8_z",),
        notes="First protected form is the SqueezeNet-style channel concatenation branch.",
    ),
    SupportEntry(
        support_id="ONNX_FLOAT_REFERENCE_STATIC",
        schema_source="mixed",
        domain="ai.onnx",
        op_type="multiple",
        opset_range="11+",
        schema_form="static-shape float or QDQ graph executed by generated C reference backend",
        lowering="c99_reference_runtime",
        backend="c-reference",
        capability_type="reference_float",
        status="ok",
        notes="Verification backend only; not a CMSIS-NN int8 delivery claim.",
    ),
    SupportEntry(
        support_id="ONNX_FLOAT_NO_QDQ_BLOCKED",
        schema_source="mixed",
        domain="ai.onnx",
        op_type="multiple",
        opset_range="11+",
        schema_form="internal float32 graph without QDQ quantization section",
        lowering="reject_without_quantization",
        backend="reject",
        capability_type="negative",
        status="blocked",
        notes="Protects the CMSIS-NN int8 product boundary.",
    ),
    SupportEntry(
        support_id="ONNX_SQUEEZENET_QDQ_BLOCKED",
        schema_source="mixed",
        domain="ai.onnx",
        op_type="multiple",
        opset_range="10+",
        schema_form="SqueezeNet int8 with QDQ/Concat/GlobalAveragePool/Softmax boundary gaps",
        lowering="blocked_import_boundary",
        backend="reject",
        capability_type="import_boundary",
        status="blocked",
    ),
    SupportEntry(
        support_id="ONNX_SSD_MOBILENET_UNSUPPORTED",
        schema_source="mixed",
        domain="ai.onnx",
        op_type="multiple",
        opset_range="10+",
        schema_form="SSD-MobileNet detection graph with multi-output heads/dynamic or postprocess ops",
        lowering="unsupported_detection_boundary",
        backend="reject",
        capability_type="import_boundary",
        status="unsupported",
    ),
    SupportEntry(
        support_id="ONNX_EFFICIENTNET_LITE_UNSUPPORTED",
        schema_source="mixed",
        domain="ai.onnx",
        op_type="multiple",
        opset_range="10+",
        schema_form="EfficientNet-Lite quantized graph forms outside current layout/op whitelist",
        lowering="unsupported_import_boundary",
        backend="reject",
        capability_type="import_boundary",
        status="unsupported",
    ),
)

SUPPORT_MAP: dict[str, SupportEntry] = {
    entry.support_id: entry for entry in SUPPORT_MATRIX
}

VALID_SUPPORT_STATUSES = {"ok", "blocked", "unsupported", "oversize"}
VALID_BACKENDS = {"cmsis-nn", "c-reference", "reject"}
VALID_CAPABILITY_TYPES = {
    "cmsis_int8",
    "reference_float",
    "negative",
    "import_boundary",
}
VALID_SCHEMA_SOURCES = {"official", "extension", "mixed"}

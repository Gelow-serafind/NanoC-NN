from __future__ import annotations

from onnx import TensorProto
from onnx.onnx_pb import ValueInfoProto

from .model import ConversionError, TensorInfo


def tensor_dtype_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return f"UNKNOWN_{elem_type}"


def tensor_info_from_value_info(
    value_info: ValueInfoProto,
    *,
    batch_size: int,
    strict: bool,
    warnings: list[str],
    role: str,
) -> TensorInfo:
    tensor_type = value_info.type.tensor_type
    elem_type = tensor_dtype_name(tensor_type.elem_type)
    raw_shape: list[int | str | None] = []
    resolved_shape: list[int | str | None] = []
    dynamic_axes: list[int] = []

    for axis, dim in enumerate(tensor_type.shape.dim):
        if dim.HasField("dim_value") and dim.dim_value > 0:
            value: int | str | None = int(dim.dim_value)
            resolved: int | str | None = value
        elif dim.HasField("dim_param") and dim.dim_param:
            value = dim.dim_param
            resolved = batch_size if axis == 0 else None
            dynamic_axes.append(axis)
        else:
            value = None
            resolved = batch_size if axis == 0 else None
            dynamic_axes.append(axis)

        raw_shape.append(value)
        resolved_shape.append(resolved)

        if axis == 0 and value != resolved and resolved == batch_size:
            warnings.append(
                f"{role} '{value_info.name}' has dynamic batch axis; fixed to {batch_size}."
            )
        elif axis != 0 and value != resolved:
            message = (
                f"{role} '{value_info.name}' has dynamic non-batch axis {axis}; "
                "static C buffer size is unknown."
            )
            if strict:
                raise ConversionError(message)
            warnings.append(message)

    return TensorInfo(
        name=value_info.name,
        elem_type=elem_type,
        shape=resolved_shape,
        raw_shape=raw_shape,
        dynamic_axes=dynamic_axes,
    )


def shape_to_text(shape: list[int | str | None]) -> str:
    if not shape:
        return "[]"
    return "[" + ", ".join("unknown" if item is None else str(item) for item in shape) + "]"


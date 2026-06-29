import onnx
from onnx import helper

from nanoc_onnx_converter.parser import extract_attributes, normalize_attributes


def test_extract_attributes_returns_plain_values() -> None:
    node = helper.make_node(
        "Conv",
        inputs=["x", "w"],
        outputs=["y"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        group=1,
    )

    assert extract_attributes(node.attribute) == {
        "group": 1,
        "kernel_shape": [3, 3],
        "strides": [2, 2],
    }


def test_normalize_gemm_defaults() -> None:
    assert normalize_attributes(op_type="Gemm", attributes={}, weight_infos=[]) == {
        "alpha": 1.0,
        "beta": 1.0,
        "transA": 0,
        "transB": 0,
    }


def test_tensor_attribute_is_serializable() -> None:
    tensor = helper.make_tensor(
        name="shape",
        data_type=onnx.TensorProto.INT64,
        dims=[2],
        vals=[1, 3],
    )
    node = helper.make_node("Constant", inputs=[], outputs=["shape"], value=tensor)

    assert extract_attributes(node.attribute) == {
        "value": {
            "name": "shape",
            "data_type": "INT64",
            "dims": [2],
            "element_count": 2,
            "values": [1, 3],
            "preview": None,
        }
    }


def test_normalize_new_operator_defaults() -> None:
    assert normalize_attributes(op_type="Add", attributes={}, weight_infos=[]) == {
        "broadcast": "numpy"
    }
    assert normalize_attributes(op_type="Cast", attributes={"to": 1}, weight_infos=[]) == {
        "to": 1,
        "to_dtype": "FLOAT",
        "saturate": 1,
    }
    assert normalize_attributes(
        op_type="BatchNormalization",
        attributes={},
        weight_infos=[],
    ) == {
        "epsilon": 1e-5,
        "momentum": 0.9,
        "training_mode": 0,
    }

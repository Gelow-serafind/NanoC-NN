from nanoc_onnx_converter.naming import make_unique_c_names, sanitize_c_identifier


def test_sanitize_c_identifier_adds_prefix_and_cleans_invalid_chars() -> None:
    assert sanitize_c_identifier("conv1.weight", prefix="ex0") == "ex0_conv1_weight"
    assert sanitize_c_identifier("1.bad-name", prefix="") == "_1_bad_name"


def test_make_unique_c_names_resolves_cleaning_collisions() -> None:
    names = ["conv1.w", "conv1_w", "conv1/w"]
    assert make_unique_c_names(names, prefix="m") == {
        "conv1.w": "m_conv1_w",
        "conv1_w": "m_conv1_w_2",
        "conv1/w": "m_conv1_w_3",
    }


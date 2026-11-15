import numpy as np
from PIL import Image

from model_registry._scale import prepare_model_input, restore_model_result
from model_registry.base import ModelResult, TileRequest
from model_registry.cellvit import CellViTAdapter


def test_scale_policy_resamples_rgb_and_restores_integer_ids():
    request = TileRequest(
        "tile.png", "slide", output_size=4, border_ratio=0.25, mpp=0.5
    )
    tile = Image.new("RGB", (6, 6), (10, 20, 30))

    scaled_tile, scaled_request, scale, warning = prepare_model_input(
        tile, request, source_mpp=0.5, reference_mpp=0.25
    )

    assert scaled_tile.size == (12, 12)
    assert scaled_request.output_size == 8
    assert scale == 2
    assert warning is None

    native_labels = np.zeros((8, 8), dtype=np.int32)
    native_labels[2:6, 2:6] = 17
    result = ModelResult(
        native_labels, (native_labels > 0).astype(np.uint8),
        "cellsam", "1.0", output_size=8,
        instance_classes={17: "cell"},
        class_probabilities={17: {"cell": 0.9}},
    )
    restored = restore_model_result(
        result, request, source_mpp=0.5, reference_mpp=0.25,
        scale=scale, input_shape=(6, 6, 3),
    )

    assert restored.instance_mask.shape == (4, 4)
    assert set(np.unique(restored.instance_mask)) == {0, 17}
    assert restored.scale_factor == 2
    assert restored.instance_classes == {17: "cell"}
    assert restored.class_probabilities == {17: {"cell": 0.9}}
    assert restored.metadata["scale_policy"]["label_interpolation"] == "nearest"


def test_scale_policy_marks_missing_mpp_unverified():
    request = TileRequest("tile.png", "slide", output_size=4)
    tile = Image.new("RGB", (4, 4))

    scaled_tile, scaled_request, scale, warning = prepare_model_input(
        tile, request, source_mpp=None, reference_mpp=0.25
    )

    assert scaled_tile is tile
    assert scaled_request.output_size == 4
    assert scale == 1
    assert warning and "scale_unverified" in warning


def test_cellvit_adapter_preserves_native_instance_types():
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:6, 2:6] = 3
    adapter = CellViTAdapter(
        predictor=lambda _image, _request: {
            "instance_map": labels,
            "instance_classes": {3: {"class_id": 2, "label": "Inflammatory"}},
            "class_probabilities": {3: {"Inflammatory": 0.8}},
        }
    )
    adapter.load()
    request = TileRequest("tile.png", "slide", output_size=8)

    raw = adapter.predict(Image.new("RGB", (8, 8)), request)
    result = adapter.postprocess(raw, request)

    assert result.instance_classes == {
        3: {"class_id": 2, "label": "Inflammatory"}
    }
    assert result.class_probabilities == {3: {"Inflammatory": 0.8}}

"""Physical-scale preprocessing shared by model adapters."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image

from .base import ModelResult, TileRequest
from .tile_io import border_pixels


def prepare_model_input(
    tile: Image.Image,
    request: TileRequest,
    *,
    source_mpp: float | None,
    reference_mpp: float | None,
) -> tuple[Image.Image, TileRequest, float, str | None]:
    """Resample RGB to the requested physical scale before inference."""
    if source_mpp is None or reference_mpp is None:
        return tile, replace(request, mpp=source_mpp), 1.0, (
            "scale_unverified: source MPP or model reference MPP is unavailable"
        )

    scale = source_mpp / reference_mpp
    scaled_size = max(1, int(round(request.output_size * scale)))
    scaled_request = replace(request, output_size=scaled_size, mpp=source_mpp)
    border = border_pixels(scaled_size, request.border_ratio)
    expected = scaled_size + 2 * border
    if tile.size != (expected, expected):
        resampling = (
            Image.Resampling.BOX if scale < 1 else Image.Resampling.BILINEAR
        )
        tile = tile.resize((expected, expected), resampling)
    return tile, scaled_request, scale, None


def restore_model_result(
    result: ModelResult,
    request: TileRequest,
    *,
    source_mpp: float | None,
    reference_mpp: float | None,
    scale: float,
    input_shape: tuple[int, ...],
) -> ModelResult:
    """Restore scaled instance labels to canonical input coordinates."""
    labels = result.instance_mask
    if scale != 1.0:
        labels = Image.fromarray(labels.astype("int32"), mode="I").resize(
            (request.output_size, request.output_size), Image.Resampling.NEAREST
        )
        labels = np.asarray(labels, dtype="int32")
        result.instance_mask = labels
        result.binary_mask = (labels > 0).astype("uint8")
    visible_ids = {int(value) for value in np.unique(labels) if value > 0}
    if result.instance_classes is not None:
        result.instance_classes = {
            key: value for key, value in result.instance_classes.items()
            if int(key) in visible_ids
        }
    if result.class_probabilities is not None:
        result.class_probabilities = {
            key: value for key, value in result.class_probabilities.items()
            if int(key) in visible_ids
        }

    result.output_size = request.output_size
    result.output_shape = (request.output_size, request.output_size)
    result.input_shape = input_shape
    result.source_mpp = source_mpp
    result.reference_mpp = reference_mpp
    result.scale_factor = scale
    result.resize_scale = scale
    result.metadata = {
        **dict(result.metadata),
        "scale_policy": {
            "source_mpp": source_mpp,
            "reference_mpp": reference_mpp,
            "output_pixels_per_input_pixel": scale,
            "rgb_interpolation": "box" if scale < 1 else "bilinear",
            "label_interpolation": "nearest",
            "labels_restored_to_input_coordinates": True,
        },
    }
    return result

"""CellViT-Inference adapter for single 1024-pixel patches."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ._adapter_utils import checked_checkpoint, result, rgb_array
from .base import ModelAdapter, TileRequest


class CellViTAdapter(ModelAdapter):
    model_name = "cellvit"
    version = "1.1"

    def __init__(self, *, predictor: Any = None):
        self._predictor = predictor
        self._model = None
        self._postprocessor = None
        self._torch = None
        self._functional = None
        self._checkpoint: Path | None = None
        self._device = "cpu"
        self._patch_size = 16
        self._mean = (0.5, 0.5, 0.5)
        self._std = (0.5, 0.5, 0.5)
        self._label_map: dict[int, str] = {}
        self._builtin_checkpoint = False
        self._last_padding = (0, 0)
        self.reference_mpp: float | None = None

    def load(self, checkpoint_path: Path | None = None, *, device: str = "cpu") -> None:
        self._device = device
        if self._predictor is not None:
            return

        try:
            import torch
            import torch.nn.functional as functional
            from cellvit.inference.postprocessing_numpy import DetectionCellPostProcessor
            from cellvit.config.config import TYPE_NUCLEI_DICT_PANNUKE
            from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
            from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
            from cellvit.utils.cache_models import cache_cellvit_sam_h
            from cellvit.utils.tools import unflatten_dict
        except ImportError as exc:
            raise RuntimeError(
                "CellViT requires the official cellvit-inference package, "
                "PyTorch, and its model dependencies."
            ) from exc

        self._builtin_checkpoint = checkpoint_path is None
        self._checkpoint = checked_checkpoint(
            checkpoint_path, self.model_name
        ) if checkpoint_path is not None else Path(cache_cellvit_sam_h())
        checkpoint = torch.load(str(self._checkpoint), map_location="cpu")
        run_config = unflatten_dict(checkpoint["config"], ".")
        num_nuclei_classes = int(run_config["data"]["num_nuclei_classes"])
        num_tissue_classes = int(run_config["data"]["num_tissue_classes"])
        architecture = checkpoint["arch"]
        if architecture == "CellViT256":
            model = CellViT256(
                model256_path=None,
                num_nuclei_classes=num_nuclei_classes,
                num_tissue_classes=num_tissue_classes,
                regression_loss=run_config["model"].get("regression_loss", False),
            )
        elif architecture == "CellViTSAM":
            model = CellViTSAM(
                model_path=None,
                num_nuclei_classes=num_nuclei_classes,
                num_tissue_classes=num_tissue_classes,
                vit_structure=run_config["model"]["backbone"],
                regression_loss=run_config["model"].get("regression_loss", False),
            )
        else:
            raise RuntimeError(
                f"unsupported CellViT checkpoint architecture {architecture!r}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device)

        transformations = run_config.get("transformations", {})
        normalization = transformations.get("normalize", {})
        self._mean = tuple(normalization.get("mean", (0.5, 0.5, 0.5)))
        self._std = tuple(normalization.get("std", (0.5, 0.5, 0.5)))
        label_map = run_config["data"].get("label_map", TYPE_NUCLEI_DICT_PANNUKE)
        self._label_map = {
            int(key): str(value) for key, value in label_map.items()
        }
        self._patch_size = int(model.patch_size)
        self._torch = torch
        self._functional = functional
        self._model = model
        self._postprocessor = DetectionCellPostProcessor(
            wsi=None, nr_types=num_nuclei_classes
        )
        if self._builtin_checkpoint:
            self.reference_mpp = 0.25

    def predict(self, tile: Any, request: TileRequest) -> Any:
        image = rgb_array(tile)
        if self._predictor is not None and self._model is None:
            if hasattr(self._predictor, "predict_instances"):
                return self._predictor.predict_instances(image)
            return self._predictor(image, request)
        if self._model is None or self._torch is None or self._functional is None:
            raise RuntimeError("Call load() before predict()")
        height, width = image.shape[:2]
        pad_height = (-height) % self._patch_size
        pad_width = (-width) % self._patch_size
        self._last_padding = (pad_height, pad_width)
        if pad_height or pad_width:
            image = np.pad(
                image,
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode="reflect" if min(height, width) > 1 else "edge",
            )

        torch = self._torch
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div_(255.0)
        mean = torch.tensor(self._mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self._std, dtype=tensor.dtype).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            predictions = self._model.forward(tensor, retrieve_tokens=True)
            predictions["nuclei_binary_map"] = self._functional.softmax(
                predictions["nuclei_binary_map"], dim=1
            ).permute(0, 2, 3, 1)
            predictions["nuclei_type_map"] = self._functional.softmax(
                predictions["nuclei_type_map"], dim=1
            ).permute(0, 2, 3, 1)
            predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
            instance_maps, cells_per_image = self._postprocessor.post_process_batch(
                predictions
            )

        instance_map = instance_maps[0].detach().cpu().numpy().astype("int32")
        instance_map = instance_map[:height, :width]
        instance_classes: dict[int, Any] = {}
        class_probabilities: dict[int, Any] = {}
        for instance_id, cell in cells_per_image[0].items():
            if not np.any(instance_map == int(instance_id)):
                continue
            class_id = int(cell["type"])
            class_name = self._label_map.get(class_id, str(class_id))
            instance_classes[int(instance_id)] = {
                "class_id": class_id,
                "label": class_name,
            }
            class_probabilities[int(instance_id)] = {
                class_name: float(cell["type_prob"])
            }
        return {
            "instance_map": instance_map,
            "instance_classes": instance_classes,
            "class_probabilities": class_probabilities,
            "patch_padding": self._last_padding,
        }

    def postprocess(self, raw_output: Any, request: TileRequest):
        start = time.perf_counter()
        instance_classes = None
        class_probabilities = None
        if isinstance(raw_output, dict):
            labels = raw_output.get("instance_map")
            instance_classes = raw_output.get("instance_classes")
            class_probabilities = raw_output.get("class_probabilities")
            if instance_classes is not None and not isinstance(instance_classes, dict):
                raise RuntimeError("CellViT instance_classes must map instance IDs to classes")
            if class_probabilities is not None and not isinstance(class_probabilities, dict):
                raise RuntimeError("CellViT class_probabilities must map instance IDs to scores")
        else:
            labels = raw_output
        if labels is None:
            raise RuntimeError("CellViT did not return an instance map")
        checkpoint_id = str(self._checkpoint) if self._checkpoint else None
        return result(
            self.model_name, self.version, labels, request, checkpoint_id,
            self._device, start,
            {
                "upstream": "TIO-IKIM/CellViT-Inference",
                "upstream_revision": "67cd98e72865520df8b8fd6e2141e1a8c5234a25",
                "checkpoint_source": (
                    "official CellViT-SAM-H-x40-AMP cache"
                    if self._builtin_checkpoint
                    else "user-supplied checkpoint"
                ),
                "patch_padding_bottom_right": (
                    raw_output.get("patch_padding", (0, 0))
                    if isinstance(raw_output, dict)
                    else (0, 0)
                ),
                "reference_mpp": self.reference_mpp,
                "native_output_fields": (
                    sorted(raw_output) if isinstance(raw_output, dict) else []
                ),
                "native_classes": self._label_map,
                "cell_scope": "nucleus instances and PanNuke-style nucleus classes",
                "whole_cell_validated": False,
                "postprocessing": "official CellViT HoVer postprocessor",
            },
            instance_classes=instance_classes,
            class_probabilities=class_probabilities,
        )

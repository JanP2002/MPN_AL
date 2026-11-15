"""CellSAM adapter using the cell-oriented automatic instance path."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from .base import ModelAdapter, TileRequest
from ._adapter_utils import checked_checkpoint, result, rgb_array

class CellSAMAdapter(ModelAdapter):
    model_name="cellsam"; version="1.0"
    def __init__(self, *, predictor: Any=None, model_variant: str="cellsam_general"):
        self._predictor=predictor; self._model=None; self._checkpoint=None
        self._device="cpu"; self.model_variant=model_variant; self._builtin_checkpoint=False
    def load(self,checkpoint_path:Path|None=None,*,device:str="cpu")->None:
        self._builtin_checkpoint=checkpoint_path is None
        self._checkpoint=checked_checkpoint(checkpoint_path,self.model_name) if checkpoint_path else None
        self._device=device
        if self._predictor is not None:
            return
        try:
            from cellSAM import get_local_model, get_model, segment_cellular_image
        except ImportError as exc:
            raise RuntimeError(
                "CellSAM dependencies are missing. Install the pinned official "
                "CellSAM package in an environment with PyTorch."
            ) from exc
        self._model = (
            get_local_model(str(self._checkpoint))
            if self._checkpoint
            else get_model(model=self.model_variant, version="1.2")
        )
        if self._builtin_checkpoint:
            self._checkpoint = (
                Path.home() / ".deepcell" / "models" / "cellsam_v1.2"
                / f"{self.model_variant}.pt"
            )
        self._predictor = segment_cellular_image
    def predict(self,tile:Any,request:TileRequest)->Any:
        del request
        if self._predictor is None: raise RuntimeError("Call load() before predict()")
        image=rgb_array(tile)
        if self._model is not None:
            masks, *_ = self._predictor(image, self._model, device=self._device)
        else:
            masks=self._predictor(image, model_path=str(self._checkpoint) if self._checkpoint else None)
        if isinstance(masks,dict): masks=masks.get("masks",masks.get("labels"))
        if masks is None: raise RuntimeError("CellSAM returned no instance map")
        return masks
    def postprocess(self,raw_output:Any,request:TileRequest):
        start=time.perf_counter()
        return result(
            self.model_name,self.version,raw_output,request,
            str(self._checkpoint) if self._checkpoint else "CellSAM v1.2 cellsam_general",
            self._device,start,
            {"upstream":"vanvalenlab/cellSAM",
             "upstream_revision":"362a8e521bcd7313d5cb280bcd4cfc12977c1e9f",
             "upstream_version":"v1.2",
             "model_variant":self.model_variant,
             "checkpoint_source":"official CellSAM loader" if self._builtin_checkpoint else "user override",
             "checkpoint_access":"DeepCell token may be required",
             "cell_scope":"cell segmentation; marrow whole-cell masks not validated",
             "whole_cell_validated":False,
             "postprocessing":"upstream labels; nearest-neighbor resize; center crop context"},
        )

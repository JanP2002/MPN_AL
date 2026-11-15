"""Cellpose-SAM adapter for instance segmentation."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from .base import ModelAdapter, TileRequest
from ._adapter_utils import checked_checkpoint, result, rgb_array

class CellposeSAMAdapter(ModelAdapter):
    model_name="cellpose_sam"; version="1.0"
    def __init__(self,*,diameter:float|None=None,flow_threshold:float=0.4,cellprob_threshold:float=0.0): self.diameter=diameter; self.flow_threshold=flow_threshold; self.cellprob_threshold=cellprob_threshold; self._model=None; self._checkpoint=None; self._device="cpu"
    def load(self,checkpoint_path:Path|None=None,*,device:str="cpu")->None:
        self._checkpoint=checked_checkpoint(checkpoint_path,self.model_name) if checkpoint_path else None; self._device=device
        try: from cellpose import models
        except ImportError as exc: raise RuntimeError("Cellpose-SAM requires the cellpose package; install the pinned Cellpose-SAM environment.") from exc
        try:
            pretrained = str(self._checkpoint) if self._checkpoint else "cpsam_v2"
            self._model=models.CellposeModel(pretrained_model=pretrained, gpu=device.startswith("cuda"))
        except Exception as exc: raise RuntimeError(f"Could not load Cellpose-SAM checkpoint: {exc}") from exc
    def predict(self,tile:Any,request:TileRequest)->Any:
        del request
        if self._model is None: raise RuntimeError("Call load() before predict()")
        masks, *_ = self._model.eval(rgb_array(tile), diameter=self.diameter, flow_threshold=self.flow_threshold, cellprob_threshold=self.cellprob_threshold, channels=[0,0])
        return masks
    def postprocess(self,raw_output:Any,request:TileRequest):
        start=time.perf_counter()
        return result(
            self.model_name,self.version,raw_output,request,
            str(self._checkpoint) if self._checkpoint else "Cellpose cpsam_v2",
            self._device,start,
            {"upstream":"MouseLand/cellpose",
             "upstream_revision":"a54cb48849b7e225a81e8e43dcb042d42427f543",
             "pretrained_model":str(self._checkpoint) if self._checkpoint else "cpsam_v2",
             "weight_source":"Cellpose built-in model loader" if not self._checkpoint else "user override",
             "license_note":"Cellpose model training data is CC-BY-NC; review upstream terms",
             "cell_scope":"cell/nucleus segmentation; marrow whole-cell masks not validated",
             "whole_cell_validated":False,
             "diameter":self.diameter,"flow_threshold":self.flow_threshold,
             "cellprob_threshold":self.cellprob_threshold,"channels":[0,0],
             "postprocessing":"instance labels; nearest-neighbor resize; center crop context"},
        )

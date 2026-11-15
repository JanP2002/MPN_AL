import argparse
import json

from PIL import Image
import pytest

from model_registry.patho_sam import PathoSAMAdapter
from model_registry.run_foundation_model import _create_adapter, _parser, run


def test_create_adapter_uses_pathosam_builtin_without_checkpoint(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(
        PathoSAMAdapter,
        "load",
        lambda self, checkpoint, **kwargs: loaded.append(checkpoint),
    )
    args = argparse.Namespace(
        model="pathosam", seed=0, checkpoint=None, checkpoint_root=tmp_path / "weights"
    )

    adapter = _create_adapter(args, "cpu", {})

    assert isinstance(adapter, PathoSAMAdapter)
    assert args.checkpoint is None
    assert loaded == [None]


def test_pathosam_calls_upstream_with_builtin_weights():
    import numpy as np

    from model_registry.base import TileRequest

    calls = []

    def segment(image, **kwargs):
        calls.append((image, kwargs))
        return np.ones((8, 8), dtype=np.int32)

    adapter = PathoSAMAdapter(predictor=segment)
    adapter.load(device="cpu")
    request = TileRequest(input_path="tile.png", image_id="slide", output_size=8)

    raw = adapter.predict(Image.new("RGB", (8, 8)), request)
    result = adapter.postprocess(raw, request)

    assert calls[0][1] == {
        "model_type": "vit_b_histopathology",
        "checkpoint_path": None,
        "device": "cpu",
        "output_choice": "instances",
        "tile_shape": (384, 384),
        "halo": (64, 64),
        "batch_size": 1,
    }
    assert result.instance_mask.shape == (8, 8)
    assert result.checkpoint_id == "micro_sam:vit_b_histopathology"
    assert result.metadata["weights"] == "PathoSAM built-in weights"


def test_checkpoint_discovery_requires_a_single_file(tmp_path):
    from model_registry.run_foundation_model import _resolve_checkpoint

    model_dir = tmp_path / "sam"
    model_dir.mkdir()
    first = model_dir / "vit_b.pth"
    first.touch()

    assert _resolve_checkpoint("sam", None, tmp_path) == first.resolve()

    (model_dir / "vit_h.pth").touch()
    with pytest.raises(ValueError, match="Multiple checkpoint files found"):
        _resolve_checkpoint("sam", None, tmp_path)


def test_checkpoint_root_environment_variable(tmp_path, monkeypatch):
    from model_registry.run_foundation_model import _resolve_checkpoint

    checkpoint = tmp_path / "cellsam" / "model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    monkeypatch.setenv("MPN_AL_CHECKPOINT_ROOT", str(tmp_path))

    assert _resolve_checkpoint("cellsam", None) == checkpoint.resolve()


def test_png_directory_runs_each_tile_with_filename_coordinates(tmp_path):
    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    Image.new("RGB", (1024, 1024), (10, 20, 30)).save(
        input_dir / "slideA_x128_y256.png"
    )
    Image.new("RGB", (1024, 1024), (40, 50, 60)).save(
        input_dir / "slideA_x640_y256.png"
    )
    output_dir = tmp_path / "runs"
    args = _parser().parse_args([
        "--model", "sam",
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--dry-run",
    ])

    run_dirs = run(args)

    assert len(run_dirs) == 2
    metadata = [
        json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    assert [(item["tile"]["image_id"], item["tile"]["x"], item["tile"]["y"])
            for item in metadata] == [("slideA", 128, 256), ("slideA", 640, 256)]
    assert all(item["tile"]["requested_box"] == item["tile"]["readable_box"]
               for item in metadata)
    with Image.open(run_dirs[0] / "input.png") as image:
        assert image.size == (1024, 1024)
        assert image.getpixel((0, 0)) == (10, 20, 30)


def test_default_output_is_grouped_by_model(tmp_path, monkeypatch):
    input_path = tmp_path / "tile.png"
    Image.new("RGB", (1024, 1024), (10, 20, 30)).save(input_path)
    monkeypatch.chdir(tmp_path)
    args = _parser().parse_args([
        "--model", "sam", "--input", str(input_path), "--dry-run",
    ])

    destination = run(args)

    assert destination.parent == tmp_path / "segment_result" / "sam" / "result"
    assert (destination / "instance_mask.tif").is_file()
    assert (destination / "run.json").is_file()


def test_native_instance_classes_are_serialized(tmp_path, monkeypatch):
    import numpy as np

    from model_registry.base import ModelResult
    import model_registry.run_foundation_model as runner

    input_path = tmp_path / "tile.png"
    Image.new("RGB", (1024, 1024), (10, 20, 30)).save(input_path)
    monkeypatch.chdir(tmp_path)

    class Adapter:
        def predict(self, tile, request):
            return np.zeros((request.output_size, request.output_size), dtype=np.int32)

        def postprocess(self, raw, request):
            labels = np.zeros((request.output_size, request.output_size), dtype=np.int32)
            labels[1, 1] = 4
            return ModelResult(
                labels, (labels > 0).astype(np.uint8), "cellsam", "test",
                output_size=request.output_size, instance_classes={4: "unknown"},
                class_probabilities={4: {"unknown": 1.0}},
            )

    monkeypatch.setattr(runner, "_create_adapter", lambda *_args: Adapter())
    args = _parser().parse_args(["--model", "cellsam", "--input", str(input_path)])

    destination = run(args)

    instance_metadata = json.loads(
        (destination / "instances.json").read_text(encoding="utf-8")
    )
    run_metadata = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert instance_metadata["instances"] == [{
        "instance_id": 4,
        "predicted_class": "unknown",
        "class_probabilities": {"unknown": 1.0},
    }]
    assert run_metadata["validation"]["status"] == "not_assessed"
    assert "scale_unverified" in run_metadata["inference"]["warnings"][0]

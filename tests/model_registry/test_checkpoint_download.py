import hashlib
import io

import pytest

from model_registry import _checkpoint_download


class _Response:
    def __init__(self, content):
        self._stream = io.BytesIO(content)
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size=-1):
        return self._stream.read(size)


def test_sam_checkpoint_is_cached_and_hash_checked(tmp_path, monkeypatch):
    content = b"official checkpoint test data"
    expected_prefix = hashlib.sha256(content).hexdigest()[:6]
    monkeypatch.setitem(
        _checkpoint_download.SAM_CHECKPOINTS,
        "vit_b",
        ("test.pth", "https://official.example/test.pth", expected_prefix),
    )
    monkeypatch.setattr(
        _checkpoint_download, "urlopen", lambda *_args, **_kwargs: _Response(content)
    )

    checkpoint = _checkpoint_download.download_sam_checkpoint("vit_b", tmp_path)

    assert checkpoint.read_bytes() == content
    assert _checkpoint_download.download_sam_checkpoint("vit_b", tmp_path) == checkpoint

    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="unexpected SHA-256 prefix"):
        _checkpoint_download.download_sam_checkpoint("vit_b", tmp_path)


def test_sam_checkpoint_rejects_truncated_download(tmp_path, monkeypatch):
    monkeypatch.setitem(
        _checkpoint_download.SAM_CHECKPOINTS,
        "vit_b",
        ("test.pth", "https://official.example/test.pth", "deadbe"),
    )
    monkeypatch.setattr(
        _checkpoint_download, "urlopen", lambda *_args, **_kwargs: _Response(b"wrong")
    )

    with pytest.raises(OSError, match="SHA-256 prefix mismatch"):
        _checkpoint_download.download_sam_checkpoint("vit_b", tmp_path)

    assert not list((tmp_path / "sam").glob("*.pth"))

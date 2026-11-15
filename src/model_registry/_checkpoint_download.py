"""Download first-party model checkpoints into the shared local cache."""
from __future__ import annotations

import os
import tempfile
import hashlib
from pathlib import Path
from urllib.request import urlopen


SAM_CHECKPOINTS = {
    "vit_b": ("sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "01ec64"),
    "vit_l": ("sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "0b3195"),
    "vit_h": ("sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "4b8939"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sam_checkpoint(model_type: str, cache_root: Path) -> Path:
    """Fetch the official SAM checkpoint once; partial files are never reused."""
    try:
        filename, url, expected_sha256_prefix = SAM_CHECKPOINTS[model_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported SAM model type {model_type!r}; "
            f"choose one of {', '.join(SAM_CHECKPOINTS)}"
        ) from exc

    destination = cache_root / "sam" / filename
    if destination.is_file():
        if not _sha256(destination).startswith(expected_sha256_prefix):
            raise RuntimeError(
                f"cached {model_type} checkpoint has an unexpected SHA-256 "
                f"prefix; remove or replace {destination}"
            )
        return destination.resolve()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{filename}.", suffix=".part", dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(url, timeout=60) as response:
                expected_length = response.headers.get("Content-Length")
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
                    downloaded += len(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
        if downloaded == 0:
            raise OSError(f"official SAM checkpoint download was empty: {url}")
        if expected_length is not None and downloaded != int(expected_length):
            raise OSError(
                f"incomplete SAM checkpoint download: received {downloaded} "
                f"of {expected_length} bytes from {url}"
            )
        if not _sha256(temporary_path).startswith(expected_sha256_prefix):
            raise OSError(
                f"official {model_type} checkpoint SHA-256 prefix mismatch "
                f"after download from {url}"
            )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination.resolve()

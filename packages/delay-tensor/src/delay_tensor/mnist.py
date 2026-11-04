"""Full MNIST: fetch the four IDX files once, then raster-scans them.

MNIST is the one dataset here that is not synthetic, so it needs its own
module rather than a branch inside ``datasets``: four IDX files to fetch and
cache, the IDX binary format to parse, and a 28x28 image to turn into the
784-point waveform the tensor consumes.

The raster scan is the whole reason this module exists. A delay tensor reads
one sample per slot clock, so a digit has to arrive as a *sequence*, and the
only ordering that preserves an image is the one a CRT uses: row by row,
left to right. ``raster_scan`` fixes that order and nothing else.

Downloaded files are cached under ``~/.cache/delay-tensor/mnist`` (override
with ``DELAY_TENSOR_DATA``), so the network is touched once per machine.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Tried in order; any of the three serves the same upstream files. Keeping
# several means a dead mirror is an inconvenience rather than a broken run.
_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://raw.githubusercontent.com/fgnt/mnist/master",
)

_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    # The test split is published as t10k-*, not test-*: only the training
    # half kept its full word.
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

_IMAGE_MAGIC = 2051  # 0x00000803: unsigned bytes, 3 dimensions
_LABEL_MAGIC = 2049  # 0x00000801: unsigned bytes, 1 dimension
_SIDE = 28


class MnistError(RuntimeError):
    """A fetch failed, or a file on disk was not the MNIST it claimed to be."""


@dataclass(frozen=True)
class Mnist:
    """The canonical 60k/10k split, as images rather than waveforms."""

    train_images: np.ndarray  # (60000, 28, 28), uint8
    train_labels: np.ndarray  # (60000,), int64
    test_images: np.ndarray  # (10000, 28, 28), uint8
    test_labels: np.ndarray  # (10000,), int64

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(str(digit) for digit in range(10))


def cache_dir() -> Path:
    """Where the IDX files live; ``DELAY_TENSOR_DATA`` overrides the default."""
    override = os.environ.get("DELAY_TENSOR_DATA")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "delay-tensor" / "mnist"


def load_mnist(
    *,
    directory: Path | None = None,
    timeout: float = 60.0,
) -> Mnist:
    """Return MNIST, downloading anything not already cached.

    Every file is parsed and shape-checked even when it came off disk, so a
    truncated download surfaces here instead of as a confusing shape error
    three layers deeper.
    """
    target = directory if directory is not None else cache_dir()
    target.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name in _FILES:
        path = target / name
        if not path.exists():
            _download(name, path, timeout=timeout)
        paths[name] = path

    train_images = _read_images(paths["train-images-idx3-ubyte.gz"])
    test_images = _read_images(paths["t10k-images-idx3-ubyte.gz"])
    train_labels = _read_labels(paths["train-labels-idx1-ubyte.gz"])
    test_labels = _read_labels(paths["t10k-labels-idx1-ubyte.gz"])

    if len(train_images) != len(train_labels):
        raise MnistError(
            f"train images and labels disagree: {len(train_images)} vs {len(train_labels)}"
        )
    if len(test_images) != len(test_labels):
        raise MnistError(
            f"test images and labels disagree: {len(test_images)} vs {len(test_labels)}"
        )
    return Mnist(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )


def raster_scan(images: np.ndarray, *, scale: bool = True) -> np.ndarray:
    """Flatten ``(n, 28, 28)`` images to ``(n, 784)`` waveforms, row-major.

    Row-major is not a convenience: it is the slot order. Pixel ``(r, c)``
    lands at ``r * 28 + c``, so the tensor reads the digit the way a scanning
    beam would, and slot ``s`` always means the same place in the image.

    ``scale`` divides by 255 so drive amplitudes land in ``[0, 1]``, which is
    the range the coupling and decay terms were tuned around. Turning it off
    yields raw 0-255 counts for callers that want them.
    """
    if images.ndim != 3 or images.shape[1:] != (_SIDE, _SIDE):
        raise ValueError(f"expected (n, 28, 28) images, got shape {images.shape}")
    flat = images.reshape(len(images), _SIDE * _SIDE).astype(np.float64)
    return flat / 255.0 if scale else flat


def _download(name: str, destination: Path, *, timeout: float) -> None:
    """Fetch one file, trying each mirror; write via a temp file.

    A partial body must never be left at the final path -- the next call
    would then find it "cached" and parse garbage.
    """
    last_error: Exception | None = None
    payload: bytes | None = None
    for mirror in _MIRRORS:
        url = f"{mirror}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read()
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc

    if payload is None:
        raise MnistError(f"could not fetch {name} from any mirror: {last_error}")

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def _read_images(path: Path) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    if len(raw) < 16:
        raise MnistError(f"{path}: truncated IDX image header")
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != _IMAGE_MAGIC:
        raise MnistError(f"{path}: bad image magic {magic}, expected {_IMAGE_MAGIC}")
    if (rows, cols) != (_SIDE, _SIDE):
        raise MnistError(f"{path}: expected {_SIDE}x{_SIDE} images, got {rows}x{cols}")

    values = np.frombuffer(raw[16:], dtype=np.uint8)
    if values.size != count * rows * cols:
        raise MnistError(
            f"{path}: header claims {count} images but body holds {values.size // (rows * cols)}"
        )
    return values.reshape(count, rows, cols)


def _read_labels(path: Path) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    if len(raw) < 8:
        raise MnistError(f"{path}: truncated IDX label header")
    magic, count = struct.unpack(">II", raw[:8])
    if magic != _LABEL_MAGIC:
        raise MnistError(f"{path}: bad label magic {magic}, expected {_LABEL_MAGIC}")

    values = np.frombuffer(raw[8:], dtype=np.uint8)
    if values.size != count:
        raise MnistError(f"{path}: header claims {count} labels but body holds {values.size}")
    # uint8 cannot go below 0, so only the upper bound can fail.
    if values.size and int(values.max()) > 9:
        raise MnistError(f"{path}: labels outside 0-9")
    return values.astype(np.int64)

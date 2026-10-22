"""Download, verify and load the Raptor Maps InfraredSolarModules dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASS_INDEX, CLASSES

log = logging.getLogger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/RaptorMaps/InfraredSolarModules/master/2020-02-14_InfraredSolarModules.zip"
)
DATASET_SHA256 = "b82c706bc719b045ac4f8930570d81767a8a170d0998ca3e09283b585db05b5e"  # pragma: allowlist secret
IMAGE_SHAPE = (40, 24)  # rows, cols as stored in the source JPEGs
METADATA_FILE = "module_metadata.json"


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, corrupted or does not match the schema."""


@dataclass(frozen=True)
class Catalog:
    root: Path
    module_ids: np.ndarray  # int64
    labels: np.ndarray  # int64 class index
    paths: list[Path]
    raw_metadata: dict[str, dict]

    def __len__(self) -> int:
        return len(self.module_ids)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(data_dir: Path, url: str = DATASET_URL, expected_sha256: str = DATASET_SHA256) -> Path:
    """Fetch the zip once, verify its checksum and extract it. Returns the dataset root."""
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    root = data_dir / "InfraredSolarModules"
    if (root / METADATA_FILE).exists():
        log.info("dataset already present at %s", root)
        return root

    archive = data_dir / "InfraredSolarModules.zip"
    if not archive.exists() or sha256_of(archive) != expected_sha256:
        log.info("downloading %s", url)
        tmp = archive.with_suffix(".part")
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.replace(archive)

    actual = sha256_of(archive)
    if actual != expected_sha256:
        archive.unlink(missing_ok=True)
        raise DatasetError(f"checksum mismatch for {archive}: {actual}")

    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if m.startswith("InfraredSolarModules/")]
        zf.extractall(data_dir, members=members)
    log.info("extracted %d files into %s", len(members), root)
    return root


def resolve_root(data_dir: Path) -> Path:
    """Accept either the download folder (with InfraredSolarModules/ inside) or the dataset root itself."""
    nested = data_dir / "InfraredSolarModules"
    return nested if (nested / METADATA_FILE).exists() else data_dir


def load_catalog(root: Path) -> Catalog:
    meta_path = root / METADATA_FILE
    if not meta_path.exists():
        raise DatasetError(f"{meta_path} not found, run `pvtriage download` first")
    raw = json.loads(meta_path.read_text())

    ids, labels, paths = [], [], []
    for key in sorted(raw, key=int):
        record = raw[key]
        cls = record.get("anomaly_class")
        if cls not in CLASS_INDEX:
            raise DatasetError(f"unknown class {cls!r} for module {key}")
        path = root / record["image_filepath"]
        if not path.exists():
            raise DatasetError(f"missing image {path}")
        ids.append(int(key))
        labels.append(CLASS_INDEX[cls])
        paths.append(path)
    return Catalog(root, np.asarray(ids, dtype=np.int64), np.asarray(labels, dtype=np.int64), paths, raw)


def load_images(paths: list[Path]) -> np.ndarray:
    """Load grayscale thermal crops as uint8 array of shape (n, 40, 24)."""
    out = np.empty((len(paths), *IMAGE_SHAPE), dtype=np.uint8)
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"))
        if arr.shape != IMAGE_SHAPE:
            raise DatasetError(f"unexpected shape {arr.shape} for {path}")
        out[i] = arr
    return out


def stratified_split(
    labels: np.ndarray, seed: int, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> np.ndarray:
    """Return an array of 'train' / 'val' / 'test' per row, stratified by class and reproducible."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    split = np.empty(len(labels), dtype=object)
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_train = int(round(fractions[0] * len(idx)))
        n_val = max(1, int(round(fractions[1] * len(idx)))) if len(idx) >= 3 else 0
        split[idx[:n_train]] = "train"
        split[idx[n_train : n_train + n_val]] = "val"
        split[idx[n_train + n_val :]] = "test"
    return split


def make_sample(root: Path, out_dir: Path, per_class: int, seed: int) -> Path:
    """Copy a small stratified subset (images plus metadata) used by tests and CI into out_dir."""
    catalog = load_catalog(root)
    rng = np.random.default_rng(seed)
    images_dir = out_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    metadata: dict[str, dict] = {}
    for cls_idx, cls in enumerate(CLASSES):
        rows = np.flatnonzero(catalog.labels == cls_idx)
        for row in sorted(rng.choice(rows, size=min(per_class, len(rows)), replace=False)):
            key = str(catalog.module_ids[row])
            shutil.copy2(catalog.paths[row], images_dir / catalog.paths[row].name)
            metadata[key] = {"image_filepath": f"images/{catalog.paths[row].name}", "anomaly_class": cls}
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=1, sort_keys=True))
    return out_dir

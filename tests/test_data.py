import json

import numpy as np
import pytest

from pvtriage.config import CLASSES
from pvtriage.data import DatasetError, load_catalog, load_images, resolve_root, sha256_of, stratified_split


def test_sample_catalog_is_complete(sample_dir):
    catalog = load_catalog(resolve_root(sample_dir))
    assert len(catalog) == 240
    assert np.bincount(catalog.labels, minlength=len(CLASSES)).tolist() == [20] * len(CLASSES)
    images = load_images(catalog.paths[:5])
    assert images.shape == (5, 40, 24) and images.dtype == np.uint8


def test_split_is_reproducible_and_stratified():
    labels = np.repeat(np.arange(4), [100, 50, 20, 10])
    a, b = stratified_split(labels, seed=1), stratified_split(labels, seed=1)
    assert (a == b).all()
    for cls in range(4):
        parts = a[labels == cls]
        assert {"train", "val", "test"} <= set(parts)
        assert abs((parts == "train").mean() - 0.7) < 0.06


def test_split_differs_by_seed():
    labels = np.repeat(np.arange(3), 50)
    assert not (stratified_split(labels, 1) == stratified_split(labels, 2)).all()


def test_unknown_class_is_rejected(tmp_path):
    root = tmp_path / "InfraredSolarModules"
    (root / "images").mkdir(parents=True)
    (root / "module_metadata.json").write_text(
        json.dumps({"1": {"image_filepath": "images/1.jpg", "anomaly_class": "Lightning"}})
    )
    with pytest.raises(DatasetError, match="unknown class"):
        load_catalog(root)


def test_missing_metadata_is_explicit(tmp_path):
    with pytest.raises(DatasetError, match="pvtriage download"):
        load_catalog(tmp_path)


def test_sha256(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # pragma: allowlist secret
    assert sha256_of(f) == expected

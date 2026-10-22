import numpy as np
import pytest

from pvtriage.thermal import DESCRIPTOR_NAMES, describe, describe_batch


def _get(vec, name):
    return vec[DESCRIPTOR_NAMES.index(name)]


def test_uniform_module_has_no_hot_area():
    vec = describe(np.full((40, 24), 90, dtype=np.uint8))
    assert _get(vec, "contrast_index") == 0
    assert _get(vec, "hot_fraction") == 0
    assert _get(vec, "hot_blobs") == 0


def test_single_hot_cell_is_one_blob():
    img = np.full((40, 24), 80, dtype=np.uint8)
    img[10:14, 5:9] = 200
    vec = describe(img)
    assert _get(vec, "hot_blobs") == 1
    assert _get(vec, "hot_fraction") == pytest.approx(16 / (40 * 24))
    assert _get(vec, "contrast_index") > 0.3


def test_one_warm_third_looks_like_a_bypass_diode():
    img = np.full((40, 24), 80, dtype=np.uint8)
    img[:, 16:] = 120  # one of three substrings warmer
    vec = describe(img)
    assert _get(vec, "band_spread") == pytest.approx(40, abs=1)
    assert _get(vec, "band_hot_count") == 1


def test_two_separate_hot_spots():
    img = np.full((40, 24), 80, dtype=np.uint8)
    img[2:5, 2:5] = 220
    img[30:33, 18:21] = 220
    assert _get(describe(img), "hot_blobs") == 2


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        describe(np.zeros((2, 40, 24), dtype=np.uint8))


def test_batch_shape():
    out = describe_batch(np.zeros((3, 40, 24), dtype=np.uint8))
    assert out.shape == (3, len(DESCRIPTOR_NAMES))

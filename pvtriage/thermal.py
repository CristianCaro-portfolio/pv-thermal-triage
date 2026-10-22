"""Interpretable thermal descriptors for one module crop.

The source crops are 8 bit grayscale renders of a thermal frame, not radiometric
temperatures, so every value here is relative to the module itself. The
descriptors explain a decision to a technician ("one third of the module is
hotter than the rest") and feed the baseline model.
"""

from __future__ import annotations

import numpy as np

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "median_level",
    "p99_level",
    "contrast_index",
    "hot_fraction",
    "hot_blobs",
    "band_spread",
    "band_hot_count",
    "edge_energy",
)


def _robust_spread(img: np.ndarray, center: float) -> float:
    mad = float(np.median(np.abs(img - center)))
    return max(1.4826 * mad, 2.0)


def _count_blobs(mask: np.ndarray) -> int:
    """4-connected component count, small and dependency free for 40x24 masks."""
    seen = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    blobs = 0
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            blobs += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return blobs


def _band_profile(img: np.ndarray) -> tuple[float, int]:
    """Spread between the hottest and coolest third of the module.

    Bypass diodes split a module into three substrings, so a diode fault shows
    up as one band clearly warmer than the others. Both orientations are checked
    because crops are not consistently rotated.
    """
    best_spread, best_hot = 0.0, 0
    for axis in (0, 1):
        bands = np.array_split(img, 3, axis=axis)
        means = np.array([b.mean() for b in bands])
        spread = float(means.max() - means.min())
        if spread > best_spread:
            base = float(np.median(means))
            best_spread = spread
            best_hot = int(np.sum(means - base > 0.5 * spread)) if spread > 0 else 0
    return best_spread, best_hot


def describe(img: np.ndarray) -> np.ndarray:
    """Return the descriptor vector for a single (40, 24) uint8 crop."""
    if img.ndim != 2:
        raise ValueError(f"expected a 2D crop, got shape {img.shape}")
    x = img.astype(np.float32)
    median = float(np.median(x))
    p99 = float(np.percentile(x, 99))
    spread = _robust_spread(x, median)
    hot = x > median + max(3.0 * spread, 8.0)
    band_spread, band_hot = _band_profile(x)
    gy, gx = np.gradient(x)
    return np.array(
        [
            median,
            p99,
            (p99 - median) / 255.0,
            float(hot.mean()),
            float(_count_blobs(hot)),
            band_spread,
            float(band_hot),
            float(np.mean(np.hypot(gx, gy))),
        ],
        dtype=np.float32,
    )


def describe_batch(images: np.ndarray) -> np.ndarray:
    return np.stack([describe(img) for img in images]) if len(images) else np.empty((0, len(DESCRIPTOR_NAMES)))

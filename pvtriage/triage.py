"""Turn model outputs into decisions a maintenance team can act on.

A finding is auto accepted only when two independent signals agree: the CNN
classifier and the labelled precedents returned by DuckDB. The confidence bar
for auto acceptance is calibrated on the validation split to hit a target
accuracy, everything below it goes to a person.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ACTIONS, CLASSES, LOSS_FRACTION, SAFETY_CLASSES

LOSS_VECTOR = np.array([LOSS_FRACTION[c] for c in CLASSES])
SAFETY_MASK = np.array([c in SAFETY_CLASSES for c in CLASSES])


def fuse(cnn_probs: np.ndarray, precedent_votes: np.ndarray, weight: float = 0.5) -> np.ndarray:
    if cnn_probs.shape != precedent_votes.shape:
        raise ValueError(f"shape mismatch {cnn_probs.shape} vs {precedent_votes.shape}")
    fused = (1.0 - weight) * cnn_probs + weight * precedent_votes
    return fused / fused.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class Calibration:
    threshold: float
    coverage: float
    auto_accuracy: float
    target: float
    reached_target: bool


def calibrate(confidence: np.ndarray, agree: np.ndarray, correct: np.ndarray, target: float) -> Calibration:
    """Lowest confidence threshold whose auto accepted set meets the target accuracy.

    Lower threshold means more automation, so the scan keeps the most coverage
    that still meets the target. If nothing meets it, everything goes to review.
    """
    if not (len(confidence) == len(agree) == len(correct)):
        raise ValueError("inputs must have the same length")
    candidates = np.unique(confidence[agree])
    best = Calibration(1.01, 0.0, 0.0, target, False)
    for tau in candidates:  # ascending
        auto = agree & (confidence >= tau)
        if not auto.any():
            continue
        acc = float(correct[auto].mean())
        if acc >= target:
            return Calibration(float(tau), float(auto.mean()), acc, target, True)
    return best


def decide(confidence: np.ndarray, agree: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(agree & (confidence >= threshold), "auto", "review")


def expected_kwh_lost(probs: np.ndarray, kwh_per_module_day: float) -> np.ndarray:
    """Probability weighted expected loss, so an uncertain diode still ranks high."""
    return probs @ LOSS_VECTOR * kwh_per_module_day


def safety_flags(probs: np.ndarray, min_prob: float = 0.25) -> np.ndarray:
    """Flag any module with a meaningful chance of being a safety relevant class."""
    return probs[:, SAFETY_MASK].sum(axis=1) >= min_prob


def action_for(label: str, decision: str) -> str:
    base = ACTIONS[label]
    return f"confirm on site, then {base}" if decision == "review" and label != "No-Anomaly" else base


def energy_capture_curve(priority: np.ndarray, true_loss: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Share of the true energy loss found when visiting the top x fraction of modules."""
    order = np.argsort(-priority, kind="stable")
    cum = np.cumsum(true_loss[order])
    total = cum[-1] if len(cum) and cum[-1] > 0 else 1.0
    idx = np.clip((fractions * len(order)).astype(int) - 1, 0, len(order) - 1)
    return cum[idx] / total

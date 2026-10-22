import numpy as np
import pytest

from pvtriage import triage
from pvtriage.config import CLASS_INDEX, CLASSES


def _onehot(names):
    out = np.zeros((len(names), len(CLASSES)))
    for i, n in enumerate(names):
        out[i, CLASS_INDEX[n]] = 1.0
    return out


def test_fuse_is_a_distribution():
    a = np.random.default_rng(0).dirichlet(np.ones(len(CLASSES)), size=5)
    b = _onehot(["Diode"] * 5)
    fused = triage.fuse(a, b)
    assert np.allclose(fused.sum(1), 1)
    with pytest.raises(ValueError):
        triage.fuse(a, b[:, :3])


def test_calibrate_keeps_most_coverage_that_meets_target():
    conf = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    agree = np.ones(6, dtype=bool)
    correct = np.array([False, True, True, True, True, True])
    cal = triage.calibrate(conf, agree, correct, target=0.99)
    assert cal.threshold == 0.6 and cal.reached_target
    assert cal.coverage == pytest.approx(5 / 6)


def test_calibrate_sends_everything_to_review_when_target_is_unreachable():
    cal = triage.calibrate(np.array([0.9, 0.95]), np.array([True, True]), np.array([False, False]), 0.9)
    assert not cal.reached_target
    assert (triage.decide(np.array([0.99]), np.array([True]), cal.threshold) == "review").all()


def test_disagreement_always_goes_to_review():
    assert triage.decide(np.array([0.99]), np.array([False]), 0.1)[0] == "review"


def test_expected_loss_uses_class_fractions():
    kwh = triage.expected_kwh_lost(_onehot(["Offline-Module", "Diode", "No-Anomaly"]), 2.0)
    assert kwh.tolist() == pytest.approx([2.0, 0.66, 0.0])


def test_safety_flag_catches_uncertain_hot_spots():
    p = np.zeros((1, len(CLASSES)))
    p[0, CLASS_INDEX["No-Anomaly"]] = 0.7
    p[0, CLASS_INDEX["Hot-Spot"]] = 0.3
    assert triage.safety_flags(p)[0]


def test_energy_capture_perfect_order():
    true_loss = np.array([0, 0, 5.0, 0, 1.0])
    curve = triage.energy_capture_curve(true_loss, true_loss, np.array([0.2, 0.4, 1.0]))
    assert curve.tolist() == pytest.approx([5 / 6, 1.0, 1.0])


def test_review_action_asks_for_confirmation():
    assert triage.action_for("Diode", "review").startswith("confirm on site")
    assert triage.action_for("No-Anomaly", "review") == "none"

"""Metrics and figures written after each run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .config import CLASS_INDEX, CLASSES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_pred = probs.argmax(axis=1)
    normal = CLASS_INDEX["No-Anomaly"]
    anomaly_true, anomaly_pred = y_true != normal, y_pred != normal
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(CLASSES)), zero_division=0)),
        "anomaly_recall": float(recall_score(anomaly_true, anomaly_pred, zero_division=0)),
        "anomaly_precision": float(precision_score(anomaly_true, anomaly_pred, zero_division=0)),
    }


def per_class_f1(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    scores = f1_score(y_true, probs.argmax(1), average=None, labels=range(len(CLASSES)), zero_division=0)
    return {c: float(s) for c, s in zip(CLASSES, scores, strict=True)}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_results(
    out: Path,
    y_true: np.ndarray,
    fused_probs: np.ndarray,
    coverage_curve: list[tuple[float, float]],
    calibrated: tuple[float, float],
    capture_fractions: np.ndarray,
    capture_curves: dict[str, np.ndarray],
    subtitle: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure(figsize=(15, 5.2), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    cm = confusion_matrix(y_true, fused_probs.argmax(1), labels=range(len(CLASSES)), normalize="true")
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f7fc", "#2a78d6", "#0d3a73"])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8, color=INK_2)
    for i in range(len(CLASSES)):
        v = cm[i, i]
        ax.text(i, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK)
    ax.set_title("Recall per class (row normalized)", fontsize=11, color=INK, loc="left")
    ax.set_xlabel("predicted", color=INK_2, fontsize=9)
    ax.set_ylabel("true", color=INK_2, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = fig.add_subplot(gs[1])
    _style(ax)
    cov = np.array(coverage_curve).reshape(-1, 2)
    if len(cov):
        ax.plot(cov[:, 0] * 100, cov[:, 1] * 100, color=SERIES[0], linewidth=2)
    ax.scatter([calibrated[0] * 100], [calibrated[1] * 100], s=60, color=SERIES[1], zorder=3, edgecolor=SURFACE)
    ax.annotate(
        f"calibrated: {calibrated[0]:.0%} auto\nat {calibrated[1]:.1%} accuracy",
        (calibrated[0] * 100, calibrated[1] * 100),
        xytext=(-110, -42),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    ax.set_xlabel("modules auto accepted (%)", color=INK_2, fontsize=9)
    ax.set_ylabel("accuracy on auto accepted (%)", color=INK_2, fontsize=9)
    ax.set_title("Automation vs accuracy (test split)", fontsize=11, color=INK, loc="left")

    ax = fig.add_subplot(gs[2])
    _style(ax)
    for color, (name, curve) in zip(SERIES, capture_curves.items(), strict=False):
        ax.plot(capture_fractions * 100, curve * 100, color=color, linewidth=2, label=name)
    ax.set_xlabel("modules visited, highest priority first (%)", color=INK_2, fontsize=9)
    ax.set_ylabel("share of true energy loss found (%)", color=INK_2, fontsize=9)
    ax.set_title("Energy loss captured by triage order", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")

    fig.text(0.01, 1.03, subtitle, ha="left", va="bottom", fontsize=9, color=INK_2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_architecture(out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    stages = [
        ("Drone IR crops", "Raptor Maps\n20k modules, 12 classes\nsha256 verified"),
        ("Thermal descriptors", "contrast index\nhot area, blobs\nsubstring band spread"),
        ("ThermalNet CNN", "PyTorch on CPU\nclass probabilities\n64d embedding"),
        ("DuckDB 2.0", "NEAREST join precedents\nVARIANT source metadata\ntrigger keeps batch rollups"),
        ("Triage", "CNN and precedents\nmust agree, calibrated\nthreshold, safety review"),
        ("Work orders", "kWh per day at risk\nNASA POWER irradiance\nMLflow run tracking"),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.4), facecolor=SURFACE)
    ax.set_xlim(0, len(stages) * 2.7)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    for i, (title, body) in enumerate(stages):
        x = i * 2.7 + 0.15
        box = FancyBboxPatch(
            (x, 0.35), 2.25, 2.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=SERIES[0], lw=1.6
        )
        ax.add_patch(box)
        ax.text(x + 1.125, 2.45, title, ha="center", va="center", fontsize=10.5, color=INK, weight="bold")
        ax.text(x + 1.125, 1.35, body, ha="center", va="center", fontsize=8.8, color=INK_2, linespacing=1.5)
        if i < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 2.28, 1.6), (x + 2.82, 1.6), arrowstyle="-|>", mutation_scale=14, color=SERIES[1], lw=1.6
                )
            )
    ax.text(0.15, 3.05, "pv-thermal-triage", fontsize=12, color=INK, weight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out

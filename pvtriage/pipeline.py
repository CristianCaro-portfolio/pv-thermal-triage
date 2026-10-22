"""End to end run: data, model, precedent index, triage, evaluation."""

from __future__ import annotations

import csv
import json
import logging
import platform
import time
from dataclasses import asdict
from pathlib import Path

import duckdb
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from . import irradiance, report, triage
from .config import CLASSES, Settings
from .data import load_catalog, load_images, resolve_root, stratified_split
from .model import predict, train
from .store import TriageStore
from .thermal import DESCRIPTOR_NAMES, describe_batch
from .tracking import Tracker

log = logging.getLogger(__name__)


def _selective_curve(confidence: np.ndarray, agree: np.ndarray, correct: np.ndarray) -> list[tuple[float, float]]:
    points = []
    for tau in np.linspace(0.0, 1.0, 101):
        auto = agree & (confidence >= tau)
        if auto.sum() >= 10:
            points.append((float(auto.mean()), float(correct[auto].mean())))
    return points


def run(settings: Settings, offline: bool = False) -> dict:
    t0 = time.perf_counter()
    if settings.num_threads:
        torch.set_num_threads(settings.num_threads)
    out = settings.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(settings.tracking, settings.tracking_uri)

    catalog = load_catalog(resolve_root(settings.data_dir))
    images = load_images(catalog.paths)
    split = stratified_split(catalog.labels, settings.seed)
    masks = {name: split == name for name in ("train", "val", "test")}
    log.info("loaded %d modules: %s", len(catalog), {k: int(v.sum()) for k, v in masks.items()})

    descriptors = describe_batch(images)

    irr = irradiance.load(settings.site_lat, settings.site_lon, settings.irradiance_year, settings.data_dir, offline)
    kwh_module = irradiance.module_kwh_per_day(irr, settings.module_wp, settings.performance_ratio)

    with tracker.run(f"run-seed{settings.seed}"):
        tracker.params(
            {
                "seed": settings.seed,
                "epochs": settings.epochs,
                "k_neighbors": settings.k_neighbors,
                "target_auto_accuracy": settings.target_auto_accuracy,
                "n_modules": len(catalog),
                "duckdb_version": duckdb.__version__,
                "torch_version": torch.__version__,
            }
        )

        # Baseline on interpretable descriptors only, to show what the CNN adds.
        baseline = HistGradientBoostingClassifier(max_iter=300, random_state=settings.seed, class_weight="balanced")
        baseline.fit(descriptors[masks["train"]], catalog.labels[masks["train"]])
        base_probs = np.zeros((int(masks["test"].sum()), len(CLASSES)))
        base_probs[:, baseline.classes_] = baseline.predict_proba(descriptors[masks["test"]])

        result = train(
            images[masks["train"]],
            catalog.labels[masks["train"]],
            images[masks["val"]],
            catalog.labels[masks["val"]],
            n_classes=len(CLASSES),
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            lr=settings.learning_rate,
            embedding_dim=settings.embedding_dim,
            seed=settings.seed,
            on_epoch=lambda row: tracker.metrics(row, step=int(row["epoch"])),
        )
        probs, emb = predict(result.model, images)

        store = TriageStore(settings.db_path, settings.embedding_dim)
        store.load_modules(catalog.module_ids, split, catalog.labels, catalog.raw_metadata)
        store.load_reference(catalog.module_ids[masks["train"]], catalog.labels[masks["train"]], emb[masks["train"]])

        def precedents(mask: np.ndarray, approx: bool = False) -> tuple[np.ndarray, float]:
            store.load_queries(catalog.module_ids[mask], emb[mask])
            start = time.perf_counter()
            ids, votes = store.precedent_votes(settings.k_neighbors, approx=approx)
            elapsed = time.perf_counter() - start
            if not np.array_equal(ids, catalog.module_ids[mask]):
                raise RuntimeError("precedent search did not return every query module")
            return votes, elapsed

        # Calibrate the auto accept threshold on validation only.
        val_votes, _ = precedents(masks["val"])
        val_fused = triage.fuse(probs[masks["val"]], val_votes)
        val_agree = probs[masks["val"]].argmax(1) == val_votes.argmax(1)
        val_correct = val_fused.argmax(1) == catalog.labels[masks["val"]]
        calibration = triage.calibrate(val_fused.max(1), val_agree, val_correct, settings.target_auto_accuracy)

        # Test split: simulated arrival of inspection batches.
        test_idx = np.flatnonzero(masks["test"])
        test_votes, nearest_seconds = precedents(masks["test"])
        approx_votes, approx_seconds = precedents(masks["test"], approx=True)
        # Same search in numpy as a correctness and speed reference for the SQL path.
        start = time.perf_counter()
        ref_emb, ref_labels = emb[masks["train"]], catalog.labels[masks["train"]]
        sims = emb[masks["test"]] @ ref_emb.T
        top = np.argpartition(-sims, settings.k_neighbors - 1, axis=1)[:, : settings.k_neighbors]
        numpy_votes = np.zeros_like(test_votes)
        for row in range(len(top)):
            np.add.at(numpy_votes[row], ref_labels[top[row]], np.maximum(sims[row, top[row]], 0.0))
        numpy_seconds = time.perf_counter() - start
        y_test = catalog.labels[test_idx]
        cnn_test = probs[test_idx]
        fused = triage.fuse(cnn_test, test_votes)
        confidence = fused.max(1)
        agree = cnn_test.argmax(1) == test_votes.argmax(1)
        decision = triage.decide(confidence, agree, calibration.threshold)
        predicted = fused.argmax(1)
        correct = predicted == y_test
        expected_kwh = triage.expected_kwh_lost(fused, kwh_module)
        safety = triage.safety_flags(fused)
        decision = np.where(safety, "review", decision)
        auto = decision == "auto"
        true_loss_test = triage.LOSS_VECTOR[y_test] * kwh_module

        n_batches = 0
        for start in range(0, len(test_idx), settings.ingest_batch_size):
            rows = []
            for j in range(start, min(start + settings.ingest_batch_size, len(test_idx))):
                row = test_idx[j]
                label = CLASSES[predicted[j]]
                rows.append(
                    (
                        n_batches,
                        int(catalog.module_ids[row]),
                        label,
                        float(confidence[j]),
                        CLASSES[int(test_votes[j].argmax())],
                        float(test_votes[j].max()),
                        str(decision[j]),
                        bool(safety[j]),
                        float(descriptors[row, DESCRIPTOR_NAMES.index("contrast_index")]),
                        float(descriptors[row, DESCRIPTOR_NAMES.index("band_spread")]),
                        float(expected_kwh[j]),
                        triage.action_for(label, str(decision[j])),
                    )
                )
            store.insert_findings(rows)
            n_batches += 1
        rollup_ok = store.rollup_matches_scan()

        true_loss = true_loss_test
        fractions = np.linspace(0.01, 1.0, 100)
        cnn_only_loss = triage.expected_kwh_lost(cnn_test, kwh_module)
        contrast_rank = descriptors[test_idx, DESCRIPTOR_NAMES.index("contrast_index")]
        curves = {
            "fused triage": triage.energy_capture_curve(expected_kwh, true_loss, fractions),
            "CNN only": triage.energy_capture_curve(cnn_only_loss, true_loss, fractions),
            "hottest first (contrast index)": triage.energy_capture_curve(contrast_rank, true_loss, fractions),
        }
        anomaly_share = float((y_test != 0).mean())
        at = int(np.ceil(anomaly_share * 100)) - 1

        metrics = {
            "test_modules": len(test_idx),
            "fused": report.classification_metrics(y_test, fused),
            "cnn_only": report.classification_metrics(y_test, cnn_test),
            "precedents_only": report.classification_metrics(y_test, test_votes),
            "descriptor_baseline": report.classification_metrics(y_test, base_probs),
            "per_class_f1_fused": report.per_class_f1(y_test, fused),
            "triage": {
                "threshold": calibration.threshold,
                "val_coverage": calibration.coverage,
                "val_auto_accuracy": calibration.auto_accuracy,
                "target_reached_on_val": calibration.reached_target,
                "test_auto_share": float(auto.mean()),
                "test_auto_accuracy": float(correct[auto].mean()) if auto.any() else None,
                "test_review_accuracy": float(correct[~auto].mean()) if (~auto).any() else None,
                "safety_flagged": int(safety.sum()),
                "auto_cleared_anomalies": int(np.sum(auto & (predicted == 0) & (y_test != 0))),
                "auto_cleared_anomaly_share": float(
                    np.sum(auto & (predicted == 0) & (y_test != 0)) / max(1, np.sum(y_test != 0))
                ),
                "auto_cleared_kwh_day": float(np.sum(true_loss_test[auto & (predicted == 0)])),
                "missed_safety_modules": int(
                    np.sum(np.isin(y_test, np.flatnonzero(triage.SAFETY_MASK)) & auto & ~correct)
                ),
            },
            "energy": {
                "irradiance_source": irr.source,
                "mean_kwh_m2_day": irr.mean_kwh_m2_day,
                "healthy_module_kwh_day": kwh_module,
                "true_kwh_day_at_risk": float(true_loss.sum()),
                "estimated_kwh_day_at_risk": float(expected_kwh.sum()),
                "anomaly_share": anomaly_share,
                "loss_captured_at_anomaly_share": {k: float(v[at]) for k, v in curves.items()},
                "loss_captured_top_10pct": {k: float(v[9]) for k, v in curves.items()},
            },
            "duckdb": {
                "version": duckdb.__version__,
                "reference_rows": int(masks["train"].sum()),
                "nearest_join_seconds": nearest_seconds,
                "approx_nearest_join_seconds": approx_seconds,
                "approx_top1_agreement": float((approx_votes.argmax(1) == test_votes.argmax(1)).mean()),
                "numpy_bruteforce_seconds": numpy_seconds,
                "numpy_top1_agreement": float((numpy_votes.argmax(1) == test_votes.argmax(1)).mean()),
                "ingest_batches": n_batches,
                "trigger_rollups_match_rescan": rollup_ok,
            },
            "training": {"seconds": result.seconds, "epochs": settings.epochs, "history": result.history},
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cpu_threads": torch.get_num_threads(),
                "machine": platform.machine(),
            },
            "settings": {k: str(v) for k, v in asdict(settings).items()},
            "wall_seconds": time.perf_counter() - t0,
        }

        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        orders = store.work_orders(limit=50)
        with (out / "work_orders.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(orders[0].keys()) if orders else ["module_id"])
            writer.writeheader()
            writer.writerows(orders)
        example = orders[0]["module_id"] if orders else None
        if example is not None:
            store.load_queries(np.array([example]), emb[np.flatnonzero(catalog.module_ids == example)])
            metrics["example_precedents"] = {
                "module_id": int(example),
                "neighbors": store.precedents_for(int(example), k=3),
            }
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        rollups = {"batches": store.batch_rollup(), "classes": store.class_rollup()}
        (out / "rollups.json").write_text(json.dumps(rollups, indent=2, default=str))
        store.close()

        figure = report.plot_results(
            out / "results.png",
            y_test,
            fused,
            _selective_curve(confidence, agree, correct),
            (float(auto.mean()), float(correct[auto].mean()) if auto.any() else 0.0),
            fractions,
            curves,
            f"{len(test_idx)} held out modules, {settings.epochs} epochs on CPU, k={settings.k_neighbors}, "
            f"DuckDB {duckdb.__version__}, irradiance: {irr.source}",
        )
        tracker.metrics({f"test_{k}": v for k, v in metrics["fused"].items()})
        tracker.metrics({"test_auto_share": metrics["triage"]["test_auto_share"]})
        for path in (out / "metrics.json", out / "work_orders.csv", figure):
            tracker.artifact(path)

    return metrics


def summary_lines(metrics: dict) -> list[str]:
    f, t, e, d = metrics["fused"], metrics["triage"], metrics["energy"], metrics["duckdb"]
    lines = [
        f"test modules: {metrics['test_modules']}",
        f"fused accuracy {f['accuracy']:.3f}, macro F1 {f['macro_f1']:.3f}, anomaly recall {f['anomaly_recall']:.3f}",
        f"cnn only macro F1 {metrics['cnn_only']['macro_f1']:.3f}, precedents only "
        f"{metrics['precedents_only']['macro_f1']:.3f}, descriptor baseline "
        f"{metrics['descriptor_baseline']['macro_f1']:.3f}",
        f"auto accepted {t['test_auto_share']:.1%} at accuracy {t['test_auto_accuracy']}",
        f"energy found at anomaly share: {e['loss_captured_at_anomaly_share']}",
        f"duckdb {d['version']} nearest join {d['nearest_join_seconds']:.2f}s, "
        f"rollups ok {d['trigger_rollups_match_rescan']}",
    ]
    return lines


def write_summary(metrics: dict, path: Path) -> None:
    path.write_text("\n".join(summary_lines(metrics)) + "\n")

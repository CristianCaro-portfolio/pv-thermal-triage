"""Thin MLflow wrapper so tracking can be switched off in tests and CI."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, enabled: bool, uri: str, experiment: str = "pv-thermal-triage") -> None:
        self.enabled = enabled
        self._mlflow = None
        if not enabled:
            return
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        try:
            import mlflow
        except ImportError:
            log.warning("mlflow not installed, tracking disabled")
            self.enabled = False
            return
        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self._mlflow = mlflow

    @contextlib.contextmanager
    def run(self, name: str):
        if not self.enabled:
            yield self
            return
        with self._mlflow.start_run(run_name=name):
            yield self

    def params(self, values: dict) -> None:
        if self.enabled:
            self._mlflow.log_params(values)

    def metrics(self, values: dict, step: int | None = None) -> None:
        if self.enabled:
            clean = {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}
            self._mlflow.log_metrics(clean, step=step)

    def artifact(self, path: Path) -> None:
        if self.enabled and path.exists():
            self._mlflow.log_artifact(str(path))

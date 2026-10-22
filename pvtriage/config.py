"""Runtime settings and the engineering assumptions used for triage.

Every number that is an assumption rather than a measurement lives here so it
can be reviewed in one place and overridden per fleet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CLASSES: tuple[str, ...] = (
    "No-Anomaly",
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Shadowing",
    "Diode",
    "Diode-Multi",
    "Vegetation",
    "Soiling",
    "Offline-Module",
)
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

# Fraction of module output assumed lost while the defect is present.
# These are planning assumptions, not measurements: a bypassed diode removes one
# of three substrings, an offline module removes all of it, single cell hot spots
# cost little. Tune them with your own string level monitoring data.
LOSS_FRACTION: dict[str, float] = {
    "No-Anomaly": 0.0,
    "Cell": 0.03,
    "Cell-Multi": 0.10,
    "Cracking": 0.05,
    "Hot-Spot": 0.05,
    "Hot-Spot-Multi": 0.10,
    "Shadowing": 0.15,
    "Diode": 0.33,
    "Diode-Multi": 0.66,
    "Vegetation": 0.20,
    "Soiling": 0.05,
    "Offline-Module": 1.0,
}

# Classes where a hot area can escalate into a safety problem (backsheet burn,
# junction box failure). These always go to a person, whatever the energy loss.
SAFETY_CLASSES: frozenset[str] = frozenset({"Hot-Spot", "Hot-Spot-Multi", "Cell-Multi", "Diode-Multi"})

ACTIONS: dict[str, str] = {
    "No-Anomaly": "none",
    "Cell": "monitor at next survey",
    "Cell-Multi": "IV curve test, warranty claim if confirmed",
    "Cracking": "visual check, warranty claim",
    "Hot-Spot": "on site check of thin film module",
    "Hot-Spot-Multi": "on site check of thin film module",
    "Shadowing": "review row spacing or nearby obstruction",
    "Diode": "replace junction box or module",
    "Diode-Multi": "replace module",
    "Vegetation": "dispatch vegetation crew",
    "Soiling": "add to cleaning schedule",
    "Offline-Module": "check connectors, fuse and string",
}


def _env(name: str, default: str) -> str:
    return os.environ.get(f"PVT_{name}", default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data/raw")))
    out_dir: Path = field(default_factory=lambda: Path(_env("OUT_DIR", "reports")))
    db_path: Path = field(default_factory=lambda: Path(_env("DB_PATH", "reports/pvtriage.duckdb")))
    seed: int = int(_env("SEED", "7"))
    epochs: int = int(_env("EPOCHS", "14"))
    batch_size: int = 128
    learning_rate: float = 2e-3
    embedding_dim: int = 64
    k_neighbors: int = int(_env("K_NEIGHBORS", "15"))
    # Fraction of auto accepted decisions that must be right on the validation split.
    target_auto_accuracy: float = float(_env("TARGET_AUTO_ACCURACY", "0.97"))
    ingest_batch_size: int = 500
    # Reference site used only for the irradiance that converts loss fractions into kWh.
    site_name: str = _env("SITE_NAME", "La Loma, Cesar, Colombia")
    site_lat: float = float(_env("SITE_LAT", "9.6"))
    site_lon: float = float(_env("SITE_LON", "-73.6"))
    irradiance_year: int = int(_env("IRRADIANCE_YEAR", "2025"))
    module_wp: float = float(_env("MODULE_WP", "350"))
    performance_ratio: float = float(_env("PERFORMANCE_RATIO", "0.80"))
    tracking: bool = _env("TRACKING", "1") == "1"
    tracking_uri: str = _env("MLFLOW_TRACKING_URI", "sqlite:///reports/mlflow.db")
    num_threads: int = int(_env("NUM_THREADS", "0"))

"""Daily irradiance from the NASA POWER API, used to turn loss fractions into kWh.

NASA POWER is free and needs no key. A cached response is kept next to the
data so reruns are offline, and a bundled one year sample lets tests and CI
run without network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETER = "ALLSKY_SFC_SW_DWN"  # kWh/m2/day, all sky surface shortwave downward irradiance
SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "sample" / "nasa_power_daily.json"


@dataclass(frozen=True)
class IrradianceSummary:
    source: str
    days: int
    mean_kwh_m2_day: float
    p10_kwh_m2_day: float
    missing_days: int


def fetch(lat: float, lon: float, year: int, cache_dir: Path, timeout: int = 60) -> dict:
    cache = cache_dir / f"nasa_power_{lat:.2f}_{lon:.2f}_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import requests

    params = {
        "parameters": PARAMETER,
        "community": "RE",
        "latitude": lat,
        "longitude": lon,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
    }
    resp = requests.get(POWER_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    return payload


def summarize(payload: dict, source: str) -> IrradianceSummary:
    header = payload.get("header", {})
    fill = header.get("fill_value", -999.0)
    try:
        series = payload["properties"]["parameter"][PARAMETER]
    except KeyError as exc:
        raise ValueError("NASA POWER payload has no ALLSKY_SFC_SW_DWN series") from exc
    values = sorted(float(v) for v in series.values() if v is not None and float(v) != fill and float(v) >= 0)
    missing = len(series) - len(values)
    if not values:
        raise ValueError("NASA POWER payload has no valid days")
    p10 = values[max(0, int(0.1 * len(values)) - 1)]
    return IrradianceSummary(source, len(values), sum(values) / len(values), p10, missing)


def load(lat: float, lon: float, year: int, cache_dir: Path, offline: bool = False) -> IrradianceSummary:
    """Live or cached NASA POWER data, falling back to the bundled sample."""
    if not offline:
        try:
            return summarize(fetch(lat, lon, year, cache_dir), f"NASA POWER {lat},{lon} {year}")
        except Exception as exc:  # network, HTTP or schema problems all fall back
            log.warning("NASA POWER unavailable (%s), using bundled sample", exc)
    return summarize(json.loads(SAMPLE_FILE.read_text()), f"bundled sample {SAMPLE_FILE.name}")


def module_kwh_per_day(irr: IrradianceSummary, module_wp: float, performance_ratio: float) -> float:
    """Energy of a healthy module on an average day.

    Peak sun hours equal daily irradiation in kWh/m2 because STC is 1 kW/m2.
    Using horizontal irradiance for a tilted array is a simplification that is
    close at low latitudes such as the Colombian Caribbean.
    """
    return module_wp / 1000.0 * irr.mean_kwh_m2_day * performance_ratio

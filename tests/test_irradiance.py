import json

import pytest

from pvtriage import irradiance


def test_bundled_sample_summary():
    irr = irradiance.summarize(json.loads(irradiance.SAMPLE_FILE.read_text()), "sample")
    assert irr.days == 365
    assert 3.0 < irr.mean_kwh_m2_day < 8.0
    assert irr.p10_kwh_m2_day <= irr.mean_kwh_m2_day


def test_fill_values_are_ignored():
    payload = {
        "header": {"fill_value": -999.0},
        "properties": {"parameter": {"ALLSKY_SFC_SW_DWN": {"20250101": 5.0, "20250102": -999.0, "20250103": 3.0}}},
    }
    irr = irradiance.summarize(payload, "t")
    assert irr.days == 2 and irr.missing_days == 1
    assert irr.mean_kwh_m2_day == pytest.approx(4.0)


def test_bad_payload_raises():
    with pytest.raises(ValueError):
        irradiance.summarize({"properties": {}}, "t")


def test_offline_uses_sample(tmp_path):
    assert irradiance.load(0, 0, 2025, tmp_path, offline=True).source.startswith("bundled sample")


def test_module_energy():
    irr = irradiance.IrradianceSummary("t", 1, 5.0, 4.0, 0)
    assert irradiance.module_kwh_per_day(irr, 400, 0.8) == pytest.approx(1.6)

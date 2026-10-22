import json

from pvtriage.config import Settings
from pvtriage.pipeline import run


def test_end_to_end_on_sample(tmp_path, sample_dir):
    settings = Settings(data_dir=sample_dir, out_dir=tmp_path, db_path=tmp_path / "t.duckdb", epochs=1, tracking=False)
    settings.k_neighbors = 5
    settings.ingest_batch_size = 10
    metrics = run(settings, offline=True)

    assert metrics["test_modules"] == 36
    assert metrics["duckdb"]["trigger_rollups_match_rescan"]
    assert metrics["duckdb"]["ingest_batches"] == 4
    for key in ("metrics.json", "work_orders.csv", "results.png", "rollups.json"):
        assert (tmp_path / key).exists()
    json.loads((tmp_path / "metrics.json").read_text())

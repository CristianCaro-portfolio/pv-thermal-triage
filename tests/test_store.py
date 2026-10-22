import numpy as np

from pvtriage.config import CLASS_INDEX, CLASSES
from pvtriage.store import TriageStore


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _store():
    store = TriageStore(":memory:", embedding_dim=4)
    ref = _unit([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0], [0, 0.95, 0.05, 0], [0, 0, 1, 0]])
    labels = np.array([CLASS_INDEX[c] for c in ("Diode", "Diode", "Soiling", "Soiling", "No-Anomaly")])
    store.load_reference(np.arange(5), labels, ref)
    return store


def test_nearest_join_votes_follow_closest_precedents():
    store = _store()
    store.load_queries(np.array([100, 101]), _unit([[1, 0.05, 0, 0], [0.05, 1, 0, 0]]))
    ids, votes = store.precedent_votes(k=2)
    assert ids.tolist() == [100, 101]
    assert np.allclose(votes.sum(1), 1)
    assert CLASSES[votes[0].argmax()] == "Diode"
    assert CLASSES[votes[1].argmax()] == "Soiling"


def test_precedents_for_returns_sorted_neighbours():
    store = _store()
    store.load_queries(np.array([7]), _unit([[1, 0, 0, 0]]))
    hits = store.precedents_for(7, k=3)
    assert [h[0] for h in hits][:2] == [0, 1]
    assert hits[0][2] >= hits[1][2] >= hits[2][2]


def _finding(batch, module, label="Diode", decision="auto", kwh=0.5):
    return (batch, module, label, 0.9, label, 0.8, decision, False, 0.1, 3.0, kwh, "x")


def test_trigger_rollup_tracks_every_insert():
    store = _store()
    store.insert_findings([_finding(0, 1), _finding(0, 2, "No-Anomaly", kwh=0.0)])
    store.insert_findings([_finding(0, 3, decision="review"), _finding(1, 4)])
    rollup = {r[0]: r for r in store.batch_rollup()}
    assert rollup[0][1:5] == (3, 2, 1, 0)
    assert rollup[0][5] == 1.0
    assert rollup[1][1] == 1
    assert store.rollup_matches_scan()
    classes = {r[0]: r[1] for r in store.class_rollup()}
    assert classes == {"Diode": 3, "No-Anomaly": 1}


def test_variant_metadata_round_trip():
    store = TriageStore(":memory:", embedding_dim=4)
    store.load_modules(
        np.array([5]),
        np.array(["test"]),
        np.array([CLASS_INDEX["Cell"]]),
        {"5": {"image_filepath": "images/5.jpg", "anomaly_class": "Cell"}},
    )
    keys = store.con.execute("SELECT variant_keys(source_meta) FROM modules").fetchone()[0]
    assert sorted(keys) == ["anomaly_class", "image_filepath"]


def test_rejects_wrong_embedding_width():
    store = TriageStore(":memory:", embedding_dim=4)
    try:
        store.load_queries(np.array([1]), np.zeros((1, 3), dtype=np.float32))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_second_merge_trigger_is_still_unsupported_in_duckdb_2_0():
    """Documents why class_rollup is a view (see docs/adr-001). Remove the view when this starts failing."""
    import duckdb
    import pytest

    con = duckdb.connect()
    con.execute("CREATE TABLE f (k INT); CREATE TABLE a (n BIGINT); CREATE TABLE b (n BIGINT)")
    for target in ("a", "b"):
        con.execute(
            f"CREATE TRIGGER t_{target} AFTER INSERT ON f REFERENCING NEW TABLE AS n "
            f"FOR EACH STATEMENT MERGE INTO {target} x USING (SELECT count(*) AS c FROM n) d "
            "ON x.n = d.c WHEN NOT MATCHED THEN INSERT VALUES (d.c)"
        )
    with pytest.raises(duckdb.Error):
        con.execute("INSERT INTO f VALUES (1)")

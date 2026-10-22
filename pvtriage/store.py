"""DuckDB 2.0 storage: module catalog, embeddings, precedent search and rollups.

Three DuckDB 2.0 features carry real weight here:
  * NEAREST joins find the most similar labelled modules for every new crop,
    which gives the technician precedents and gives triage a second opinion.
  * A trigger keeps per batch rollups current as findings are inserted, so
    the inspection dashboard never rescans the findings table.
  * VARIANT stores the raw source metadata without forcing a schema on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from .config import CLASSES

SCHEMA = """
CREATE TABLE IF NOT EXISTS modules (
    module_id   BIGINT PRIMARY KEY,
    split       VARCHAR NOT NULL,
    label       VARCHAR NOT NULL,
    source_meta VARIANT
);

CREATE TABLE IF NOT EXISTS reference_embeddings (
    module_id BIGINT PRIMARY KEY,
    label     VARCHAR NOT NULL,
    emb       FLOAT[{dim}] NOT NULL
);

CREATE TABLE IF NOT EXISTS query_embeddings (
    module_id BIGINT PRIMARY KEY,
    emb       FLOAT[{dim}] NOT NULL
);

CREATE TABLE IF NOT EXISTS findings (
    batch_id         INTEGER NOT NULL,
    module_id        BIGINT NOT NULL,
    predicted        VARCHAR NOT NULL,
    confidence       DOUBLE NOT NULL,
    precedent_label  VARCHAR NOT NULL,
    precedent_share  DOUBLE NOT NULL,
    decision         VARCHAR NOT NULL,
    safety_flag      BOOLEAN NOT NULL,
    contrast_index   DOUBLE NOT NULL,
    band_spread      DOUBLE NOT NULL,
    expected_kwh_day DOUBLE NOT NULL,
    action           VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_rollup (
    batch_id     INTEGER PRIMARY KEY,
    n_modules    BIGINT,
    n_anomalies  BIGINT,
    n_review     BIGINT,
    n_safety     BIGINT,
    kwh_day_lost DOUBLE
);

-- DuckDB 2.0.0 fails to bind two MERGE based statement triggers on the same
-- table, so the class rollup is a view computed on read. It is small.
CREATE VIEW IF NOT EXISTS class_rollup AS
    SELECT predicted,
           count(*) AS n_modules,
           count(*) FILTER (WHERE decision = 'review') AS n_review,
           sum(expected_kwh_day) AS kwh_day_lost
    FROM findings GROUP BY predicted;
"""

TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS findings_batch_rollup
    AFTER INSERT ON findings
    REFERENCING NEW TABLE AS n
    FOR EACH STATEMENT
        MERGE INTO batch_rollup r
        USING (
            SELECT batch_id,
                   count(*) AS n_modules,
                   count(*) FILTER (WHERE predicted <> 'No-Anomaly') AS n_anomalies,
                   count(*) FILTER (WHERE decision = 'review') AS n_review,
                   count(*) FILTER (WHERE safety_flag) AS n_safety,
                   sum(expected_kwh_day) AS kwh_day_lost
            FROM n GROUP BY batch_id
        ) d
        ON r.batch_id = d.batch_id
        WHEN MATCHED THEN UPDATE SET
            n_modules = r.n_modules + d.n_modules,
            n_anomalies = r.n_anomalies + d.n_anomalies,
            n_review = r.n_review + d.n_review,
            n_safety = r.n_safety + d.n_safety,
            kwh_day_lost = r.kwh_day_lost + d.kwh_day_lost
        WHEN NOT MATCHED THEN INSERT
            VALUES (d.batch_id, d.n_modules, d.n_anomalies, d.n_review, d.n_safety, d.kwh_day_lost);

"""

FINDING_COLUMNS = (
    "batch_id",
    "module_id",
    "predicted",
    "confidence",
    "precedent_label",
    "precedent_share",
    "decision",
    "safety_flag",
    "contrast_index",
    "band_spread",
    "expected_kwh_day",
    "action",
)


class TriageStore:
    def __init__(self, path: Path | str = ":memory:", embedding_dim: int = 64) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()  # every run rebuilds the store from source data
            path = str(path)
        self.dim = embedding_dim
        self.con = duckdb.connect(path)
        self.con.execute(SCHEMA.format(dim=embedding_dim))
        self.con.execute(TRIGGERS)

    def close(self) -> None:
        self.con.close()

    # ---- loading -------------------------------------------------------
    def load_modules(self, module_ids: np.ndarray, splits: np.ndarray, labels: np.ndarray, raw: dict) -> None:
        import pyarrow as pa

        staged = pa.table(  # noqa: F841 - referenced by name in SQL below
            {
                "module_id": pa.array(module_ids.astype(np.int64)),
                "split": pa.array([str(s) for s in splits]),
                "label": pa.array([CLASSES[int(y)] for y in labels]),
                "meta": pa.array([json.dumps(raw.get(str(int(m)), {})) for m in module_ids]),
            }
        )
        self.con.execute("INSERT INTO modules SELECT module_id, split, label, meta::JSON::VARIANT FROM staged")

    def _load_embeddings(self, table: str, columns: dict[str, object]) -> None:
        import pyarrow as pa

        emb = np.ascontiguousarray(columns.pop("emb"), dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"expected embeddings of shape (n, {self.dim}), got {emb.shape}")
        arrays = {k: pa.array(v) for k, v in columns.items()}
        arrays["emb"] = pa.FixedSizeListArray.from_arrays(pa.array(emb.ravel()), self.dim)
        staged = pa.table(arrays)  # noqa: F841 - referenced by name in SQL below
        self.con.execute(f"INSERT INTO {table} SELECT * FROM staged")

    def load_reference(self, module_ids: np.ndarray, labels: np.ndarray, emb: np.ndarray) -> None:
        names = [CLASSES[int(y)] for y in labels]
        self._load_embeddings(
            "reference_embeddings", {"module_id": module_ids.astype(np.int64), "label": names, "emb": emb}
        )

    def load_queries(self, module_ids: np.ndarray, emb: np.ndarray) -> None:
        self.con.execute("DELETE FROM query_embeddings")
        self._load_embeddings("query_embeddings", {"module_id": module_ids.astype(np.int64), "emb": emb})

    # ---- precedent search ---------------------------------------------
    def precedent_votes(self, k: int, approx: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Similarity weighted class votes from the k nearest labelled modules.

        Returns (module_ids, votes) where votes has one column per class and
        each row sums to 1.
        """
        nearest = "APPROX NEAREST" if approx else "NEAREST"
        sql = f"""
            WITH hits AS (
                SELECT q.module_id AS query_id, r.label,
                       array_cosine_similarity(q.emb, r.emb) AS sim
                FROM query_embeddings q
                INNER JOIN reference_embeddings r
                    {nearest} {int(k)} BY SIMILARITY array_cosine_similarity(q.emb, r.emb)
            )
            SELECT query_id, label, sum(greatest(sim, 0.0)) AS weight
            FROM hits GROUP BY query_id, label
        """
        rows = self.con.execute(sql).fetchall()
        ids = np.array(sorted({r[0] for r in rows}), dtype=np.int64)
        pos = {m: i for i, m in enumerate(ids)}
        votes = np.zeros((len(ids), len(CLASSES)), dtype=np.float64)
        class_pos = {c: i for i, c in enumerate(CLASSES)}
        for query_id, label, weight in rows:
            votes[pos[query_id], class_pos[label]] += weight
        totals = votes.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return ids, votes / totals

    def precedents_for(self, module_id: int, k: int = 3) -> list[tuple[int, str, float]]:
        sql = f"""
            SELECT r.module_id, r.label, array_cosine_similarity(q.emb, r.emb) AS sim
            FROM (SELECT * FROM query_embeddings WHERE module_id = ?) q
            INNER JOIN reference_embeddings r
                NEAREST {int(k)} BY SIMILARITY array_cosine_similarity(q.emb, r.emb)
            ORDER BY sim DESC
        """
        return [(int(a), str(b), float(c)) for a, b, c in self.con.execute(sql, [module_id]).fetchall()]

    # ---- findings ------------------------------------------------------
    def insert_findings(self, rows: list[tuple]) -> None:
        """Insert one inspection batch as a single statement, so the trigger fires once per batch."""
        import pyarrow as pa

        if not rows:
            return
        columns = list(zip(*rows, strict=True))
        staged = pa.table({name: pa.array(col) for name, col in zip(FINDING_COLUMNS, columns, strict=True)})  # noqa: F841
        self.con.execute(f"INSERT INTO findings ({', '.join(FINDING_COLUMNS)}) SELECT * FROM staged")

    def batch_rollup(self) -> list[tuple]:
        return self.con.execute("SELECT * FROM batch_rollup ORDER BY batch_id").fetchall()

    def class_rollup(self) -> list[tuple]:
        return self.con.execute("SELECT * FROM class_rollup ORDER BY kwh_day_lost DESC").fetchall()

    def rollup_matches_scan(self) -> bool:
        """Check that trigger maintained totals equal a full rescan of findings."""
        diff = self.con.execute(
            """
            SELECT count(*) FROM (
                SELECT batch_id, count(*) AS n, round(sum(expected_kwh_day), 6) AS kwh
                FROM findings GROUP BY batch_id
            ) s FULL OUTER JOIN batch_rollup r USING (batch_id)
            WHERE s.n IS DISTINCT FROM r.n_modules OR s.kwh IS DISTINCT FROM round(r.kwh_day_lost, 6)
            """
        ).fetchone()[0]
        return diff == 0

    def work_orders(self, limit: int) -> list[dict]:
        cur = self.con.execute(
            """
            SELECT f.*, m.label AS true_label
            FROM findings f JOIN modules m USING (module_id)
            WHERE f.predicted <> 'No-Anomaly' OR f.decision = 'review'
            ORDER BY f.safety_flag DESC, f.expected_kwh_day DESC, f.confidence DESC
            LIMIT ?
            """,
            [limit],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

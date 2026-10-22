# ADR 001: DuckDB 2.0 for precedent search and rollups

Status: accepted, October 2026

## Context

Every flagged module needs two things from storage. First, the most similar
modules that a person already labelled, so a technician can compare and so
triage has a second opinion that does not come from the classifier alone.
Second, running totals per inspection batch (modules, anomalies, review load,
kWh at risk) that stay current while findings stream in.

The usual answer is a vector database next to a warehouse. For one solar site
inspection that is two services to run for a few thousand vectors.

## Decision

Use DuckDB 2.0.0 (released 21 October 2026) as the only store.

* `NEAREST k BY SIMILARITY array_cosine_similarity(...)` joins every new
  embedding to its k most similar labelled modules in plain SQL. The vote is a
  similarity weighted group by over the join output.
* An `AFTER INSERT ... REFERENCING NEW TABLE ... FOR EACH STATEMENT` trigger
  merges each inserted batch into `batch_rollup`. The pipeline checks after
  every run that the trigger totals match a full rescan.
* The raw source metadata goes into a `VARIANT` column, so the source schema can
  change without a migration.

## Consequences

* One embedded file, no server, runs in CI.
* Exact NEAREST is a brute force scan. 3,001 queries against 13,999 reference
  rows took about 17 s on 2 CPU threads, against 5 to 8 s for a numpy
  reference that the pipeline runs next to it (both return the same top class
  for every query). That is fine for a nightly batch per site, not for an
  interactive tool. `APPROX NEAREST` gave the same answers and no speed up
  without an index. At fleet scale I would partition by site and survey or add
  an ANN index.
* DuckDB 2.0.0 fails with an internal binder error when two statement level
  triggers on the same table both run a MERGE over the transition table. The class
  rollup is therefore a view, and only the batch rollup is trigger maintained.
  This is pinned with a test so an upgrade that fixes it is easy to spot.
* Files written by 2.0 cannot be read by 1.x. The store is rebuilt on each run,
  so this does not matter here, but it would for a shared file.

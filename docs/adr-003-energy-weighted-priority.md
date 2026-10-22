# ADR 003: Rank work orders by expected kWh at risk

Status: accepted, October 2026

## Context

Drone thermography reports usually rank findings by temperature delta. The
hottest spot is not always the most expensive one: an offline module is warm
everywhere and has little contrast, while a single hot cell can look dramatic
and cost almost nothing.

## Decision

Each finding gets an expected loss in kWh per day:

    expected_kwh = sum over classes (p_class x loss_fraction_class) x healthy_module_kwh_day

* `p_class` is the fused probability, so an uncertain diode fault still ranks
  high.
* `loss_fraction_class` is a reviewed planning assumption in `config.py`
  (for example one third of output for one bypassed substring).
* `healthy_module_kwh_day` comes from NASA POWER daily irradiance for the
  configured site, module rating and performance ratio. NASA POWER is free and
  needs no key; the response is cached and a one year sample is bundled for
  offline runs.

Safety classes are ordered first regardless of energy.

The ranking is evaluated by how much of the true energy loss (true class loss
fraction) is found when crews visit the top x percent of modules, against
ranking by the contrast index alone.

## Consequences

* The ranking is in units a site manager already uses.
* Loss fractions are assumptions, so absolute kWh numbers are planning
  estimates. The ranking comparison is still fair because every method is
  scored with the same fractions.
* MLflow records each run's parameters, metrics and artifacts in a local SQLite
  store so different assumptions can be compared side by side.

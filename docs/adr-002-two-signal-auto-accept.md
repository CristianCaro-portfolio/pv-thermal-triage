# ADR 002: Auto accept only when two signals agree, with a calibrated bar

Status: accepted, October 2026

## Context

A classifier that is right most of the time still sends trucks to the wrong
modules and, worse, can quietly mark a real hot spot as fine. Operations teams
want automation for the easy cases and a short, honest review queue for the
rest.

## Decision

A finding is auto accepted only when:

1. the CNN top class equals the top class of the DuckDB precedent vote, and
2. the fused confidence (average of CNN probabilities and precedent vote) is
   at or above a threshold, and
3. the fused probability of any safety class (thin film hot spots, multi cell
   hot spots, multiple bypass diodes) is below 0.25.

The threshold is the lowest value that reaches a target accuracy (default 97
percent) on the validation split, which gives the most automation that still
meets the target. The test split is never used for this choice, and the model
is the last epoch, not the best validation epoch, so validation stays clean for
calibration.

## Consequences

* The trade between automation and accuracy is one documented number, not a
  hidden constant.
* Precedents make every decision explainable: the work order can show the three
  most similar labelled modules.
* The guarantee is empirical on one split, not a formal bound. A conformal
  procedure would give a finite sample guarantee and is the next step if the
  review rate has to be contractually bounded.

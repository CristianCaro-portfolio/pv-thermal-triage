"""Command line entry points."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from . import data
from .config import Settings
from .pipeline import run as run_pipeline
from .pipeline import summary_lines, write_summary

app = typer.Typer(add_completion=False, help="Thermal triage for solar PV drone inspections.")


def _logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(asctime)s %(name)s %(message)s")


@app.command()
def download(data_dir: Path = typer.Option(Path("data/raw"), help="Where to put the dataset.")) -> None:
    """Download and verify the InfraredSolarModules dataset (about 15 MB)."""
    _logging(True)
    root = data.download(data_dir)
    typer.echo(f"dataset ready at {root}")


@app.command()
def run(
    data_dir: Path = typer.Option(None, help="Folder that contains InfraredSolarModules/."),
    out_dir: Path = typer.Option(None, help="Where metrics, figures and the DuckDB file go."),
    epochs: int = typer.Option(None, help="Training epochs."),
    offline: bool = typer.Option(False, help="Skip the NASA POWER call and use the bundled irradiance."),
    no_tracking: bool = typer.Option(False, help="Disable MLflow tracking."),
    verbose: bool = typer.Option(True),
) -> None:
    """Train, index, triage and evaluate."""
    _logging(verbose)
    settings = Settings()
    if data_dir:
        settings.data_dir = data_dir
    if out_dir:
        settings.out_dir = out_dir
        settings.db_path = out_dir / "pvtriage.duckdb"
    if epochs:
        settings.epochs = epochs
    if no_tracking:
        settings.tracking = False
    metrics = run_pipeline(settings, offline=offline)
    write_summary(metrics, settings.out_dir / "summary.txt")
    typer.echo("\n".join(summary_lines(metrics)))


@app.command()
def smoke(out_dir: Path = typer.Option(Path("reports/smoke"))) -> None:
    """Fast end to end run on the bundled sample, used in CI."""
    _logging(False)
    settings = Settings(
        data_dir=Path("data/sample"), out_dir=out_dir, db_path=out_dir / "pvtriage.duckdb", epochs=2, tracking=False
    )
    settings.k_neighbors = 5
    settings.ingest_batch_size = 10
    metrics = run_pipeline(settings, offline=True)
    assert metrics["duckdb"]["trigger_rollups_match_rescan"], "trigger rollups drifted from findings"
    typer.echo(json.dumps({k: metrics[k] for k in ("test_modules", "fused", "triage")}, indent=2))


@app.command("make-sample")
def make_sample(
    data_dir: Path = typer.Option(Path("data/raw")),
    out_dir: Path = typer.Option(Path("data/sample")),
    per_class: int = typer.Option(20),
    seed: int = typer.Option(7),
) -> None:
    """Rebuild the small stratified sample committed to the repo."""
    root = data.make_sample(data.resolve_root(data_dir), out_dir, per_class, seed)
    typer.echo(f"sample written to {root}")


@app.command()
def architecture(out: Path = typer.Option(Path("docs/architecture.png"))) -> None:
    """Render the architecture diagram."""
    from .report import plot_architecture

    typer.echo(plot_architecture(out))

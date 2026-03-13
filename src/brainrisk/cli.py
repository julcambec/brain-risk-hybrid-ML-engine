"""
Command-line interface for the brainrisk package.

Provides three commands:

- ``brainrisk generate-synthetic`` — generate synthetic data to a directory.
- ``brainrisk preprocess`` — run the preprocessing pipeline from a config.
- ``brainrisk demo-preprocessing`` — generate synthetic data then run the
  full preprocessing pipeline end-to-end (the primary demo entry point).
"""

from __future__ import annotations

import click

from brainrisk.utils.logging import setup_logger


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG-level logging.")
def cli(verbose: bool) -> None:
    """brainrisk — Neuroimaging ML pipeline for subtype discovery and modeling."""
    level = "DEBUG" if verbose else "INFO"
    setup_logger(name="brainrisk", level=level)


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    default="artifacts/synthetic",
    show_default=True,
    help="Directory to write synthetic data.",
)
@click.option("--n-subjects", "-n", default=50, show_default=True, help="Number of subjects.")
@click.option("--seed", "-s", default=42, show_default=True, help="Random seed.")
def generate_synthetic(output_dir: str, n_subjects: int, seed: int) -> None:
    """Generate synthetic ROI features, volumetric data, clinical data, and labels."""
    from brainrisk.data.synthetic import (
        generate_clinical_data,
        generate_labels,
        generate_roi_features,
        generate_volumetric_data,
    )

    click.echo(f"Generating synthetic data for {n_subjects} subjects (seed={seed}) ...")

    labels_df = generate_labels(n_subjects=n_subjects, seed=seed, output_dir=output_dir)
    generate_roi_features(
        n_subjects=n_subjects, n_features=400, n_sites=3, seed=seed, output_dir=output_dir
    )
    generate_clinical_data(n_subjects=n_subjects, n_sites=3, seed=seed, output_dir=output_dir)
    generate_volumetric_data(
        n_subjects=n_subjects,
        shape=(64, 64, 64),
        seed=seed,
        output_dir=output_dir,
        labels=labels_df,
    )

    click.echo(f"Synthetic data written to {output_dir}/")


@cli.command()
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True),
    help="Path to preprocessing YAML config.",
)
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Output directory (overrides config).",
)
def preprocess(config: str | None, output_dir: str | None) -> None:
    """Run the preprocessing pipeline (shared core → ROI branch + DL branch)."""
    from brainrisk.preprocessing.pipeline import run_pipeline

    report = run_pipeline(config_path=config, output_dir=output_dir)
    _print_report_summary(report)


@cli.command(name="demo-preprocessing")
@click.option(
    "--config",
    "-c",
    default="configs/preprocessing.yaml",
    show_default=True,
    type=click.Path(exists=True),
    help="Path to preprocessing YAML config.",
)
@click.option(
    "--output-dir",
    "-o",
    default="artifacts",
    show_default=True,
    help="Output directory.",
)
@click.option("--n-subjects", "-n", default=50, show_default=True, help="Number of subjects.")
def demo_preprocessing(config: str, output_dir: str, n_subjects: int) -> None:
    """
    Generate synthetic data and run the full preprocessing pipeline (demo mode).

    This is the primary demo entry point. It exercises the complete pipeline
    on synthetic data without requiring real MRI scans or FreeSurfer.
    """
    from brainrisk.preprocessing.pipeline import run_pipeline

    overrides = {"n_subjects": n_subjects, "mode": "demo"}
    report = run_pipeline(config_path=config, config_overrides=overrides, output_dir=output_dir)
    _print_report_summary(report)


def _print_report_summary(report: dict) -> None:
    """Print a human-readable summary of the pipeline QC report."""
    click.echo("")
    click.echo("=" * 60)
    click.echo("  PREPROCESSING PIPELINE — SUMMARY")
    click.echo("=" * 60)
    click.echo(f"  Mode: {report.get('mode', 'unknown')}")

    roi = report.get("roi_branch", {})
    if roi:
        shape = roi.get("roi_shape", [])
        warnings = roi.get("schema_warnings", [])
        click.echo(f"  ROI branch:  {shape[0]} subjects × {shape[1]} columns")
        if warnings:
            click.echo(f"    Warnings: {len(warnings)}")
        else:
            click.echo("    Schema validation: PASSED")

    dl = report.get("dl_branch", {})
    if dl:
        n = dl.get("n_subjects", 0)
        target = dl.get("target_shape", [])
        vol_warnings = dl.get("volume_warnings", [])
        click.echo(f"  DL branch:   {n} subjects → {'×'.join(str(d) for d in target)}")
        if vol_warnings:
            click.echo(f"    Volume warnings: {len(vol_warnings)}")
        else:
            click.echo("    Volume QC: PASSED")

    report_path = report.get("report_path", "")
    if report_path:
        click.echo(f"  QC report:   {report_path}")

    click.echo("=" * 60)
    click.echo("")

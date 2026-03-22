"""
Command-line interface for the brainrisk package.

Provides five commands:

- ``brainrisk generate-synthetic``: generate synthetic data to a directory.
- ``brainrisk preprocess``: run the preprocessing pipeline from a config.
- ``brainrisk demo-preprocessing``: generate synthetic data then run the
  full preprocessing pipeline end-to-end (the primary demo entry point).
- ``brainrisk demo-ml``: generate synthetic data, run HYDRA clustering and
  ROI baselines, and print a summary (the ML-track demo entry point).
- ``brainrisk demo-dl``: generate synthetic data and run a minimal MINiT
  training epoch (the DL-track demo entry point).
"""

from __future__ import annotations

import click

from brainrisk.utils.logging import setup_logger


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG-level logging.")
def cli(verbose: bool) -> None:
    """brainrisk: Neuroimaging ML pipeline for subtype discovery and modeling."""
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
    generate_clinical_data(
        n_subjects=n_subjects,
        n_sites=3,
        seed=seed,
        output_dir=output_dir,
        labels=labels_df,
    )
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


@cli.command(name="demo-ml")
@click.option(
    "--output-dir",
    "-o",
    default="artifacts",
    show_default=True,
    help="Output directory for synthetic data and results.",
)
@click.option("--n-subjects", "-n", default=50, show_default=True, help="Number of subjects.")
@click.option("--seed", "-s", default=42, show_default=True, help="Random seed.")
def demo_ml(output_dir: str, n_subjects: int, seed: int) -> None:
    """
    Generate synthetic data, run HYDRA clustering, and evaluate ROI baselines.

    This is the ML-track demo entry point. It exercises semi-supervised
    clustering, classification (subtype / sex / maternal substance use),
    and regression (family income) on synthetic data.
    """
    from pathlib import Path

    import numpy as np

    from brainrisk.baselines.classifiers import run_classification_suite
    from brainrisk.baselines.regressors import run_regression_suite
    from brainrisk.clustering.hydra import HydraClusterer
    from brainrisk.data.clinical import get_feature_columns, load_and_merge, prepare_task
    from brainrisk.data.splits import subject_split
    from brainrisk.data.synthetic import (
        generate_clinical_data,
        generate_labels,
        generate_roi_features,
    )

    synthetic_dir = Path(output_dir) / "synthetic"

    # --- Generate synthetic data ---
    click.echo(f"Generating synthetic data ({n_subjects} subjects, seed={seed}) ...")
    labels_df = generate_labels(n_subjects=n_subjects, seed=seed, output_dir=synthetic_dir)
    generate_roi_features(
        n_subjects=n_subjects, n_features=400, n_sites=3, seed=seed, output_dir=synthetic_dir
    )
    generate_clinical_data(
        n_subjects=n_subjects,
        n_sites=3,
        seed=seed,
        output_dir=synthetic_dir,
        labels=labels_df,
    )

    roi_path = synthetic_dir / "roi" / "features.csv"
    clinical_path = synthetic_dir / "labels" / "clinical.csv"

    # --- Load and merge ---
    merged = load_and_merge(roi_path, clinical_path)
    feature_cols = get_feature_columns(merged)
    click.echo(f"Loaded {len(merged)} subjects × {len(feature_cols)} ROI features.")

    # --- Quick HYDRA clustering demo ---
    click.echo("\nRunning HYDRA clustering (k=3) ...")
    X_all, _, _ = prepare_task(merged, "hydra_subtype", feature_cols)

    # Designate subtype 3 as pseudo-controls for the demo.
    subtypes = merged["hydra_subtype"].to_numpy()
    control_mask = subtypes == 3
    patient_features = X_all[~control_mask]
    control_features = X_all[control_mask]

    clusterer = HydraClusterer(n_clusters=2, covariate_correction=False, seed=seed)
    clusterer.fit(patient_features, control_features)

    labels = clusterer.labels_
    if labels is None:
        raise RuntimeError("HydraClusterer.fit() did not populate labels_.")

    n_per_cluster = np.bincount(labels)[1:]  # skip index 0
    click.echo(
        f"  HYDRA found {len(n_per_cluster)} clusters among "
        f"{len(patient_features)} patients (sizes: {list(n_per_cluster)})"
    )

    # --- Train/test split ---
    train_df, test_df = subject_split(
        merged, test_size=0.2, stratify_col="hydra_subtype", seed=seed
    )
    click.echo(f"\nTrain/test split: {len(train_df)} / {len(test_df)} subjects.")

    # --- Classification baselines ---
    click.echo("\nRunning classification baselines ...")
    clf_tasks = [
        ("HYDRA subtype (3-class)", "hydra_subtype"),
        ("Sex (binary)", "sex"),
        ("Maternal substance use", "maternal_substance_use"),
    ]

    all_clf_results: list[dict] = []
    for task_name, target_col in clf_tasks:
        X_train, y_train, _ = prepare_task(train_df, target_col, feature_cols)
        X_test, y_test, _ = prepare_task(test_df, target_col, feature_cols)
        results = run_classification_suite(
            X_train,
            X_test,
            y_train,
            y_test,
            task_name=task_name,
            seed=seed,
        )
        all_clf_results.extend(results)

    # --- Regression baselines ---
    click.echo("Running regression baselines ...")
    X_train, y_train, _ = prepare_task(train_df, "family_income_k", feature_cols)
    X_test, y_test, _ = prepare_task(test_df, "family_income_k", feature_cols)
    all_reg_results = run_regression_suite(
        X_train,
        X_test,
        y_train,
        y_test,
        task_name="Family income",
        seed=seed,
    )

    # --- Print summary ---
    _print_ml_summary(all_clf_results, all_reg_results)


@cli.command(name="demo-dl")
@click.option(
    "--output-dir",
    "-o",
    default="artifacts",
    show_default=True,
    help="Output directory for synthetic data and training artifacts.",
)
@click.option("--n-subjects", "-n", default=20, show_default=True, help="Number of subjects.")
@click.option("--seed", "-s", default=42, show_default=True, help="Random seed.")
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True),
    help="Optional YAML config override (default: demo_dl.yaml).",
)
def demo_dl(output_dir: str, n_subjects: int, seed: int, config: str | None) -> None:
    """
    Generate synthetic data and run a minimal MINiT training epoch.

    This is the DL-track demo entry point. It generates synthetic 64³
    volumes, trains a small MINiT model for 4 epochs, and saves
    checkpoints and a training log. Designed to complete in under 2 minutes
    on a CPU-only machine.
    """
    try:
        import torch  # noqa: F401
    except ImportError as err:
        click.echo(
            "ERROR: PyTorch is required for the DL track.\nInstall with: pip install -e '.[dl]'"
        )
        raise SystemExit(1) from err

    from pathlib import Path

    from brainrisk.data.synthetic import (
        generate_clinical_data,
        generate_labels,
        generate_volumetric_data,
    )
    from brainrisk.deeplearning.trainer import run_training
    from brainrisk.utils.config import load_config

    synthetic_dir = Path(output_dir) / "synthetic"

    # --- Generate synthetic data ---
    click.echo(f"Generating synthetic data ({n_subjects} subjects, seed={seed}) ...")
    labels_df = generate_labels(n_subjects=n_subjects, seed=seed, output_dir=synthetic_dir)
    generate_clinical_data(
        n_subjects=n_subjects,
        n_sites=2,
        seed=seed,
        output_dir=synthetic_dir,
        labels=labels_df,
    )
    generate_volumetric_data(
        n_subjects=n_subjects,
        shape=(64, 64, 64),
        seed=seed,
        output_dir=synthetic_dir,
        labels=labels_df,
    )

    # --- Build training config ---
    if config is not None:
        training_config = load_config(config)
    else:
        # Inline demo config (small model for fast CPU execution)
        training_config = {
            "seed": seed,
            "model": {
                "volume_size": 64,
                "block_size": 16,
                "patch_size": 8,
                "in_channels": 1,
                "num_classes": 3,
                "embed_dim": 32,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_dim": 64,
                "dropout": 0.1,
            },
            "training": {
                "epochs": 4,
                "batch_size": 4,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "warmup_epochs": 0,
                "label_column": "hydra_subtype",
                "test_size": 0.2,
                "num_workers": 0,
                "early_stopping_patience": 0,
            },
            "augmentation": {"mixup_alpha": 0.0},
            "checkpoint": {
                "save_dir": str(Path(output_dir) / "dl" / "checkpoints"),
                "save_every": 1,
            },
            "logging": {
                "log_path": str(Path(output_dir) / "dl" / "training_log.jsonl"),
                "wandb_enabled": False,
            },
            "freeze_strategy": "full",
        }

    # Inject data paths from the just-generated synthetic data
    training_config["data"] = {
        "manifest_path": str(synthetic_dir / "volumes" / "manifest.csv"),
        "clinical_path": str(synthetic_dir / "labels" / "clinical.csv"),
        "label_column": training_config.get("training", {}).get("label_column", "hydra_subtype"),
        "test_size": training_config.get("training", {}).get("test_size", 0.2),
    }
    training_config["seed"] = seed

    # --- Train ---
    click.echo("")
    click.echo("=" * 60)
    click.echo("  DEEP LEARNING TRACK — MINiT DEMO")
    click.echo("=" * 60)

    summary = run_training(training_config)

    # --- Print summary ---
    click.echo("")
    click.echo("=" * 60)
    click.echo("  DL DEMO — SUMMARY")
    click.echo("=" * 60)
    click.echo(f"  Best val accuracy:  {summary['best_val_acc']:.4f}")
    click.echo(f"  Final epoch:        {summary['final_epoch']}")
    click.echo(f"  Training log:       {summary['log_path']}")
    click.echo(f"  Checkpoints:        {summary['checkpoint_dir']}")
    click.echo("=" * 60)
    click.echo("")


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


def _print_ml_summary(
    clf_results: list[dict],
    reg_results: list[dict],
) -> None:
    """Print a formatted summary of ML baseline results."""
    click.echo("")
    click.echo("=" * 68)
    click.echo("  ML BASELINES — SUMMARY")
    click.echo("=" * 68)

    # Classification table
    click.echo("")
    click.echo(f"  {'Task':<30s} {'Model':<16s} {'F1-macro':>8s} {'Accuracy':>8s}")
    click.echo(f"  {'─' * 30} {'─' * 16} {'─' * 8} {'─' * 8}")
    for r in clf_results:
        m = r["metrics"]
        click.echo(
            f"  {r['task']:<30s} {r['model_type']:<16s} {m['f1_macro']:>8.3f} {m['accuracy']:>8.3f}"
        )

    # Regression table
    click.echo("")
    click.echo(f"  {'Task':<30s} {'Model':<16s} {'R²':>8s} {'MAE':>8s}")
    click.echo(f"  {'─' * 30} {'─' * 16} {'─' * 8} {'─' * 8}")
    for r in reg_results:
        m = r["metrics"]
        click.echo(f"  {r['task']:<30s} {r['model_type']:<16s} {m['r2']:>8.3f} {m['mae']:>8.2f}")

    click.echo("")
    click.echo("=" * 68)
    click.echo("")

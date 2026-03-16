"""
Unit and integration tests for the baselines module:
data splits, clinical loading, classifiers, regressors, and interpretation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brainrisk.baselines.classifiers import (
    build_pipeline as build_clf_pipeline,
)
from brainrisk.baselines.classifiers import (
    evaluate as clf_evaluate,
)
from brainrisk.baselines.classifiers import (
    run_classification_suite,
)
from brainrisk.baselines.classifiers import (
    tune_and_evaluate as clf_tune,
)
from brainrisk.baselines.interpretation import (
    extract_feature_importance,
    map_to_atlas,
    parse_roi_name,
)
from brainrisk.baselines.regressors import (
    build_pipeline as build_reg_pipeline,
)
from brainrisk.baselines.regressors import (
    evaluate as reg_evaluate,
)
from brainrisk.baselines.regressors import (
    run_regression_suite,
)
from brainrisk.baselines.regressors import (
    tune_and_evaluate as reg_tune,
)
from brainrisk.data.clinical import get_feature_columns, load_and_merge, prepare_task
from brainrisk.data.splits import subject_split

# ---------------
# data/splits.py
# ---------------


def test_subject_split_produces_disjoint_ids(synthetic_bundle: dict) -> None:
    roi_df = pd.read_csv(str(synthetic_bundle["roi"]))
    train, test = subject_split(roi_df, test_size=0.25, seed=42)

    train_ids = set(train["subject_id"])
    test_ids = set(test["subject_id"])
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == roi_df["subject_id"].nunique()


def test_subject_split_respects_test_size(synthetic_bundle: dict) -> None:
    roi_df = pd.read_csv(str(synthetic_bundle["roi"]))
    n_total = roi_df["subject_id"].nunique()
    _, test = subject_split(roi_df, test_size=0.25, seed=42)
    # Allow some rounding tolerance
    assert abs(test["subject_id"].nunique() - round(n_total * 0.25)) <= 1


def test_subject_split_stratified(synthetic_bundle: dict) -> None:
    clinical_df = pd.read_csv(str(synthetic_bundle["clinical"]))
    train, _ = subject_split(clinical_df, test_size=0.25, stratify_col="hydra_subtype", seed=42)
    # Both splits should have all subtypes
    assert set(train["hydra_subtype"].unique()) == set(clinical_df["hydra_subtype"].unique())


def test_subject_split_raises_on_missing_column() -> None:
    df = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
    with pytest.raises(KeyError, match="subject_id"):
        subject_split(df)


# -----------------
# data/clinical.py
# -----------------


def test_load_and_merge_returns_combined_dataframe(synthetic_bundle: dict) -> None:
    merged = load_and_merge(synthetic_bundle["roi"], synthetic_bundle["clinical"])
    assert "subject_id" in merged.columns
    assert "hydra_subtype" in merged.columns
    # Should have ROI feature columns too
    feature_cols = get_feature_columns(merged)
    assert len(feature_cols) > 0


def test_get_feature_columns_excludes_metadata(synthetic_bundle: dict) -> None:
    merged = load_and_merge(synthetic_bundle["roi"], synthetic_bundle["clinical"])
    feature_cols = get_feature_columns(merged)
    assert "subject_id" not in feature_cols
    assert "site" not in feature_cols
    assert "hydra_subtype" not in feature_cols
    assert "sex" not in feature_cols


def test_prepare_task_returns_correct_shapes(synthetic_bundle: dict) -> None:
    merged = load_and_merge(synthetic_bundle["roi"], synthetic_bundle["clinical"])
    feature_cols = get_feature_columns(merged)
    X, y, names = prepare_task(merged, "hydra_subtype", feature_cols)
    assert X.shape[0] == len(y)
    assert X.shape[1] == len(names)
    assert len(names) == len(feature_cols)


# -------------------------
# baselines/classifiers.py
# -------------------------


def test_build_clf_pipeline_valid_types() -> None:
    for model_type in ("dummy", "hist_boosting", "logistic"):
        pipe = build_clf_pipeline(model_type)
        assert "scaler" in pipe.named_steps
        assert "clf" in pipe.named_steps


def test_build_clf_pipeline_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_clf_pipeline("xgboost")


def test_clf_evaluate_returns_valid_metrics() -> None:
    y_true = np.array([1, 1, 2, 2, 3, 3])
    y_pred = np.array([1, 2, 2, 2, 3, 1])
    metrics = clf_evaluate(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_clf_tune_and_evaluate_runs(synthetic_bundle: dict) -> None:
    """Smoke test: tune_and_evaluate completes on synthetic data."""
    merged = load_and_merge(synthetic_bundle["roi"], synthetic_bundle["clinical"])
    feature_cols = get_feature_columns(merged)
    train, test = subject_split(merged, test_size=0.25, seed=42)

    X_train, y_train, _ = prepare_task(train, "hydra_subtype", feature_cols)
    X_test, y_test, _ = prepare_task(test, "hydra_subtype", feature_cols)

    result = clf_tune(
        X_train,
        X_test,
        y_train,
        y_test,
        model_type="dummy",
        seed=42,
    )
    assert "metrics" in result
    assert "fitted_pipeline" in result
    assert result["model_type"] == "dummy"


def test_run_classification_suite_returns_three_models(synthetic_bundle: dict) -> None:
    merged = load_and_merge(synthetic_bundle["roi"], synthetic_bundle["clinical"])
    feature_cols = get_feature_columns(merged)
    train, test = subject_split(merged, test_size=0.25, seed=42)

    X_train, y_train, _ = prepare_task(train, "sex", feature_cols)
    X_test, y_test, _ = prepare_task(test, "sex", feature_cols)

    results = run_classification_suite(
        X_train,
        X_test,
        y_train,
        y_test,
        task_name="sex",
        n_iter=3,
        cv=2,
        seed=42,
    )
    assert len(results) == 3
    model_types = {r["model_type"] for r in results}
    assert model_types == {"dummy", "hist_boosting", "logistic"}
    assert all("task" in r for r in results)


# ------------------------
# baselines/regressors.py
# ------------------------


def test_build_reg_pipeline_valid_types() -> None:
    for model_type in ("dummy", "hist_boosting", "ridge"):
        pipe = build_reg_pipeline(model_type)
        assert "scaler" in pipe.named_steps
        assert "reg" in pipe.named_steps


def test_build_reg_pipeline_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_reg_pipeline("svr")


def test_reg_evaluate_returns_valid_metrics() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
    metrics = reg_evaluate(y_true, y_pred)
    assert "r2" in metrics
    assert "mae" in metrics
    assert "mse" in metrics
    assert metrics["r2"] > 0.9  # predictions are close


def test_run_regression_suite_returns_three_models(synthetic_bundle: dict) -> None:
    merged = load_and_merge(synthetic_bundle["roi"], synthetic_bundle["clinical"])
    feature_cols = get_feature_columns(merged)
    train, test = subject_split(merged, test_size=0.25, seed=42)

    X_train, y_train, _ = prepare_task(train, "family_income_k", feature_cols)
    X_test, y_test, _ = prepare_task(test, "family_income_k", feature_cols)

    results = run_regression_suite(
        X_train,
        X_test,
        y_train,
        y_test,
        task_name="income",
        n_iter=3,
        cv=2,
        seed=42,
    )
    assert len(results) == 3
    model_types = {r["model_type"] for r in results}
    assert model_types == {"dummy", "hist_boosting", "ridge"}


def test_regression_dummy_baseline_has_zero_r2() -> None:
    """Dummy regressor should have R-squared near 0 (predicts the mean)."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (50, 10))
    y_train = rng.normal(100, 20, 50)
    X_test = rng.normal(0, 1, (15, 10))
    y_test = rng.normal(100, 20, 15)

    result = reg_tune(
        X_train,
        X_test,
        y_train,
        y_test,
        model_type="dummy",
        seed=42,
    )
    # Dummy R-squared is typically around 0 (or slightly negative)
    assert result["metrics"]["r2"] < 0.3


# ----------------------------
# baselines/interpretation.py
# ----------------------------


def test_parse_roi_name_cortical() -> None:
    parsed = parse_roi_name("lh_bankssts_thickness")
    assert parsed["hemisphere"] == "lh"
    assert parsed["region"] == "bankssts"
    assert parsed["measure"] == "thickness"


def test_parse_roi_name_subcortical() -> None:
    parsed = parse_roi_name("Left-Hippocampus_volume")
    assert parsed["hemisphere"] == "Left"
    assert parsed["region"] == "Hippocampus"
    assert parsed["measure"] == "volume"


def test_parse_roi_name_generic() -> None:
    parsed = parse_roi_name("morphometry_feature_001")
    assert parsed["hemisphere"] is None
    assert parsed["region"] is None
    assert parsed["measure"] is None


def test_extract_feature_importance_from_logistic(synthetic_bundle: dict) -> None:
    """Feature importance extraction works on a fitted logistic pipeline."""
    merged = load_and_merge(synthetic_bundle["roi"], synthetic_bundle["clinical"])
    feature_cols = get_feature_columns(merged)
    train, _ = subject_split(merged, test_size=0.25, seed=42)

    X_train, y_train, _ = prepare_task(train, "sex", feature_cols)

    pipe = build_clf_pipeline("logistic", seed=42)
    pipe.fit(X_train, y_train)

    imp_df = extract_feature_importance(pipe, feature_cols, method="model")
    assert "feature" in imp_df.columns
    assert "importance" in imp_df.columns
    assert len(imp_df) == len(feature_cols)
    # Should be sorted by abs_importance descending
    assert imp_df["abs_importance"].is_monotonic_decreasing


def test_map_to_atlas_annotates_features() -> None:
    imp_df = pd.DataFrame(
        {
            "feature": [
                "lh_bankssts_thickness",
                "rh_cuneus_area",
                "Left-Hippocampus_volume",
                "morphometry_feature_001",
            ],
            "importance": [0.5, 0.3, 0.2, 0.1],
            "abs_importance": [0.5, 0.3, 0.2, 0.1],
        }
    )
    atlas_df = map_to_atlas(imp_df, top_n=4)
    assert "hemisphere" in atlas_df.columns
    assert "region" in atlas_df.columns
    assert "measure" in atlas_df.columns
    assert len(atlas_df) == 4
    assert atlas_df.iloc[0]["region"] == "bankssts"


# -----------------------
# Integration: full flow
# -----------------------


def test_full_baselines_flow_on_synthetic_bundle(synthetic_bundle: dict) -> None:
    """Integration: load data → split → classify → regress → interpret."""
    merged = load_and_merge(synthetic_bundle["roi"], synthetic_bundle["clinical"])
    feature_cols = get_feature_columns(merged)
    train, test = subject_split(merged, test_size=0.25, stratify_col="hydra_subtype", seed=42)

    # Classification: subtype
    X_train, y_train, _ = prepare_task(train, "hydra_subtype", feature_cols)
    X_test, y_test, _ = prepare_task(test, "hydra_subtype", feature_cols)
    clf_results = run_classification_suite(
        X_train,
        X_test,
        y_train,
        y_test,
        task_name="subtype",
        n_iter=3,
        cv=2,
        seed=42,
    )
    assert len(clf_results) == 3

    # Regression: income
    X_train_r, y_train_r, _ = prepare_task(train, "family_income_k", feature_cols)
    X_test_r, y_test_r, _ = prepare_task(test, "family_income_k", feature_cols)
    reg_results = run_regression_suite(
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        task_name="income",
        n_iter=3,
        cv=2,
        seed=42,
    )
    assert len(reg_results) == 3

    # Interpretation: from the logistic model
    logistic_result = next(r for r in clf_results if r["model_type"] == "logistic")
    imp_df = extract_feature_importance(
        logistic_result["fitted_pipeline"], feature_cols, method="model"
    )
    atlas_df = map_to_atlas(imp_df, top_n=10)
    assert len(atlas_df) == 10

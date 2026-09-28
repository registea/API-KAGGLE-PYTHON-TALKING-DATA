import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.ml.evaluation import evaluate_binary_predictions
from training.job.utils.ml.fit_models import CustomPredictor, EstimatorFit


def generate_calibration_predictions(
    classifier_name: str,
    hyperparameters: dict[str, Any],
    monotone_constraints: dict[str, Any],
    seed: int,
    features: list[str],
    X: pd.DataFrame,
    y: np.ndarray,
    inner_splits: int,
    inner_test_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate time-based OOF predictions for fitting the calibrator.
    """

    inner_splitter = TimeSeriesSplit(
        n_splits=inner_splits,
        test_size=inner_test_size,
        gap=0,
    )

    calibration_probabilities = []
    calibration_targets = []

    for train_index, validation_index in inner_splitter.split(X, y):
        estimator = EstimatorFit(
            classifier_name=classifier_name,
            hyperparameters=hyperparameters,
            monotone_constraints=monotone_constraints,
            seed=seed,
            features=features,
        )

        estimator.fit(
            X=X.iloc[train_index],
            y=y[train_index],
        )

        probabilities = estimator.predict_proba(X=X.iloc[validation_index])

        calibration_probabilities.append(probabilities)
        calibration_targets.append(y[validation_index])

    return (
        np.concatenate(calibration_probabilities),
        np.concatenate(calibration_targets),
    )


def cross_validation(
    classifier_name: str,
    hyperparameters: dict[str, Any],
    monotone_constraints: dict[str, Any],
    seed: int,
    target_col: str,
    features: list[str],
    df_data: pd.DataFrame,
    logger: logging.Logger,
    inner_splits: int = 3,
    inner_test_size: int = 50_000,
) -> list[dict[str, Any]]:
    """ """

    # Extract model inputs
    X = df_data[features].copy()
    y = df_data[target_col].to_numpy()

    # Create outer splitter
    outer_splitter = TimeSeriesSplit(
        n_splits=4,
        test_size=100000,
        gap=0,
    )

    # Retain fold results for comparison and downstream model selection
    results = []

    # Split data over successive folds
    for fold, (train_index, test_index) in enumerate(
        outer_splitter.split(X, y), start=1
    ):
        # log fold
        logger.info(f"Cross validation fold: {fold}")

        # Capture outer fold data
        x_fold_train = X.iloc[train_index].copy()
        y_fold_train = y[train_index]
        x_fold_test = X.iloc[test_index].copy()
        y_fold_test = y[test_index]

        # Generate calibration data using only the outer training fold
        calibration_probabilities, calibration_y = (
            generate_calibration_predictions(
                classifier_name=classifier_name,
                hyperparameters=hyperparameters,
                monotone_constraints=monotone_constraints,
                seed=seed,
                features=features,
                X=x_fold_train,
                y=y_fold_train,
                inner_splits=inner_splits,
                inner_test_size=inner_test_size,
            )
        )

        # Fit outer model (base + calibrator)
        estimator_fold = CustomPredictor(
            classifier_name=classifier_name,
            hyperparameters=hyperparameters,
            monotone_constraints=monotone_constraints,
            seed=seed,
            features=features,
        )
        estimator_fold.fit(
            X=x_fold_train,
            y=y_fold_train,
            calibration_probabilities=calibration_probabilities,
            calibration_y=calibration_y,
        )

        # Generate raw and calibrated probabilities on unseen future observations
        raw_probabilities = estimator_fold.predict_raw_proba(x_fold_test)
        calibrated_probabilities = estimator_fold.predict_proba(x_fold_test)

        # Evaluate metrics
        raw_metrics = evaluate_binary_predictions(
            y_true=y_fold_test,
            y_probability=raw_probabilities,
        )
        calibrated_metrics = evaluate_binary_predictions(
            y_true=y_fold_test,
            y_probability=calibrated_probabilities,
        )

        # Trace
        results.append(
            {"fold": fold, "raw": raw_metrics, "calibrated": calibrated_metrics}
        )

        logger.info(
            "Raw - CV fold %s: prevalence=%.4f%%, AUROC=%.4f, AP=%.4f, "
            "log-loss=%.6f, Brier=%.6f",
            fold,
            100 * raw_metrics["prevalence"],
            raw_metrics["roc_auc"],
            raw_metrics["average_precision"],
            raw_metrics["log_loss"],
            raw_metrics["brier_score"],
        )
        for top_metrics in raw_metrics["top_fraction_metrics"]:
            logger.info(
                "CV fold %s top %.2f%%: precision=%.4f, recall=%.4f, "
                "lift=%.1fx (%s/%s positives)",
                fold,
                100 * top_metrics["fraction"],
                top_metrics["precision"],
                top_metrics["recall"],
                top_metrics["lift"],
                top_metrics["positives"],
                top_metrics["selected"],
            )

        logger.info(
            "Calibrated - CV fold %s: prevalence=%.4f%%, AUROC=%.4f, AP=%.4f, "
            "log-loss=%.6f, Brier=%.6f",
            fold,
            100 * calibrated_metrics["prevalence"],
            calibrated_metrics["roc_auc"],
            calibrated_metrics["average_precision"],
            calibrated_metrics["log_loss"],
            calibrated_metrics["brier_score"],
        )
        for top_metrics in calibrated_metrics["top_fraction_metrics"]:
            logger.info(
                "CV fold %s top %.2f%%: precision=%.4f, recall=%.4f, "
                "lift=%.1fx (%s/%s positives)",
                fold,
                100 * top_metrics["fraction"],
                top_metrics["precision"],
                top_metrics["recall"],
                top_metrics["lift"],
                top_metrics["positives"],
                top_metrics["selected"],
            )

    return results

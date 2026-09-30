"""
Fit a binary classifier and evaluate.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Any, Dict


# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.logging import setup_logger
from training.job.utils.ml.cross_validation import (
    cross_validation,
    generate_calibration_predictions,
)
from training.job.utils.ml.evaluation import evaluate_binary_predictions
from training.job.utils.ml.fit_models import CustomPredictor


def run(input_files: Dict[str, Path], output_dir: Path, config: dict) -> int:
    """
    Fit and evaluate a demonstration binary classifier.

    Select the first eligible numeric predictor and evaluate on a stratified holdout. Refit preprocessing and the
    model on all labelled rows before saving the model and evaluation metrics.

    :param input_files: Processed data across various splits.
    :param output_dir: Directory in which to save the model and metrics.
    :param config: Training settings including target, join key, split size and seed.

    :return: Path to the fitted model.joblib artifact.
    """

    # Create a custom logger
    logger = setup_logger()

    # ------------------------------------------------------------------------------------------------------------------
    # Load processed data

    # Provide an actionable failure when the data process node has not produced an input
    for split, input_file in input_files.items():
        if not input_file.is_file():
            raise FileNotFoundError(
                f"Processed data split not found: {split}. Run data_process "
                "first or pass --processed-data."
            )

    df_train = duckdb.read_csv(str(input_files["train"]))
    df_val = duckdb.read_csv(str(input_files["val"]))
    df_test = duckdb.read_csv(str(input_files["test"]))
    # df_full = pd.concat(
    #     [df_train.df(), df_val.df(), df_test.df()],
    #     axis=0,
    #     ignore_index=True,
    # )

    classifier_name = "xgb"
    target_col = "is_attributed"
    features = ["channel"]  # ["app", "device", "os", "channel"] ["channel"]
    seed = 42
    inner_splits = 4
    inner_test_size = 50000
    hyperparameters: dict[str, Any] = {}
    monotone_constraints: dict[str, Any] = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Cross validation

    cross_validation(
        classifier_name=classifier_name,
        hyperparameters=hyperparameters,
        monotone_constraints=monotone_constraints,
        target_col=target_col,
        features=features,
        seed=seed,
        df_data=df_train.df(),
        logger=logger,
        inner_splits=inner_splits,
        inner_test_size=inner_test_size,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Feature selection

    # ------------------------------------------------------------------------------------------------------------------
    # Hyperparameter tuning

    # ------------------------------------------------------------------------------------------------------------------
    # Test set evaluation

    # Combine train and validation data
    df_combined_train = pd.concat(
        [df_train.df(), df_val.df()],
        axis=0,
        ignore_index=True,
    )
    X_train = df_combined_train[features]
    y_train = df_combined_train[[target_col]].to_numpy()

    # Calculate OOF probabilities for calibration
    calibration_probabilities, calibration_y = generate_calibration_predictions(
        classifier_name=classifier_name,
        hyperparameters=hyperparameters,
        monotone_constraints=monotone_constraints,
        seed=seed,
        features=features,
        X=X_train,
        y=y_train,
        inner_splits=inner_splits,
        inner_test_size=inner_test_size,
    )

    # Create model which can be fit on the test set
    eval_model = CustomPredictor(
        seed=seed,
        classifier_name=classifier_name,
        hyperparameters=hyperparameters,
        monotone_constraints=monotone_constraints,
        features=features,
    )
    eval_model.fit(
        X=X_train,
        y=y_train,
        calibration_probabilities=calibration_probabilities,
        calibration_y=calibration_y,
    )

    # Generate inference
    probabilities = eval_model.predict_proba(df_test.df()[features])

    # Evaluate and log performance
    eval_metrics = evaluate_binary_predictions(
        y_true=df_test.df()[[target_col]],
        y_probability=probabilities,
    )
    logger.info(
        "Test Set: prevalence=%.4f%%, AUROC=%.4f, AP=%.4f, "
        "PR AUC=%.4f, "
        "log-loss=%.6f, Brier=%.6f",
        100 * eval_metrics["prevalence"],
        eval_metrics["roc_auc"],
        eval_metrics["average_precision"],
        eval_metrics["pr_auc"],
        eval_metrics["log_loss"],
        eval_metrics["brier_score"],
    )

    # # ------------------------------------------------------------------------------------------------------------------
    # # Full build

    # # All labelled competition data
    # X_full = df_full[features]
    # y_full = df_full[target_col].to_numpy()

    # # Honest temporal OOF probabilities for calibration
    # calibration_probabilities, calibration_y = (
    #     generate_calibration_predictions(
    #         X=X_full,
    #         y=y_full,
    #         classifier_name=classifier_name,
    #         hyperparameters=hyperparameters,
    #         monotone_constraints=monotone_constraints,
    #         seed=seed,
    #         features=features,
    #         inner_splits=inner_splits,
    #         inner_test_size=inner_test_size,
    #     )
    # )

    # # Base model is fitted on all labelled rows;
    # # calibrator is fitted on temporal OOF predictions
    # final_model = CustomPredictor(
    #     classifier_name=classifier_name,
    #     hyperparameters=hyperparameters,
    #     monotone_constraints=monotone_constraints,
    #     seed=seed,
    #     features=features,
    # )

    # final_model.fit(
    #     X=X_full,
    #     y=y_full,
    #     calibration_probabilities=calibration_probabilities,
    #     calibration_y=calibration_y,
    # )

    # # Kaggle's unlabelled test.csv
    # df_kaggle_test = pd.read_csv(test_path)

    # submission = pd.DataFrame(
    #     {
    #         "click_id": df_kaggle_test["click_id"],
    #         "is_attributed": final_model.predict_proba(
    #             X=df_kaggle_test[features]
    #         ),
    #     }
    # )

    # submission.to_csv("submission.csv", index=False)

    return 1

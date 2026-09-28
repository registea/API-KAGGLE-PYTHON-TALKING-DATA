"""
Fit a binary classifier and evaluate.
"""

import duckdb
from pathlib import Path
from typing import Dict


# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.logging import setup_logger
from training.job.utils.ml.cross_validation import cross_validation


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
    # df_val = duckdb.read_csv(str(input_files["val"]))
    # df_test = duckdb.read_csv(str(input_files["test"]))

    # ------------------------------------------------------------------------------------------------------------------
    # Cross validation

    cross_validation(
        classifier_name="xgb",
        hyperparameters={},
        monotone_constraints={},
        target_col="is_attributed",
        features=["channel"],
        seed=42,
        df_data=df_train.df(),
        logger=logger,
        inner_splits=4,
        inner_test_size=50000,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Feature selection

    # ------------------------------------------------------------------------------------------------------------------
    # Hyperparameter tuning

    # # ------------------------------------------------------------------------------------------------------------------
    # # Refit on all available labelled rows after held-out evaluation, then save

    # output_dir.mkdir(parents=True, exist_ok=True)
    # model_path = output_dir / "model.joblib"

    # # Keep holdout probabilities for metrics, then use all labelled rows for final scoring.
    # model.fit(x, y)
    # joblib.dump(model, model_path)

    # # Record enough context to interpret the demonstration validation scores
    # metrics = {
    #     "feature": feature,
    #     "target": target,
    #     "train_rows": len(x_train),
    #     "validation_rows": len(x_test),
    #     "roc_auc": float(roc_auc_score(y_test, probabilities)),
    #     "average_precision": float(
    #         average_precision_score(y_test, probabilities)
    #     ),
    #     "random_seed": config["random_seed"],
    #     "refit_rows": len(x),
    #     "note": (
    #         "One-feature demonstration with a random stratified split; "
    #         "not a competition-ready model."
    #     ),
    # }
    # (output_dir / "metrics.json").write_text(
    #     json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    # )
    # logger.info(
    #     "Fitted feature %s; ROC AUC %.4f; saved model to %s",
    #     feature,
    #     metrics["roc_auc"],
    #     model_path,
    # )

    return 1

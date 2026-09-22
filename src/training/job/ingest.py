"""
Join competition transaction and identity tables into a flat training file.
"""

from pathlib import Path

import duckdb

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality
from training.job.utils.logging import setup_logger


def run(
    input_dir: Path, output_dir: Path, config: dict
) -> Path:
    """
    Ingest data into the pipeline and save them for future use.

    :param input_dir: Directory containing the configured raw CSV files.
    :param output_dir: Directory in which to write training.csv.
    :param config: Training configuration containing file names, join key and target.
    :param max_rows: Maximum transaction rows to read; None loads all rows.

    :return: Path to the processed training CSV.
    """
    logger = setup_logger()

    # Capture data path
    train_path = input_dir / config["train_file"]

    # Load training data
    df_train = duckdb.read_csv(train_path)

    # Take the sample
    df_sample = df_train.query(
        "df_train",
        f"""
        SELECT *
        FROM df_train
        USING SAMPLE reservoir({config["sample_rows"]} ROWS) REPEATABLE ({config["random_seed"]})
        """,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Persist the stage output for this run or a later training job

    # Create the Kaggle working directory or its local equivalent when required
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.csv"
    df_sample.to_csv(str(output_path))
    logger.info(
        "Ingested training data with %s rows and %s columns to %s",
        len(df_sample),
        len(df_sample.columns),
        output_path,
    )
    return output_path

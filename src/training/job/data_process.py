"""
Join competition transaction and identity tables into a flat training file.
"""

import duckdb
from pathlib import Path
from typing import Dict

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.logging import setup_logger


def run(input_dir: Path, output_dir: Path, config: dict) -> Dict[str, Path]:
    """
    Join transaction and identity data and save the training table.

    Validate unique join keys before a left join so transactions without identity data remain available for training.

    :param input_dir: Directory containing the configured raw CSV files.
    :param output_dir: Directory in which to write training.csv.
    :param config: Training configuration containing file names, join key and target.

    :return: Path to the processed training CSV.
    """

    # Create a custom logger
    logger = setup_logger()

    # ------------------------------------------------------------------------------------------------------------------
    # Load ingested data

    # Provide an actionable failure when the ingestion node has not produced an input
    if not input_dir.is_file():
        raise FileNotFoundError(
            f"Ingested data not found: {input_dir}. Run data_process "
            "first or pass --processed-data."
        )

    # Load the ingested training table
    df_data = duckdb.read_csv(str(input_dir))

    logger.info(f"Loaded source data: {len(df_data)} rows.")

    # ------------------------------------------------------------------------------------------------------------------
    # Create training splits

    # Sort the data in time order
    df_sorted = df_data.order("click_time ASC")

    # Capture specific splits
    df_split = df_sorted.query(
        "df_sorted",
        """
        WITH ranked AS (
            SELECT
                *,
                row_number() OVER (
                    ORDER BY click_time
                ) AS split_row,
                count(*) OVER () AS total_rows
            FROM df_sorted
        )
        SELECT
            * EXCLUDE(split_row, total_rows),
            CASE
                WHEN split_row <= floor(total_rows * 0.7) THEN 'train'
                WHEN split_row <= floor(total_rows * 0.85) THEN 'validation'
                ELSE 'test'
            END AS dataset
        FROM ranked
        ORDER By split_row
        """,
    )

    # Training portion
    df_train = df_split.query(
        "df_split",
        """
        SELECT
            * EXCLUDE(dataset)
        FROM df_split
        WHERE "dataset" == 'train'
        """,
    )

    # Validation portion
    df_val = df_split.query(
        "df_split",
        """
        SELECT
            * EXCLUDE(dataset)
        FROM df_split
        WHERE "dataset" == 'validation'
        """,
    )

    # Testing portion
    df_test = df_split.query(
        "df_split",
        """
        SELECT
            * EXCLUDE(dataset)
        FROM df_split
        WHERE "dataset" == 'test'
        """,
    )

    # log row counts per split
    logger.info(f"Train data: {len(df_train)} rows.")
    logger.info(f"Validation data: {len(df_val)} rows.")
    logger.info(f"Test data: {len(df_test)} rows.")

    # Stratification check
    df_fraud_check = df_split.query(
        "df_split",
        """

        SELECT
            date_trunc('hour', click_time) AS time_bucket,
            count(*) AS observations,
            sum(is_attributed) AS positives,
            100.0 * avg(is_attributed) AS pct_attributed
        FROM df_split
        GROUP BY time_bucket
        HAVING count(*) >= 1000
        ORDER BY time_bucket
        """,
    ).df()
    df_fraud_check

    # ------------------------------------------------------------------------------------------------------------------
    # Exploration

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(12, 5))

    # ax.plot(
    #     df_fraud_check["time_bucket"],
    #     df_fraud_check["pct_attributed"],
    # )

    # ax.set(
    #     title="Attribution rate over time",
    #     xlabel="Time",
    #     ylabel="Attributed (%)",
    # )
    # fig.tight_layout()
    # fig.savefig(
    #     output_dir / "attribution_rate_over_time.png",
    #     dpi=150,
    # )
    # plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # Feature engineering

    # ------------------------------------------------------------------------------------------------------------------
    # Persist the stage output for this run or a later training job

    # Create the Kaggle working directory or its local equivalent when required
    output_dir.mkdir(parents=True, exist_ok=True)
    output_train_path = output_dir / "train_split.csv"
    output_val_path = output_dir / "val_split.csv"
    output_test_path = output_dir / "test_split.csv"
    df_train.to_csv(str(output_train_path))
    df_val.to_csv(str(output_val_path))
    df_test.to_csv(str(output_test_path))

    # ------------------------------------------------------------------------------------------------------------------
    # Save output for future runs

    # if config.get("publish_processed_dataset", False):
    #     import kagglehub

    #     # Training data
    #     kagglehub.dataset_upload(
    #         config["processed_split_train_handle"],
    #         str(output_dir),
    #         version_notes=f"Generated by the data_process node: {len(df_train)} rows.",
    #     )
    #     logger.info("Saved training split.")

    #     # Validation data
    #     kagglehub.dataset_upload(
    #         config["processed_split_val_handle"],
    #         str(output_dir),
    #         version_notes=f"Generated by the data_process node: {len(df_val)} rows.",
    #     )
    #     logger.info("Saved validation split.")

    #     # test data
    #     kagglehub.dataset_upload(
    #         config["processed_split_test_handle"],
    #         str(output_dir),
    #         version_notes=f"Generated by the data_process node: {len(df_test)} rows.",
    #     )
    #     logger.info("Saved test split.")

    return {
        "train": output_train_path,
        "val": output_val_path,
        "test": output_test_path,
    }

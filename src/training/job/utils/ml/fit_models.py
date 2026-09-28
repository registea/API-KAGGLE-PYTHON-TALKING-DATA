import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Any


# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.ml.algorithms import base_classifier


class EstimatorFit:
    def __init__(
        self,
        classifier_name: str,
        hyperparameters: dict[str, Any],
        monotone_constraints: dict[str, Any],
        seed: int,
        features: list[str],
    ):
        # Capture in instance
        self.classifier_name = classifier_name
        self.hyperparameters = hyperparameters
        self.monotone_constraints = monotone_constraints
        self.seed = seed
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        """
        Fit a chosen classifier on training data.
        """

        # Step 1: Extract features from X data
        x_train = X[self.features]

        # Step 2: Load a base classifier and fit
        self.classifier_ = base_classifier(
            classifier_name=self.classifier_name,
            hyperparameters=self.hyperparameters,
            monotone_constraints=self.monotone_constraints,
            seed=self.seed,
            features=self.features,
        )
        self.classifier_.fit(x_train, y)

        return self

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict the probability of fraud.
        """

        probabilities = self.classifier_.predict_proba(X[self.features])
        classes = self.classifier_.classes_

        if 1 not in classes:
            raise ValueError("Prediction contains no positive class 1.")

        positive_index = list(classes).index(1)
        return probabilities[:, positive_index]

    def predict(self, X: pd.DataFrame):
        """
        Capture the class representing the largest probability.
        """

        return self.classifier_.predict(X[self.features])


class ProbabilityCalibratorFit:
    def __init__(
        self,
        seed: int,
    ):
        # Save for instance
        self.seed = seed

    # Prepare probabilities
    def prepare_model_scores_(
        self,
        raw_probabilities: np.ndarray,
    ):
        """
        Clip and convert raw probabilities
        """

        # Clip raw probabilities
        clipped_probabilities = np.clip(
            np.asarray(raw_probabilities).reshape(-1), 1e-6, 1 - 1e-6
        )

        # Convert raw probabilities to a score
        return np.log(
            clipped_probabilities / (1 - clipped_probabilities)
        ).reshape(-1, 1)

    # Fit calibrator
    def fit(
        self,
        raw_probabilities: np.ndarray,
        y: pd.Series | np.ndarray,
    ):
        """
        Fit probability calibrator based on raw predictions
        """

        # Convert to scores
        scores = self.prepare_model_scores_(raw_probabilities=raw_probabilities)

        # Fit logistic regression calibrator
        self.calibrator_ = LogisticRegression(
            random_state=self.seed,
            max_iter=1000,
        )
        self.calibrator_.fit(scores, np.asarray(y).reshape(-1))

    def predict_proba(self, raw_probabilities: np.ndarray):
        """
        Generate calibrated probabilities.
        """

        # Convert to scores
        scores = self.prepare_model_scores_(raw_probabilities=raw_probabilities)

        # Return calibrated probabilities
        return self.calibrator_.predict_proba(scores)[:, 1]


class CustomPredictor:
    def __init__(
        self,
        seed: int,
        classifier_name: str,
        hyperparameters: dict[str, Any],
        monotone_constraints: dict[str, Any],
        features: list[str],
    ):
        # Store inputs
        self.seed = seed
        self.classifier_name = classifier_name
        self.hyperparameters = hyperparameters
        self.monotone_constraints = monotone_constraints
        self.features = features

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        calibration_probabilities: np.ndarray,
        calibration_y: pd.Series | np.ndarray,
    ):
        """
        Fit a custom set of ML models
        """

        # Fit base estimator
        self.base_estimator_ = EstimatorFit(
            classifier_name=self.classifier_name,
            hyperparameters=self.hyperparameters,
            monotone_constraints=self.monotone_constraints,
            seed=self.seed,
            features=self.features,
        )
        self.base_estimator_.fit(X=X, y=y)

        # Fit probability calibrator
        self.probability_calibrator_ = ProbabilityCalibratorFit(seed=self.seed)
        self.probability_calibrator_.fit(
            raw_probabilities=calibration_probabilities, y=calibration_y
        )

        # Place holder for other stuff

        return self

    def predict_raw_proba(self, X: pd.DataFrame):
        """
        Predict probability of positive class with base estimator
        """

        return self.base_estimator_.predict_proba(X=X)

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict the probability of fraud.
        """

        raw_probabilities = self.predict_raw_proba(X=X)

        return self.probability_calibrator_.predict_proba(
            raw_probabilities=raw_probabilities
        )

    def predict(self, X: pd.DataFrame, threshold: float = 0.5):
        """
        Predict classes by thresholding the calibrated probability.
        """

        probabilities = self.predict_proba(X=X)

        return (probabilities >= threshold).astype(int)

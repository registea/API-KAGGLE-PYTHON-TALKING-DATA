from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Any
from xgboost import XGBClassifier


def create_monotone_constraints(
    monotone_dict: dict[str, int], features: list[str]
):
    """
    Parse monotone constraints to a classifier..

    Args:
        monotone_dict: Directions of constraints per feature, 0 indicates no constraint.
        features: Feature names used to order constraints.

    Returns:
        List of ordered constraints.
    """

    # Valid directions
    valid_directions = {-1, 0, 1}

    # Defensive block on values provided
    invalid_directions = {
        feature: direction
        for feature, direction in monotone_dict.items()
        if direction not in valid_directions
    }
    if invalid_directions:
        raise ValueError(
            f"Monotonic constraints must be -1, 0, or 1: {invalid_directions}"
        )

    # Defensive block on features not stated
    unknown_features = set(monotone_dict) - set(features)
    if unknown_features:
        raise ValueError(
            f"Constraints reference unknown features: {sorted(unknown_features)}"
        )

    # On success, return mapping
    return [monotone_dict.get(feature, 0) for feature in features]


def base_classifier(
    classifier_name: str,
    seed: int,
    hyperparameters: dict[str, Any] | None = None,
    monotone_constraints: dict[str, Any] | None = None,
    features: list[str] | None = None,
) -> (
    LGBMClassifier | RandomForestClassifier | LogisticRegression | XGBClassifier
):
    """
    Create the configured classifier.

    Args:
        model_name: Name of the algorithm to use.
        seed: Seed for reproducibility.
        hyperparameters: hyperparameters passed to run the specific model.
        monotone_constraints: Constraint configuration for supported models.
        features: Feature names used to order XGBoost constraints.

    Returns:
        Configured classifier model.
    """

    # Defensive input protection
    hyperparameters = hyperparameters or {}
    monotone_constraints = monotone_constraints or {}
    features = features or []

    # Map the available algorithms
    model_mapping = {
        "xgb": XGBClassifier,
        "lgb": LGBMClassifier,
        "rf": RandomForestClassifier,
        "logistic": LogisticRegression,
    }

    # Select model based on the name
    classifier = model_mapping.get(classifier_name)
    if classifier is None:
        raise ValueError(
            f"Provided classifier name '{classifier_name}' "
            f"is not part of the valid list of models: {list(model_mapping)}"
            ""
        )

    # Configure algorithms
    enabled_models = monotone_constraints.get("MODELS", [])
    constraint_mapping = monotone_constraints.get("CONSTRAINTS", {})

    # Conditionally add monotone constraints
    if classifier_name in enabled_models and constraint_mapping:
        # Collect constraints
        constraints = create_monotone_constraints(
            monotone_dict=constraint_mapping,
            features=features,
        )

        # Xgboost syntax
        if classifier_name == "xgb":
            hyperparameters["monotone_constraints"] = (
                f"({','.join(map(str, constraints))})"
            )

        # lightgbm syntax
        if classifier_name == "lgb":
            hyperparameters["monotone_constraints"] = constraints

        # Random forest syntax
        if classifier_name == "rf":
            hyperparameters["monotonic_cst"] = constraints

    # Return configured classifier
    return classifier(random_state=seed, **hyperparameters)

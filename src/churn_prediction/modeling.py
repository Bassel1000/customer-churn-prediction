"""Model construction and chronological data splitting."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary_value",
    "unique_products",
    "active_months",
    "average_order_value",
    "average_items_per_order",
    "country",
]
NUMERIC_FEATURES = FEATURE_COLUMNS[:-1]
CATEGORICAL_FEATURES = ["country"]


def split_chronologically(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and final-test partitions by snapshot date."""
    snapshot_dates = sorted(pd.to_datetime(dataset["snapshot_date"]).unique())
    if len(snapshot_dates) < 3:
        raise ValueError("At least three snapshot dates are required for train/validation/test splits.")

    train_dates = snapshot_dates[:-2]
    validation_date = snapshot_dates[-2]
    test_date = snapshot_dates[-1]
    dates = pd.to_datetime(dataset["snapshot_date"])
    return (
        dataset.loc[dates.isin(train_dates)].copy(),
        dataset.loc[dates == validation_date].copy(),
        dataset.loc[dates == test_date].copy(),
    )


def feature_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs and normalize nullable string values for scikit-learn."""
    missing_columns = set(FEATURE_COLUMNS).difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing model features: {sorted(missing_columns)}")

    features = dataset[FEATURE_COLUMNS].copy()
    features["country"] = features["country"].astype("object")
    features["country"] = features["country"].where(features["country"].notna(), None)
    return features


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing shared by all candidate models."""
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_models(random_seed: int) -> dict[str, Pipeline]:
    """Return explainable baseline and nonlinear candidate classifiers."""
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=random_seed),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }

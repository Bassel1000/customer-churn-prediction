"""Tests for chronological model-data preparation."""

import pandas as pd

from churn_prediction.modeling import feature_matrix, split_chronologically


def test_split_chronologically_keeps_the_latest_snapshot_for_testing() -> None:
    dataset = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2011-06-01", "2011-07-01", "2011-08-01", "2011-09-01"]),
            "country": ["UK"] * 4,
            "recency_days": [1] * 4,
            "frequency": [1] * 4,
            "monetary_value": [1.0] * 4,
            "unique_products": [1] * 4,
            "active_months": [1] * 4,
            "average_order_value": [1.0] * 4,
            "average_items_per_order": [1.0] * 4,
        }
    )

    train, validation, test = split_chronologically(dataset)

    assert train["snapshot_date"].max() == pd.Timestamp("2011-07-01")
    assert validation["snapshot_date"].iloc[0] == pd.Timestamp("2011-08-01")
    assert test["snapshot_date"].iloc[0] == pd.Timestamp("2011-09-01")
    assert feature_matrix(test).columns.tolist()[-1] == "country"

"""Tests for leakage-safe customer features."""

import pandas as pd

from churn_prediction.features import build_customer_features


def test_features_use_only_the_history_before_the_snapshot() -> None:
    transactions = pd.DataFrame(
        {
            "CustomerID": [1, 1, 1],
            "InvoiceNo": ["A", "B", "FUTURE"],
            "StockCode": ["X", "Y", "Z"],
            "InvoiceDate": pd.to_datetime(["2011-01-05", "2011-01-20", "2011-02-02"]),
            "Quantity": [2, 3, 100],
            "LineTotal": [10.0, 30.0, 1000.0],
            "Country": ["UK", "UK", "UK"],
        }
    )

    features = build_customer_features(transactions, pd.Timestamp("2011-02-01"), 30)

    assert len(features) == 1
    assert features.loc[0, "frequency"] == 2
    assert features.loc[0, "monetary_value"] == 40.0
    assert features.loc[0, "recency_days"] == 12

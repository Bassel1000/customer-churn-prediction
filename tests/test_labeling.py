"""Tests for future-window churn labels."""

import pandas as pd

from churn_prediction.labeling import add_churn_labels


def test_labels_customers_without_future_orders_as_churned() -> None:
    features = pd.DataFrame({"CustomerID": [1, 2]})
    transactions = pd.DataFrame(
        {
            "CustomerID": [1, 2],
            "InvoiceDate": pd.to_datetime(["2011-02-15", "2011-05-02"]),
        }
    )

    labeled = add_churn_labels(features, transactions, pd.Timestamp("2011-02-01"), 90)

    assert labeled["churned"].tolist() == [0, 1]

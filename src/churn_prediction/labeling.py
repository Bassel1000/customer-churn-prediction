"""Time-safe churn-label generation."""

import pandas as pd


def add_churn_labels(
    customer_features: pd.DataFrame,
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    churn_window_days: int,
) -> pd.DataFrame:
    """Label active customers as churned when they do not buy in the future window."""
    future_end = snapshot_date + pd.Timedelta(days=churn_window_days)
    future_transactions = transactions.loc[
        (transactions["InvoiceDate"] >= snapshot_date)
        & (transactions["InvoiceDate"] < future_end)
    ]
    retained_customers = set(future_transactions["CustomerID"].unique())

    labeled = customer_features.copy()
    labeled["churned"] = (~labeled["CustomerID"].isin(retained_customers)).astype("int8")
    return labeled

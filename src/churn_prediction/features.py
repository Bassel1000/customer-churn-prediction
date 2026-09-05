"""Leakage-safe customer feature engineering."""

import pandas as pd


def build_customer_features(
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    history_window_days: int,
) -> pd.DataFrame:
    """Build one row per active customer using only data before ``snapshot_date``."""
    history_start = snapshot_date - pd.Timedelta(days=history_window_days)
    history = transactions.loc[
        (transactions["InvoiceDate"] >= history_start)
        & (transactions["InvoiceDate"] < snapshot_date)
    ].copy()

    if history.empty:
        return pd.DataFrame()

    customer_features = history.groupby("CustomerID").agg(
        recency_days=("InvoiceDate", lambda dates: (snapshot_date - dates.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary_value=("LineTotal", "sum"),
        unique_products=("StockCode", "nunique"),
        active_months=("InvoiceDate", lambda dates: dates.dt.to_period("M").nunique()),
    )

    order_totals = history.groupby(["CustomerID", "InvoiceNo"], as_index=False).agg(
        order_value=("LineTotal", "sum"),
        order_items=("Quantity", "sum"),
    )
    order_features = order_totals.groupby("CustomerID").agg(
        average_order_value=("order_value", "mean"),
        average_items_per_order=("order_items", "mean"),
    )

    latest_country = (
        history.sort_values("InvoiceDate")
        .groupby("CustomerID", as_index=True)["Country"]
        .last()
        .rename("country")
    )

    features = customer_features.join(order_features).join(latest_country).reset_index()
    features["snapshot_date"] = snapshot_date
    return features

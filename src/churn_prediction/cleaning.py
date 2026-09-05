"""Deterministic transaction cleaning rules."""

from collections.abc import Mapping

import pandas as pd

from churn_prediction.validation import validate_required_columns


def clean_transactions(transactions: pd.DataFrame) -> tuple[pd.DataFrame, Mapping[str, int]]:
    """Return valid purchase transactions and row counts for every cleaning step."""
    validate_required_columns(transactions)
    cleaned = transactions.copy()
    report: dict[str, int] = {"input_rows": len(cleaned)}

    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")
    # Excel infers mixed values for identifiers such as StockCode. Normalize all
    # categorical/text fields so Arrow can write a stable Parquet schema.
    for column in ("InvoiceNo", "StockCode", "Description", "Country"):
        cleaned[column] = cleaned[column].astype("string")

    cancellation_mask = cleaned["InvoiceNo"].astype("string").str.upper().str.startswith("C", na=False)
    report["cancelled_invoice_rows_removed"] = int(cancellation_mask.sum())
    cleaned = cleaned.loc[~cancellation_mask].copy()

    missing_customer_mask = cleaned["CustomerID"].isna()
    report["missing_customer_rows_removed"] = int(missing_customer_mask.sum())
    cleaned = cleaned.loc[~missing_customer_mask].copy()

    invalid_date_mask = cleaned["InvoiceDate"].isna()
    report["invalid_date_rows_removed"] = int(invalid_date_mask.sum())
    cleaned = cleaned.loc[~invalid_date_mask].copy()

    invalid_quantity_mask = cleaned["Quantity"].isna() | (cleaned["Quantity"] <= 0)
    report["non_positive_or_invalid_quantity_rows_removed"] = int(invalid_quantity_mask.sum())
    cleaned = cleaned.loc[~invalid_quantity_mask].copy()

    invalid_price_mask = cleaned["UnitPrice"].isna() | (cleaned["UnitPrice"] <= 0)
    report["non_positive_or_invalid_price_rows_removed"] = int(invalid_price_mask.sum())
    cleaned = cleaned.loc[~invalid_price_mask].copy()

    cleaned["CustomerID"] = cleaned["CustomerID"].astype("int64")
    cleaned["LineTotal"] = cleaned["Quantity"] * cleaned["UnitPrice"]
    cleaned = cleaned.sort_values("InvoiceDate").reset_index(drop=True)

    report["output_rows"] = len(cleaned)
    report["unique_customers"] = int(cleaned["CustomerID"].nunique())
    report["min_invoice_date"] = cleaned["InvoiceDate"].min().isoformat()
    report["max_invoice_date"] = cleaned["InvoiceDate"].max().isoformat()
    return cleaned, report

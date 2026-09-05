"""Schema checks for the UCI Online Retail source data."""

import pandas as pd

REQUIRED_COLUMNS = frozenset(
    {
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    }
)


def validate_required_columns(transactions: pd.DataFrame) -> None:
    """Raise a clear error when the source schema is incompatible."""
    missing_columns = REQUIRED_COLUMNS.difference(transactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Source data is missing required columns: {missing}")

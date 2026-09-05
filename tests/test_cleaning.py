"""Tests for the transaction-cleaning rules."""

import pandas as pd

from churn_prediction.cleaning import clean_transactions


def test_clean_transactions_removes_invalid_rows_and_adds_line_total() -> None:
    transactions = pd.DataFrame(
        {
            "InvoiceNo": ["10001", "C10002", "10003", "10004", "10005"],
            "StockCode": ["A", "B", "C", "D", "E"],
            "Description": ["One", "Two", "Three", "Four", "Five"],
            "Quantity": [2, 2, -1, 1, 3],
            "InvoiceDate": ["2011-01-01", "2011-01-01", "2011-01-02", "invalid", "2011-01-03"],
            "UnitPrice": [4.5, 4.5, 4.5, 4.5, 0],
            "CustomerID": [12345, 12345, 12345, 12345, None],
            "Country": ["UK", "UK", "UK", "UK", "UK"],
        }
    )

    cleaned, report = clean_transactions(transactions)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "LineTotal"] == 9.0
    assert report["cancelled_invoice_rows_removed"] == 1
    assert report["missing_customer_rows_removed"] == 1


def test_clean_transactions_rejects_missing_columns() -> None:
    transactions = pd.DataFrame({"InvoiceNo": ["10001"]})

    try:
        clean_transactions(transactions)
    except ValueError as error:
        assert "CustomerID" in str(error)
    else:
        raise AssertionError("Expected a missing-column validation error.")


def test_clean_transactions_normalizes_mixed_stock_codes_to_strings() -> None:
    transactions = pd.DataFrame(
        {
            "InvoiceNo": [10001, "10002"],
            "StockCode": [12345, "POST"],
            "Description": ["Product", "Postage"],
            "Quantity": [1, 1],
            "InvoiceDate": ["2011-01-01", "2011-01-02"],
            "UnitPrice": [1.0, 2.0],
            "CustomerID": [12345, 12346],
            "Country": ["UK", "UK"],
        }
    )

    cleaned, _ = clean_transactions(transactions)

    assert str(cleaned["StockCode"].dtype) == "string"
    assert cleaned["StockCode"].tolist() == ["12345", "POST"]

"""Functions for reading immutable source data."""

from pathlib import Path

import pandas as pd


def load_transactions(path: Path) -> pd.DataFrame:
    """Load the UCI Online Retail workbook without changing the source file."""
    if not path.is_file():
        raise FileNotFoundError(f"Source data file was not found: {path}")

    return pd.read_excel(path)

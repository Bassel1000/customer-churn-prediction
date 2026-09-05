"""Build a time-safe customer churn training dataset from clean transactions."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn_prediction.features import build_customer_features
from churn_prediction.labeling import add_churn_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "clean_transactions.parquet",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "modeling.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "customer_churn_dataset.parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "training_dataset_report.json",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    transactions = pd.read_parquet(args.input)
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])

    history_window_days = int(config["history_window_days"])
    churn_window_days = int(config["churn_window_days"])
    snapshot_dates = [pd.Timestamp(date) for date in config["snapshot_dates"]]

    latest_required_date = max(snapshot_dates) + pd.Timedelta(days=churn_window_days)
    if transactions["InvoiceDate"].max() < latest_required_date:
        raise ValueError(
            "The cleaned data does not cover the full future churn window for the "
            f"latest snapshot. Required through {latest_required_date.date()}."
        )

    datasets = []
    for snapshot_date in snapshot_dates:
        features = build_customer_features(transactions, snapshot_date, history_window_days)
        datasets.append(add_churn_labels(features, transactions, snapshot_date, churn_window_days))

    training_dataset = pd.concat(datasets, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    training_dataset.to_parquet(args.output, index=False)

    report = {
        "rows": len(training_dataset),
        "unique_customers": int(training_dataset["CustomerID"].nunique()),
        "churn_rate": float(training_dataset["churned"].mean()),
        "snapshot_dates": [date.date().isoformat() for date in snapshot_dates],
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report['rows']:,} customer snapshots to {args.output}")
    print(f"Wrote training-dataset report to {args.report}")


if __name__ == "__main__":
    main()

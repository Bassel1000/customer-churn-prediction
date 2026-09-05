"""Create validated, cleaned transaction data from the raw UCI workbook."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn_prediction.cleaning import clean_transactions
from churn_prediction.data_loading import load_transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx",
        help="Path to the immutable raw Excel workbook.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "clean_transactions.parquet",
        help="Path for cleaned transaction data.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "data_quality_report.json",
        help="Path for the generated data-quality report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transactions = load_transactions(args.input)
    cleaned, report = clean_transactions(transactions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {report['output_rows']:,} cleaned rows to {args.output}")
    print(f"Wrote data-quality report to {args.report}")


if __name__ == "__main__":
    main()

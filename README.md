# Customer Churn Prediction from Retail Transactions

An end-to-end command-line machine-learning project using UCI Online Retail transactions. It converts raw invoices into customer snapshots and predicts whether a customer will make no purchase in the following 90 days.

## Design

- Raw source: `data/raw/Online Retail.xlsx`, treated as immutable.
- Features: 180-day purchase history: recency, frequency, monetary value, product variety, active months, order value, order size, and country.
- Target: `churned = 1` when an active customer makes no purchase during the next 90 days.
- Time safety: features end before each snapshot; labels use only later purchases; splits are chronological.
- Selection: Logistic Regression and Random Forest are compared on August 2011 PR-AUC, then the winner is evaluated on September.

## Setup and run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_project.py
python -m pytest
python -m ruff check .
```

You can also run each stage separately:

```powershell
python scripts/run_pipeline.py
python scripts/build_training_dataset.py
python scripts/train_model.py
```

## Outputs

```text
data/interim/clean_transactions.parquet
data/processed/customer_churn_dataset.parquet
artifacts/models/churn_model.joblib
artifacts/metrics/model_metrics.json
artifacts/metrics/holdout_predictions.csv
artifacts/figures/*.png
artifacts/reports/*.json
artifacts/reports/model_card.md
```

Generated data, models, metrics, and figures are intentionally excluded from Git and can be recreated with the commands above.

## Layout

```text
configs/                 horizons and snapshot dates
scripts/                 pipeline entry points
src/churn_prediction/    reusable cleaning, features, labels, models, metrics, charts
tests/                   business-rule and time-split tests
data/raw/                source workbook only
data/interim/            generated cleaned transactions
data/processed/          generated customer-level training data
artifacts/               generated reports, figures, metrics, and model
```

## Limitations

This predicts a lack of future purchases in this historical dataset; it does not prove a customer has permanently left. Use it to prioritize retention outreach, not to automate adverse decisions, and monitor performance before use with another retailer or time period.

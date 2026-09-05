"""Metrics and report generation for churn classifiers."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Calculate classification metrics from churn probabilities."""
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": threshold,
        "churn_rate": float(y_true.mean()),
    }
    return metrics


def write_model_card(path: Path, model_name: str, metrics: dict[str, float], dataset_rows: int) -> None:
    """Write a concise, reproducible statement of model purpose and limits."""
    content = f"""# Customer Churn Model Card

## Purpose
Prioritize retail customers likely to make no purchase during the next 90 days.

## Model
- Selected model: `{model_name}`
- Training rows (including validation period): {dataset_rows:,}
- Final evaluation: chronological September 2011 holdout

## Final-holdout metrics
- ROC-AUC: {metrics['roc_auc']:.3f}
- PR-AUC: {metrics['pr_auc']:.3f}
- Precision at 0.50: {metrics['precision']:.3f}
- Recall at 0.50: {metrics['recall']:.3f}
- F1 at 0.50: {metrics['f1']:.3f}

## Limitations
This is a historical transaction model, not evidence that a customer has permanently left. It should support retention prioritization, not automated adverse decisions. Performance should be monitored after deployment and before use with a different retailer or period.
"""
    path.write_text(content, encoding="utf-8")

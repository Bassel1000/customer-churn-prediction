"""Save evaluation figures as artifacts; never open interactive windows."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def save_evaluation_figures(
    y_true: pd.Series, probabilities: np.ndarray, output_directory: Path
) -> None:
    """Create ROC, precision-recall, confusion-matrix, and class-balance figures."""
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= 0.5).astype(int)

    fig, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=axis)
    axis.set_title("ROC Curve — Final Holdout")
    fig.tight_layout()
    fig.savefig(output_directory / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=axis)
    axis.set_title("Precision-Recall Curve — Final Holdout")
    fig.tight_layout()
    fig.savefig(output_directory / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, ax=axis, colorbar=False)
    axis.set_title("Confusion Matrix at 0.50 Threshold")
    fig.tight_layout()
    fig.savefig(output_directory / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    counts = y_true.value_counts().reindex([0, 1], fill_value=0)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.bar(["Retained", "Churned"], counts.values, color=["#4C78A8", "#E45756"])
    axis.set_title("Final-Holdout Class Distribution")
    axis.set_ylabel("Customer snapshots")
    fig.tight_layout()
    fig.savefig(output_directory / "class_distribution.png", dpi=150)
    plt.close(fig)


def save_feature_importance(model: object, output_directory: Path) -> None:
    """Save top absolute feature importances for the selected pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        importances = classifier.feature_importances_

    importance_table = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_table = importance_table.nlargest(15, "importance").sort_values("importance")
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(importance_table["feature"], importance_table["importance"], color="#4C78A8")
    axis.set_title("Top Feature Importances")
    axis.set_xlabel("Absolute coefficient / feature importance")
    fig.tight_layout()
    fig.savefig(output_directory / "feature_importance.png", dpi=150)
    plt.close(fig)

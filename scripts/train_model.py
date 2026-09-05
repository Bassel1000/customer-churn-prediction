"""Train, select, and evaluate churn models with chronological splits."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn_prediction.evaluation import calculate_metrics, write_model_card
from churn_prediction.modeling import (
    build_candidate_models,
    feature_matrix,
    split_chronologically,
)
from churn_prediction.visualization import (
    save_evaluation_figures,
    save_feature_importance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "processed" / "customer_churn_dataset.parquet")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "modeling.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pd.read_parquet(args.input)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train, validation, test = split_chronologically(dataset)

    validation_metrics: dict[str, dict[str, float]] = {}
    candidates = build_candidate_models(int(config["random_seed"]))
    for name, model in candidates.items():
        model.fit(feature_matrix(train), train["churned"])
        probabilities = model.predict_proba(feature_matrix(validation))[:, 1]
        validation_metrics[name] = calculate_metrics(validation["churned"], probabilities)

    selected_name = max(validation_metrics, key=lambda name: validation_metrics[name]["pr_auc"])
    selected_model = candidates[selected_name]
    train_and_validation = pd.concat([train, validation], ignore_index=True)
    selected_model.fit(feature_matrix(train_and_validation), train_and_validation["churned"])

    test_probabilities = selected_model.predict_proba(feature_matrix(test))[:, 1]
    test_metrics = calculate_metrics(test["churned"], test_probabilities)
    predictions = test[["CustomerID", "snapshot_date", "churned"]].copy()
    predictions["churn_probability"] = test_probabilities
    predictions["predicted_churn"] = (test_probabilities >= 0.5).astype(int)
    predictions = predictions.sort_values("churn_probability", ascending=False)

    model_directory = PROJECT_ROOT / "artifacts" / "models"
    metrics_directory = PROJECT_ROOT / "artifacts" / "metrics"
    figure_directory = PROJECT_ROOT / "artifacts" / "figures"
    report_directory = PROJECT_ROOT / "artifacts" / "reports"
    for directory in (model_directory, metrics_directory, figure_directory, report_directory):
        directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected_model, model_directory / "churn_model.joblib")
    predictions.to_csv(metrics_directory / "holdout_predictions.csv", index=False)
    metrics = {
        "selected_model": selected_name,
        "validation_metrics": validation_metrics,
        "final_holdout_metrics": test_metrics,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
    }
    (metrics_directory / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_evaluation_figures(test["churned"], test_probabilities, figure_directory)
    save_feature_importance(selected_model, figure_directory)
    write_model_card(report_directory / "model_card.md", selected_name, test_metrics, len(train_and_validation))

    print(f"Selected {selected_name} using validation PR-AUC.")
    print(f"Final holdout PR-AUC: {test_metrics['pr_auc']:.3f}")
    print(f"Saved model and evaluation artifacts under {PROJECT_ROOT / 'artifacts'}")


if __name__ == "__main__":
    main()

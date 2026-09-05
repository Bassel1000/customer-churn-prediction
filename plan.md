# Intelligent Customer Churn Prediction Platform — Plan

## Goal
Identify customers at risk of cancelling and give the retention team a prioritised, explainable intervention list.

## Data
- Customer profile, subscription, product usage, support, billing, and cancellation history.
- Define a prediction horizon (for example, churn in the next 30 days) and document the churn label.
- Keep raw data immutable; validate schema, unique customer IDs, dates, and missingness before processing.

## Build plan
1. Create the repository structure from the blueprint and configuration for paths, target, and random seed.
2. Produce reproducible EDA: target distribution, tenure/usage/support patterns, and segment-level churn.
3. Build leakage-safe features: tenure, recency, engagement trends, billing status, payment failures, and support-contact counts.
4. Train a logistic-regression baseline, then compare Random Forest and gradient-boosted models using a time-aware holdout if dates exist.
5. Evaluate ROC-AUC, PR-AUC, recall and precision at the retention team's capacity; calibrate the selected model if needed.
6. Generate SHAP global and individual explanations, a ranked high-risk list, and segment-level churn report.
7. Serve predictions through a small FastAPI endpoint or Streamlit dashboard.
8. Add unit tests, a reproducible command-line pipeline, containerisation, and deployment instructions.

## Success criteria
- Better PR-AUC and recall-at-capacity than the logistic baseline.
- Every high-risk prediction includes top contributing factors.
- Retention list can be reproduced from versioned input and model artefacts.

## Deliverables
- Figures: target distribution, cohort/segment churn, feature importance, ROC/PR curves, confusion matrix, SHAP summary.
- Metrics report with threshold rationale and estimated retention value assumptions.
- API/dashboard, tests, README, Dockerfile, and model card describing limitations.

## Risks and safeguards
- Prevent leakage from post-cancellation events and future usage.
- Check performance across key customer segments; do not use sensitive attributes for unfair treatment.
- Track prediction drift and intervention outcomes after deployment.

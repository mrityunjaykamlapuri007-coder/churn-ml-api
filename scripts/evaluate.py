#!/usr/bin/env python
"""
scripts/evaluate.py — Standalone evaluation script
Run evaluation on saved model against test data without retraining.
Usage:
    python scripts/evaluate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import COLUMNS_PKL, PIPELINE_PKL
from src.data import clean_data, load_raw_data, split_data
from src.features import add_features


def main():
    print("Loading saved model and running evaluation ...\n")

    # Load saved artifacts
    pipeline = joblib.load(PIPELINE_PKL)
    columns = joblib.load(COLUMNS_PKL)

    # Prepare data (same preprocessing as training)
    df = load_raw_data()
    X, y = clean_data(df)
    X = add_features(X)
    _, X_test, _, y_test = split_data(X, y)

    # Align columns
    X_test = X_test.reindex(columns=columns, fill_value=0)

    # Predict
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS (Saved Pipeline)")
    print("=" * 50)
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print("\n" + str(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])))


if __name__ == "__main__":
    main()

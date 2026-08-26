#!/usr/bin/env python
"""
scripts/train.py — Training CLI Script
Usage:
    python scripts/train.py
    python scripts/train.py --config configs/model.yaml
"""
import os
import sys

import joblib
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import UTC

from src.config import COLUMNS_PKL, MODEL_DIR, MODEL_PKL, PIPELINE_PKL
from src.data import clean_data, load_raw_data, split_data
from src.features import add_features
from src.logger import get_logger
from src.models import evaluate_models, save_plots, train_all_models

logger = get_logger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("  CHURN MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: ETL Extract ──
    logger.info("Step 1/7: Loading raw data ...")
    df = load_raw_data()

    # ── Step 2: ETL Transform ──
    logger.info("Step 2/7: Cleaning data ...")
    X, y = clean_data(df)

    # ── Step 3: Feature Engineering ──
    logger.info("Step 3/7: Engineering features ...")
    X = add_features(X)

    # ── Step 4: Split ──
    logger.info("Step 4/7: Splitting data ...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Step 5: Training ──
    logger.info("Step 5/7: Training models ...")
    models = train_all_models(X_train, y_train)

    # ── Step 6: Evaluation ──
    logger.info("Step 6/7: Evaluating models ...")
    df_results = evaluate_models(models, X_test, y_test)
    logger.info("\n" + df_results.to_string(index=False))

    best_rf = models.get("Random Forest")
    save_plots(models, X_train, X_test, y_test, df_results, best_rf=best_rf)

    # ── Step 7: Save Artifacts ──
    logger.info("Step 7/7: Saving model artifacts ...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_rf, MODEL_PKL)
    logger.info(f"Saved -> {MODEL_PKL}")

    joblib.dump(list(X_train.columns), COLUMNS_PKL)
    logger.info(f"Saved -> {COLUMNS_PKL}")

    pipeline = Pipeline([("model", best_rf)])
    joblib.dump(pipeline, PIPELINE_PKL)
    logger.info(f"Saved -> {PIPELINE_PKL}")

    # ── Step 7b: Save Model Metadata ──
    import json
    from datetime import datetime

    from src.config import METADATA_JSON

    # Extract metrics for the chosen model
    rf_row = df_results[df_results["Model"] == "Random Forest"]
    metrics = {}
    if not rf_row.empty:
        metrics = {
            "accuracy": round(float(rf_row["Accuracy"].iloc[0]), 4),
            "precision": round(float(rf_row["Precision"].iloc[0]), 4),
            "recall": round(float(rf_row["Recall"].iloc[0]), 4),
            "f1_score": round(float(rf_row["F1"].iloc[0]), 4),
            "roc_auc": round(float(rf_row["ROC-AUC"].iloc[0]), 4),
        }

    metadata = {
        "version": "1.0.0",
        "algorithm": "RandomForestClassifier",
        "trained_at": datetime.now(UTC).isoformat(),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(X_train.columns),
        "metrics": metrics,
        "hyperparameters": best_rf.get_params() if best_rf else {},
    }

    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved -> {METADATA_JSON}")

    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

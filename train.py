#!/usr/bin/env python
"""
train.py — Main Orchestrator
Runs the full ML pipeline: ETL → Feature Engineering → Training → Evaluation → Save.
Usage:  python train.py
"""
import os
import joblib

from src.config import MODEL_DIR, MODEL_PKL, PIPELINE_PKL, COLUMNS_PKL
from src.data import load_raw_data, clean_data, split_data
from src.features import add_features
from src.models import train_all_models, evaluate_models, save_plots

from sklearn.pipeline import Pipeline


def main():
    # ── 1. ETL: Extract ──
    df = load_raw_data()

    # ── 2. ETL: Transform & Clean ──
    X, y = clean_data(df)

    # ── 3. Feature Engineering ──
    X = add_features(X)

    # ── 4. ETL: Split ──
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── 5. Model Training ──
    models = train_all_models(X_train, y_train)

    # ── 6. Evaluation ──
    df_results = evaluate_models(models, X_test, y_test)
    print("\n" + df_results.to_string(index=False))

    best_rf = models.get("Random Forest")
    save_plots(models, X_train, X_test, y_test, df_results, best_rf=best_rf)

    # ── 7. Save Artifacts ──
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_rf, MODEL_PKL)
    print(f"\nSaved {MODEL_PKL}")

    joblib.dump(list(X_train.columns), COLUMNS_PKL)
    print(f"Saved {COLUMNS_PKL}")

    pipeline = Pipeline([("model", best_rf)])
    joblib.dump(pipeline, PIPELINE_PKL)
    print(f"Saved {PIPELINE_PKL}")

    print("\nPipeline complete! All artifacts saved to model/")


if __name__ == "__main__":
    main()

"""
Model Evaluation
Computes metrics and generates comparison plots.
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import PLOTS_DIR


def evaluate_models(
    models: dict, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Evaluate all models and return a results DataFrame.

    Args:
        models: dict of {name: fitted_estimator}
        X_test: Test features
        y_test: Test labels

    Returns:
        DataFrame with Accuracy, Precision, Recall, F1, ROC-AUC per model.
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
        }
        rows.append(row)

        print(f"\n{'='*40}")
        print(f"  {name}")
        print(f"{'='*40}")
        for k, v in row.items():
            if k != "Model":
                print(f"  {k:12s}: {v:.4f}")

    return pd.DataFrame(rows)


def save_plots(
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_results: pd.DataFrame,
    best_rf=None,
) -> None:
    """Generate and save all evaluation plots to disk.

    Args:
        models: dict of {name: fitted_estimator}
        X_train: Training features (for feature importance)
        X_test: Test features
        y_test: Test labels
        df_results: Metrics DataFrame from evaluate_models()
        best_rf: The best Random Forest model (for feature importance)
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── 1. Bar Comparison ──
    df_results.set_index("Model").plot(kind="bar", figsize=(12, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # ── 2. Feature Importance (RF) ──
    if best_rf is not None:
        fi = pd.DataFrame({
            "feature": X_train.columns,
            "importance": best_rf.feature_importances_,
        }).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi.head(15), x="importance", y="feature")
        plt.title("Top 15 Features Influencing Churn")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "feature_importance.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")

    # ── 3. Confusion Matrices ──
    plt.figure(figsize=(12, 10))
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.subplot(2, 2, i + 1)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(name)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # ── 4. ROC Curves ──
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # ── 5. Precision-Recall Curves ──
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(rec, prec, label=name)
    plt.title("Precision-Recall Curve Comparison")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "precision_recall_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

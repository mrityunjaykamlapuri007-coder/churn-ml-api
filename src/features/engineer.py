"""
Feature Engineering
Creates derived features and applies one-hot encoding.
"""
import pandas as pd
from src.config import SERVICE_INDICATOR_COLS

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features and one-hot encode categoricals.

    Steps:
        1. Create avg_monthly_spend, is_new_customer, high_monthly_charge
        2. One-hot encode all categorical (object) columns
        3. Create long_term_contract and total_services_used from dummies

    Args:
        X: Raw feature DataFrame (before encoding).

    Returns:
        Transformed DataFrame with new features and dummies.
    """
    X = X.copy()

    # Derived numeric features
    X["avg_monthly_spend"] = X["TotalCharges"] / (X["tenure"] + 1)
    X["is_new_customer"] = (X["tenure"] < 12).astype(int)
    X["high_monthly_charge"] = (
        X["MonthlyCharges"] > X["MonthlyCharges"].median()
    ).astype(int)
    print("[FEATURES] Added: avg_monthly_spend, is_new_customer, high_monthly_charge")

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    print(f"[FEATURES] One-hot encoded -> {X.shape[1]} columns")

    # Post-encoding derived features
    if "Contract_One year" in X.columns and "Contract_Two year" in X.columns:
        X["long_term_contract"] = X["Contract_One year"] + X["Contract_Two year"]
        print("[FEATURES] Added: long_term_contract")

    service_cols = [c for c in SERVICE_INDICATOR_COLS if c in X.columns]
    if service_cols:
        X["total_services_used"] = X[service_cols].sum(axis=1)  # type: ignore
        print(f"[FEATURES] Added: total_services_used (from {len(service_cols)} services)")

    print(f"[FEATURES] Final shape: {X.shape}")
    return X

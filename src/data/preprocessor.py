"""
ETL — Transform & Load
Handles data cleaning, type coercion, encoding, and train/test splitting.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TARGET_COL, DROP_COLS, NUMERIC_COERCE_COLS,
    TARGET_MAP, TEST_SIZE, RANDOM_STATE,
)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw data and return features (X) and target (y).

    Steps:
        1. Drop ID columns
        2. Coerce numeric columns
        3. Drop rows with NaN
        4. Map target to binary integers

    Returns:
        (X, y) — feature DataFrame and target Series.
    """
    df = df.copy()

    # Drop irrelevant columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    print(f"[ETL] Dropped columns: {DROP_COLS}")

    # Coerce types
    for col in NUMERIC_COERCE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[ETL] Coerced to numeric: {NUMERIC_COERCE_COLS}")

    # Handle missing values
    n_before = len(df)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)
    print(f"[ETL] Dropped {n_dropped} rows with missing values")

    # Encode target
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAP)
    print(f"[ETL] Target '{TARGET_COL}' mapped: {TARGET_MAP}")

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[ETL] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

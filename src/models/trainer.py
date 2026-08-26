"""
Model Training
Trains LR, Random Forest, XGBoost, and Stacking classifiers.
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE,
    RF_PARAM_GRID, RF_SEARCH_ITER, RF_SEARCH_CV, RF_SCORING,
    XGB_PARAM_GRID, XGB_SEARCH_ITER, XGB_SEARCH_CV, XGB_SCORING,
)


def _train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression model."""
    print("\n>>> Training Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    return lr


def _train_random_forest(X_train, y_train):
    """Train a Random Forest with RandomizedSearchCV."""
    print("\n>>> Training Random Forest (RandomizedSearchCV) ...")
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        param_distributions=RF_PARAM_GRID,
        n_iter=RF_SEARCH_ITER,
        scoring=RF_SCORING,
        cv=RF_SEARCH_CV,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_search.fit(X_train, y_train)
    print(f"    Best params: {rf_search.best_params_}")
    return rf_search.best_estimator_


def _train_xgboost(X_train, y_train):
    """Train an XGBoost model with RandomizedSearchCV."""
    print("\n>>> Training XGBoost (RandomizedSearchCV) ...")
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    xgb_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=XGB_PARAM_GRID,
        n_iter=XGB_SEARCH_ITER,
        scoring=XGB_SCORING,
        cv=XGB_SEARCH_CV,
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    xgb_search.fit(X_train, y_train)
    print(f"    Best params: {xgb_search.best_params_}")
    return xgb_search.best_estimator_


def _train_stacking(lr, best_rf, best_xgb, X_train, y_train):
    """Train a Stacking Classifier combining LR, RF, and XGBoost."""
    print("\n>>> Training Stacking Classifier ...")
    stack = StackingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", best_rf),
            ("xgb", best_xgb),
        ],
        final_estimator=LogisticRegression(),
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    return stack


def train_all_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    """Train all models and return them in a dict.

    Returns:
        dict mapping model name -> fitted estimator.
    """
    lr = _train_logistic_regression(X_train, y_train)
    best_rf = _train_random_forest(X_train, y_train)
    best_xgb = _train_xgboost(X_train, y_train)
    stack = _train_stacking(lr, best_rf, best_xgb, X_train, y_train)

    models = {
        "Logistic Regression": lr,
        "Random Forest": best_rf,
        "XGBoost": best_xgb,
        "Stacking": stack,
    }
    print(f"\nTrained {len(models)} models successfully.")
    return models

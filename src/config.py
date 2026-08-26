"""
Central configuration for the Churn ML Pipeline.
All paths, hyperparameters, and constants are defined here.
"""
import os

# ── Paths ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

RAW_DATA_FILE = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── Model Artifacts ────────────────────────────
MODEL_PKL = os.path.join(MODEL_DIR, "churn_model.pkl")
PIPELINE_PKL = os.path.join(MODEL_DIR, "full_pipeline.pkl")
COLUMNS_PKL = os.path.join(MODEL_DIR, "model_columns.pkl")

# ── Data Config ────────────────────────────────
TARGET_COL = "Churn"
DROP_COLS = ["customerID"]
NUMERIC_COERCE_COLS = ["TotalCharges"]
TARGET_MAP = {"Yes": 1, "No": 0}

# ── Train/Test Split ──────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Hyperparameter Search ─────────────────────
RF_PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [6, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
RF_SEARCH_ITER = 10
RF_SEARCH_CV = 3
RF_SCORING = "recall"

XGB_PARAM_GRID = {
    "n_estimators": [300, 500, 700],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.7, 0.85, 1],
    "colsample_bytree": [0.7, 0.85, 1],
    "gamma": [0, 0.1, 0.3],
    "min_child_weight": [1, 3, 5],
}
XGB_SEARCH_ITER = 15
XGB_SEARCH_CV = 3
XGB_SCORING = "roc_auc"

# ── Feature Engineering ───────────────────────
SERVICE_INDICATOR_COLS = [
    "PhoneService_Yes",
    "MultipleLines_Yes",
    "OnlineSecurity_Yes",
    "OnlineBackup_Yes",
    "DeviceProtection_Yes",
    "TechSupport_Yes",
    "StreamingTV_Yes",
    "StreamingMovies_Yes",
]

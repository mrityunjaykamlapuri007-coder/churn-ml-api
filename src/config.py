"""
Central configuration for the Churn ML Pipeline.
All paths, hyperparameters, and constants are defined here.
"""
import os
import yaml

# ── Paths ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")

RAW_DATA_FILE = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── Load Model Config ──────────────────────────
with open(os.path.join(CONFIGS_DIR, "model.yaml"), "r") as f:
    model_config = yaml.safe_load(f)

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
RF_PARAM_GRID = model_config["random_forest"]["param_grid"]
# YAML parses 'null' as None, but we need to ensure it's handled correctly
if None in RF_PARAM_GRID.get("max_depth", []):
    pass # YAML handles this natively

RF_SEARCH_ITER = model_config["random_forest"]["search"]["n_iter"]
RF_SEARCH_CV = model_config["random_forest"]["search"]["cv"]
RF_SCORING = model_config["random_forest"]["search"]["scoring"]

XGB_PARAM_GRID = model_config["xgboost"]["param_grid"]
XGB_SEARCH_ITER = model_config["xgboost"]["search"]["n_iter"]
XGB_SEARCH_CV = model_config["xgboost"]["search"]["cv"]
XGB_SCORING = model_config["xgboost"]["search"]["scoring"]

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

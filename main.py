"""
main.py — FastAPI Churn Prediction API
Production-grade API with rate limiting, health checks, model metadata,
config-driven CORS, and structured logging.
"""
import json
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import COLUMNS_PKL, METADATA_JSON, PIPELINE_PKL, app_config
from src.logger import get_logger

logger = get_logger(__name__)

# ── 1. Rate Limiter ──
rate_limit_config = app_config.get("rate_limit", {})
limiter = Limiter(key_func=get_remote_address, default_limits=[rate_limit_config.get("default", "60/minute")])

# ── 2. FastAPI App ──
api_config = app_config.get("api", {})
app = FastAPI(
    title=api_config.get("title", "Customer Churn Prediction API"),
    description="Predict customer churn probability using a trained ML pipeline.",
    version=api_config.get("version", "1.0.0"),
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# ── 3. CORS Middleware (Config-Driven) ──
cors_config = app_config.get("cors", {})
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config.get("allowed_origins", ["*"]),
    allow_credentials=cors_config.get("allow_credentials", True),
    allow_methods=cors_config.get("allow_methods", ["*"]),
    allow_headers=cors_config.get("allow_headers", ["*"]),
)

# ── 4. Global Exception Handler ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )

# ── 5. Load Artifacts ──
_model_ready = False
_model_metadata = {}

try:
    logger.info("Loading model artifacts...")
    pipeline = joblib.load(PIPELINE_PKL)
    columns = joblib.load(COLUMNS_PKL)
    _model_ready = True

    # Load metadata if available
    try:
        with open(METADATA_JSON) as f:
            _model_metadata = json.load(f)
        logger.info(f"Model metadata loaded: v{_model_metadata.get('version', 'unknown')}")
    except FileNotFoundError:
        _model_metadata = {"version": api_config.get("version", "1.0.0"), "note": "metadata not found, using defaults"}
        logger.warning("Model metadata file not found; using defaults.")

    logger.info("Model artifacts loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    pipeline = None
    columns = None


# ── 6. Request Schema ──
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ── 7. Endpoints ──
@app.get("/")
@limiter.limit(rate_limit_config.get("default", "60/minute"))
def home(request: Request):
    logger.info("Root endpoint called.")
    return {
        "service": "Customer Churn Prediction API",
        "version": api_config.get("version", "1.0.0"),
        "status": "running",
    }


@app.get("/health")
@limiter.limit(rate_limit_config.get("default", "60/minute"))
def health(request: Request):
    """Dedicated health check endpoint with model readiness status."""
    logger.info("Health check endpoint called.")
    return {
        "status": "healthy" if _model_ready else "degraded",
        "model_loaded": _model_ready,
        "model_version": _model_metadata.get("version", "unknown"),
        "model_trained_at": _model_metadata.get("trained_at", "unknown"),
        "model_algorithm": _model_metadata.get("algorithm", "unknown"),
        "model_metrics": _model_metadata.get("metrics", {}),
        "api_version": api_config.get("version", "1.0.0"),
    }


@app.post("/predict")
@limiter.limit(rate_limit_config.get("predict", "30/minute"))
def predict(data: CustomerData, request: Request):
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is degraded. Please try again later.",
        )

    logger.info(f"Prediction requested for customer with tenure {data.tenure} and contract {data.Contract}.")
    start_time = time.time()

    try:
        input_df = pd.DataFrame([data.model_dump()])

        # Apply feature engineering (prevents training-serving skew)
        from src.features.engineer import add_features
        input_df = add_features(input_df, is_train=False)

        # Align columns with the model's expected input
        input_df = input_df.reindex(columns=columns, fill_value=0)

        assert pipeline is not None  # Guarded by _model_ready check above
        prob = pipeline.predict_proba(input_df)[0][1]

        # Risk classification from config
        risk_thresholds = app_config.get("risk_thresholds", {})
        if prob > risk_thresholds.get("high", {}).get("min_probability", 0.7):
            risk = "High"
            action = risk_thresholds.get("high", {}).get("action", "Immediate retention call")
        elif prob > risk_thresholds.get("medium", {}).get("min_probability", 0.5):
            risk = "Medium"
            action = risk_thresholds.get("medium", {}).get("action", "Offer discount")
        else:
            risk = "Low"
            action = risk_thresholds.get("low", {}).get("action", "No action needed")

        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Prediction successful: Probability={prob:.2f}, Risk={risk}, Latency={latency_ms}ms")

        return {
            "churn_probability": round(float(prob), 4),
            "risk_level": risk,
            "recommended_action": action,
            "model_version": _model_metadata.get("version", "unknown"),
            "latency_ms": latency_ms,
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing prediction.")

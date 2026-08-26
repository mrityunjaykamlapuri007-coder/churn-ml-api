"""
main.py — FastAPI Churn Prediction API
Loads the saved pipeline and serves predictions via REST endpoints.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd

from src.config import PIPELINE_PKL, COLUMNS_PKL
from src.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using a trained ML pipeline.",
    version="1.0.0",
)

# ── 1. CORS Middleware (Essential for Production) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In strict prod, replace "*" with specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. Global Exception Handler ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )

# ── 3. Load Artifacts ──
try:
    logger.info("Loading model artifacts...")
    pipeline = joblib.load(PIPELINE_PKL)
    columns = joblib.load(COLUMNS_PKL)
    logger.info("Model artifacts loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    raise RuntimeError("Could not load model artifacts. Did you run train.py?")


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


@app.get("/")
def home():
    logger.info("Health check endpoint called.")
    return {"message": "Churn prediction API running", "status": "healthy"}


@app.post("/predict")
def predict(data: CustomerData):
    logger.info(f"Prediction requested for customer with tenure {data.tenure} and contract {data.Contract}.")
    
    try:
        input_df = pd.DataFrame([data.model_dump()])
        input_df = input_df.reindex(columns=columns, fill_value=0)

        prob = pipeline.predict_proba(input_df)[0][1]

        if prob > 0.7:
            risk = "High"
            action = "Immediate retention call"
        elif prob > 0.5:
            risk = "Medium"
            action = "Offer discount"
        else:
            risk = "Low"
            action = "No action needed"

        logger.info(f"Prediction successful: Probability={prob:.2f}, Risk={risk}")
        
        return {
            "churn_probability": float(prob),
            "risk_level": risk,
            "recommended_action": action,
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing prediction.")

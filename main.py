from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

pipeline = joblib.load("model/full_pipeline.pkl")
columns = joblib.load("model/model_columns.pkl")


from pydantic import BaseModel

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
    return {"message": "Churn prediction API running"}

@app.post("/predict")
def predict(data: CustomerData):

    input_df = pd.DataFrame([data.dict()])
    
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

    return {
        "churn_probability": float(prob),
        "risk_level": risk,
        "recommended_action": action
    }

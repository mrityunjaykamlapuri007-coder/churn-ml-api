"""
tests/test_api.py — Integration tests for the FastAPI endpoints
Run: pytest tests/test_api.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_service_info(self):
        response = client.get("/")
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_model_status(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]

    def test_health_returns_model_metadata(self):
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert "model_version" in data
        assert "api_version" in data

    def test_health_returns_metrics(self):
        response = client.get("/health")
        data = response.json()
        assert "model_metrics" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    SAMPLE_CUSTOMER = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }

    def test_predict_returns_200(self):
        response = client.post("/predict", json=self.SAMPLE_CUSTOMER)
        assert response.status_code == 200

    def test_predict_returns_probability(self):
        response = client.post("/predict", json=self.SAMPLE_CUSTOMER)
        data = response.json()
        assert "churn_probability" in data
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_returns_risk_level(self):
        response = client.post("/predict", json=self.SAMPLE_CUSTOMER)
        data = response.json()
        assert data["risk_level"] in ["Low", "Medium", "High"]

    def test_predict_returns_action(self):
        response = client.post("/predict", json=self.SAMPLE_CUSTOMER)
        data = response.json()
        assert "recommended_action" in data

    def test_predict_returns_model_version(self):
        """Prediction response should include model version."""
        response = client.post("/predict", json=self.SAMPLE_CUSTOMER)
        data = response.json()
        assert "model_version" in data

    def test_predict_returns_latency(self):
        """Prediction response should include latency metric."""
        response = client.post("/predict", json=self.SAMPLE_CUSTOMER)
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_predict_invalid_payload(self):
        """Missing fields should return 422."""
        response = client.post("/predict", json={"gender": "Male"})
        assert response.status_code == 422

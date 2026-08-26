# API Reference

## Base URL

- **Local**: `http://localhost:8000`
- **Production**: `https://churn-ml-api-6cir.onrender.com`

---

## `GET /`

Health check endpoint.

**Response**:
```json
{
  "message": "Churn prediction API running"
}
```

---

## `POST /predict`

Predict customer churn probability.

### Request Body

| Field            | Type    | Example              | Description                     |
|-----------------|---------|----------------------|---------------------------------|
| gender          | string  | "Male"               | Male / Female                   |
| SeniorCitizen   | int     | 0                    | 0 = No, 1 = Yes                |
| Partner         | string  | "Yes"                | Has partner                     |
| Dependents      | string  | "No"                 | Has dependents                  |
| tenure          | int     | 12                   | Months with company             |
| PhoneService    | string  | "Yes"                | Has phone service               |
| MultipleLines   | string  | "No"                 | Has multiple lines              |
| InternetService | string  | "Fiber optic"        | DSL / Fiber optic / No          |
| OnlineSecurity  | string  | "No"                 | Has online security             |
| OnlineBackup    | string  | "No"                 | Has online backup               |
| DeviceProtection| string  | "No"                 | Has device protection           |
| TechSupport     | string  | "No"                 | Has tech support                |
| StreamingTV     | string  | "No"                 | Has streaming TV                |
| StreamingMovies | string  | "No"                 | Has streaming movies            |
| Contract        | string  | "Month-to-month"     | Contract type                   |
| PaperlessBilling| string  | "Yes"                | Uses paperless billing          |
| PaymentMethod   | string  | "Electronic check"   | Payment method                  |
| MonthlyCharges  | float   | 70.0                 | Monthly charge amount           |
| TotalCharges    | float   | 840.0                | Total charges to date           |

### Response

```json
{
  "churn_probability": 0.73,
  "risk_level": "High",
  "recommended_action": "Immediate retention call"
}
```

### Risk Levels

| Probability | Risk Level | Action                    |
|-------------|-----------|---------------------------|
| > 0.7       | High      | Immediate retention call  |
| > 0.5       | Medium    | Offer discount            |
| ≤ 0.5       | Low       | No action needed          |

### Error Responses

| Code | Description                  |
|------|------------------------------|
| 422  | Validation error (missing/invalid fields) |
| 500  | Internal server error        |

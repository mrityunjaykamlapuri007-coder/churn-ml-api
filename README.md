# Customer Churn Prediction API

A production-grade machine learning pipeline and REST API for predicting customer churn. Built with a modular, configuration-driven architecture, this project serves as a reference template for scalable ML deployments.

## Architecture & Tooling

- **Dependency Management**: `uv` (lightning-fast, deterministic builds via `uv.lock`)
- **Core ML**: `scikit-learn`, `xgboost`, `pandas`
- **API Framework**: `FastAPI` (with CORS, structured logging, and global exception handling)
- **Frontend**: `Streamlit`
- **CI/CD**: GitHub Actions (automated linting, testing, and Docker builds)
- **Code Quality**: `ruff`, `pytest`, pre-commit hooks

## Project Structure

```text
churn-ml-api/
├── .github/workflows/ci.yaml    # CI/CD pipeline definition
├── configs/                     # Configuration files (YAML)
│   ├── data.yaml                # Data paths and split configurations
│   ├── features.yaml            # Feature engineering settings
│   ├── model.yaml               # Hyperparameters and search spaces
│   └── app.yaml                 # API and deployment settings
├── data/                        # Raw data directory (gitignored)
├── model/                       # Saved model artifacts (.pkl)
├── reports/                     # Generated evaluation plots and metrics
├── src/                         # Core source code
│   ├── config.py                # Central Python configuration loader
│   ├── logger.py                # Centralized structured logging
│   ├── data/                    # ETL layer (loader, preprocessor)
│   ├── features/                # Feature engineering layer
│   └── models/                  # Model training and evaluation layer
├── scripts/                     # Internal scripts
├── tests/                       # Automated test suite
├── main.py                      # FastAPI application entry point
├── app.py                       # Streamlit frontend entry point
├── Dockerfile                   # Container definition
├── pyproject.toml               # Project metadata and dependencies
└── uv.lock                      # Deterministic dependency lockfile
```

## Quick Start

### 1. Installation

This project uses `uv` for dependency management. Ensure `uv` is installed on your system.

```bash
git clone https://github.com/mrityunjaykamlapuri007-coder/churn-ml-api.git
cd churn-ml-api
uv sync
```

### 2. Download Dataset

The project relies on the Telco Customer Churn dataset from Kaggle.

```bash
uv run churn-data
```

### 3. Train the Model

Execute the full ETL, feature engineering, and model training pipeline. The best model will be automatically saved to the `model/` directory.

```bash
uv run churn-train
```

### 4. Start the API

Launch the FastAPI backend server.

```bash
uv run churn-serve
```
The API documentation will be available at `http://localhost:8000/docs`.

### 5. Start the Web Interface

Launch the Streamlit frontend.

```bash
uv run streamlit run app.py
```

## Model Performance

The training pipeline evaluates multiple algorithms using `RandomizedSearchCV`. The current baseline metrics on the test set (1,407 records) are as follows:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.64 | 0.54 | 0.59 | 0.84 |
| Random Forest (Chosen) | 0.73 | 0.49 | 0.80 | 0.61 | 0.84 |
| XGBoost | 0.80 | 0.66 | 0.53 | 0.59 | 0.84 |
| Stacking Classifier | 0.79 | 0.64 | 0.54 | 0.58 | 0.84 |

**Note:** The Random Forest model is selected as the primary artifact due to its superior Recall (80%). In churn prediction, minimizing false negatives (failing to identify a churning customer) is the primary business objective.

## Testing and Quality Assurance

Run the automated test suite:

```bash
uv run pytest tests/ -v
```

Run the code linter:

```bash
uv run ruff check .
```

## Docker Deployment

The application is fully containerized and optimized for production environments.

```bash
docker build -t churn-ml-api .
docker run -d -p 8000:8000 churn-ml-api
```

## License

MIT License

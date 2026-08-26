# Customer Churn Prediction -- ML Pipeline + API

A **production-grade** machine learning project for predicting customer churn using the Telco Customer Churn dataset. Built with a modular, config-driven architecture that serves as a reference template for structuring any ML project.

---

## 🏗️ Project Structure

```
churn-ml-api/
│
├── .github/workflows/ci.yaml    # CI/CD — lint, test, Docker build
├── .githooks/pre-commit          # Local git hooks for code quality
├── .pre-commit-config.yaml       # Pre-commit framework config
│
├── configs/                      # ⚙️ Configuration (YAML)
│   ├── data.yaml                 #   Data paths, columns, splits
│   ├── features.yaml             #   Feature engineering settings
│   ├── model.yaml                #   Hyperparameters & search spaces
│   └── app.yaml                  #   API & deployment settings
│
├── data/                         # 📦 Raw data (gitignored)
├── model/                        # 💾 Saved .pkl model artifacts
├── reports/                      # 📊 Generated plots & reports
├── notebooks/                    # 📓 Jupyter notebooks (EDA)
│
├── src/                          # 🧠 Core source code
│   ├── config.py                 #   Central Python config
│   ├── data/                     #   ETL layer
│   │   ├── loader.py             #     Extract — load CSV
│   │   └── preprocessor.py       #     Transform — clean & split
│   ├── features/                 #   Feature layer
│   │   └── engineer.py           #     Derived features + encoding
│   └── models/                   #   Model layer
│       ├── trainer.py            #     Train all models
│       └── evaluator.py          #     Metrics + plots
│
├── scripts/                      # 🔧 CLI entry points
│   ├── download_data.py          #   Download dataset from Kaggle
│   ├── train.py                  #   Full training pipeline
│   ├── evaluate.py               #   Evaluate saved model
│   └── serve.py                  #   Start API server
│
├── tests/                        # ✅ Automated tests
│   ├── test_data.py              #   ETL tests
│   ├── test_features.py          #   Feature tests
│   └── test_api.py               #   API integration tests
│
├── docs/                         # 📖 Documentation
│   ├── architecture.md           #   System architecture overview
│   └── api.md                    #   API endpoint reference
│
├── main.py                       # FastAPI application
├── app.py                        # Streamlit frontend
├── train.py                      # Quick-run orchestrator
├── Dockerfile                    # Container definition
├── Makefile                      # Common commands
├── pyproject.toml                # Python project metadata
├── requirements.txt              # Pip dependencies
├── AGENTS.md                     # AI agent instructions
├── CHANGELOG.md                  # Release history
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/churn-ml-api.git
cd churn-ml-api
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

### 2. Download Data

```bash
python scripts/download_data.py
```

### 3. Train Models

```bash
python scripts/train.py
```

### 4. Start API

```bash
python scripts/serve.py
# API docs → http://localhost:8000/docs
```

### 5. Start Frontend

```bash
streamlit run app.py
```

---

## 📊 Models

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---------------------|----------|-----------|--------|------|---------|
| Logistic Regression | 0.80     | 0.64      | 0.54   | 0.59 | 0.84    |
| Random Forest       | 0.73     | 0.49      | 0.80   | 0.61 | 0.84    |
| XGBoost             | 0.80     | 0.66      | 0.53   | 0.59 | 0.84    |
| Stacking            | 0.79     | 0.64      | 0.54   | 0.58 | 0.84    |

> **Random Forest** is used as the primary model (highest recall for churn detection).

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 🐳 Docker

```bash
docker build -t churn-ml-api .
docker run -p 8000:8000 churn-ml-api
```

---

## 📁 Key Design Principles

| Principle | How It's Applied |
|-----------|-----------------|
| **Config-driven** | All hyperparams in `configs/*.yaml`, not hardcoded |
| **Modular ETL** | Separate Extract → Transform → Load stages |
| **Testable** | Each layer has dedicated unit tests |
| **CI/CD ready** | GitHub Actions pipeline with lint + test + Docker |
| **Documented** | Architecture docs, API reference, inline comments |
| **Reproducible** | Fixed random seeds, pinned dependencies |

---

## 📜 License

MIT

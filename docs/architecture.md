# Architecture Overview

## Project Structure

```
churn-ml-api/
│
├── .github/                    # CI/CD pipelines
│   └── workflows/
│       └── ci.yaml             # Lint, test, build on every push & PR
│
├── configs/                    # Configuration files (YAML)
│   ├── data.yaml               # Data source, columns, split config
│   ├── features.yaml           # Feature engineering config
│   ├── model.yaml              # Hyperparameters & search spaces
│   └── app.yaml                # API, Streamlit & deployment config
│
├── data/                       # Raw data (not checked into git)
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── docs/                       # Documentation
│   ├── architecture.md         # This file — system overview
│   └── api.md                  # API endpoint reference
│
├── model/                      # Saved model artifacts (.pkl)
│   ├── churn_model.pkl
│   ├── full_pipeline.pkl
│   └── model_columns.pkl
│
├── notebooks/                  # Jupyter notebooks (EDA, experiments)
│   └── customer-churn.ipynb
│
├── reports/                    # Generated evaluation reports
│   └── plots/                  # Saved comparison plots
│
├── scripts/                    # CLI entry points
│   ├── download_data.py        # Download dataset from Kaggle
│   ├── train.py                # Run full training pipeline
│   ├── evaluate.py             # Evaluate saved model
│   └── serve.py                # Start FastAPI server
│
├── src/                        # Core source code
│   ├── config.py               # Central config (paths, constants)
│   ├── data/                   # ETL layer
│   │   ├── loader.py           #   Extract — load CSV
│   │   └── preprocessor.py     #   Transform — clean, split
│   ├── features/               # Feature layer
│   │   └── engineer.py         #   Derived features + encoding
│   └── models/                 # Model layer
│       ├── trainer.py          #   Train LR, RF, XGB, Stacking
│       └── evaluator.py        #   Metrics + plot generation
│
├── tests/                      # Automated tests
│   ├── test_data.py            # ETL tests
│   ├── test_features.py        # Feature engineering tests
│   └── test_api.py             # API integration tests
│
├── main.py                     # FastAPI application
├── app.py                      # Streamlit frontend
├── train.py                    # Quick-run training orchestrator
├── Dockerfile                  # Container definition
├── Makefile                    # Common commands shortcut
├── pyproject.toml              # Python project metadata
├── requirements.txt            # Pip dependencies
├── CHANGELOG.md                # Release history
├── AGENTS.md                   # AI agent instructions
└── README.md                   # Project overview
```

## Data Flow

```
┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Kaggle  │──▶│  loader.py   │──▶│ preprocessor  │──▶│  engineer.py │
│ CSV Data │    │  (Extract)   │    │  (Transform)  │    │  (Features)  │
└──────────┘    └──────────────┘    └───────────────┘    └──────┬───────┘
                                                                │
                    ┌──────────────┐    ┌───────────────┐       │
                    │ evaluator.py │◀──│  trainer.py   │◀──────┘
                    │  (Metrics)   │    │   (Models)    │
                    └──────┬───────┘    └───────┬───────┘
                           │                    │
                    ┌──────▼───────┐    ┌───────▼──────┐
                    │   reports/   │    │    model/    │
                    │   (Plots)    │    │   (.pkl)     │
                    └──────────────┘    └───────┬──────┘
                                                │
                                        ┌───────▼──────┐
                                        │   main.py    │
                                        │  (FastAPI)   │
                                        └───────┬──────┘
                                                │
                                        ┌───────▼──────┐
                                        │    app.py    │
                                        │ (Streamlit)  │
                                        └──────────────┘
```

## Models Trained

| Model               | Purpose                              |
|---------------------|--------------------------------------|
| Logistic Regression | Baseline, interpretable              |
| Random Forest       | Primary model (best recall)          |
| XGBoost             | Gradient boosting alternative        |
| Stacking            | Ensemble of all three above          |

## Key Design Decisions

1. **Config-driven**: All hyperparameters live in `configs/` YAML files, not hardcoded
2. **Modular ETL**: Separate extract/transform layers for testability
3. **Feature engineering post-encoding**: One-hot encoding before creating contract/service aggregates
4. **Pipeline serialization**: Save sklearn Pipeline (not raw model) for consistent inference
5. **Column alignment**: `reindex()` at inference to handle missing/extra features gracefully

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] - 2026-08-26

### Added
- Dedicated `/health` endpoint with model readiness status, version, and metrics
- Model metadata (`metadata.json`) saved during training with version, timestamp, hyperparameters, and test metrics
- API rate limiting via `slowapi` (60 req/min default, 30 req/min for `/predict`)
- Prediction response latency tracking (`latency_ms` field)
- Model version included in prediction responses

### Changed
- CORS origins now config-driven from `configs/app.yaml` (no more wildcard `*`)
- Risk thresholds now read from config instead of hardcoded
- Root `/` endpoint now returns structured service info (name, version, status)
- API gracefully degrades (503) if model fails to load instead of crashing

### Infrastructure
- Added `slowapi>=0.1.9` to production dependencies
- Extended test suite with `/health` endpoint tests and new response field validation

## [1.0.0] - 2026-08-26

### Added
- Full ML training pipeline with modular `src/` architecture
- ETL pipeline: `src/data/loader.py` (Extract), `src/data/preprocessor.py` (Transform)
- Feature engineering: `src/features/engineer.py`
- Model training: Logistic Regression, Random Forest, XGBoost, Stacking Classifier
- Model evaluation with metrics and 5 comparison plots
- FastAPI prediction API (`main.py`)
- Streamlit frontend (`app.py`)
- Config-driven architecture using YAML files in `configs/`
- CLI scripts: `train.py`, `serve.py`, `evaluate.py`, `download_data.py`
- Unit and integration tests in `tests/`
- GitHub Actions CI/CD pipeline
- Docker support with `Dockerfile` and `compose.yaml`
- Pre-commit hooks for code quality
- Full documentation in `docs/`

### Infrastructure
- `pyproject.toml` for modern Python packaging
- `Makefile` with common project commands
- `.githooks/pre-commit` for local linting
- `.pre-commit-config.yaml` for pre-commit framework

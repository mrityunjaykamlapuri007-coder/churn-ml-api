# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

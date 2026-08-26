# AGENTS.md

Instructions for AI agents working on this codebase.

## Project Overview
Customer Churn Prediction ML pipeline with FastAPI serving and Streamlit frontend.

## Key Conventions
- All config values live in `configs/*.yaml` — never hardcode paths or hyperparameters
- Core source code goes in `src/` with clear subpackages: `data/`, `features/`, `models/`
- CLI scripts go in `scripts/` — each should be runnable standalone
- Tests go in `tests/` — mirror the `src/` structure
- Model artifacts are saved to `model/` directory
- Plots and reports are saved to `reports/` directory

## Running the Project
1. `python scripts/download_data.py` — Download the dataset
2. `python scripts/train.py` — Train all models
3. `python scripts/serve.py` — Start the API
4. `streamlit run app.py` — Start the frontend
5. `pytest tests/ -v` — Run tests

## Code Style
- Use Ruff for linting (`ruff check .`)
- Follow PEP 8 with 100-char line length
- Type hints encouraged but not mandatory
- Docstrings for all public functions

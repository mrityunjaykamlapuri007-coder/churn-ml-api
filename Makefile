# ──────────────────────────────────────────────────────────────
# Makefile — Common project commands
# Usage: make <target>
# ──────────────────────────────────────────────────────────────

.PHONY: help install data train evaluate serve test lint docker clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	uv sync

data:  ## Download the dataset from Kaggle
	python scripts/download_data.py

train:  ## Run the full training pipeline
	python scripts/train.py

evaluate:  ## Evaluate the saved model
	python scripts/evaluate.py

serve:  ## Start the FastAPI server
	python scripts/serve.py --reload

streamlit:  ## Start the Streamlit frontend
	streamlit run app.py

test:  ## Run all tests
	pytest tests/ -v

lint:  ## Lint code with Ruff
	ruff check src/ scripts/ tests/ main.py app.py

docker:  ## Build Docker image
	docker build -t churn-ml-api .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 churn-ml-api

clean:  ## Remove generated files
	rm -rf plots/ __pycache__ src/__pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

all: data train test  ## Full pipeline: download → train → test

# GridPulse developer tasks.  Run `make help` for the list.
.DEFAULT_GOAL := help
.PHONY: help setup probe ingest build quality train anomalies export all \
        test lint format api app dagster dbt docker clean

PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install everything
	$(PY) -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements-dev.txt
	./.venv/bin/pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
	./.venv/bin/pip install -e . --no-deps
	@echo "Now: cp .env.example .env and add your API keys."

probe:      ## Validate API keys and response contracts
	gridpulse probe
ingest:     ## Extract EIA and weather into bronze
	gridpulse ingest
build:      ## Build the silver and gold warehouse layers
	gridpulse build
quality:    ## Run the data quality suite
	gridpulse quality
train:      ## Train and evaluate every forecasting model
	gridpulse train
anomalies:  ## Fit and score the anomaly detectors
	gridpulse anomalies
export:     ## Write the deployment artifact for the public app
	gridpulse export
all:        ## Run the entire pipeline end to end
	gridpulse all

test:    ## Run the test suite with coverage
	pytest -v --cov=gridpulse --cov-report=term-missing
lint:    ## Lint with ruff
	ruff check src tests orchestration app.py
format:  ## Auto-format with ruff
	ruff format src tests app.py
	ruff check --fix src tests orchestration app.py

api:      ## Serve the FastAPI app on :8000
	uvicorn gridpulse.api.main:app --reload --port 8000
app:      ## Serve the Streamlit dashboard on :8501
	streamlit run app.py
dagster:  ## Open the Dagster UI on :3000
	dagster dev -f orchestration/dagster_app/definitions.py
dbt:      ## Build and test the dbt marts
	cd dbt/gridpulse && dbt deps --profiles-dir . && dbt build --profiles-dir .

docker:  ## Build and run the local Docker stack
	docker compose up --build

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf dbt/gridpulse/target dbt/gridpulse/logs

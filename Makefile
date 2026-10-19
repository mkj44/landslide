.PHONY: help install test clean data detection susceptibility warning all

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest

help:
	@echo "Landslide Detection, Prediction & Warning Pipeline"
	@echo "--------------------------------------------------"
	@echo "make install         - Install pinned dependencies"
	@echo "make test            - Run unit and integration tests"
	@echo "make data            - Download and inspect raw datasets"
	@echo "make preprocess      - Run feature engineering and preprocessing"
	@echo "make detection       - Train Stage A detection models"
	@echo "make susceptibility  - Train Stage B susceptibility & risk models"
	@echo "make warning         - Run Stage C early-warning engine"
	@echo "make evaluate        - Compute full evaluation metrics and figures"
	@echo "make all             - Run full pipeline from data to evaluation"
	@echo "make clean           - Remove caches, temporary outputs, and bytecode"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) tests/ -v

data:
	$(PYTHON) scripts/download_data.py

preprocess:
	$(PYTHON) scripts/preprocess.py

detection:
	$(PYTHON) scripts/train_detection.py

susceptibility:
	$(PYTHON) scripts/train_susceptibility.py

warning:
	$(PYTHON) scripts/warn.py --batch data/processed/stage_b_features.csv --map

evaluate:
	$(PYTHON) scripts/evaluate.py

all: data preprocess detection susceptibility warning evaluate
	@echo "Pipeline execution completed successfully."

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	rm -rf .pytest_cache .coverage htmlcov

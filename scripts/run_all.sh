#!/usr/bin/env bash
set -e
echo "Running Landslide Detection, Prediction & Warning End-to-End Pipeline..."
python scripts/download_data.py
python scripts/preprocess.py
python scripts/train_detection.py
python scripts/train_susceptibility.py
python scripts/generate_alerts.py
python scripts/evaluate.py
echo "End-to-End Pipeline Completed."

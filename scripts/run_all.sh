#!/usr/bin/env bash
# End-to-end master execution pipeline for Landslide Detection, Prediction & Warning System

set -euo pipefail

echo "=========================================================================="
echo "LANDSLIDE MULTI-STAGE ML PIPELINE (STAGES A, B, C): FULL EXECUTION"
echo "=========================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

PYTHON="${ROOT_DIR}/.venv/bin/python"
PYTEST="${ROOT_DIR}/.venv/bin/pytest"

# 1. Verification of unit test suite
echo ""
echo "[Step 1/5] Running automated pytest suite across all phases..."
"$PYTEST" tests/ -v

# 2. Stage A Detection Benchmark
echo ""
echo "[Step 2/5] Running Stage A Sentinel-2 multi-spectral detection benchmark..."
"$PYTHON" scripts/train_detection.py

# 3. Stage B Susceptibility & Spatial Cross-Validation Benchmark
echo ""
echo "[Step 3/5] Running Stage B 5-fold Spatial Block CV & SHAP interpretability..."
"$PYTHON" scripts/train_susceptibility.py

# 4. Stage C Early Warning Regional Assessment & Static Risk Mapping
echo ""
echo "[Step 4/5] Running Stage C Early Warning batch assessment & risk mapping..."
"$PYTHON" scripts/warn.py --batch data/processed/stage_b_features.csv --map

# 5. Master Report Compilation
echo ""
echo "[Step 5/5] Compiling final scientific evaluation report and summary figures..."
"$PYTHON" scripts/evaluate.py

echo ""
echo "=========================================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "All figures generated in   : results/figures/"
echo "All tables generated in    : results/tables/"
echo "Alert records logged in    : results/alerts/"
echo "Final Markdown Report in   : results/final_evaluation_report.md"
echo "=========================================================================="

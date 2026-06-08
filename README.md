# UPI Fraud Detection — Project Overview

## Project purpose

This repository contains code and data used for a basic UPI (Unified Payments Interface) fraud-detection exploration. The goal is to provide reproducible analysis, a trained model artifact, evaluation results, and a minimal dashboard for inspection.

## Quick start

1. Create and activate a Python virtual environment (recommended):
   - Windows (PowerShell):

     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

   - Linux / macOS (bash):

     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. Install dependencies (replace or extend as needed):

   ```bash
   pip install -U pip
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter dash
   ```

3. Run the dashboard:

   ```bash
   python dash_app.py
   ```

## Repository structure and important files

- `transactions.csv`, `users.csv`, `merchants.csv` — raw data CSVs used for analysis and feature engineering.
- `fraud_labels.csv` — labels used for supervised training.
- `upimine.ipynb`, `upimine working 1.ipynb` — exploratory notebooks and model development.
- `best_model.txt` — exported model identifier or metadata for the selected model.
- `evaluation_results.csv`, `evaluation_results.json` — evaluation metrics and outputs.
- `dash_app.py` — minimal Dash app for visualizing insights and checking model outputs.

## Data overview

The dataset contains transaction records and additional lookup tables for users and merchants. Typical fields include timestamp, amount, user and merchant identifiers, and label information indicating fraudulent transactions. Before running experiments, inspect the CSVs for memory and datatype considerations.

## Reproducibility and running experiments

1. Inspect the notebooks (`upimine.ipynb`) for the data-preprocessing and model-training pipeline.
2. If you want to re-run training, convert the notebook or extract the relevant Python script sections. Use a fixed random seed for deterministic results.
3. Save trained models and metadata in a dedicated `models/` folder (not present by default). Record preprocessing steps so the model can be reapplied consistently.

## Evaluation

- Use `evaluation_results.csv` and `evaluation_results.json` for stored metrics.
- Common classification metrics include precision, recall, F1-score, AUC-ROC, and confusion matrices. Prefer threshold-based analysis when prioritizing precision or recall.

## Notes and best practices

- Data leakage: ensure that time-based splits are used when appropriate (train on past, test on future) to avoid optimistic estimates.
- Class imbalance: fraud data is typically imbalanced. Use appropriate sampling, class weights, or specialized metrics.
- Feature logging: keep a record of features and transformations applied to enable reproducibility.

## Extending the project

- Add `requirements.txt` to pin dependency versions for reproducible environments.
- Add a `models/` directory to store trained models and metadata.

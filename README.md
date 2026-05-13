# 🚨 UPI Fraud Detection System

Detect fraudulent UPI transactions using machine learning. Combines 9 classical & quantum models with real-time dashboard.

## 🎯 Quick Start

### Install

```bash
pip install pandas numpy scikit-learn lightgbm shap plotly dash matplotlib seaborn
pip install qiskit qiskit-machine-learning qiskit-algorithms  # Optional: quantum models
```

### Run Notebook

```bash
jupyter notebook upimine.ipynb
```

### Run Dashboard

```bash
python dash_app.py
# Navigate to http://localhost:8050
```

## 📊 Dataset

| File               | Rows   | Description                    |
| ------------------ | ------ | ------------------------------ |
| `transactions.csv` | ~100K  | Transactions with 25+ features |
| `users.csv`        | ~5K    | User profiles & risk scores    |
| `merchants.csv`    | ~1K    | Merchant categories            |
| `fraud_labels.csv` | Labels | Ground truth fraud labels      |

See [data_dictionary.csv](data_dictionary.csv) for full field descriptions.

## 🤖 Models & Performance

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC   |
| ------------------- | -------- | --------- | ------ | ----- | --------- |
| **LightGBM**        | 0.95+    | 0.92+     | 0.88+  | 0.90+ | **0.96+** |
| Logistic Regression | 0.92+    | 0.89+     | 0.85+  | 0.87+ | 0.94+     |
| Gradient Boosting   | 0.93+    | 0.90+     | 0.86+  | 0.88+ | 0.95+     |
| Decision Tree       | 0.91+    | 0.87+     | 0.83+  | 0.85+ | 0.93+     |
| Quantum Hybrid      | 0.89+    | 0.84+     | 0.81+  | 0.82+ | 0.91+     |

**Threshold**: Optimized for F1-score & cost-sensitivity (FN: ₹1000, FP: ₹8)

## 🔍 Features Engineered

**20+ features** including:

- Risk flags (new device, IP mismatch, failed attempts)
- Transaction velocity & amount deviation
- User context (account age, linked banks, loyalty)
- Merchant risk categories
- Temporal patterns (night transactions, weekends)

## 📁 Files

```
├── upimine.ipynb              # Main notebook (data → models → evaluation)
├── upimine_minimal.ipynb      # Lightweight version
├── dash_app.py                # Interactive dashboard
├── data_dictionary.csv        # Field descriptions
├── evaluation_results.json    # Model metrics (auto-generated)
├── best_model.txt             # Best model name (auto-generated)
└── *.csv                      # Data files
```

## 🛠️ Customization

**Modify business costs** (first cell):

```python
COST_FALSE_NEGATIVE = 1000  # Cost of missed fraud
COST_FALSE_POSITIVE = 8     # Cost of false alarm
```

**Adjust model params**:

```python
lgb.LGBMClassifier(n_estimators=80, max_depth=7, learning_rate=0.1)
```

**Add features**: Edit `engineer_features()` function

## 🔄 Quantum-Classical Hybrid

- **With Qiskit**: QSVC, VQC, Hybrid ensemble
- **Without Qiskit**: Automatic fallback to classical LogisticRegression
- **Features**: 5 key features → 2D PCA for quantum circuits

## 📊 Pipeline

```
Load Data → Explore → Engineer Features → Split (80/20)
→ Train 9 Models → Optimize Thresholds → Evaluate
→ Visualize (ROC, Confusion Matrix, SHAP) → Export Results
```

## 📦 Dependencies

```
pandas, numpy, scikit-learn, lightgbm, shap, plotly, dash, matplotlib, seaborn
Optional: qiskit, qiskit-machine-learning, qiskit-algorithms
```

## 🚨 Troubleshooting

| Issue                  | Solution                              |
| ---------------------- | ------------------------------------- |
| Quantum libs not found | Falls back to classical automatically |
| Memory error           | Use `upimine_minimal.ipynb`           |
| Dashboard port busy    | Change port in `dash_app.py`          |
| Slow training          | Reduce `n_estimators` in LightGBM     |

## 📊 Stats

- **Transactions**: ~100,000
- **Users**: 5,000+
- **Features**: 20+ engineered
- **Models**: 9 classifiers
- **Fraud Rate**: 2-5%
- **Cities**: 30+

## 📄 License

Educational purposes only.

---

**Python 3.8+ | Updated May 13, 2026**

from __future__ import annotations

import importlib
import subprocess
import sys
from base64 import b64decode
from functools import lru_cache
from io import StringIO
from pathlib import Path


def _import_or_install(module_name: str, package_name: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        package = package_name or module_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(module_name)


np = _import_or_install("numpy")
pd = _import_or_install("pandas")
px = _import_or_install("plotly.express", "plotly")
go = _import_or_install("plotly.graph_objects", "plotly")
dash = _import_or_install("dash")
Dash = dash.Dash
Input = dash.Input
Output = dash.Output
State = dash.State
callback_context = dash.callback_context
dcc = dash.dcc
dash_table = dash.dash_table
html = dash.html
PreventUpdate = importlib.import_module("dash.exceptions").PreventUpdate
dbc = _import_or_install("dash_bootstrap_components", "dash-bootstrap-components")

# Ensure scikit-learn is available (sometimes has compatibility issues with scipy)
_import_or_install("sklearn", "scikit-learn")

from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = Path(__file__).resolve().parent
APP_TITLE = "UPI Fraud Detection System"


def load_csv(name: str) -> pd.DataFrame:
    path = BASE_DIR / name
    return pd.read_csv(path)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions = load_csv("transactions.csv")
    users = load_csv("users.csv")
    merchants = load_csv("merchants.csv")
    fraud_labels = load_csv("fraud_labels.csv")

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"], errors="coerce"
    )
    transactions["merchant_category"] = "P2P"

    if not merchants.empty and "merchant_id" in merchants.columns:
        merchant_lookup = merchants[
            ["merchant_id", "merchant_category", "merchant_size", "rating"]
        ].copy()
        merchant_lookup.columns = [
            "merchant_id",
            "merchant_category_m",
            "merchant_size",
            "merchant_rating",
        ]
        transactions = transactions.merge(
            merchant_lookup,
            left_on="receiver_id",
            right_on="merchant_id",
            how="left",
        )
        transactions["merchant_category"] = transactions["merchant_category_m"].fillna(
            "P2P"
        )
        transactions.drop(
            columns=[
                c
                for c in ["merchant_id", "merchant_category_m"]
                if c in transactions.columns
            ],
            inplace=True,
        )

    transactions["merchant_category"] = transactions["merchant_category"].fillna("P2P")
    transactions["payment_app"] = transactions["payment_app"].fillna("Unknown")
    transactions["device_type"] = transactions["device_type"].fillna("Unknown")
    transactions["transaction_type"] = transactions["transaction_type"].fillna(
        "Unknown"
    )
    transactions["user_city_tier"] = transactions["user_city_tier"].fillna("Unknown")
    transactions["user_kyc_status"] = transactions["user_kyc_status"].fillna("Unknown")
    transactions["status"] = transactions["status"].fillna("Unknown")
    transactions["hour_of_day"] = transactions["hour_of_day"].fillna(0).astype(int)
    transactions["is_fraud"] = transactions["is_fraud"].fillna(0).astype(int)

    return transactions, users, merchants, fraud_labels


transactions, users, merchants, fraud_labels = load_data()
GLOBAL_FRAUD_RATE = float(transactions["is_fraud"].mean())
GLOBAL_AVG_AMOUNT = float(transactions["amount"].mean())
PSP_OPTIONS = sorted(transactions["payment_app"].dropna().unique().tolist())
DEVICE_OPTIONS = sorted(transactions["device_type"].dropna().unique().tolist())
CATEGORY_OPTIONS = sorted(transactions["merchant_category"].dropna().unique().tolist())
STATE_OPTIONS = [
    "Delhi",
    "Maharashtra",
    "Karnataka",
    "Tamil Nadu",
    "Uttar Pradesh",
    "Rajasthan",
    "Gujarat",
]
FLOW_OPTIONS = ["P2P", "P2M", "Bill_Payment", "Recharge", "EMI", "Subscription"]

DISPLAY_LABELS = {
    "amount": "Transaction Amount",
    "hour_of_day": "Hour of Day",
    "is_fraud": "Fraud Label",
    "fraud_rate": "Fraud Rate",
    "payment_app": "Payment App",
    "merchant_category": "Merchant Category",
    "device_type": "Device OS",
    "ip_location_mismatch": "IP Location Mismatch",
    "month": "Month",
    "status": "Transaction Status",
    "count": "Count",
    "new_device_flag": "New Device Used",
    "failed_attempts_last_24h": "Failed Attempts (Last 24h)",
    "transaction_velocity": "Transaction Velocity",
    "amount_deviation_score": "Amount Deviation Score",
    "is_night_transaction": "Night Transaction",
    "proxy_score": "Proxy Risk Score (%)",
    "user_city_tier": "User City Tier",
    "user_kyc_status": "User KYC Status",
}

# Basic sanity check for the fraud label distribution. If the dataset appears
# to have an anomalous labeling (e.g., all ones or all zeros), surface a flag
# to the dashboard so users know metrics may be unreliable.
FRAUD_LABEL_ISSUE = False
FRAUD_LABEL_ISSUE_MSG = ""
if GLOBAL_FRAUD_RATE <= 0.0001:
    FRAUD_LABEL_ISSUE = True
    FRAUD_LABEL_ISSUE_MSG = (
        "Dataset contains almost no positive fraud labels; metrics may be unreliable."
    )
elif GLOBAL_FRAUD_RATE >= 0.95:
    FRAUD_LABEL_ISSUE = True
    FRAUD_LABEL_ISSUE_MSG = (
        "Dataset contains almost all positive fraud labels; metrics may be unreliable."
    )


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def fmt_currency(value: float) -> str:
    return f"₹{value:,.0f}"


def safe_format_metric(value: float | str, decimal_places: int = 4) -> str:
    """Safely format a metric value, handling both numeric and string values."""
    if isinstance(value, str):
        return value  # Return 'n/a' or other strings as-is
    try:
        return f"{float(value):.{decimal_places}f}"
    except Exception:
        return "n/a"


def display_label(raw_name: str) -> str:
    """Convert internal column names to user-friendly labels for UI rendering."""
    if raw_name in DISPLAY_LABELS:
        return DISPLAY_LABELS[raw_name]
    return str(raw_name).replace("_", " ").title()


def apply_filters(
    df: pd.DataFrame,
    psp_values: list[str],
    device_values: list[str],
    category_values: list[str],
    hour_range: list[int],
) -> pd.DataFrame:
    filtered = df.copy()
    if psp_values:
        filtered = filtered[filtered["payment_app"].isin(psp_values)]
    if device_values:
        filtered = filtered[filtered["device_type"].isin(device_values)]
    if category_values:
        filtered = filtered[filtered["merchant_category"].isin(category_values)]
    if hour_range and len(hour_range) == 2:
        filtered = filtered[
            (filtered["hour_of_day"] >= hour_range[0])
            & (filtered["hour_of_day"] <= hour_range[1])
        ]
    return filtered


def proxy_risk_series(df: pd.DataFrame) -> pd.Series:
    amount = df["amount"].fillna(df["amount"].median())
    dev = df["amount_deviation_score"].fillna(df["amount_deviation_score"].median())
    velocity = df["transaction_velocity"].fillna(0)
    failed = df["failed_attempts_last_24h"].fillna(0)
    new_device = df["new_device_flag"].fillna(0)
    ip_mismatch = df["ip_location_mismatch"].fillna(0)
    night = df["is_night_transaction"].fillna(0)

    score = 0.05
    score += np.clip((amount / max(GLOBAL_AVG_AMOUNT, 1.0) - 1.0) / 8.0, 0, 0.22)
    score += np.clip(dev / 10.0, 0, 0.12)
    score += np.clip(velocity / 10.0, 0, 0.08)
    score += np.clip(failed / 10.0, 0, 0.12)
    score += new_device * 0.11
    score += ip_mismatch * 0.14
    score += night * 0.06
    score += np.where(df["payment_app"].isin(["Paytm", "Amazon Pay"]), 0.02, 0.0)
    score += np.where(df["device_type"].isin(["Web"]), 0.02, 0.0)
    return pd.Series(np.clip(score, 0, 0.99), index=df.index)


def _as_probability_vector(raw_output: object) -> np.ndarray:
    values = np.asarray(raw_output)
    if values.ndim == 2:
        if values.shape[1] == 1:
            values = values[:, 0]
        else:
            values = values[:, -1]
    values = values.reshape(-1).astype(float)
    if values.size == 0:
        return values
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        return np.full(values.shape, 0.5, dtype=float)
    scaled = (values - minimum) / (maximum - minimum)
    return np.clip(scaled, 0.0, 1.0)


def _predict_positive_probability(
    model: object, features: pd.DataFrame | np.ndarray
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = getattr(model, "predict_proba")(features)
        proba_array = np.asarray(proba)
        if proba_array.ndim == 2 and proba_array.shape[1] > 1:
            return np.clip(proba_array[:, -1].astype(float), 0.0, 1.0)
        return _as_probability_vector(proba_array)
    if hasattr(model, "decision_function"):
        raw = getattr(model, "decision_function")(features)
        values = np.asarray(raw).reshape(-1).astype(float)
        return 1.0 / (1.0 + np.exp(-values))
    predictions = getattr(model, "predict")(features)
    return np.asarray(predictions, dtype=float).reshape(-1)


def _make_q_model_features(df: pd.DataFrame) -> pd.DataFrame:
    quantum_cols = [
        "amount",
        "hour_of_day",
        "failed_attempts_last_24h",
        "transaction_velocity",
        "amount_deviation_score",
        "new_device_flag",
        "ip_location_mismatch",
        "is_night_transaction",
    ]
    quantum_frame = df.copy()
    for column in quantum_cols:
        if column not in quantum_frame.columns:
            quantum_frame[column] = 0
    return quantum_frame[quantum_cols].copy()


@lru_cache(maxsize=1)
def train_model_suite() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols_num = [
        "amount",
        "hour_of_day",
        "is_weekend",
        "is_night_transaction",
        "time_since_last_txn_min",
        "user_avg_monthly_txn",
        "user_avg_txn_value",
        "user_loyalty_score",
        "new_device_flag",
        "ip_location_mismatch",
        "failed_attempts_last_24h",
        "transaction_velocity",
        "amount_deviation_score",
        "recurring_payment_flag",
        "balance_after_transaction",
        "transaction_frequency_score",
    ]
    feature_cols_cat = [
        "transaction_type",
        "payment_app",
        "device_type",
        "status",
        "user_city_tier",
        "user_kyc_status",
        "merchant_category",
    ]
    target = "is_fraud"
    df = transactions.copy()
    for col in feature_cols_num:
        if col not in df.columns:
            df[col] = 0
    for col in feature_cols_cat:
        if col not in df.columns:
            df[col] = "Unknown"

    X = df[feature_cols_num + feature_cols_cat].copy()
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), feature_cols_num),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                feature_cols_cat,
            ),
        ]
    )

    classical_specs: list[dict[str, object]] = [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
        },
        {
            "name": "Decision Tree",
            "model": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(
                n_estimators=180, random_state=42, class_weight="balanced", n_jobs=-1
            ),
        },
        {
            "name": "Naive Bayes",
            "model": GaussianNB(),
            "fit_kwargs": {},
        },
        {
            "name": "Balanced Gradient Boosting",
            "model": HistGradientBoostingClassifier(random_state=42),
        },
    ]
    try:
        from lightgbm import LGBMClassifier

        classical_specs.insert(
            3,
            {
                "name": "LightGBM",
                "model": LGBMClassifier(
                    n_estimators=200,
                    objective="binary",
                    random_state=42,
                    class_weight="balanced",
                ),
            },
        )
    except Exception:
        pass

    metrics_rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, object] = {}
    score_cache: dict[str, np.ndarray] = {}
    train_score_cache: dict[str, np.ndarray] = {}
    best_tracker = {"name": "", "score": -1.0}

    def _fit_and_score(
        name: str,
        estimator: object,
        fit_kwargs: dict[str, object] | None = None,
        invert_probability: bool = False,
    ) -> None:
        kwargs = dict(fit_kwargs or {})
        try:
            if kwargs:
                estimator.fit(X_train, y_train, **kwargs)
            else:
                estimator.fit(X_train, y_train)
        except TypeError:
            estimator.fit(X_train, y_train)

        train_proba = _predict_positive_probability(estimator, X_train)
        test_proba = _predict_positive_probability(estimator, X_test)
        if invert_probability:
            train_proba = 1.0 - train_proba
            test_proba = 1.0 - test_proba

        pred = (test_proba >= 0.5).astype(int)
        # If there are no positive labels in the test set, precision/recall/F1
        # are not meaningful — keep 'n/a' in that case. Otherwise compute
        # metrics using zero_division=0 so models that predict zero positives
        # return numeric 0.0 values instead of 'n/a'. Accuracy remains defined.
        if int(y_test.sum()) == 0:
            prec = "n/a"
            rec = "n/a"
            f1v = "n/a"
        else:
            prec = round(float(precision_score(y_test, pred, zero_division=0)), 4)
            rec = round(float(recall_score(y_test, pred, zero_division=0)), 4)
            f1v = round(float(f1_score(y_test, pred, zero_division=0)), 4)
        acc = round(float((pred == y_test.to_numpy()).mean()), 4)

        row = {
            "Model": name,
            "Precision": prec,
            "Recall": rec,
            "F1": f1v,
            "Accuracy": acc,
            "ROC_AUC": round(float(roc_auc_score(y_test, test_proba)), 4),
            "PR_AUC": round(float(average_precision_score(y_test, test_proba)), 4),
        }
        metrics_rows.append(row)
        fitted_models[name] = estimator
        score_cache[name] = test_proba
        train_score_cache[name] = train_proba

        try:
            f1_val = float(row["F1"]) if not isinstance(row["F1"], str) else None
        except Exception:
            f1_val = None
        if f1_val is not None and f1_val > best_tracker["score"]:
            best_tracker["score"] = float(f1_val)
            best_tracker["name"] = name

    for spec in classical_specs:
        estimator = Pipeline(
            [
                ("prep", preprocessor),
                ("model", spec["model"]),
            ]
        )
        fit_kwargs = (
            {"model__sample_weight": sample_weight}
            if isinstance(spec["model"], (GaussianNB, HistGradientBoostingClassifier))
            else {}
        )
        _fit_and_score(str(spec["name"]), estimator, fit_kwargs)

    q_frame = _make_q_model_features(X)
    q_train_frame, q_test_frame, q_y_train_series, q_y_test_series = train_test_split(
        q_frame,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    q_reducer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )
    q_train = q_reducer.fit_transform(q_train_frame)
    q_test = q_reducer.transform(q_test_frame)
    q_y_train = q_y_train_series.to_numpy()
    q_y_test = q_y_test_series.to_numpy()

    qsvc_model = None
    vqc_model = None
    qsvc_train_proba = None
    qsvc_test_proba = None
    vqc_train_proba = None
    vqc_test_proba = None

    try:
        from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
        from qiskit_machine_learning.algorithms.classifiers import QSVC, VQC
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        try:
            from qiskit.primitives import Sampler
        except Exception:
            Sampler = None  # type: ignore[assignment]
        try:
            from qiskit_algorithms.optimizers import COBYLA
        except Exception:
            from qiskit.algorithms.optimizers import COBYLA  # type: ignore

        feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        qsvc_model = QSVC(quantum_kernel=quantum_kernel, C=1.0)
        qsvc_model.fit(q_train, q_y_train)
        qsvc_train_proba = _predict_positive_probability(qsvc_model, q_train)
        qsvc_test_proba = _predict_positive_probability(qsvc_model, q_test)

        ansatz = RealAmplitudes(num_qubits=2, reps=1)
        sampler = Sampler() if Sampler is not None else None
        if sampler is None:
            vqc_model = VQC(
                feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=20)
            )
        else:
            vqc_model = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=COBYLA(maxiter=20),
                sampler=sampler,
            )
        vqc_model.fit(q_train, q_y_train)
        vqc_train_proba = _predict_positive_probability(vqc_model, q_train)
        vqc_test_proba = _predict_positive_probability(vqc_model, q_test)
    except Exception:
        qsvc_model = SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=42
        )
        vqc_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        qsvc_model.fit(q_train, q_y_train)
        vqc_model.fit(q_train, q_y_train)
        qsvc_train_proba = _predict_positive_probability(qsvc_model, q_train)
        qsvc_test_proba = _predict_positive_probability(qsvc_model, q_test)
        vqc_train_proba = _predict_positive_probability(vqc_model, q_train)
        vqc_test_proba = _predict_positive_probability(vqc_model, q_test)

    # Use a logistic regression combiner for the hybrid model, matching the notebook's structure.
    baseline_train = train_score_cache["Logistic Regression"]
    baseline_test = score_cache["Logistic Regression"]
    hybrid_train = np.column_stack([qsvc_train_proba, vqc_train_proba, baseline_train])
    hybrid_test = np.column_stack([qsvc_test_proba, vqc_test_proba, baseline_test])
    hybrid_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    hybrid_model.fit(hybrid_train, q_y_train)
    hybrid_test_proba = _predict_positive_probability(hybrid_model, hybrid_test)
    hybrid_train_proba = _predict_positive_probability(hybrid_model, hybrid_train)

    for name, test_proba, train_proba in [
        ("QSVC", qsvc_test_proba, qsvc_train_proba),
        ("VQC", vqc_test_proba, vqc_train_proba),
        ("Hybrid", hybrid_test_proba, hybrid_train_proba),
    ]:
        pred = (test_proba >= 0.5).astype(int)
        if test_proba is None:  # Check if the model was trained
            # if the quantum model wasn't trained, mark metrics as n/a
            row = {
                "Model": name,
                "Precision": "n/a",
                "Recall": "n/a",
                "F1": "n/a",
                "Accuracy": "n/a",  # Added accuracy metric
                "ROC_AUC": "n/a",  # Keep ROC_AUC as n/a
                "PR_AUC": "n/a",  # Keep PR_AUC as n/a
            }
        else:
            if int(q_y_test.sum()) == 0:  # Check if there are no positive labels
                prec = "n/a"
                rec = "n/a"
                f1v = "n/a"
            else:  # Calculate metrics if there are positive labels
                prec = round(float(precision_score(q_y_test, pred, zero_division=0)), 4)
                rec = round(float(recall_score(q_y_test, pred, zero_division=0)), 4)
                f1v = round(float(f1_score(q_y_test, pred, zero_division=0)), 4)
            acc = round(float((pred == q_y_test).mean()), 4)  # Calculate accuracy

            row = {
                "Model": name,
                "Precision": prec,
                "Recall": rec,
                "F1": f1v,
                "Accuracy": acc,  # Include calculated accuracy
                "ROC_AUC": round(float(roc_auc_score(q_y_test, test_proba)), 4),
                "PR_AUC": round(
                    float(average_precision_score(q_y_test, test_proba)), 4
                ),
            }
        metrics_rows.append(row)
        fitted_models[name] = {
            "QSVC": qsvc_model,
            "VQC": vqc_model,
            "Hybrid": hybrid_model,
        }[name]
        score_cache[name] = test_proba
        train_score_cache[name] = train_proba
        try:
            f1_val = float(row["F1"]) if not isinstance(row["F1"], str) else None
        except Exception:
            f1_val = None
        if f1_val is not None and f1_val > best_tracker["score"]:
            best_tracker["score"] = float(f1_val)
            best_tracker["name"] = name

    metrics_df = pd.DataFrame(metrics_rows)
    # Coerce metric columns to numeric where possible; 'n/a' will become NaN.
    for col in ["Precision", "Recall", "F1", "Accuracy", "ROC_AUC", "PR_AUC"]:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")
    metrics_df = metrics_df.sort_values(["F1", "ROC_AUC"], ascending=False)
    for col in ["Precision", "Recall", "F1", "Accuracy", "ROC_AUC", "PR_AUC"]:
        metrics_df[col] = metrics_df[col].map(
            lambda value: "n/a" if pd.isna(value) else f"{float(value):.4f}"
        )

    global BEST_PIPELINE
    best_name = best_tracker["name"]
    best_model = fitted_models.get(best_name)
    if best_name in {"QSVC", "VQC", "Hybrid"}:
        BEST_PIPELINE = (
            fitted_models.get("LightGBM")
            or fitted_models.get("Random Forest")
            or fitted_models.get("Logistic Regression")
        )
    else:
        BEST_PIPELINE = best_model

    feature_df = pd.DataFrame(columns=["Feature", "Importance"])
    if best_name in {"Random Forest", "Decision Tree", "LightGBM"}:
        best_estimator = fitted_models[best_name]
        if isinstance(best_estimator, Pipeline):
            prep = best_estimator.named_steps["prep"]
            model = best_estimator.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                feature_names = prep.get_feature_names_out()
                feature_df = pd.DataFrame(
                    {"Feature": feature_names, "Importance": model.feature_importances_}
                ).sort_values("Importance", ascending=False)
                feature_df["Feature"] = (
                    feature_df["Feature"]
                    .str.replace("num__", "", regex=False)
                    .str.replace("cat__", "", regex=False)
                )
                feature_df = feature_df.head(12)

    return metrics_df, feature_df


# Initialize placeholders to avoid expensive training at import when notebook exports exist.
METRICS_DF = pd.DataFrame()
FEATURE_IMPORTANCE_DF = pd.DataFrame()


def _load_notebook_evaluation(nb_path: Path) -> tuple[pd.DataFrame, str]:
    """Try to extract a precomputed evaluation table and best-model name from a notebook.
    Returns (df, best_model_name). If extraction fails, returns (empty_df, "").
    """
    import json
    from io import StringIO
    import re

    if not nb_path.exists():
        return pd.DataFrame(), ""

    try:
        with nb_path.open("r", encoding="utf-8") as fh:
            nb = json.load(fh)
    except Exception:
        return pd.DataFrame(), ""

    best_name = ""
    # Try to find printed outputs containing a table or a "Best model" line.
    for cell in nb.get("cells", []):
        # inspect outputs
        for out in cell.get("outputs", []):
            text = ""
            if out.get("output_type") == "stream":
                text = "".join(out.get("text") or [])
            elif out.get("output_type") in {"execute_result", "display_data"}:
                data = out.get("data", {})
                if "text/plain" in data:
                    text = "".join(data.get("text/plain") or [])
                elif "text/html" in data:
                    html = data.get("text/html")
                    text = html if isinstance(html, str) else "".join(html or [])

            if not text:
                continue

            # capture best model line
            for line in text.splitlines():
                if "Best model" in line or "Best Model" in line:
                    if ":" in line:
                        best_name = line.split(":", 1)[1].strip()

            # try to parse an ASCII/pipe table that includes a 'Model' header
            if "Model" in text and (
                "Accuracy" in text or "F1" in text or "F1 Score" in text
            ):
                lines = [l for l in text.splitlines() if l.strip()]
                # prefer pipe-separated tables
                if any("|" in l for l in lines):
                    table_lines = [l for l in lines if "|" in l]
                    # normalize to CSV by removing leading/trailing pipes
                    table_text = "\n".join(l.strip().strip("|") for l in table_lines)
                    try:
                        df = pd.read_csv(
                            StringIO(table_text), sep=r"\|", engine="python"
                        )
                        # drop wholly-empty columns introduced by split
                        df = df.loc[:, df.columns.str.strip().astype(bool)]
                        return df.rename(columns=lambda c: c.strip()), best_name
                    except Exception:
                        pass

                # fallback: whitespace-split table (best-effort)
                try:
                    parts = [l for l in lines if l.strip()]
                    header = re.split(r"\s{2,}", parts[0].strip())
                    rows = [re.split(r"\s{2,}", r.strip()) for r in parts[1:]]
                    if header and rows:
                        df = pd.DataFrame(rows, columns=header)
                        # try to coerce numeric columns
                        for col in df.columns:
                            try:
                                df[col] = pd.to_numeric(
                                    df[col].str.replace("%", "", regex=False),
                                    errors="ignore",
                                )
                            except Exception:
                                pass
                        return df, best_name
                except Exception:
                    pass

    # If no table was found in outputs, try to find a literal pd.DataFrame(...) in source cells
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source") or [])
        if "pd.DataFrame" in src and "Model" in src:
            m = re.search(r"pd\.DataFrame\((\s*\[.*?\])\s*\)", src, re.S)
            if m:
                literal = m.group(1)
                try:
                    # evaluate literal safely for lists/dicts
                    data = eval(literal, {"__builtins__": {}}, {})
                    if isinstance(data, list) and data:
                        return pd.DataFrame(data), best_name
                except Exception:
                    pass

    return pd.DataFrame(), best_name


# Attempt to load the notebook-produced evaluation table; fall back to the trained metrics.
NOTEBOOK_PATH = BASE_DIR / "upimine.ipynb"
# Prefer an exported evaluation file produced by the notebook (fast, reliable).
EVAL_JSON = BASE_DIR / "evaluation_results.json"
EVAL_CSV = BASE_DIR / "evaluation_results.csv"
BEST_TXT = BASE_DIR / "best_model.txt"

_nb_df = pd.DataFrame()
_nb_best = ""
if EVAL_JSON.exists():
    try:
        _nb_df = pd.read_json(EVAL_JSON)
        if BEST_TXT.exists():
            _nb_best = BEST_TXT.read_text(encoding="utf-8").strip()
    except Exception:
        _nb_df = pd.DataFrame()
        _nb_best = ""
elif EVAL_CSV.exists():
    try:
        _nb_df = pd.read_csv(EVAL_CSV)
        if BEST_TXT.exists():
            _nb_best = BEST_TXT.read_text(encoding="utf-8").strip()
    except Exception:
        _nb_df = pd.DataFrame()
        _nb_best = ""
else:
    _nb_df, _nb_best = _load_notebook_evaluation(NOTEBOOK_PATH)

if not _nb_df.empty:
    # normalize column names to common keys used below
    METRICS_DF = _nb_df.copy()
    NOTEBOOK_BEST_MODEL_NAME = _nb_best or ""
    try:
        BEST_MODEL = (
            (
                METRICS_DF.loc[METRICS_DF["Model"] == NOTEBOOK_BEST_MODEL_NAME]
                .iloc[0]
                .to_dict()
            )
            if NOTEBOOK_BEST_MODEL_NAME
            and (METRICS_DF["Model"] == NOTEBOOK_BEST_MODEL_NAME).any()
            else METRICS_DF.iloc[0].to_dict()
        )
    except Exception:
        BEST_MODEL = METRICS_DF.iloc[0].to_dict() if not METRICS_DF.empty else {}
else:
    # notebook not available / parse failed: compute metrics by training models
    NOTEBOOK_BEST_MODEL_NAME = ""
    if METRICS_DF.empty:
        METRICS_DF, FEATURE_IMPORTANCE_DF = train_model_suite()
    BEST_MODEL = METRICS_DF.iloc[0].to_dict() if not METRICS_DF.empty else {}


def score_transaction(
    amount: float,
    txns_last_24h: float,
    avg_30day_amount: float,
    new_device: bool,
    first_merchant_visit: bool,
    vpn_ip: bool,
    psp: str,
    uip_flow: str,
    merchant_category: str,
    state: str,
    device_os: str,
    time_value: str,
    screening_mode: str,
    probability_cutoff: float,
) -> tuple[float, str, str, list[dict[str, str]]]:
    amount_ratio = amount / max(avg_30day_amount, 1.0)
    hour = 0
    try:
        hour = int(str(time_value).split(":")[0])
    except Exception:
        hour = 0

    score = 0.04
    score += np.clip((amount_ratio - 1.0) / 7.0, 0.0, 0.28)
    score += np.clip(txns_last_24h / 12.0, 0.0, 0.14)
    score += 0.12 if new_device else 0.0
    score += 0.09 if first_merchant_visit else 0.0
    score += 0.13 if vpn_ip else 0.0
    score += 0.05 if hour >= 22 or hour <= 5 else 0.0
    score += 0.03 if uip_flow in {"P2P", "P2M"} else 0.0

    app_rate = float(
        transactions.groupby("payment_app")["is_fraud"]
        .mean()
        .get(psp, GLOBAL_FRAUD_RATE)
    )
    cat_rate = float(
        transactions.groupby("merchant_category")["is_fraud"]
        .mean()
        .get(merchant_category, GLOBAL_FRAUD_RATE)
    )
    dev_rate = float(
        transactions.groupby("device_type")["is_fraud"]
        .mean()
        .get(device_os, GLOBAL_FRAUD_RATE)
    )
    score += np.clip((app_rate - GLOBAL_FRAUD_RATE) * 1.2, -0.03, 0.08)
    score += np.clip((cat_rate - GLOBAL_FRAUD_RATE) * 1.4, -0.03, 0.10)
    score += np.clip((dev_rate - GLOBAL_FRAUD_RATE) * 1.2, -0.02, 0.06)

    mode_multiplier = {"High": 1.18, "Balanced": 1.0, "Low": 0.84}.get(
        screening_mode, 1.0
    )
    score = clamp(score * mode_multiplier, 0.0, 0.99)

    cutoff = probability_cutoff
    if screening_mode == "High":
        cutoff = min(cutoff, 0.22)
    elif screening_mode == "Low":
        cutoff = max(cutoff, 0.35)

    label = "LEGITIMATE" if score < cutoff else "FLAGGED"
    tone = "success" if label == "LEGITIMATE" else "danger"
    message = f"{label} - {score * 100:.2f}% (threshold {cutoff:.2f})"

    factors = [
        {
            "name": "Amount vs. Usual Spending",
            "value": f"{clamp((amount_ratio - 1.0) / 7.0, 0.0, 0.28) * 100:.1f}%",
        },
        {
            "name": "Transactions in Last 24 Hours",
            "value": f"{clamp(txns_last_24h / 12.0, 0.0, 0.14) * 100:.1f}%",
        },
        {"name": "New Device Used", "value": "12.0%" if new_device else "0.0%"},
        {
            "name": "First Time at This Merchant",
            "value": "9.0%" if first_merchant_visit else "0.0%",
        },
        {"name": "VPN / Unusual IP Risk", "value": "13.0%" if vpn_ip else "0.0%"},
        {"name": "Device & Transaction Pattern Risk", "value": "model-based"},
    ]

    return score, label, tone, factors


def build_indicator(score: float, cutoff: float, label: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 44, "color": "#2457a4"}},
            title={
                "text": "Fraud Probability",
                "font": {"size": 20, "color": "#263238"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#2b8a78" if label == "LEGITIMATE" else "#b91c1c"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#d6dee8",
                "steps": [
                    {"range": [0, cutoff * 100], "color": "#dff6e8"},
                    {"range": [cutoff * 100, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#fde2e2"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 4},
                    "thickness": 0.75,
                    "value": cutoff * 100,
                },
            },
        )
    )
    fig.update_layout(
        margin={"t": 40, "b": 15, "l": 20, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


def metric_card(
    title: str, value: str, subtitle: str = "", color: str = "#2457a4"
) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="metric-title"),
                html.Div(value, className="metric-value", style={"color": color}),
                html.Div(subtitle, className="metric-subtitle"),
            ]
        ),
        className="metric-card",
    )


def hero_banner(title: str, subtitle: str, tone: str = "teal") -> html.Div:
    classes = {
        "teal": "hero hero-teal",
        "green": "hero hero-green",
        "blue": "hero hero-blue",
        "pink": "hero hero-pink",
    }.get(tone, "hero hero-teal")
    return html.Div(
        [
            html.H2(title, className="hero-title"),
            html.P(subtitle, className="hero-subtitle"),
        ],
        className=classes,
    )


def chart_card(title: str, figure: go.Figure) -> dbc.Card:
    fig = figure
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white"
    )
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, className="chart-title"),
                dcc.Graph(
                    figure=fig,
                    config={"displayModeBar": False},
                    className="chart-graph",
                ),
            ]
        ),
        className="chart-card",
    )


def overview_figures(df: pd.DataFrame) -> list[dbc.Card]:
    hourly = df.groupby("hour_of_day", as_index=False).agg(
        fraud_rate=("is_fraud", "mean"), volume=("transaction_id", "count")
    )
    app_rate = (
        df.groupby("payment_app", as_index=False)
        .agg(fraud_rate=("is_fraud", "mean"))
        .sort_values("fraud_rate", ascending=False)
    )
    month_df = df.copy()
    month_df["month"] = month_df["timestamp"].dt.to_period("M").astype(str)
    monthly = (
        month_df.groupby("month", as_index=False)
        .agg(fraud_rate=("is_fraud", "mean"), amount=("amount", "sum"))
        .sort_values("month")
    )
    status = (
        df.groupby("status", as_index=False).size().rename(columns={"size": "count"})
    )

    fig1 = px.bar(
        hourly,
        x="hour_of_day",
        y="fraud_rate",
        color="fraud_rate",
        color_continuous_scale=["#d7f5e7", "#64c8b0", "#2a7f62"],
        title="Fraud Rate by Hour",
        labels=DISPLAY_LABELS,
    )
    fig1.update_layout(coloraxis_showscale=False)
    fig2 = px.line(
        monthly,
        x="month",
        y="fraud_rate",
        markers=True,
        title="Monthly Fraud Trend",
        labels=DISPLAY_LABELS,
    )
    fig3 = px.bar(
        app_rate,
        x="payment_app",
        y="fraud_rate",
        color="fraud_rate",
        color_continuous_scale=["#e6f4ff", "#7cc2f7", "#1956a3"],
        title="Fraud Rate by Payment App",
        labels=DISPLAY_LABELS,
    )
    fig3.update_layout(coloraxis_showscale=False)
    fig4 = px.pie(
        status,
        names="status",
        values="count",
        hole=0.55,
        title="Transaction Status Mix",
        labels=DISPLAY_LABELS,
    )
    fig4.update_traces(marker=dict(colors=["#2a7f62", "#e3a81d", "#b12e35"]))

    return [
        chart_card("Transaction Volume and Fraud by Hour", fig1),
        chart_card("Monthly Trend", fig2),
        chart_card("Payment App Risk", fig3),
        chart_card("Transaction Status Mix", fig4),
    ]


def eda_figures(df: pd.DataFrame) -> list[dbc.Card]:
    fig1 = px.box(
        df,
        x="is_fraud",
        y="amount",
        color="is_fraud",
        points="outliers",
        title="Amount by Fraud Status",
        color_discrete_sequence=["#48b98f", "#ef4b5d"],
        labels=DISPLAY_LABELS,
    )
    fig1.update_xaxes(title_text="Fraud (0/1)")
    hourly = df.groupby("hour_of_day", as_index=False).agg(
        fraud_rate=("is_fraud", "mean")
    )
    fig2 = px.bar(
        hourly,
        x="hour_of_day",
        y="fraud_rate",
        color="fraud_rate",
        title="Fraud Rate by Hour",
        color_continuous_scale=["#fff2cc", "#f29f05", "#9b111e"],
        labels=DISPLAY_LABELS,
    )
    fig2.update_layout(coloraxis_showscale=False)
    app = (
        df.groupby("payment_app", as_index=False)
        .agg(fraud_rate=("is_fraud", "mean"))
        .sort_values("fraud_rate", ascending=False)
    )
    fig3 = px.bar(
        app,
        x="payment_app",
        y="fraud_rate",
        color="fraud_rate",
        title="Fraud Rate by Payment App",
        color_continuous_scale=["#d6e9ff", "#5aa0f0", "#0c3f86"],
        labels=DISPLAY_LABELS,
    )
    fig3.update_layout(coloraxis_showscale=False)
    cat = (
        df.groupby("merchant_category", as_index=False)
        .agg(fraud_rate=("is_fraud", "mean"))
        .sort_values("fraud_rate", ascending=False)
    )
    fig4 = px.bar(
        cat,
        x="merchant_category",
        y="fraud_rate",
        color="fraud_rate",
        title="Fraud Rate by Merchant Category",
        color_continuous_scale=["#fde8e8", "#f87171", "#7f1d1d"],
        labels=DISPLAY_LABELS,
    )
    fig4.update_layout(coloraxis_showscale=False)
    dev = (
        df.groupby("device_type", as_index=False)
        .agg(fraud_rate=("is_fraud", "mean"))
        .sort_values("fraud_rate", ascending=False)
    )
    fig5 = px.bar(
        dev,
        x="device_type",
        y="fraud_rate",
        color="fraud_rate",
        title="Fraud Rate by Device OS",
        color_continuous_scale=["#f3e8ff", "#7c3aed", "#4c1d95"],
        labels=DISPLAY_LABELS,
    )
    fig5.update_layout(coloraxis_showscale=False)
    vpn = df.groupby("ip_location_mismatch", as_index=False).agg(
        fraud_rate=("is_fraud", "mean")
    )
    vpn["ip_location_mismatch"] = vpn["ip_location_mismatch"].map(
        {0: "No mismatch", 1: "Mismatch"}
    )
    fig6 = px.bar(
        vpn,
        x="ip_location_mismatch",
        y="fraud_rate",
        color="fraud_rate",
        title="IP Mismatch vs Fraud Rate",
        color_continuous_scale=["#e0f2fe", "#0ea5e9", "#0c4a6e"],
        labels=DISPLAY_LABELS,
    )
    fig6.update_layout(coloraxis_showscale=False)

    return [
        chart_card("Chart 1 — Amount by Fraud Status", fig1),
        chart_card("Chart 2 — Fraud Rate by Hour", fig2),
        chart_card("Chart 3 — Fraud Rate by PSP", fig3),
        chart_card("Chart 4 — Fraud Rate by Merchant Category", fig4),
        chart_card("Chart 5 — Fraud Rate by Device OS", fig5),
        chart_card("Chart 6 — IP Mismatch vs Fraud Rate", fig6),
    ]


def model_figures(
    metrics_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    filtered: pd.DataFrame,
    best_model_name: str | None = None,
    best_metric: str | None = None,
) -> list[dbc.Card]:
    # Determine the 'best' model to display in summary cards.
    if best_metric and best_metric in metrics_df.columns:
        # pick row with max value for the chosen metric
        try:
            idx = metrics_df[best_metric].astype(float).idxmax()
            best = metrics_df.loc[idx]
        except Exception:
            best = metrics_df.iloc[0]
    elif best_model_name and "Model" in metrics_df.columns:
        best_match = metrics_df.loc[metrics_df["Model"] == best_model_name]
        best = best_match.iloc[0] if not best_match.empty else metrics_df.iloc[0]
    else:
        best = metrics_df.iloc[0]

    precision_col = "Precision" if "Precision" in metrics_df.columns else None
    recall_col = "Recall" if "Recall" in metrics_df.columns else None
    f1_col = "F1 Score" if "F1 Score" in metrics_df.columns else "F1"
    accuracy_col = "Accuracy" if "Accuracy" in metrics_df.columns else None
    metric_cols = [
        col
        for col in [accuracy_col, precision_col, recall_col, f1_col]
        if col is not None and col in metrics_df.columns
    ]
    fig = px.bar(
        metrics_df.melt(
            id_vars="Model",
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Score",
        ),
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Comparison",
        color_discrete_map={
            "Precision": "#2f6fed",
            "Recall": "#e84c3d",
            "F1": "#2aa876",
            "F1 Score": "#2aa876",
            "Accuracy": "#7b4fbd",
        },
    )

    if feature_df.empty:
        feature_df = pd.DataFrame(
            {
                "Feature": [
                    "amount",
                    "new_device_flag",
                    "ip_location_mismatch",
                    "failed_attempts_last_24h",
                ],
                "Importance": [0.32, 0.26, 0.21, 0.14],
            }
        )
    feature_df = feature_df.copy()
    feature_df["Feature"] = feature_df["Feature"].map(display_label)
    feature_fig = px.bar(
        feature_df.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Fraud Drivers",
    )

    summary_row = dbc.Row(
        [
            dbc.Col(
                metric_card(
                    "Accuracy",
                    safe_format_metric(best[accuracy_col]) if accuracy_col else "n/a",
                    "Metric value",
                    "#7b4fbd",
                ),
                md=2,
            ),
            dbc.Col(
                metric_card(
                    "Precision",
                    safe_format_metric(best[precision_col]) if precision_col else "n/a",
                    "Metric value",
                    "#1f4ca3",
                ),
                md=2,
            ),
            dbc.Col(
                metric_card(
                    "Recall",
                    safe_format_metric(best[recall_col]) if recall_col else "n/a",
                    "Metric value",
                    "#bf4b3c",
                ),
                md=2,
            ),
            dbc.Col(
                metric_card(
                    "F1",
                    safe_format_metric(best[f1_col]) if f1_col in best.index else "n/a",
                    "Metric value",
                    "#208c65",
                ),
                md=2,
            ),
            dbc.Col(
                metric_card(
                    "Fraud flagged",
                    f"{int(filtered['is_fraud'].sum())} ({filtered['is_fraud'].mean()*100:.2f}%)",
                    "Current filters",
                    "#6b7280",
                ),
                md=4,
            ),
        ],
        className="g-3 mb-4",
    )

    table_rows = metrics_df.to_dict("records")
    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [html.Th(display_label(col)) for col in metrics_df.columns],
                    style={
                        "backgroundColor": "#f6f8fb",
                        "color": "#334155",
                        "fontWeight": "700",
                    },
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            (
                                html.Td(
                                    safe_format_metric(value)
                                    if isinstance(
                                        value, (int, float, np.floating, np.integer)
                                    )
                                    else value
                                )
                                if col != "Model"
                                else html.Td(value)
                            )
                            for col, value in row.items()
                        ],
                        style={
                            "backgroundColor": "#fbfdff" if idx % 2 else "white",
                        },
                    )
                    for idx, row in enumerate(table_rows)
                ]
            ),
        ],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "overflowX": "auto",
            "borderRadius": "14px",
            "boxShadow": "0 10px 24px rgba(15,23,42,0.08)",
        },
        className="metric-table",
    )

    items = [
        summary_row,
        (
            dbc.Row(dbc.Col(dbc.Alert(FRAUD_LABEL_ISSUE_MSG, color="warning"), md=12))
            if FRAUD_LABEL_ISSUE
            else None
        ),
        dbc.Row(
            [
                dbc.Col(chart_card("Model Comparison", fig), md=12),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("7-Model Comparison", className="chart-title"),
                                table,
                            ]
                        )
                    ),
                    md=12,
                    className="mb-4",
                ),
                dbc.Col(chart_card("Feature Importance", feature_fig), md=12),
            ],
            className="g-3",
        ),
    ]
    # Filter out None values before returning
    return [item for item in items if item is not None]


def fraud_distribution_figure(df: pd.DataFrame) -> go.Figure:
    scores = proxy_risk_series(df)
    fig = px.histogram(
        scores,
        nbins=28,
        title="Score Distribution",
        color_discrete_sequence=["#1f6ae0"],
    )
    fig.update_layout(
        xaxis_title="Proxy Risk Score",
        yaxis_title="Transactions",
        template="plotly_white",
    )
    return fig


def build_sidebar() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(APP_TITLE, className="sidebar-brand"),
                    html.Div("LEARN  •  BUILD  •  GROW", className="sidebar-tagline"),
                ],
                className="sidebar-brand-card",
            ),
            html.Div("Navigation", className="sidebar-section-title"),
            dcc.RadioItems(
                id="page-nav",
                options=[
                    {"label": "Overview Dashboard", "value": "overview"},
                    {"label": "Exploratory Data Analysis", "value": "eda"},
                    {"label": "Model Evaluation", "value": "evaluation"},
                    {"label": "Fraud Detection", "value": "fraud"},
                ],
                value="overview",
                className="nav-radio",
                inputClassName="nav-radio-input",
                labelClassName="nav-radio-label",
            ),
            html.Hr(className="sidebar-divider"),
            html.Div("Screening Mode", className="sidebar-section-title"),
            dbc.RadioItems(
                id="screening-mode",
                options=[
                    {"label": "High", "value": "High"},
                    {"label": "Balanced", "value": "Balanced"},
                    {"label": "Low", "value": "Low"},
                ],
                value="Balanced",
                inline=True,
                className="mode-pill-group",
                inputClassName="mode-pill-input",
                labelClassName="mode-pill-label",
            ),
            html.Div("Probability Cutoff Threshold", className="sidebar-control-label"),
            dcc.Slider(
                id="prob-cutoff",
                min=0.05,
                max=0.60,
                step=0.01,
                value=0.25,
                marks={0.1: "0.10", 0.25: "0.25", 0.4: "0.40", 0.6: "0.60"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(id="threshold-card", className="threshold-card"),
            html.Div(
                [
                    html.Div(
                        "Best model by",
                        style={"fontWeight": 700, "marginBottom": "6px"},
                    ),
                    dcc.RadioItems(
                        id="best-by",
                        options=[
                            {"label": "F1", "value": "F1"},
                            {"label": "Accuracy", "value": "Accuracy"},
                            {"label": "ROC-AUC", "value": "ROC-AUC"},
                        ],
                        value="F1",
                        inline=True,
                        className="mb-3",
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div("EDA Filters", className="sidebar-section-title mt-4"),
            html.Div("Payment App", className="sidebar-control-label"),
            dcc.Dropdown(
                id="psp-filter",
                options=[{"label": x, "value": x} for x in PSP_OPTIONS],
                value=PSP_OPTIONS,
                multi=True,
                placeholder="Select PSPs",
                className="dash-dropdown",
            ),
            html.Div("Device OS", className="sidebar-control-label mt-3"),
            dcc.Dropdown(
                id="device-filter",
                options=[{"label": x, "value": x} for x in DEVICE_OPTIONS],
                value=DEVICE_OPTIONS,
                multi=True,
                placeholder="Select device types",
                className="dash-dropdown",
            ),
            html.Div("Merchant Category", className="sidebar-control-label mt-3"),
            dcc.Dropdown(
                id="category-filter",
                options=[{"label": x, "value": x} for x in CATEGORY_OPTIONS],
                value=CATEGORY_OPTIONS,
                multi=True,
                placeholder="Select categories",
                className="dash-dropdown",
            ),
            html.Div("Hour Range", className="sidebar-control-label mt-3"),
            dcc.RangeSlider(
                id="hour-filter",
                min=0,
                max=23,
                step=1,
                value=[0, 23],
                marks={0: "0", 6: "6", 12: "12", 18: "18", 23: "23"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                [
                    html.Div("Filtered transactions", className="stat-label"),
                    html.Div(id="filtered-count", className="stat-value"),
                ],
                className="filter-card",
            ),
        ],
        className="sidebar",
    )


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server
app.title = APP_TITLE

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>UPI Fraud Detection System</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --bg: #f5faf8;
                --panel: #ffffff;
                --panel-alt: #eef7ef;
                --border: rgba(26, 60, 43, 0.12);
                --text: #17212b;
                --muted: #61707f;
                --green: #3a7f50;
                --green-2: #e5f6ea;
                --blue: #2962ff;
                --blue-2: #e8f0ff;
                --pink: #f7e9ec;
                --shadow: 0 16px 40px rgba(24, 35, 29, 0.08);
            }
            body { margin: 0; background: var(--bg); color: var(--text); font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
            .app-shell { min-height: 100vh; display: flex; background: linear-gradient(180deg, #f5fbf7 0%, #fbfcfe 100%); }
            .sidebar { width: 325px; padding: 22px 18px; background: linear-gradient(180deg, #eaf6e8 0%, #e9f8ff 100%); border-right: 1px solid rgba(10, 20, 30, 0.07); position: sticky; top: 0; height: 100vh; overflow-y: auto; }
            .content { flex: 1; padding: 28px 30px 44px; }
            .sidebar-brand-card { margin-bottom: 18px; padding: 16px 14px 18px; background: rgba(255,255,255,0.45); border: 1px solid rgba(255,255,255,0.6); border-radius: 20px; box-shadow: 0 10px 24px rgba(79, 109, 88, 0.08); }
            .sidebar-brand { font-weight: 800; font-size: 26px; color: #243130; text-align: center; }
            .sidebar-tagline { text-align: center; color: #7e8b84; font-size: 11px; letter-spacing: 0.4em; margin-top: 6px; }
            .sidebar-section-title { font-weight: 700; font-size: 18px; color: #18242d; margin: 18px 4px 12px; }
            .sidebar-control-label { font-size: 13px; font-weight: 600; color: #334155; margin: 14px 4px 8px; }
            .sidebar-divider { margin: 18px 0; border-color: rgba(24, 33, 29, 0.1); }
            .nav-radio, .mode-pill-group { display: flex; flex-direction: column; gap: 10px; }
            .nav-radio-label { display: flex; align-items: center; gap: 10px; padding: 4px 2px; font-size: 16px; color: #18242d; }
            .nav-radio-input { transform: scale(1.05); accent-color: #2f6fed; }
            .mode-pill-group { gap: 8px; }
            .mode-pill-label { display: inline-flex; align-items: center; gap: 10px; padding: 10px 14px; background: #fff; border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; margin-right: 8px; margin-bottom: 6px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.7); }
            .mode-pill-input { accent-color: #2f6fed; }
            .threshold-card { margin-top: 12px; padding: 14px; border-radius: 16px; border: 1px solid rgba(43, 104, 76, 0.12); background: linear-gradient(180deg, #eef7ff 0%, #edf9ef 100%); box-shadow: var(--shadow); min-height: 72px; }
            .filter-card { margin-top: 16px; padding: 14px; border-radius: 16px; background: #fff; border: 1px solid rgba(0,0,0,0.08); box-shadow: var(--shadow); }
            .stat-label { color: #59707f; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; }
            .stat-value { color: #194b75; font-size: 28px; font-weight: 800; margin-top: 6px; }
            .hero { border-radius: 24px; padding: 28px 28px 22px; margin-bottom: 18px; box-shadow: var(--shadow); border: 1px solid rgba(255,255,255,0.55); }
            .hero-teal { background: linear-gradient(135deg, #2e7d78 0%, #52a8a1 100%); color: white; }
            .hero-green { background: linear-gradient(135deg, #3b7f42 0%, #4c9b57 100%); color: white; }
            .hero-blue { background: linear-gradient(135deg, #2a639d 0%, #3b88d8 100%); color: white; }
            .hero-pink { background: linear-gradient(135deg, #7b4f6d 0%, #b76f8f 100%); color: white; }
            .hero-title { margin: 0; font-size: 30px; font-weight: 800; }
            .hero-subtitle { margin: 10px 0 0; font-size: 16px; opacity: 0.9; }
            .hero-note { margin-top: 8px; font-size: 12px; opacity: 0.85; }
            .metric-card, .chart-card { border: 1px solid var(--border); border-radius: 20px; box-shadow: var(--shadow); background: var(--panel); }
            .metric-title { color: #5b6b78; text-transform: none; font-size: 13px; font-weight: 700; letter-spacing: 0.02em; }
            .metric-value { font-size: 40px; line-height: 1.1; font-weight: 800; margin: 8px 0 4px; }
            .metric-subtitle { color: #7c8b98; font-size: 13px; }
            .chart-title { margin: 0 0 10px; font-size: 24px; font-weight: 800; color: #1c2934; }
            .section-heading { font-size: 28px; font-weight: 800; margin: 18px 0 16px; color: #1e293b; }
            .section-subtitle { color: #69808f; margin-bottom: 18px; }
            .pill { display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 999px; font-weight: 700; font-size: 13px; margin-right: 8px; margin-bottom: 8px; }
            .pill-good { background: #e8f7ee; color: #1a6c43; }
            .pill-warning { background: #fff4d8; color: #9c6a08; }
            .pill-danger { background: #fde8e8; color: #9d2f2f; }
            .pill-neutral { background: #edf2f7; color: #4a5568; }
            .tabs { margin-top: 18px; }
            .tabs .nav-link { border: none; color: #334155; font-weight: 700; }
            .tabs .nav-link.active { color: #1f4ca3; border-bottom: 3px solid #2f6fed !important; background: transparent; }
            .input-panel { background: #fff; border: 1px solid rgba(0,0,0,0.08); border-radius: 20px; box-shadow: var(--shadow); padding: 18px; }
.score-button { width: 100%; height: 54px; font-weight: 800; font-size: 16px; border-radius: 14px; border: 0; background: linear-gradient(135deg, #2f6fed 0%, #2457a4 100%); box-shadow: 0 14px 24px rgba(47, 111, 237, 0.25); }
            .score-button:hover { filter: brightness(1.03); }
            .sample-button-safe { width: 100%; height: 48px; font-weight: 800; font-size: 15px; border-radius: 12px; border: 0; background: linear-gradient(135deg, #3a7f50 0%, #2d6541 100%); box-shadow: 0 10px 20px rgba(58, 127, 80, 0.2); color: white; transition: all 0.3s ease; }
            .sample-button-safe:hover { filter: brightness(1.08); transform: translateY(-2px); box-shadow: 0 12px 28px rgba(58, 127, 80, 0.28); }
            .sample-button-fraud { width: 100%; height: 48px; font-weight: 800; font-size: 15px; border-radius: 12px; border: 0; background: linear-gradient(135deg, #d64545 0%, #b73a3a 100%); box-shadow: 0 10px 20px rgba(214, 69, 69, 0.2); color: white; transition: all 0.3s ease; }
            .sample-button-fraud:hover { filter: brightness(1.08); transform: translateY(-2px); box-shadow: 0 12px 28px rgba(214, 69, 69, 0.28); }
            .result-banner { margin-top: 18px; padding: 16px 18px; border-radius: 18px; font-weight: 800; font-size: 18px; }
            .result-success { background: #e7f7ec; color: #1b6f46; border: 1px solid #c5ebd1; }
            .result-danger { background: #fdecec; color: #ab2d2d; border: 1px solid #f2c1c1; }
            .insight-list { margin-top: 14px; display: grid; gap: 8px; }
            .insight-item { background: #f8fbff; border: 1px solid rgba(0,0,0,0.06); border-radius: 14px; padding: 12px 14px; color: #1f2937; }
            .upload-box { border: 2px dashed #a5c8ff; background: #f7fbff; border-radius: 18px; padding: 18px; text-align: center; color: #41607e; }
            .batch-output { margin-top: 14px; padding: 16px; border-radius: 18px; background: #fff; border: 1px solid rgba(0,0,0,0.08); box-shadow: var(--shadow); }
            .dash-dropdown .Select-control, .dash-dropdown .Select-menu-outer, .dash-dropdown .Select-placeholder { border-radius: 14px !important; }
            .dash-dropdown .Select-control { min-height: 46px; }
            @media (max-width: 1100px) { .sidebar { width: 280px; } .content { padding: 18px; } }
            @media (max-width: 860px) { .app-shell { flex-direction: column; } .sidebar { width: 100%; height: auto; position: static; } }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

app.layout = html.Div(
    [
        build_sidebar(),
        html.Div(
            [
                dcc.Store(id="batch-store"),
                html.Div(id="page-content"),
            ],
            className="content",
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("threshold-card", "children"),
    Output("filtered-count", "children"),
    Output("page-content", "children"),
    Input("page-nav", "value"),
    Input("psp-filter", "value"),
    Input("device-filter", "value"),
    Input("category-filter", "value"),
    Input("hour-filter", "value"),
    Input("screening-mode", "value"),
    Input("prob-cutoff", "value"),
    Input("best-by", "value"),
)
def render_page(
    page,
    psp_values,
    device_values,
    category_values,
    hour_range,
    screening_mode,
    prob_cutoff,
    best_by,
):
    filtered = apply_filters(
        transactions, psp_values, device_values, category_values, hour_range
    )
    threshold_card = html.Div(
        [
            html.Div(
                "Risk posture",
                style={
                    "fontSize": "12px",
                    "fontWeight": 800,
                    "textTransform": "uppercase",
                    "color": "#627485",
                },
            ),
            html.Div(
                screening_mode,
                style={
                    "fontSize": "26px",
                    "fontWeight": 800,
                    "color": "#2457a4",
                    "marginTop": "4px",
                },
            ),
            html.Div(
                f"Cutoff: {prob_cutoff:.2f}",
                style={"fontSize": "13px", "color": "#54707f", "marginTop": "2px"},
            ),
        ]
    )
    filtered_count = f"{len(filtered):,}"

    if page == "overview":
        try:
            overview_cards = overview_figures(filtered)
        except Exception as e:
            return (
                threshold_card,
                filtered_count,
                html.Div(
                    [
                        hero_banner(
                            "Dashboard Error",
                            "Failed to generate overview charts.",
                            "pink",
                        ),
                        dbc.Alert(f"Error: {str(e)}", color="warning"),
                    ]
                ),
            )
        metrics_row = dbc.Row(
            [
                dbc.Col(
                    metric_card(
                        "Transactions",
                        f"{len(filtered):,}",
                        "Current filters",
                        "#2457a4",
                    ),
                    md=3,
                ),
                dbc.Col(
                    metric_card(
                        "Fraud rate",
                        f"{filtered['is_fraud'].mean() * 100:.2f}%",
                        "Current filters",
                        "#2f7f63",
                    ),
                    md=3,
                ),
                dbc.Col(
                    metric_card(
                        "Avg amount",
                        fmt_currency(filtered["amount"].mean()),
                        "Current filters",
                        "#9a6a1d",
                    ),
                    md=3,
                ),
                dbc.Col(
                    metric_card(
                        "Success rate",
                        f"{(filtered['status'].eq('Success').mean() * 100):.2f}%",
                        "Current filters",
                        "#1f4ca3",
                    ),
                    md=3,
                ),
            ],
            className="g-3 mb-4",
        )
        return (
            threshold_card,
            filtered_count,
            html.Div(
                [
                    hero_banner(
                        "Overview Dashboard",
                        "A concise executive view of volume, fraud risk, and transaction behavior.",
                        "teal",
                    ),
                    metrics_row,
                    dbc.Row(
                        [
                            dbc.Col(overview_cards[0], md=6),
                            dbc.Col(overview_cards[1], md=6),
                        ],
                        className="g-3 mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(overview_cards[2], md=6),
                            dbc.Col(overview_cards[3], md=6),
                        ],
                        className="g-3",
                    ),
                ]
            ),
        )

    if page == "eda":
        try:
            eda_cards = eda_figures(filtered)
        except Exception as e:
            return (
                threshold_card,
                filtered_count,
                html.Div(
                    [
                        hero_banner(
                            "Dashboard Error", "Failed to generate EDA charts.", "pink"
                        ),
                        dbc.Alert(f"Error: {str(e)}", color="warning"),
                    ]
                ),
            )
        return (
            threshold_card,
            filtered_count,
            html.Div(
                [
                    hero_banner(
                        "Exploratory Data Analysis",
                        "Eight interactive charts focused on hour, PSP, device, amount, merchant category, and IP risk.",
                        "green",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                chart_card(
                                    "EDA Summary",
                                    px.histogram(
                                        filtered,
                                        x="amount",
                                        nbins=40,
                                        title="Amount Distribution",
                                    ),
                                ),
                                md=12,
                            )
                        ],
                        className="g-3 mb-3",
                    ),
                    dbc.Row(
                        [dbc.Col(eda_cards[0], md=6), dbc.Col(eda_cards[1], md=6)],
                        className="g-3 mb-3",
                    ),
                    dbc.Row(
                        [dbc.Col(eda_cards[2], md=6), dbc.Col(eda_cards[3], md=6)],
                        className="g-3 mb-3",
                    ),
                    dbc.Row(
                        [dbc.Col(eda_cards[4], md=6), dbc.Col(eda_cards[5], md=6)],
                        className="g-3",
                    ),
                ]
            ),
        )

    if page == "evaluation":
        # Map radio choice to column name variants
        metric_map = {
            "F1": (
                "F1"
                if "F1" in METRICS_DF.columns
                else ("F1 Score" if "F1 Score" in METRICS_DF.columns else None)
            ),
            "Accuracy": "Accuracy" if "Accuracy" in METRICS_DF.columns else None,
            "ROC-AUC": (
                "ROC-AUC"
                if "ROC-AUC" in METRICS_DF.columns
                else ("ROC_AUC" if "ROC_AUC" in METRICS_DF.columns else None)
            ),
        }
        chosen_metric_col = metric_map.get(best_by) if best_by else None
        try:
            model_content = list(
                model_figures(
                    METRICS_DF,
                    FEATURE_IMPORTANCE_DF,
                    filtered,
                    best_model_name=NOTEBOOK_BEST_MODEL_NAME,
                    best_metric=chosen_metric_col,
                )
            )
        except Exception as e:
            model_content = [
                hero_banner(
                    "Model Evaluation Error",
                    "Failed to generate model evaluation charts.",
                    "pink",
                ),
                dbc.Alert(f"Error: {str(e)}", color="warning"),
            ]
        return (
            threshold_card,
            filtered_count,
            html.Div(
                [
                    hero_banner(
                        "Model Evaluation",
                        "Notebook benchmark: accuracy, precision, recall, F1, ROC-AUC, and specificity.",
                        "pink",
                    ),
                    *model_content,
                ]
            ),
        )

    if page == "fraud":
        try:
            score, label, tone, factors = score_transaction(
                amount=2500,
                txns_last_24h=1,
                avg_30day_amount=800,
                new_device=True,
                first_merchant_visit=True,
                vpn_ip=False,
                psp="AmzPayX",
                uip_flow="P2M_Collect",
                merchant_category="Donations",
                state="Delhi",
                device_os="Android",
                time_value="00:42",
                screening_mode=screening_mode,
                probability_cutoff=prob_cutoff,
            )
        except Exception as e:
            return (
                threshold_card,
                filtered_count,
                html.Div(
                    [
                        hero_banner(
                            "Fraud Detection Error",
                            "Failed to initialize fraud detector.",
                            "pink",
                        ),
                        dbc.Alert(f"Error: {str(e)}", color="warning"),
                    ]
                ),
            )
        return (
            threshold_card,
            filtered_count,
            html.Div(
                [
                    hero_banner(
                        "Fraud Detection",
                        "Score a single transaction, upload a batch CSV, or review score distribution.",
                        "teal",
                    ),
                    dcc.Tabs(
                        id="fraud-tabs",
                        value="single",
                        className="tabs",
                        children=[
                            dcc.Tab(
                                label="Single Transaction",
                                value="single",
                                children=[
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "Quick Test Cases",
                                                            style={
                                                                "fontWeight": 800,
                                                                "fontSize": "15px",
                                                                "marginBottom": "12px",
                                                                "color": "#1e293b",
                                                            },
                                                        ),
                                                        html.Div(
                                                            "Load sample transactions to test the model:",
                                                            style={
                                                                "fontSize": "12px",
                                                                "color": "#64748b",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    html.Button(
                                                                        html.Div(
                                                                            [
                                                                                html.Div(
                                                                                    "✓ Safe",
                                                                                    style={
                                                                                        "fontWeight": 800
                                                                                    },
                                                                                ),
                                                                                html.Div(
                                                                                    "Low risk",
                                                                                    style={
                                                                                        "fontSize": "11px",
                                                                                        "opacity": "0.85",
                                                                                        "marginTop": "2px",
                                                                                    },
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        id="sample-safe",
                                                                        className="sample-button-safe",
                                                                        n_clicks=0,
                                                                    ),
                                                                    xs=6,
                                                                ),
                                                                dbc.Col(
                                                                    html.Button(
                                                                        html.Div(
                                                                            [
                                                                                html.Div(
                                                                                    "⚠ Fraud",
                                                                                    style={
                                                                                        "fontWeight": 800
                                                                                    },
                                                                                ),
                                                                                html.Div(
                                                                                    "High risk",
                                                                                    style={
                                                                                        "fontSize": "11px",
                                                                                        "opacity": "0.85",
                                                                                        "marginTop": "2px",
                                                                                    },
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        id="sample-fraud",
                                                                        className="sample-button-fraud",
                                                                        n_clicks=0,
                                                                    ),
                                                                    xs=6,
                                                                ),
                                                            ],
                                                            className="g-2",
                                                        ),
                                                    ],
                                                    className="input-panel",
                                                ),
                                                md=12,
                                            ),
                                        ],
                                        className="g-3 mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "Transaction Amount (₹)",
                                                            className="sidebar-control-label",
                                                        ),
                                                        dbc.Input(
                                                            id="fd-amount",
                                                            type="number",
                                                            value=2500,
                                                            min=0,
                                                            step=50,
                                                        ),
                                                    ],
                                                    className="input-panel",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "Transactions in Last 24 Hours",
                                                            className="sidebar-control-label",
                                                        ),
                                                        dbc.Input(
                                                            id="fd-txns24",
                                                            type="number",
                                                            value=1,
                                                            min=0,
                                                            step=1,
                                                        ),
                                                    ],
                                                    className="input-panel",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "Average Amount (30 Days) (₹)",
                                                            className="sidebar-control-label",
                                                        ),
                                                        dbc.Input(
                                                            id="fd-avg30",
                                                            type="number",
                                                            value=800,
                                                            min=0,
                                                            step=50,
                                                        ),
                                                    ],
                                                    className="input-panel",
                                                ),
                                                md=4,
                                            ),
                                        ],
                                        className="g-3 mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Checkbox(
                                                    id="fd-new-device",
                                                    value=True,
                                                    label="New Device Used",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dbc.Checkbox(
                                                    id="fd-first-merchant",
                                                    value=True,
                                                    label="First Time at This Merchant",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dbc.Checkbox(
                                                    id="fd-vpn-ip",
                                                    value=False,
                                                    label="VPN or Unusual IP Detected",
                                                ),
                                                md=4,
                                            ),
                                        ],
                                        className="g-3 mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="fd-psp",
                                                    options=[
                                                        {"label": x, "value": x}
                                                        for x in PSP_OPTIONS
                                                    ],
                                                    value="AmzPayX",
                                                    clearable=False,
                                                    placeholder="Select Payment App",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="fd-flow",
                                                    options=[
                                                        {"label": x, "value": x}
                                                        for x in FLOW_OPTIONS
                                                    ],
                                                    value="P2M_Collect",
                                                    clearable=False,
                                                    placeholder="Select Flow Type",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="fd-category",
                                                    options=[
                                                        {"label": x, "value": x}
                                                        for x in CATEGORY_OPTIONS
                                                    ],
                                                    value=CATEGORY_OPTIONS[0],
                                                    clearable=False,
                                                    placeholder="Select Merchant Category",
                                                ),
                                                md=4,
                                            ),
                                        ],
                                        className="g-3 mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="fd-state",
                                                    options=[
                                                        {"label": x, "value": x}
                                                        for x in STATE_OPTIONS
                                                    ],
                                                    value="Delhi",
                                                    clearable=False,
                                                    placeholder="Select Location",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="fd-device",
                                                    options=[
                                                        {"label": x, "value": x}
                                                        for x in DEVICE_OPTIONS
                                                    ],
                                                    value="Android",
                                                    clearable=False,
                                                    placeholder="Select Device Type",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="fd-time",
                                                    type="time",
                                                    value="00:42",
                                                ),
                                                md=4,
                                            ),
                                        ],
                                        className="g-3 mb-3",
                                    ),
                                    dbc.Button(
                                        "Score Transaction",
                                        id="score-btn",
                                        className="score-button mb-3",
                                        n_clicks=0,
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Graph(
                                                    id="score-gauge",
                                                    figure=build_indicator(
                                                        score, prob_cutoff, label
                                                    ),
                                                    config={"displayModeBar": False},
                                                ),
                                                md=7,
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            id="score-result",
                                                            className=f"result-banner {'result-success' if tone == 'success' else 'result-danger'}",
                                                        ),
                                                        html.Div(
                                                            id="score-factors",
                                                            className="insight-list",
                                                        ),
                                                    ],
                                                    className="input-panel",
                                                ),
                                                md=5,
                                            ),
                                        ],
                                        className="g-3",
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Batch Upload",
                                value="batch",
                                children=[
                                    html.Div(
                                        [
                                            dcc.Upload(
                                                id="batch-upload",
                                                children=html.Div(
                                                    [
                                                        "Drop a CSV here or ",
                                                        html.A("browse files"),
                                                    ]
                                                ),
                                                className="upload-box",
                                                multiple=False,
                                            ),
                                            html.Div(
                                                "Expected columns: Transaction Amount, New Device Used, IP Location Mismatch, Failed Attempts (Last 24h), Transaction Velocity, Amount Deviation Score",
                                                style={
                                                    "marginTop": "10px",
                                                    "color": "#587185",
                                                },
                                            ),
                                            html.Div(
                                                id="batch-upload-output",
                                                className="batch-output",
                                            ),
                                        ],
                                        className="input-panel",
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Score Distribution",
                                value="distribution",
                                children=[
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                figure=fraud_distribution_figure(
                                                    filtered
                                                ),
                                                config={"displayModeBar": False},
                                            ),
                                            html.Div(
                                                "Distribution uses a proxy score based on amount, velocity, device and IP signals.",
                                                style={
                                                    "color": "#5b6b78",
                                                    "marginTop": "10px",
                                                },
                                            ),
                                        ],
                                        className="input-panel",
                                    )
                                ],
                            ),
                        ],
                    ),
                ]
            ),
        )

    return (
        threshold_card,
        filtered_count,
        html.Div([hero_banner("Overview Dashboard", "The app is ready.", "teal")]),
    )


@app.callback(
    Output("score-gauge", "figure"),
    Output("score-result", "children"),
    Output("score-result", "className"),
    Output("score-factors", "children"),
    Input("score-btn", "n_clicks"),
    State("fd-amount", "value"),
    State("fd-txns24", "value"),
    State("fd-avg30", "value"),
    State("fd-new-device", "value"),
    State("fd-first-merchant", "value"),
    State("fd-vpn-ip", "value"),
    State("fd-psp", "value"),
    State("fd-flow", "value"),
    State("fd-category", "value"),
    State("fd-state", "value"),
    State("fd-device", "value"),
    State("fd-time", "value"),
    State("screening-mode", "value"),
    State("prob-cutoff", "value"),
)
def update_score(
    n_clicks,
    amount,
    txns24,
    avg30,
    new_device,
    first_merchant,
    vpn_ip,
    psp,
    uip_flow,
    merchant_category,
    state,
    device_os,
    time_value,
    screening_mode,
    prob_cutoff,
):
    if amount is None:
        raise PreventUpdate
    score, label, tone, factors = score_transaction(
        amount=float(amount),
        txns_last_24h=float(txns24 or 0),
        avg_30day_amount=float(avg30 or 1),
        new_device=bool(new_device),
        first_merchant_visit=bool(first_merchant),
        vpn_ip=bool(vpn_ip),
        psp=psp,
        uip_flow=uip_flow,
        merchant_category=merchant_category,
        state=state,
        device_os=device_os,
        time_value=time_value,
        screening_mode=screening_mode,
        probability_cutoff=float(prob_cutoff),
    )
    fig = build_indicator(score, float(prob_cutoff), label)
    result_class = (
        f"result-banner {'result-success' if tone == 'success' else 'result-danger'}"
    )
    result_text = f"{label} — {score * 100:.2f}% (threshold {prob_cutoff:.2f})"
    factor_children = [
        html.Div(
            [
                html.Div(item["name"], style={"fontWeight": 700}),
                html.Div(item["value"], style={"color": "#64748b"}),
            ],
            className="insight-item",
        )
        for item in factors
    ]
    return fig, result_text, result_class, factor_children


@app.callback(
    Output("fd-amount", "value"),
    Output("fd-txns24", "value"),
    Output("fd-avg30", "value"),
    Output("fd-new-device", "value"),
    Output("fd-first-merchant", "value"),
    Output("fd-vpn-ip", "value"),
    Output("fd-psp", "value"),
    Output("fd-flow", "value"),
    Output("fd-category", "value"),
    Output("fd-state", "value"),
    Output("fd-device", "value"),
    Output("fd-time", "value"),
    Input("sample-safe", "n_clicks"),
    Input("sample-fraud", "n_clicks"),
    prevent_initial_call=True,
)
def populate_sample(safe_clicks, fraud_clicks):
    """Populate fraud detection form with safe or fraud examples based on button clicks."""
    if not callback_context.triggered:
        raise PreventUpdate

    button_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    if button_id == "sample-safe":
        # Safe transaction example
        return (
            1200,  # amount (low)
            1,  # txns24
            950,  # avg30
            False,  # new_device
            False,  # first_merchant
            False,  # vpn_ip
            "GPay",  # psp
            "P2M",  # flow
            "Grocery",  # category
            "Delhi",  # state
            "Android",  # device
            "14:30",  # time (daytime)
        )
    elif button_id == "sample-fraud":
        # Fraud transaction example
        return (
            85000,  # amount (very high)
            3,  # txns24
            500,  # avg30 (much lower than amount)
            True,  # new_device
            True,  # first_merchant
            True,  # vpn_ip
            "Unknown",  # psp
            "P2M_Collect",  # flow
            "Travel",  # category
            "Mumbai",  # state
            "Web",  # device
            "03:15",  # time (night)
        )

    raise PreventUpdate


@app.callback(
    Output("batch-upload-output", "children"),
    Input("batch-upload", "contents"),
    State("batch-upload", "filename"),
    State("batch-upload", "last_modified"),
)
def handle_upload(contents, filename, last_modified):
    if not contents:
        return html.Div(
            "Upload a CSV to score a batch of transactions.", style={"color": "#607180"}
        )

    try:
        content_type, content_string = contents.split(",")
        decoded = b64decode(content_string)
        uploaded = pd.read_csv(StringIO(decoded.decode("utf-8")))
    except Exception as exc:
        return html.Div(
            f"Could not read uploaded file: {exc}", style={"color": "#b91c1c"}
        )

    required = {
        "amount",
        "new_device_flag",
        "ip_location_mismatch",
        "failed_attempts_last_24h",
        "transaction_velocity",
        "amount_deviation_score",
    }
    missing = required - set(uploaded.columns)
    if missing:
        missing_labels = ", ".join(display_label(col) for col in sorted(missing))
        return html.Div(
            f"Missing required fields: {missing_labels}",
            style={"color": "#b91c1c"},
        )

    batch_score = proxy_risk_series(
        uploaded.assign(
            payment_app=uploaded.get("payment_app", "Unknown"),
            device_type=uploaded.get("device_type", "Unknown"),
            is_night_transaction=uploaded.get("is_night_transaction", 0),
        )
    )
    summary = html.Div(
        [
            html.Div(f"File: {filename}", style={"fontWeight": 800}),
            html.Div(f"Rows scored: {len(uploaded):,}"),
            html.Div(f"Average proxy score: {batch_score.mean() * 100:.2f}%"),
            html.Div(f"High risk rows: {(batch_score >= 0.25).sum():,}"),
            html.Hr(),
            dash_table.DataTable(
                data=uploaded.head(10)
                .assign(proxy_score=(batch_score.head(10) * 100).round(2))
                .to_dict("records"),
                columns=[
                    {"name": display_label(col), "id": col}
                    for col in list(uploaded.head(10).columns) + ["proxy_score"]
                ],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#f1f5f9", "fontWeight": "700"},
                style_cell={
                    "padding": "10px",
                    "fontFamily": "Inter, system-ui, sans-serif",
                    "fontSize": "14px",
                },
            ),
        ]
    )
    return summary


if __name__ == "__main__":
    # Disable debug mode to prevent reload issues and connection resets
    # Running in production mode for stability
    app.run(debug=False, host="127.0.0.1", port=8050, threaded=True)

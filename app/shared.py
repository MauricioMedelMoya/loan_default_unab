from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"
METRICS_PATH = ROOT_DIR / ".generated_assets" / "metricas_modelo_final.csv"
DB_PATH = BASE_DIR / "prediction_monitoring.db"
API_HOST = "127.0.0.1"
API_PORT = 8051
API_URL = f"http://{API_HOST}:{API_PORT}"

FEATURE_FIELDS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

NUMERIC_FIELDS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORY_OPTIONS = {
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": [
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
    ],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}

DEFAULT_FEATURES = {
    "person_age": 30,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 11.0,
    "loan_percent_income": 0.20,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6,
}

TOP_VARIABLES = [
    {
        "name": "loan_percent_income",
        "description": "Mide cuanto del ingreso queda comprometido por el prestamo.",
    },
    {
        "name": "person_income",
        "description": "Resume la capacidad de pago general del solicitante.",
    },
    {
        "name": "loan_int_rate",
        "description": "Captura el costo financiero asociado al credito.",
    },
]

metadata = joblib.load(METADATA_PATH) if METADATA_PATH.exists() else {"threshold": 0.5}
THRESHOLD = float(metadata.get("threshold", 0.5))

if METRICS_PATH.exists():
    metrics_row = pd.read_csv(METRICS_PATH).iloc[0].to_dict()
else:
    metrics_row = metadata.get("metrics", {})

MODEL_NAME = str(metrics_row.get("Modelo", "Random Forest"))
MODEL_METRICS = {
    "Accuracy": float(metrics_row.get("Accuracy", 0)),
    "Precision": float(metrics_row.get("Precision", 0)),
    "Recall": float(metrics_row.get("Recall", 0)),
    "F1-score": float(metrics_row.get("F1", 0)),
    "ROC-AUC": float(metrics_row.get("ROC_AUC", 0)),
}


def load_model():
    return joblib.load(MODEL_PATH)


def build_feature_frame(payload: dict) -> pd.DataFrame:
    row = DEFAULT_FEATURES.copy()
    row.update(payload)
    return pd.DataFrame([row], columns=FEATURE_FIELDS)

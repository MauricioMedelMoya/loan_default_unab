from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from flask import Flask, request

from shared import (
    API_HOST,
    API_PORT,
    CATEGORY_OPTIONS,
    DB_PATH,
    DEFAULT_FEATURES,
    FEATURE_FIELDS,
    MODEL_METRICS,
    MODEL_NAME,
    NUMERIC_FIELDS,
    THRESHOLD,
    TOP_VARIABLES,
    build_feature_frame,
    load_model,
)


MODEL = load_model()
BACKEND_APP = Flask(__name__)


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                person_age REAL NOT NULL,
                person_income REAL NOT NULL,
                person_home_ownership TEXT NOT NULL,
                person_emp_length REAL NOT NULL,
                loan_intent TEXT NOT NULL,
                loan_grade TEXT NOT NULL,
                loan_amnt REAL NOT NULL,
                loan_int_rate REAL NOT NULL,
                loan_percent_income REAL NOT NULL,
                cb_person_default_on_file TEXT NOT NULL,
                cb_person_cred_hist_length REAL NOT NULL,
                probability REAL NOT NULL,
                threshold_value REAL NOT NULL,
                risk_label TEXT NOT NULL
            )
            """
        )


def normalize_payload(payload: dict) -> tuple[dict, list[str]]:
    errors: list[str] = []
    normalized: dict = {}

    for field in FEATURE_FIELDS:
        value = payload.get(field, DEFAULT_FEATURES.get(field))
        if value is None:
            errors.append(f"Falta el campo {field}.")
            continue

        if field in NUMERIC_FIELDS:
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"El campo {field} debe ser numerico.")
                continue

            if number < 0:
                errors.append(f"El campo {field} no puede ser negativo.")
                continue

            if field == "loan_percent_income" and number > 1:
                errors.append("loan_percent_income debe estar entre 0 y 1.")
                continue

            normalized[field] = number
            continue

        if value not in CATEGORY_OPTIONS[field]:
            errors.append(
                f"El campo {field} debe ser una de estas opciones: "
                + ", ".join(CATEGORY_OPTIONS[field])
            )
            continue

        normalized[field] = value

    return normalized, errors


def insert_prediction(features: dict, probability: float, risk_label: str) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    record = {
        "created_at": timestamp,
        **features,
        "probability": probability,
        "threshold_value": THRESHOLD,
        "risk_label": risk_label,
    }

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO prediction_logs (
                created_at,
                person_age,
                person_income,
                person_home_ownership,
                person_emp_length,
                loan_intent,
                loan_grade,
                loan_amnt,
                loan_int_rate,
                loan_percent_income,
                cb_person_default_on_file,
                cb_person_cred_hist_length,
                probability,
                threshold_value,
                risk_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["created_at"],
                record["person_age"],
                record["person_income"],
                record["person_home_ownership"],
                record["person_emp_length"],
                record["loan_intent"],
                record["loan_grade"],
                record["loan_amnt"],
                record["loan_int_rate"],
                record["loan_percent_income"],
                record["cb_person_default_on_file"],
                record["cb_person_cred_hist_length"],
                record["probability"],
                record["threshold_value"],
                record["risk_label"],
            ),
        )
        record["id"] = cursor.lastrowid

    return record


def fetch_recent_predictions(limit: int = 5) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                person_income,
                loan_amnt,
                loan_grade,
                probability,
                risk_label
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def build_summary() -> dict:
    with get_connection() as connection:
        totals = connection.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                AVG(probability) AS average_probability,
                SUM(CASE WHEN risk_label = 'Alto riesgo' THEN 1 ELSE 0 END) AS high_risk_predictions,
                MAX(created_at) AS last_prediction_at
            FROM prediction_logs
            """
        ).fetchone()

    total_predictions = int(totals["total_predictions"] or 0)
    high_risk_predictions = int(totals["high_risk_predictions"] or 0)
    average_probability = float(totals["average_probability"] or 0)
    high_risk_rate = (
        float(high_risk_predictions / total_predictions) if total_predictions else 0.0
    )

    return {
        "total_predictions": total_predictions,
        "average_probability": average_probability,
        "high_risk_predictions": high_risk_predictions,
        "high_risk_rate": high_risk_rate,
        "last_prediction_at": totals["last_prediction_at"],
    }


@BACKEND_APP.get("/health")
def health() -> tuple[dict, int]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "database": str(DB_PATH),
    }, 200


@BACKEND_APP.get("/metrics")
def metrics() -> tuple[dict, int]:
    return {
        "model_name": MODEL_NAME,
        "threshold": THRESHOLD,
        "metrics": MODEL_METRICS,
        "top_variables": TOP_VARIABLES,
        "feature_fields": FEATURE_FIELDS,
    }, 200


@BACKEND_APP.get("/monitoring/summary")
def monitoring_summary() -> tuple[dict, int]:
    return build_summary(), 200


@BACKEND_APP.get("/monitoring/recent")
def monitoring_recent() -> tuple[dict, int]:
    limit = request.args.get("limit", default=5, type=int) or 5
    limit = max(1, min(limit, 20))
    return {"items": fetch_recent_predictions(limit)}, 200


@BACKEND_APP.post("/predict")
def predict() -> tuple[dict, int]:
    payload = request.get_json(silent=True) or {}
    features, errors = normalize_payload(payload)
    if errors:
        return {"status": "error", "errors": errors}, 400

    feature_frame = build_feature_frame(features)
    probability = float(MODEL.predict_proba(feature_frame)[0, 1])
    risk_label = "Alto riesgo" if probability >= THRESHOLD else "Bajo riesgo"
    record = insert_prediction(features, probability, risk_label)

    return {
        "status": "ok",
        "prediction_id": record["id"],
        "created_at": record["created_at"],
        "probability": probability,
        "threshold": THRESHOLD,
        "risk_label": risk_label,
        "target_meaning": "loan_status = 1 corresponde a incumplimiento",
    }, 200


def run() -> None:
    init_db()
    BACKEND_APP.run(host=API_HOST, port=API_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    run()

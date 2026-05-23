from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import error, request

import plotly.graph_objects as go
from dash import MATCH, Dash, Input, Output, State, ctx, dcc, html, no_update

from shared import (
    API_HOST,
    API_PORT,
    API_URL,
    CATEGORY_OPTIONS,
    DEFAULT_FEATURES,
)


BASE_DIR = Path(__file__).resolve().parent
BACKEND_PATH = BASE_DIR / "backend.py"
APP_TITLE = "Loan Default Monitoring"


def api_get(path: str) -> dict:
    with request.urlopen(f"{API_URL}{path}", timeout=4) as response:
        return json.loads(response.read().decode("utf-8"))


def api_post(path: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{API_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=8) as response:
            return response.getcode(), json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8")
        return exc.code, json.loads(payload or "{}")


def is_api_available() -> bool:
    try:
        api_get("/health")
        return True
    except Exception:
        return False


def ensure_api_running() -> None:
    if is_api_available():
        return

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [sys.executable, str(BACKEND_PATH)],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )

    for _ in range(20):
        if is_api_available():
            return
        time.sleep(0.3)

    raise RuntimeError(
        f"No fue posible iniciar la API en http://{API_HOST}:{API_PORT}."
    )


def safe_api_get(path: str, default: dict) -> dict:
    try:
        return api_get(path)
    except Exception:
        return default


ensure_api_running()
INITIAL_SUMMARY = safe_api_get(
    "/monitoring/summary",
    {
        "total_predictions": 0,
        "average_probability": 0.0,
        "high_risk_predictions": 0,
        "high_risk_rate": 0.0,
        "last_prediction_at": None,
    },
)
INITIAL_RECENT = safe_api_get("/monitoring/recent?limit=5", {"items": []})
INITIAL_METRICS = safe_api_get(
    "/metrics",
    {
        "model_name": "Random Forest",
        "threshold": 0.5,
        "metrics": {
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1-score": 0.0,
            "ROC-AUC": 0.0,
        },
        "top_variables": [
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
        ],
    },
)


def build_metrics_figure(metrics: dict) -> go.Figure:
    labels = list(metrics.keys())
    values = [float(value) for value in metrics.values()]

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=["#1d4ed8", "#0f766e", "#f59e0b", "#0f172a", "#7c3aed"],
                line=dict(color="#ffffff", width=1.2),
            ),
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#e2e8f0"),
        showlegend=False,
        font=dict(color="#0f172a"),
    )
    return figure


PERFORMANCE_FIGURE = build_metrics_figure(INITIAL_METRICS["metrics"])

app = Dash(__name__)
app.title = APP_TITLE
server = app.server
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            input[type="number"]::-webkit-outer-spin-button,
            input[type="number"]::-webkit-inner-spin-button {
                -webkit-appearance: none;
                margin: 0;
            }

            input[type="number"] {
                -moz-appearance: textfield;
                appearance: textfield;
            }
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
</html>
"""


APP_STYLE = {
    "minHeight": "100vh",
    "background": "linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%)",
    "padding": "28px 18px 40px",
    "fontFamily": '"Segoe UI", "Helvetica Neue", Arial, sans-serif',
    "color": "#0f172a",
}
WRAP_STYLE = {"maxWidth": "1180px", "margin": "0 auto"}
CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "22px",
    "padding": "28px",
    "border": "1px solid #e2e8f0",
    "boxShadow": "0 24px 50px rgba(15, 23, 42, 0.08)",
}
LABEL_STYLE = {
    "display": "block",
    "marginBottom": "8px",
    "fontWeight": "600",
    "color": "#0f172a",
}
INPUT_STYLE = {
    "width": "100%",
    "padding": "14px 14px",
    "minHeight": "50px",
    "borderRadius": "14px",
    "border": "1px solid #cbd5e1",
    "backgroundColor": "#f8fafc",
    "fontSize": "15px",
    "boxSizing": "border-box",
}
BUTTON_STYLE = {
    "width": "100%",
    "padding": "15px 18px",
    "border": "none",
    "borderRadius": "16px",
    "background": "linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%)",
    "color": "#ffffff",
    "fontSize": "15px",
    "fontWeight": "700",
    "cursor": "pointer",
    "boxShadow": "0 16px 30px rgba(29, 78, 216, 0.22)",
}
STEPPER_BUTTON_STYLE = {
    "width": "46px",
    "height": "50px",
    "border": "1px solid #cbd5e1",
    "backgroundColor": "#ffffff",
    "borderRadius": "14px",
    "fontSize": "22px",
    "fontWeight": "700",
    "color": "#0f172a",
    "cursor": "pointer",
    "lineHeight": "1",
}


def metric_card(title: str, value: str, tone: str = "#0f172a") -> html.Div:
    return html.Div(
        [
            html.Div(
                title,
                style={"fontSize": "13px", "color": "#64748b",
                       "marginBottom": "8px"},
            ),
            html.Div(
                value,
                style={"fontSize": "28px", "fontWeight": "800", "color": tone},
            ),
        ],
        style={
            "padding": "18px",
            "borderRadius": "18px",
            "backgroundColor": "#f8fbff",
            "border": "1px solid #dbeafe",
        },
    )


def numeric_input(component_id: str, label: str, value, minimum=0, maximum=None, step=None):
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            html.Div(
                [
                    html.Button(
                        "-",
                        id={"type": "number-minus", "field": component_id},
                        n_clicks=0,
                        style=STEPPER_BUTTON_STYLE,
                    ),
                    dcc.Input(
                        id={"type": "number-input", "field": component_id},
                        type="text",
                        inputMode="decimal",
                        value=value,
                        min=minimum,
                        max=maximum,
                        step=step,
                        style={**INPUT_STYLE, "textAlign": "center"},
                    ),
                    html.Button(
                        "+",
                        id={"type": "number-plus", "field": component_id},
                        n_clicks=0,
                        style=STEPPER_BUTTON_STYLE,
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "46px minmax(0, 1fr) 46px",
                    "gap": "10px",
                    "alignItems": "center",
                },
            ),
        ]
    )


def dropdown_input(component_id: str, label: str, options: list[str], value: str):
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            dcc.Dropdown(
                id=component_id,
                options=[{"label": item, "value": item} for item in options],
                value=value,
                clearable=False,
                style={"fontSize": "15px"},
            ),
        ]
    )


def format_timestamp(value: str | None) -> str:
    if not value:
        return "Sin registros"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%d-%m-%Y %H:%M")
    except ValueError:
        return value


def render_result_box(message: str, color: str = "#475569") -> list:
    return [
        html.P(
            message,
            style={"margin": 0, "color": color,
                   "fontSize": "16px", "lineHeight": "1.7"},
        )
    ]


def render_recent_table(items: list[dict]) -> html.Div:
    if not items:
        return html.Div(
            "Aun no hay predicciones almacenadas. El historial aparecera aqui.",
            style={"color": "#64748b", "fontSize": "15px"},
        )

    header_cells = [
        "Fecha",
        "Ingreso",
        "Prestamo",
        "Grade",
        "Probabilidad",
        "Clase",
    ]
    rows = []
    for item in items:
        rows.append(
            html.Tr(
                [
                    html.Td(format_timestamp(item.get("created_at")),
                            style={"padding": "12px 10px"}),
                    html.Td(f"{float(item.get('person_income', 0)):,.0f}", style={
                            "padding": "12px 10px"}),
                    html.Td(f"{float(item.get('loan_amnt', 0)):,.0f}",
                            style={"padding": "12px 10px"}),
                    html.Td(item.get("loan_grade", "-"),
                            style={"padding": "12px 10px"}),
                    html.Td(f"{float(item.get('probability', 0)):.2%}",
                            style={"padding": "12px 10px"}),
                    html.Td(item.get("risk_label", "-"),
                            style={"padding": "12px 10px", "fontWeight": "700"}),
                ]
            )
        )

    return html.Div(
        style={"overflowX": "auto"},
        children=[
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(
                                    label,
                                    style={
                                        "textAlign": "left",
                                        "padding": "10px",
                                        "fontSize": "13px",
                                        "color": "#475569",
                                        "borderBottom": "1px solid #e2e8f0",
                                    },
                                )
                                for label in header_cells
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                style={"width": "100%", "borderCollapse": "collapse"},
            )
        ],
    )


app.layout = html.Div(
    style=APP_STYLE,
    children=[
        dcc.Interval(id="monitor_interval", interval=8000, n_intervals=0),
        dcc.Store(id="prediction_signal"),
        html.Div(
            style=WRAP_STYLE,
            children=[
                html.Div(
                    style={"marginBottom": "24px"},
                    children=[
                        html.Div(
                            "Prediccion de incumplimiento",
                            style={
                                "display": "inline-block",
                                "padding": "7px 12px",
                                "borderRadius": "999px",
                                "backgroundColor": "#dbeafe",
                                "color": "#1d4ed8",
                                "fontWeight": "700",
                                "fontSize": "12px",
                                "letterSpacing": "0.04em",
                                "textTransform": "uppercase",
                                "marginBottom": "14px",
                            },
                        ),
                        html.H1(
                            "Sistema de evaluacion de riesgo crediticio",
                            style={"margin": "0 0 10px 0",
                                   "fontSize": "2.3rem", "lineHeight": "1.15"},
                        ),
                        html.P(
                            "Ingresa los datos del solicitante, estima la probabilidad de default y revisa el desempeno general del modelo junto con el historial reciente de predicciones.",
                            style={"margin": 0, "fontSize": "16px",
                                   "lineHeight": "1.8", "color": "#334155"},
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(320px, 1.08fr) minmax(300px, 0.92fr)",
                        "gap": "22px",
                    },
                    children=[
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H2("Datos del solicitante",
                                        style={"marginTop": 0}),
                                html.P(
                                    "Usa los botones laterales o escribe directamente los valores para simular distintos perfiles de riesgo.",
                                    style={
                                        "marginTop": 0, "color": "#475569", "lineHeight": "1.7"},
                                ),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                                        "gap": "14px",
                                    },
                                    children=[
                                        numeric_input(
                                            "person_age", "Edad", DEFAULT_FEATURES["person_age"]),
                                        numeric_input(
                                            "person_income", "Ingreso anual", DEFAULT_FEATURES["person_income"]),
                                        dropdown_input(
                                            "person_home_ownership",
                                            "Tipo de vivienda",
                                            CATEGORY_OPTIONS["person_home_ownership"],
                                            DEFAULT_FEATURES["person_home_ownership"],
                                        ),
                                        numeric_input(
                                            "person_emp_length",
                                            "Anios de empleo",
                                            DEFAULT_FEATURES["person_emp_length"],
                                            step=0.5,
                                        ),
                                        dropdown_input(
                                            "loan_intent",
                                            "Proposito del prestamo",
                                            CATEGORY_OPTIONS["loan_intent"],
                                            DEFAULT_FEATURES["loan_intent"],
                                        ),
                                        dropdown_input(
                                            "loan_grade",
                                            "Grade del prestamo",
                                            CATEGORY_OPTIONS["loan_grade"],
                                            DEFAULT_FEATURES["loan_grade"],
                                        ),
                                        numeric_input(
                                            "loan_amnt", "Monto del prestamo", DEFAULT_FEATURES["loan_amnt"]),
                                        numeric_input(
                                            "loan_int_rate",
                                            "Tasa de interes",
                                            DEFAULT_FEATURES["loan_int_rate"],
                                            step=0.1,
                                        ),
                                        numeric_input(
                                            "loan_percent_income",
                                            "Porcentaje del ingreso comprometido",
                                            DEFAULT_FEATURES["loan_percent_income"],
                                            maximum=1,
                                            step=0.01,
                                        ),
                                        dropdown_input(
                                            "cb_person_default_on_file",
                                            "Default previo",
                                            CATEGORY_OPTIONS["cb_person_default_on_file"],
                                            DEFAULT_FEATURES["cb_person_default_on_file"],
                                        ),
                                        numeric_input(
                                            "cb_person_cred_hist_length",
                                            "Historial crediticio (anios)",
                                            DEFAULT_FEATURES["cb_person_cred_hist_length"],
                                        ),
                                    ],
                                ),
                                html.Button(
                                    "Predecir riesgo",
                                    id="predict_button",
                                    n_clicks=0,
                                    style={**BUTTON_STYLE,
                                           "marginTop": "20px"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={**CARD_STYLE,
                                   "display": "grid", "gap": "18px"},
                            children=[
                                html.Div(
                                    [
                                        html.H2("Resultado", style={
                                                "margin": "0 0 8px 0"}),
                                        html.P(
                                            "La clase positiva es loan_status = 1, es decir, incumplimiento del prestamo.",
                                            style={
                                                "margin": 0, "color": "#475569", "lineHeight": "1.7"},
                                        ),
                                    ]
                                ),
                                dcc.Loading(
                                    color="#1d4ed8",
                                    children=html.Div(
                                        id="result_box",
                                        style={
                                            "background": "linear-gradient(180deg, #f8fbff 0%, #eff6ff 100%)",
                                            "border": "1px solid #dbeafe",
                                            "borderRadius": "18px",
                                            "padding": "24px",
                                            "minHeight": "250px",
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "justifyContent": "center",
                                        },
                                        children=render_result_box(
                                            "Ingresa los valores y presiona el boton para generar una prediccion."
                                        ),
                                    ),
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "Explicabilidad destacada",
                                            style={"fontWeight": "700",
                                                   "marginBottom": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            item["name"],
                                                            style={
                                                                "fontWeight": "700", "marginBottom": "4px"},
                                                        ),
                                                        html.Div(
                                                            item["description"],
                                                            style={
                                                                "color": "#64748b", "lineHeight": "1.6"},
                                                        ),
                                                    ],
                                                    style={
                                                        "padding": "14px 16px",
                                                        "borderRadius": "16px",
                                                        "backgroundColor": "#f8fafc",
                                                        "border": "1px solid #e2e8f0",
                                                    },
                                                )
                                                for item in INITIAL_METRICS["top_variables"]
                                            ],
                                            style={"display": "grid",
                                                   "gap": "10px"},
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                           "gap": "22px", "marginTop": "22px"},
                    children=[
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H2("Desempeno del modelo",
                                        style={"marginTop": 0}),
                                html.P(
                                    (
                                        f"{INITIAL_METRICS['model_name']} mantiene un threshold de "
                                        f"{float(INITIAL_METRICS['threshold']):.2f}. El grafico resume "
                                        "las metricas principales obtenidas en la evaluacion final."
                                    ),
                                    style={"color": "#475569",
                                           "lineHeight": "1.7"},
                                ),
                                dcc.Graph(
                                    figure=PERFORMANCE_FIGURE,
                                    config={"displayModeBar": False, "responsive": True},
                                    style={"height": "340px"},
                                ),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(auto-fit, minmax(140px, 1fr))",
                                        "gap": "12px",
                                    },
                                    children=[
                                        metric_card(name, f"{value:.4f}")
                                        for name, value in INITIAL_METRICS["metrics"].items()
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H2("Monitoreo persistente",
                                        style={"marginTop": 0}),
                                html.P(
                                    "Cada inferencia queda almacenada en SQLite, lo que permite seguir volumen de uso, distribucion de riesgo y ultimas predicciones.",
                                    style={"color": "#475569",
                                           "lineHeight": "1.7"},
                                ),
                                html.Div(
                                    id="summary_cards",
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
                                        "gap": "12px",
                                        "marginBottom": "18px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "Ultimas predicciones",
                                            style={"fontWeight": "700",
                                                   "marginBottom": "10px"},
                                        ),
                                        html.Div(id="recent_predictions"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("result_box", "children"),
    Output("prediction_signal", "data"),
    Input("predict_button", "n_clicks"),
    State({"type": "number-input", "field": "person_age"}, "value"),
    State({"type": "number-input", "field": "person_income"}, "value"),
    State("person_home_ownership", "value"),
    State({"type": "number-input", "field": "person_emp_length"}, "value"),
    State("loan_intent", "value"),
    State("loan_grade", "value"),
    State({"type": "number-input", "field": "loan_amnt"}, "value"),
    State({"type": "number-input", "field": "loan_int_rate"}, "value"),
    State({"type": "number-input", "field": "loan_percent_income"}, "value"),
    State("cb_person_default_on_file", "value"),
    State({"type": "number-input", "field": "cb_person_cred_hist_length"}, "value"),
    prevent_initial_call=True,
)
def request_prediction(
    n_clicks,
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
):
    payload = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }

    try:
        status_code, response = api_post("/predict", payload)
    except Exception:
        return render_result_box(
            "No fue posible comunicarse con la API de prediccion.",
            "#b91c1c",
        ), {"updated_at": time.time()}

    if status_code >= 400:
        errors = response.get("errors") or ["La API rechazo la solicitud."]
        return render_result_box(" ".join(errors), "#b91c1c"), {
            "updated_at": time.time()
        }

    probability = float(response["probability"])
    risk_label = response["risk_label"]
    tone = "#b91c1c" if risk_label == "Alto riesgo" else "#047857"
    tone_bg = "#fef2f2" if risk_label == "Alto riesgo" else "#ecfdf5"

    children = [
        html.Div(
            risk_label,
            style={
                "display": "inline-flex",
                "alignItems": "center",
                "padding": "7px 12px",
                "borderRadius": "999px",
                "backgroundColor": tone_bg,
                "color": tone,
                "fontWeight": "800",
                "marginBottom": "16px",
            },
        ),
        html.Div(
            "Probabilidad estimada de default",
            style={"fontSize": "14px", "color": "#64748b",
                   "marginBottom": "6px"},
        ),
        html.Div(
            f"{probability:.2%}",
            style={"fontSize": "46px", "fontWeight": "800", "lineHeight": "1.05"},
        ),
        html.P(
            f"Prediccion registrada con ID {response['prediction_id']} y umbral operativo {response['threshold']:.2f}.",
            style={"marginBottom": "8px",
                   "color": "#334155", "lineHeight": "1.7"},
        ),
        html.P(
            response["target_meaning"],
            style={"margin": 0, "color": "#64748b", "lineHeight": "1.7"},
        ),
    ]
    return children, {"updated_at": time.time()}


@app.callback(
    Output({"type": "number-input", "field": MATCH}, "value"),
    Input({"type": "number-minus", "field": MATCH}, "n_clicks"),
    Input({"type": "number-plus", "field": MATCH}, "n_clicks"),
    State({"type": "number-input", "field": MATCH}, "value"),
    State({"type": "number-input", "field": MATCH}, "min"),
    State({"type": "number-input", "field": MATCH}, "max"),
    State({"type": "number-input", "field": MATCH}, "step"),
    prevent_initial_call=True,
)
def adjust_numeric_input(_, __, current_value, minimum, maximum, step):
    triggered = ctx.triggered_id
    if not triggered:
        return no_update

    step_value = float(step) if step not in (None, "", "any") else 1.0
    try:
        base_value = float(current_value if current_value not in (None, "") else 0)
    except (TypeError, ValueError):
        base_value = 0.0

    next_value = base_value + (
        step_value if triggered["type"] == "number-plus" else -step_value
    )

    if minimum is not None:
        next_value = max(float(minimum), next_value)
    if maximum is not None:
        next_value = min(float(maximum), next_value)

    precision = 0
    step_text = str(step_value)
    if "." in step_text:
        precision = len(step_text.rstrip("0").split(".")[1])

    if precision == 0:
        return int(round(next_value))
    return round(next_value, precision)


@app.callback(
    Output("summary_cards", "children"),
    Output("recent_predictions", "children"),
    Input("monitor_interval", "n_intervals"),
    Input("prediction_signal", "data"),
)
def refresh_monitoring(_, __):
    summary = safe_api_get("/monitoring/summary", INITIAL_SUMMARY)
    recent = safe_api_get("/monitoring/recent?limit=5", INITIAL_RECENT)

    cards = [
        metric_card(
            "Predicciones totales",
            str(int(summary.get("total_predictions", 0))),
        ),
        metric_card(
            "Promedio de riesgo",
            f"{float(summary.get('average_probability', 0)):.2%}",
            "#1d4ed8",
        ),
        metric_card(
            "Alto riesgo",
            str(int(summary.get("high_risk_predictions", 0))),
            "#b91c1c",
        ),
        metric_card(
            "Tasa alto riesgo",
            f"{float(summary.get('high_risk_rate', 0)):.2%}",
            "#0f766e",
        ),
        metric_card(
            "Ultimo registro",
            format_timestamp(summary.get("last_prediction_at")),
        ),
    ]

    return cards, render_recent_table(recent.get("items", []))


if __name__ == "__main__":
    app.run(debug=False)

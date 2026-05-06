from pathlib import Path

import joblib
import pandas as pd
from dash import Dash, Input, Output, State, dcc, html


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"
METRICS_PATH = ROOT_DIR / ".generated_assets" / "metricas_modelo_final.csv"

model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH) if METADATA_PATH.exists() else {"threshold": 0.5}
THRESHOLD = float(metadata.get("threshold", 0.5))

if METRICS_PATH.exists():
    metrics_row = pd.read_csv(METRICS_PATH).iloc[0].to_dict()
else:
    metrics_row = metadata.get("metrics", {})

MODEL_NAME = str(metrics_row.get("Modelo", "Random Forest"))
METRICS = {
    "Accuracy": float(metrics_row.get("Accuracy", 0)),
    "Precision": float(metrics_row.get("Precision", 0)),
    "Recall": float(metrics_row.get("Recall", 0)),
    "F1-score": float(metrics_row.get("F1", 0)),
    "ROC-AUC": float(metrics_row.get("ROC_AUC", 0)),
}

TOP_VARIABLES = [
    "loan_percent_income",
    "person_income",
    "loan_int_rate",
]

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

CARD_STYLE = {
    "backgroundColor": "#FFFFFF",
    "borderRadius": "8px",
    "padding": "24px",
    "boxShadow": "0 14px 40px rgba(15, 23, 42, 0.08)",
    "border": "1px solid #E2E8F0",
}

LABEL_STYLE = {
    "display": "block",
    "fontWeight": "600",
    "marginBottom": "8px",
    "color": "#0F172A",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "14px 12px",
    "minHeight": "50px",
    "lineHeight": "1.4",
    "borderRadius": "8px",
    "border": "1px solid #CBD5E1",
    "backgroundColor": "#F8FAFC",
    "fontSize": "15px",
    "boxSizing": "border-box",
}

DROPDOWN_STYLE = {
    "fontSize": "15px",
}

app = Dash(__name__)
app.title = "Loan Default Demo"


def build_feature_frame(
    age,
    income,
    employment_length,
    home_ownership,
    loan_intent,
    loan_grade,
    loan_amount,
    interest_rate,
    percent_income,
    default_on_file,
    credit_history_length,
):
    row = DEFAULT_FEATURES.copy()
    row.update(
        {
            "person_age": age,
            "person_income": income,
            "person_emp_length": employment_length,
            "person_home_ownership": home_ownership,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amount,
            "loan_int_rate": interest_rate,
            "loan_percent_income": percent_income,
            "cb_person_default_on_file": default_on_file,
            "cb_person_cred_hist_length": credit_history_length,
        }
    )
    return pd.DataFrame([row], columns=DEFAULT_FEATURES.keys())


def metric_card(name, value):
    return html.Div(
        [
            html.Div(name, style={"fontSize": "13px", "color": "#475569", "marginBottom": "6px"}),
            html.Div(f"{value:.4f}", style={"fontSize": "24px", "fontWeight": "700", "color": "#0F172A"}),
        ],
        style={
            "backgroundColor": "#F8FAFC",
            "border": "1px solid #E2E8F0",
            "borderRadius": "8px",
            "padding": "16px",
        },
    )


def numeric_input(component_id, label, value, minimum=0, maximum=None, step=None):
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            dcc.Input(
                id=component_id,
                type="number",
                value=value,
                min=minimum,
                max=maximum,
                step=step,
                style=INPUT_STYLE,
            ),
        ]
    )


def dropdown_input(component_id, label, options, value):
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            dcc.Dropdown(
                id=component_id,
                options=[{"label": option, "value": option} for option in options],
                value=value,
                clearable=False,
                style=DROPDOWN_STYLE,
            ),
        ]
    )


app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": "linear-gradient(135deg, #F8FAFC 0%, #E0F2FE 100%)",
        "padding": "32px 20px",
        "fontFamily": "Segoe UI, Arial, sans-serif",
    },
    children=[
        html.Div(
            style={"maxWidth": "1080px", "margin": "0 auto"},
            children=[
                html.Div(
                    style={"marginBottom": "24px"},
                    children=[
                        html.Div(
                            "Predicción de incumplimiento",
                            style={
                                "display": "inline-block",
                                "padding": "6px 12px",
                                "borderRadius": "999px",
                                "backgroundColor": "#DBEAFE",
                                "color": "#1D4ED8",
                                "fontWeight": "600",
                                "fontSize": "13px",
                                "marginBottom": "14px",
                            },
                        ),
                        html.H1(
                            "Demo local para evaluar riesgo de default",
                            style={"margin": "0 0 10px 0", "color": "#0F172A"},
                        ),
                        html.P(
                            "Esta aplicación usa el pipeline final de preprocesamiento y Random Forest para estimar la probabilidad de incumplimiento de un préstamo.",
                            style={"margin": 0, "color": "#334155", "fontSize": "16px"},
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(320px, 1.15fr) minmax(300px, 0.85fr)",
                        "gap": "20px",
                    },
                    children=[
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H2("Datos del solicitante", style={"marginTop": 0, "color": "#0F172A"}),
                                html.Div(
                                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "14px"},
                                    children=[
                                        numeric_input("person_age", "Edad", 30),
                                        numeric_input("person_income", "Ingreso anual", 60000),
                                        numeric_input("person_emp_length", "Años de empleo", 5, step=0.5),
                                        numeric_input("cb_person_cred_hist_length", "Historial crediticio (años)", 6),
                                        dropdown_input("person_home_ownership", "Tipo de vivienda", ["RENT", "MORTGAGE", "OWN", "OTHER"], "RENT"),
                                        dropdown_input("loan_intent", "Propósito del préstamo", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"], "PERSONAL"),
                                        dropdown_input("loan_grade", "Grado del préstamo", ["A", "B", "C", "D", "E", "F", "G"], "B"),
                                        dropdown_input("cb_person_default_on_file", "Default previo registrado", ["N", "Y"], "N"),
                                        numeric_input("loan_amnt", "Monto del préstamo", 10000),
                                        numeric_input("loan_int_rate", "Tasa de interés", 11.0, step=0.1),
                                        numeric_input("loan_percent_income", "Porcentaje del ingreso comprometido", 0.20, maximum=1, step=0.01),
                                    ],
                                ),
                                html.Button(
                                    "Predecir riesgo",
                                    id="predict_button",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "18px",
                                        "width": "100%",
                                        "padding": "14px",
                                        "border": "none",
                                        "borderRadius": "8px",
                                        "backgroundColor": "#0F172A",
                                        "color": "#FFFFFF",
                                        "fontSize": "15px",
                                        "fontWeight": "700",
                                        "cursor": "pointer",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H2("Resultado", style={"marginTop": 0, "color": "#0F172A"}),
                                html.Div(
                                    id="result_box",
                                    style={
                                        "backgroundColor": "#F8FAFC",
                                        "border": "1px solid #E2E8F0",
                                        "borderRadius": "8px",
                                        "padding": "20px",
                                        "minHeight": "260px",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "center",
                                    },
                                    children=[
                                        html.P(
                                            "Ingresa los valores y presiona el botón para obtener la predicción.",
                                            style={"margin": 0, "color": "#475569", "fontSize": "16px"},
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style={**CARD_STYLE, "marginTop": "20px"},
                    children=[
                        html.H2("Explicación del modelo", style={"marginTop": 0, "color": "#0F172A"}),
                        html.P(
                            "Variables más importantes según el análisis del notebook:",
                            style={"color": "#334155"},
                        ),
                        html.Div(
                            style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "marginBottom": "14px"},
                            children=[
                                html.Span(
                                    item,
                                    style={
                                        "padding": "8px 12px",
                                        "borderRadius": "999px",
                                        "backgroundColor": "#E0F2FE",
                                        "color": "#075985",
                                        "fontWeight": "600",
                                    },
                                )
                                for item in TOP_VARIABLES
                            ],
                        ),
                        html.P(
                            "En general, el riesgo aumenta cuando el préstamo representa una mayor carga sobre el ingreso, cuando la tasa de interés es más alta y cuando el nivel de ingreso no compensa adecuadamente el monto solicitado.",
                            style={"marginBottom": 0, "color": "#334155", "lineHeight": "1.6"},
                        ),
                    ],
                ),
                html.Div(
                    style={**CARD_STYLE, "marginTop": "20px"},
                    children=[
                        html.H2("Monitoreo del desempeño", style={"marginTop": 0, "color": "#0F172A"}),
                        html.P(
                            f"Modelo utilizado: {MODEL_NAME} con threshold optimizado = {THRESHOLD:.2f}",
                            style={"color": "#334155", "fontWeight": "600"},
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(140px, 1fr))",
                                "gap": "14px",
                            },
                            children=[metric_card(name, value) for name, value in METRICS.items()],
                        ),
                    ],
                ),
            ],
        )
    ],
)


@app.callback(
    Output("result_box", "children"),
    Input("predict_button", "n_clicks"),
    State("person_age", "value"),
    State("person_income", "value"),
    State("person_emp_length", "value"),
    State("person_home_ownership", "value"),
    State("loan_intent", "value"),
    State("loan_grade", "value"),
    State("loan_amnt", "value"),
    State("loan_int_rate", "value"),
    State("loan_percent_income", "value"),
    State("cb_person_default_on_file", "value"),
    State("cb_person_cred_hist_length", "value"),
)
def predict_default(
    n_clicks,
    age,
    income,
    employment_length,
    home_ownership,
    loan_intent,
    loan_grade,
    loan_amount,
    interest_rate,
    percent_income,
    default_on_file,
    credit_history_length,
):
    if not n_clicks:
        return [
            html.P(
                "Ingresa los valores y presiona el botón para obtener la predicción.",
                style={"margin": 0, "color": "#475569", "fontSize": "16px"},
            )
        ]

    values = [
        age,
        income,
        employment_length,
        home_ownership,
        loan_intent,
        loan_grade,
        loan_amount,
        interest_rate,
        percent_income,
        default_on_file,
        credit_history_length,
    ]
    if any(value is None for value in values):
        return [
            html.P("Completa todos los campos para realizar la predicción.", style={"color": "#B91C1C", "fontWeight": "600"})
        ]

    features = build_feature_frame(
        age,
        income,
        employment_length,
        home_ownership,
        loan_intent,
        loan_grade,
        loan_amount,
        interest_rate,
        percent_income,
        default_on_file,
        credit_history_length,
    )
    probability = float(model.predict_proba(features)[0, 1])
    label = "Alto riesgo" if probability >= THRESHOLD else "Bajo riesgo"
    accent_color = "#B91C1C" if probability >= THRESHOLD else "#047857"
    accent_bg = "#FEF2F2" if probability >= THRESHOLD else "#ECFDF5"

    return [
        html.Div(
            style={
                "display": "inline-block",
                "padding": "6px 12px",
                "borderRadius": "999px",
                "backgroundColor": accent_bg,
                "color": accent_color,
                "fontWeight": "700",
                "marginBottom": "16px",
            },
            children=label,
        ),
        html.Div("Probabilidad estimada de default", style={"color": "#475569", "marginBottom": "6px"}),
        html.Div(
            f"{probability:.2%}",
            style={"fontSize": "42px", "fontWeight": "800", "color": "#0F172A", "lineHeight": "1.1"},
        ),
        html.P(
            "La salida corresponde a la probabilidad predicha para la clase loan_status = 1. La clasificación usa el threshold optimizado en el notebook.",
            style={"marginTop": "14px", "marginBottom": 0, "color": "#334155", "lineHeight": "1.6"},
        ),
    ]


if __name__ == "__main__":
    app.run(debug=False)

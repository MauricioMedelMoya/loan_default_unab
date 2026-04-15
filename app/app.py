from pathlib import Path

import joblib
import pandas as pd
from dash import Dash, Input, Output, State, dcc, html


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
model = joblib.load(MODEL_PATH)

MODEL_NAME = "Random Forest"
METRICS = {
    "Accuracy": 0.9337,
    "Precision": 0.9705,
    "Recall": 0.7180,
    "F1-score": 0.8254,
    "ROC-AUC": 0.9292,
}

TOP_VARIABLES = [
    "loan_percent_income",
    "person_income",
    "loan_int_rate",
]

DEFAULT_FEATURES = {
    "person_age": 30,
    "person_income": 60000,
    "person_emp_length": 5,
    "loan_amnt": 10000,
    "loan_int_rate": 11.0,
    "loan_percent_income": 0.20,
    "cb_person_cred_hist_length": 6,
    "person_home_ownership_MORTGAGE": 0,
    "person_home_ownership_OTHER": 0,
    "person_home_ownership_OWN": 0,
    "person_home_ownership_RENT": 1,
    "loan_intent_DEBTCONSOLIDATION": 0,
    "loan_intent_EDUCATION": 0,
    "loan_intent_HOMEIMPROVEMENT": 0,
    "loan_intent_MEDICAL": 0,
    "loan_intent_PERSONAL": 1,
    "loan_intent_VENTURE": 0,
    "loan_grade_A": 0,
    "loan_grade_B": 1,
    "loan_grade_C": 0,
    "loan_grade_D": 0,
    "loan_grade_E": 0,
    "loan_grade_F": 0,
    "loan_grade_G": 0,
    "cb_person_default_on_file_N": 1,
    "cb_person_default_on_file_Y": 0,
}

CARD_STYLE = {
    "backgroundColor": "#FFFFFF",
    "borderRadius": "18px",
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
    "padding": "16px 14px",
    "minHeight": "54px",
    "lineHeight": "1.4",
    "borderRadius": "12px",
    "border": "1px solid #CBD5E1",
    "backgroundColor": "#F8FAFC",
    "fontSize": "15px",
    "boxSizing": "border-box",
}

app = Dash(__name__)
app.title = "Loan Default Demo"


def build_feature_frame(income, loan_amount, interest_rate, percent_income):
    row = DEFAULT_FEATURES.copy()
    row["person_income"] = income
    row["loan_amnt"] = loan_amount
    row["loan_int_rate"] = interest_rate
    row["loan_percent_income"] = percent_income
    return pd.DataFrame([row], columns=model.feature_names_in_)


def metric_card(name, value):
    return html.Div(
        [
            html.Div(name, style={"fontSize": "13px", "color": "#475569", "marginBottom": "6px"}),
            html.Div(f"{value:.4f}", style={"fontSize": "24px", "fontWeight": "700", "color": "#0F172A"}),
        ],
        style={
            "backgroundColor": "#F8FAFC",
            "border": "1px solid #E2E8F0",
            "borderRadius": "14px",
            "padding": "16px",
        },
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
            style={"maxWidth": "980px", "margin": "0 auto"},
            children=[
                html.Div(
                    style={"marginBottom": "24px"},
                    children=[
                        html.Div(
                            "Prediccion de incumplimiento",
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
                            "Esta aplicacion usa un modelo Random Forest preentrenado para estimar la probabilidad de incumplimiento de un prestamo.",
                            style={"margin": 0, "color": "#334155", "fontSize": "16px"},
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
                        "gap": "20px",
                    },
                    children=[
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H2("Datos del solicitante", style={"marginTop": 0, "color": "#0F172A"}),
                                html.Div(
                                    style={"display": "grid", "gap": "14px"},
                                    children=[
                                        html.Div(
                                            [
                                                html.Label("Ingreso anual", style=LABEL_STYLE),
                                                dcc.Input(id="person_income", type="number", value=60000, min=0, style=INPUT_STYLE),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Monto del prestamo", style=LABEL_STYLE),
                                                dcc.Input(id="loan_amnt", type="number", value=10000, min=0, style=INPUT_STYLE),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Tasa de interes", style=LABEL_STYLE),
                                                dcc.Input(id="loan_int_rate", type="number", value=11.0, min=0, step=0.1, style=INPUT_STYLE),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Porcentaje del ingreso comprometido", style=LABEL_STYLE),
                                                dcc.Input(id="loan_percent_income", type="number", value=0.20, min=0, max=1, step=0.01, style=INPUT_STYLE),
                                            ]
                                        ),
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
                                        "borderRadius": "12px",
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
                                        "borderRadius": "16px",
                                        "padding": "20px",
                                        "minHeight": "220px",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "center",
                                    },
                                    children=[
                                        html.P(
                                            "Ingresa los valores y presiona el boton para obtener la prediccion.",
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
                        html.H2("Explicacion del modelo", style={"marginTop": 0, "color": "#0F172A"}),
                        html.P(
                            "Variables mas importantes segun el analisis del notebook:",
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
                            "En general, el riesgo aumenta cuando el prestamo representa una mayor carga sobre el ingreso, cuando la tasa de interes es mas alta y cuando el nivel de ingreso no compensa adecuadamente el monto solicitado.",
                            style={"marginBottom": 0, "color": "#334155", "lineHeight": "1.6"},
                        ),
                    ],
                ),
                html.Div(
                    style={**CARD_STYLE, "marginTop": "20px"},
                    children=[
                        html.H2("Monitoreo del desempeno", style={"marginTop": 0, "color": "#0F172A"}),
                        html.P(
                            f"Modelo utilizado: {MODEL_NAME}",
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
    State("person_income", "value"),
    State("loan_amnt", "value"),
    State("loan_int_rate", "value"),
    State("loan_percent_income", "value"),
)
def predict_default(n_clicks, income, loan_amount, interest_rate, percent_income):
    if not n_clicks:
        return [
            html.P(
                "Ingresa los valores y presiona el boton para obtener la prediccion.",
                style={"margin": 0, "color": "#475569", "fontSize": "16px"},
            )
        ]

    values = [income, loan_amount, interest_rate, percent_income]
    if any(value is None for value in values):
        return [
            html.P("Completa todos los campos para realizar la prediccion.", style={"color": "#B91C1C", "fontWeight": "600"})
        ]

    features = build_feature_frame(income, loan_amount, interest_rate, percent_income)
    probability = float(model.predict_proba(features)[0, 1])
    label = "Alto riesgo" if probability >= 0.5 else "Bajo riesgo"
    accent_color = "#B91C1C" if probability >= 0.5 else "#047857"
    accent_bg = "#FEF2F2" if probability >= 0.5 else "#ECFDF5"

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
            "La salida corresponde a la probabilidad predicha para la clase loan_status = 1, es decir, incumplimiento del prestamo.",
            style={"marginTop": "14px", "marginBottom": 0, "color": "#334155", "lineHeight": "1.6"},
        ),
    ]


if __name__ == "__main__":
    app.run(debug=True)

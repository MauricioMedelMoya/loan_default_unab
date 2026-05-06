from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "credit_risk_dataset.csv"
ASSETS_DIR = ROOT / ".generated_assets"
APP_DIR = ROOT / "app"
RANDOM_STATE = 42
THRESHOLD_GRID = [round(x / 100, 2) for x in range(20, 81, 5)]


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def metric_row(name: str, model, X_test: pd.DataFrame, y_test: pd.Series, train_seconds: float) -> dict[str, float | str]:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "Modelo": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "Tiempo_entrenamiento_seg": train_seconds,
    }


def fit_timed(model, X_train: pd.DataFrame, y_train: pd.Series):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - start


def save_barplot(df: pd.DataFrame, x: str, y: str, title: str, output: Path) -> None:
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df, x=x, y=y, color="#2563EB")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    APP_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    target = "loan_status"
    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "MLP simple": MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=120,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "MLP profunda": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=120,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "MLP regularizada": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.01,
            max_iter=120,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }

    comparison_rows = []
    trained_models = {}
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", estimator),
            ]
        )
        fitted, seconds = fit_timed(pipeline, X_train, y_train)
        trained_models[name] = fitted
        comparison_rows.append(metric_row(name, fitted, X_test, y_test, seconds))

    comparison_df = pd.DataFrame(comparison_rows).sort_values("F1", ascending=False)
    comparison_df.to_csv(ASSETS_DIR / "comparacion_modelos_ml_dl.csv", index=False)
    save_barplot(comparison_df, "Modelo", "F1", "Comparación de modelos ML y MLP", ASSETS_DIR / "grafico_modelos.png")

    balance_strategies = {
        "Sin balanceo": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)),
            ]
        ),
        "class_weight": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "RandomUnderSampler": ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
                ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)),
            ]
        ),
        "SMOTE": ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("sampler", SMOTE(random_state=RANDOM_STATE)),
                ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)),
            ]
        ),
    }

    balance_rows = []
    balance_models = {}
    for name, pipeline in balance_strategies.items():
        fitted, seconds = fit_timed(pipeline, X_train, y_train)
        balance_models[name] = fitted
        row = metric_row(name, fitted, X_test, y_test, seconds)
        row["Tecnica_balanceo"] = name
        balance_rows.append(row)

    balance_df = pd.DataFrame(balance_rows)
    balance_df.to_csv(ASSETS_DIR / "comparacion_balanceo.csv", index=False)
    save_barplot(balance_df, "Tecnica_balanceo", "F1", "Impacto de técnicas de balanceo", ASSETS_DIR / "grafico_balanceo.png")

    final_pipeline = balance_models["Sin balanceo"]
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    threshold_rows = []
    for threshold in THRESHOLD_GRID:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        threshold_rows.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(y_test, y_pred_threshold, zero_division=0),
                "Recall": recall_score(y_test, y_pred_threshold, zero_division=0),
                "F1": f1_score(y_test, y_pred_threshold, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(ASSETS_DIR / "threshold_tuning.csv", index=False)
    best_threshold = float(threshold_df.sort_values(["F1", "Recall"], ascending=False).iloc[0]["Threshold"])

    plt.figure(figsize=(9, 5))
    for metric in ["Precision", "Recall", "F1"]:
        sns.lineplot(data=threshold_df, x="Threshold", y=metric, marker="o", label=metric)
    plt.axvline(best_threshold, color="#DC2626", linestyle="--", label=f"Threshold óptimo: {best_threshold:.2f}")
    plt.title("Threshold tuning para Random Forest")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "threshold_plot.png", dpi=160)
    plt.close()

    y_pred_final = (y_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusión final")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    report = classification_report(y_test, y_pred_final, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(ASSETS_DIR / "classification_report_final.csv")

    final_metrics = {
        "Modelo": "Random Forest",
        "Threshold": best_threshold,
        "Accuracy": accuracy_score(y_test, y_pred_final),
        "Precision": precision_score(y_test, y_pred_final, zero_division=0),
        "Recall": recall_score(y_test, y_pred_final, zero_division=0),
        "F1": f1_score(y_test, y_pred_final, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }
    pd.DataFrame([final_metrics]).to_csv(ASSETS_DIR / "metricas_modelo_final.csv", index=False)

    joblib.dump(final_pipeline, APP_DIR / "model.pkl")
    joblib.dump({"threshold": best_threshold, "metrics": final_metrics}, APP_DIR / "model_metadata.pkl")
    (ASSETS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "target": target,
                "random_state": RANDOM_STATE,
                "best_threshold": best_threshold,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Comparación de modelos")
    print(comparison_df.round(4).to_string(index=False))
    print("\nComparación de balanceo")
    print(balance_df.round(4).to_string(index=False))
    print("\nMétricas finales")
    print(pd.DataFrame([final_metrics]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

# Loan Default Prediction

Proyecto de aprendizaje de máquina para predecir incumplimiento de créditos a partir de variables sociodemográficas, laborales y financieras. La segunda iteración mantiene Random Forest como modelo final, pero amplía la comparación experimental con Gradient Boosting y tres configuraciones MLP implementadas con `scikit-learn`.

## Estructura

- `notebooks/loan_default.ipynb`: análisis exploratorio, comparación de modelos, balanceo, threshold tuning, matriz de confusión y SHAP.
- `src/refinement_pipeline.py`: script reproducible para recalcular métricas, gráficos y el pipeline final.
- `app/app.py`: app Dash local para ingresar datos crudos y ejecutar inferencia con el pipeline entrenado.
- `app/model.pkl`: pipeline final serializado con preprocesamiento y Random Forest.
- `app/model_metadata.pkl`: threshold optimizado y métricas finales usadas por la app.
- `data/raw/credit_risk_dataset.csv`: dataset original.
- `.generated_assets/`: carpeta generada automáticamente con CSV y PNG para el informe. Está ignorada por Git.

## Instalación

Requiere Python 3.10 o superior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducir resultados

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe .\src\refinement_pipeline.py
```

El script genera:

- `.generated_assets/comparacion_modelos_ml_dl.csv`
- `.generated_assets/comparacion_balanceo.csv`
- `.generated_assets/threshold_tuning.csv`
- `.generated_assets/metricas_modelo_final.csv`
- gráficos PNG de comparación, balanceo, threshold tuning y matriz de confusión
- `app/model.pkl` y `app/model_metadata.pkl`

## Ejecutar el notebook

```powershell
jupyter notebook
```

Luego abrir:

```text
notebooks/loan_default.ipynb
```

## Ejecutar la app Dash

```powershell
.\.venv\Scripts\python.exe .\app\app.py
```

Abrir en el navegador:

```text
http://127.0.0.1:8050/
```

La app usa el pipeline completo, por lo que recibe variables crudas y aplica internamente imputación, escalamiento, One-Hot Encoding y predicción.

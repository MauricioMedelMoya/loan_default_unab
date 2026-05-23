# Loan Default Prediction

Proyecto de aprendizaje de maquina para predecir incumplimiento de creditos a partir de variables sociodemograficas, laborales y financieras. La version actual mantiene Random Forest como modelo final y agrega una arquitectura desacoplada con frontend Dash, backend Flask y monitoreo persistente en SQLite.

## Estructura

- `notebooks/loan_default.ipynb`: analisis exploratorio, comparacion de modelos, balanceo, threshold tuning, matriz de confusion y SHAP.
- `src/refinement_pipeline.py`: script reproducible para recalcular metricas, graficos y el pipeline final.
- `app/app.py`: frontend Dash que consume la API por HTTP.
- `app/backend.py`: API Flask con endpoints de prediccion, metricas y monitoreo.
- `app/shared.py`: configuracion compartida, carga del pipeline y constantes del proyecto.
- `app/model.pkl`: pipeline final serializado con preprocesamiento y Random Forest.
- `app/model_metadata.pkl`: threshold optimizado y metricas finales usadas por la app.
- `app/prediction_monitoring.db`: base SQLite generada automaticamente con el historial de predicciones.
- `data/raw/credit_risk_dataset.csv`: dataset original.
- `.generated_assets/`: carpeta generada automaticamente con CSV y PNG para el informe. Esta ignorada por Git.

## Instalacion

Requiere Python 3.10 o superior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducir resultados

Desde la raiz del proyecto:

```powershell
python .\src\refinement_pipeline.py
```

El script genera:

- `.generated_assets/comparacion_modelos_ml_dl.csv`
- `.generated_assets/comparacion_balanceo.csv`
- `.generated_assets/threshold_tuning.csv`
- `.generated_assets/metricas_modelo_final.csv`
- graficos PNG de comparacion, balanceo, threshold tuning y matriz de confusion
- `app/model.pkl` y `app/model_metadata.pkl`

## Ejecutar el notebook

```powershell
jupyter notebook
```

Luego abrir:

```text
notebooks/loan_default.ipynb
```

## Ejecutar la arquitectura local

```powershell
python .\app\app.py
```

Abrir en el navegador:

```text
http://127.0.0.1:8050/
```

La app inicia el frontend Dash y, si es necesario, levanta tambien la API Flask en `http://127.0.0.1:8051/`.

Endpoints principales del backend:

- `GET /health`
- `GET /metrics`
- `POST /predict`
- `GET /monitoring/summary`
- `GET /monitoring/recent`

El backend usa el pipeline completo, por lo que recibe variables crudas, aplica internamente imputacion, escalamiento, One-Hot Encoding y prediccion. Ademas, cada inferencia queda almacenada en SQLite para monitoreo persistente.

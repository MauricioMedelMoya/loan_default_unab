# Loan Default Prediction

Este proyecto tiene dos partes:
- un notebook con el análisis y el modelado
- una app web local para probar el modelo entrenado

## Archivos principales

- `notebooks/loan_default.ipynb`: notebook principal
- `app/app.py`: app local
- `app/model.pkl`: modelo entrenado usado por la app

## Requisitos

Necesitas tener instalado:
- Python 3.10 o superior
- `pip`

## Crear entorno virtual

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Instalar dependencias

Desde la carpeta raíz del proyecto:

```bash
pip install -r requirements.txt
pip install jupyter dash joblib
```

## Ejecutar el notebook

Desde la carpeta raíz del proyecto:

```bash
jupyter notebook
```

Después abre:

```text
notebooks/loan_default.ipynb
```

Si prefieres, también puedes usar:

```bash
jupyter lab
```

## Ejecutar la app

Desde la carpeta raíz del proyecto:

### Windows

```powershell
python .\app\app.py
```

### macOS

```bash
python3 app/app.py
```

Luego abre en tu navegador:

```text
http://127.0.0.1:8050/
```

## Si algo falla

Si aparece un error de dependencias, vuelve a instalar:

```bash
pip install -r requirements.txt
pip install jupyter dash joblib
```

Si PowerShell no deja activar el entorno virtual en Windows, ejecuta una vez:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

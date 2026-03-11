# MLOps — Lab 4: MLflow con PIMA Diabetes Dataset

Repositorio de la asignatura de MLOps. La rama `lab4` contiene el Lab 4, centrado en el ciclo completo de MLOps con **MLflow**: tracking, registro de modelos y servicio mediante una API REST.

## Dataset

**PIMA Indians Diabetes Dataset** — 768 muestras, 8 variables clínicas, variable objetivo `Outcome` (0 = no diabetes, 1 = diabetes).

Ruta: `data/diabetes.csv`  
EDA: `notebooks/EDA-PIMA_Diabetes.ipynb`

## Estructura del proyecto

```
experimento_prueba/        # Experimento inicial: entrenamiento simple
    1_train.py             # Entrenamiento RF + MLflow tracking
    2_register.py          # Registro en MLflow Model Registry
    3_predict_api.py       # API FastAPI para servir el modelo

experimento_completo/      # Experimento completo: búsqueda de hiperparámetros
    1_train_gridsearch.py  # Split 70/20/10 + GridSearchCV (72 combinaciones)
    2_register_best.py     # Selección por recall + registro del mejor modelo
    3_api.py               # API FastAPI con probabilidad de predicción
    4_evaluate_test.py     # Evaluación final sobre el dataset de test vía API

conclusiones/
    conclusiones.md        # Tabla de métricas, justificación médica y conclusiones

data/
    diabetes.csv

notebooks/
    EDA-PIMA_Diabetes.ipynb
```

## Requisitos

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)

```bash
uv sync
```

Dependencias principales: `mlflow`, `scikit-learn`, `fastapi`, `uvicorn`, `pandas`.

## Ejecución

### Experimento de prueba

```bash
uv run python experimento_prueba/1_train.py
uv run python experimento_prueba/2_register.py
uv run python experimento_prueba/3_predict_api.py   # API en http://localhost:8000/docs
```

### Experimento completo

```bash
# 1. Entrenamiento con GridSearch (puede tardar ~1-2 min)
uv run python experimento_completo/1_train_gridsearch.py

# 2. Registrar el mejor modelo
uv run python experimento_completo/2_register_best.py

# 3. Arrancar la API (en una terminal separada)
uv run python experimento_completo/3_api.py

# 4. Evaluar sobre el test set
uv run python experimento_completo/4_evaluate_test.py
```

### MLflow UI

```bash
# Experimento de prueba
uv run mlflow ui --backend-store-uri sqlite:///experimento_prueba/mlflow.db

# Experimento completo
uv run mlflow ui --backend-store-uri sqlite:///experimento_completo/mlflow.db
```

Interfaz disponible en `http://localhost:5000`.

## Criterio de selección del modelo

La métrica principal es **Recall** (sensibilidad). En diagnóstico de diabetes, un falso negativo (paciente diabético no detectado) es más costoso que un falso positivo. Por ello, el GridSearch optimiza directamente el recall y el modelo registrado es aquel con mayor recall en el conjunto de validación.

## Resultados (experimento completo)

| Métrica   | Valor  |
|-----------|--------|
| Recall    | 0.8519 |
| Accuracy  | 0.8052 |
| Precision | 0.6765 |
| F1        | 0.7541 |

*Evaluado sobre el 10 % del dataset reservado como test (77 muestras).*

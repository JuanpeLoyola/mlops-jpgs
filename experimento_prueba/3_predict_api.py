"""
Experimento de Prueba — Paso 3: API de predicción con FastAPI
=============================================================
Sirve el modelo registrado en MLflow a través de una API REST propia
construida con FastAPI.

Uso:
    uv run python 3_predict_api.py

Endpoint disponible:
    POST http://localhost:8000/predict
    Body: {"data": [[val1, val2, ..., val8], ...]}
"""

import mlflow.sklearn
import pandas as pd
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

HERE    = Path(__file__).parent
DB_PATH = HERE / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

MODEL_URI = "models:/Diabetes RF Model@champion"

# Cargamos el modelo una sola vez al arrancar el servidor
model = mlflow.sklearn.load_model(MODEL_URI)

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

LABELS = {0: "No Diabetes", 1: "Diabetes"}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Diabetes Prediction API")


class PredictRequest(BaseModel):
    data: list[list[float]]  # lista de filas, cada fila con 8 features

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [
                        # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
                        [6, 148, 72, 35, 0, 33.6, 0.627, 50],   # → Diabetes
                        [1,  85, 66, 29, 0, 26.6, 0.351, 31],   # → No Diabetes
                        [8, 183, 64,  0, 0, 23.3, 0.672, 32],   # → Diabetes
                    ]
                }
            ]
        }
    }


@app.post("/predict")
def predict(request: PredictRequest):
    df = pd.DataFrame(request.data, columns=FEATURE_NAMES)
    preds = model.predict(df).tolist()
    return {
        "predictions": preds,
        "labels": [LABELS[p] for p in preds],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


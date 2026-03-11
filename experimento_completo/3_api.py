"""
Experimento Completo — Paso 3: API FastAPI para el modelo champion
==================================================================
Sirve el mejor modelo registrado en MLflow a través de una API REST.

Uso:
    uv run python experimento_completo/3_api.py

Endpoints:
    GET  http://localhost:8000/health
    POST http://localhost:8000/predict
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

MODEL_URI = "models:/Diabetes RF Completo@champion"
model     = mlflow.sklearn.load_model(MODEL_URI)

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
LABELS = {0: "No Diabetes", 1: "Diabetes"}

app = FastAPI(title="Diabetes RF Completo — API")


class PredictRequest(BaseModel):
    data: list[list[float]]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [
                        # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age
                        [6, 148, 72, 35, 0, 33.6, 0.627, 50],   # → Diabetes esperado
                        [1,  85, 66, 29, 0, 26.6, 0.351, 31],   # → No Diabetes esperado
                        [8, 183, 64,  0, 0, 23.3, 0.672, 32],   # → Diabetes esperado
                    ]
                }
            ]
        }
    }


@app.post("/predict")
def predict(request: PredictRequest):
    df    = pd.DataFrame(request.data, columns=FEATURE_NAMES)
    preds = model.predict(df).tolist()
    proba = model.predict_proba(df)[:, 1].tolist()  # probabilidad de diabetes
    return {
        "predictions":   preds,
        "labels":        [LABELS[p] for p in preds],
        "probability":   [round(p, 4) for p in proba],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

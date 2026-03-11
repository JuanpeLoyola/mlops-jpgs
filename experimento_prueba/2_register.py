"""
Experimento de Prueba — Paso 2: Registro del modelo en MLflow Model Registry
=============================================================================
Uso:
    uv run python 2_register.py
"""

import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

HERE = Path(__file__).parent
DB_PATH = HERE / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

MODEL_NAME = "Diabetes RF Model"
ALIAS     = "champion"

with open(HERE / "last_model_uri.txt", "r", encoding="utf-8") as f:
    model_uri = f.readline().strip()

result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

client = MlflowClient()
client.set_registered_model_alias(
    name=result.name,
    alias=ALIAS,
    version=result.version,
)

print("=" * 55)
print(f"Modelo registrado : {result.name}")
print(f"Versión           : {result.version}")
print(f"Estado            : {result.status}")
print(f"Alias asignado    : '{ALIAS}' -> v{result.version}")
print(f"URI por alias     : models:/{result.name}@{ALIAS}")
print("=" * 55)

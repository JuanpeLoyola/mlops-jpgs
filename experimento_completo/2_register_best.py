"""
Experimento Completo — Paso 2: Selección y registro del mejor modelo
=====================================================================
Lee el ranking generado por 1_train_gridsearch.py, selecciona el mejor
modelo según recall en validación (criterio médico) y lo registra en el
MLflow Model Registry con el alias "champion".

Criterio médico:
    En diagnóstico de diabetes, un falso negativo (predecir "sano" cuando
    el paciente tiene diabetes) es más dañino que un falso positivo.
    Por eso priorizamos RECALL: queremos detectar el máximo de casos reales.

Uso:
    uv run python experimento_completo/2_register_best.py
"""

import mlflow
import pandas as pd
from pathlib import Path
from mlflow.tracking import MlflowClient

HERE    = Path(__file__).parent
DB_PATH = HERE / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

MODEL_NAME = "Diabetes RF Completo"
ALIAS      = "champion"

# ---------------------------------------------------------------------------
# Leer ranking y seleccionar el mejor por val_recall
# ---------------------------------------------------------------------------
results = pd.read_csv(HERE / "grid_results.csv")
best    = results.sort_values("val_rec", ascending=False).iloc[0]

print("=" * 60)
print("MODELO SELECCIONADO (máximo recall en validación)")
print("-" * 60)
print(f"  Run ID       : {best['run_id']}")
print(f"  n_estimators : {best['n_estimators']}")
print(f"  max_depth    : {best['max_depth']}")
print(f"  min_samples_split: {best['min_samples_split']}")
print(f"  class_weight : {best['class_weight']}")
print("-" * 60)
print(f"  CV Recall    : {best['cv_recall']:.4f}")
print(f"  Val Recall   : {best['val_rec']:.4f}  ← criterio principal")
print(f"  Val Accuracy : {best['val_acc']:.4f}")
print(f"  Val Precision: {best['val_prec']:.4f}")
print(f"  Val F1       : {best['val_f1']:.4f}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Registro en Model Registry
# ---------------------------------------------------------------------------
result = mlflow.register_model(
    model_uri=best["model_uri"],
    name=MODEL_NAME,
)

client = MlflowClient()
client.set_registered_model_alias(
    name=result.name,
    alias=ALIAS,
    version=result.version,
)

# Añadimos una descripción con el criterio de selección
client.update_model_version(
    name=result.name,
    version=result.version,
    description=(
        f"Seleccionado por máximo recall en validación ({best['val_rec']:.4f}). "
        f"Criterio médico: minimizar falsos negativos en diagnóstico de diabetes."
    ),
)

print(f"\nModelo registrado : {result.name}  v{result.version}")
print(f"Alias asignado    : '{ALIAS}'")
print(f"URI               : models:/{result.name}@{ALIAS}")

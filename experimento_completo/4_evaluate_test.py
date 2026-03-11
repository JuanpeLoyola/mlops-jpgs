"""
Experimento Completo — Paso 4: Evaluación sobre el dataset de test vía API
===========================================================================
Envía TODAS las filas del dataset de test a la API FastAPI y calcula
las métricas reales del modelo (accuracy, precision, recall, F1).

Este es el script de evaluación final: el test set no ha sido visto
en ningún momento del entrenamiento ni de la selección de modelos.

Requisito: tener la API corriendo con:
    uv run python experimento_completo/3_api.py

Uso:
    uv run python experimento_completo/4_evaluate_test.py
"""

import json
import requests
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

ENDPOINT = "http://localhost:8000/predict"
HERE     = Path(__file__).parent

# ---------------------------------------------------------------------------
# Cargar el test set guardado durante el entrenamiento
# ---------------------------------------------------------------------------
X_test = np.load(HERE / "X_test.npy")
y_test = np.load(HERE / "y_test.npy")

print(f"Test set cargado: {len(X_test)} muestras")

# ---------------------------------------------------------------------------
# Enviar todas las filas a la API en un único request
# ---------------------------------------------------------------------------
payload = {"data": X_test.tolist()}

try:
    response = requests.post(
        url=ENDPOINT,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )
    response.raise_for_status()
except requests.exceptions.ConnectionError:
    print(
        "\n[ERROR] No se puede conectar con la API.\n"
        "Arranca el servidor en otra terminal con:\n\n"
        "  uv run python experimento_completo/3_api.py\n"
    )
    raise SystemExit(1)

y_pred = np.array(response.json()["predictions"])

# ---------------------------------------------------------------------------
# Métricas finales sobre test
# ---------------------------------------------------------------------------
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 55)
print("  MÉTRICAS FINALES — DATASET DE TEST")
print("=" * 55)
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}  ← métrica principal (criterio médico)")
print(f"  F1 Score  : {f1:.4f}")
print("-" * 55)
print("  Matriz de confusión:")
print(f"    Verdaderos Negativos (TN): {tn}")
print(f"    Falsos Positivos     (FP): {fp}")
print(f"    Falsos Negativos     (FN): {fn}  ← pacientes diabéticos no detectados")
print(f"    Verdaderos Positivos (TP): {tp}")
print("-" * 55)
print("\n  Reporte completo:")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
print("=" * 55)

# ---------------------------------------------------------------------------
# Guardar resultados en CSV para incluirlos en las conclusiones
# ---------------------------------------------------------------------------
import pandas as pd

results = pd.DataFrame([{
    "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
    "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    "test_samples": len(X_test),
}])
results.to_csv(HERE / "test_results.csv", index=False)
print("Resultados guardados en test_results.csv")

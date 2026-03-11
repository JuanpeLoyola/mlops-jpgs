"""
Experimento de Prueba — Paso 1: Entrenamiento con MLflow Tracking
=================================================================
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

# Paths absolutos relativos al propio script (funcionan desde cualquier cwd)
HERE     = Path(__file__).parent
DB_PATH  = HERE / "mlflow.db"
CSV_PATH = HERE.parent / "data" / "diabetes.csv"

# ---------------------------------------------------------------------------
# Configuración MLflow
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("diabetes_prueba")

# ---------------------------------------------------------------------------
# Carga y preprocesado del dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Detectamos automáticamente las columnas numéricas donde el 0 es inválido:
# se considera inválido si la columna NO es 'Pregnancies' ni 'Outcome'
# (en todas las demás, un 0 es fisiológicamente imposible)
feature_cols = [c for c in df.columns if c not in ("Pregnancies", "Outcome")]
cols_con_ceros = [c for c in feature_cols if (df[c] == 0).any()]

for col in cols_con_ceros:
    median = df[col].replace(0, pd.NA).median()
    df[col] = df[col].replace(0, median)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split 80/20 para el experimento de prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# Entrenamiento + Tracking
# ---------------------------------------------------------------------------
run_name = f"prueba_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

params = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 5,
    "random_state": 42,
}

with mlflow.start_run(run_name=run_name) as run:

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # --- Métricas ---
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    # Solo logueamos los parámetros propios del modelo
    mlflow.log_params(params)

    # --- Log métricas ---
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # --- Log modelo ---
    input_example = X_test[:2]
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example,
    )

    # Guardamos la URI para el siguiente script
    with open(HERE / "last_model_uri.txt", "w", encoding="utf-8") as f:
        f.write(model_info.model_uri + "\n")

    print("=" * 55)
    print(f"Run ID   : {run.info.run_id}")
    print(f"Run name : {run_name}")
    print("-" * 55)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("-" * 55)
    print(f"Model URI: {model_info.model_uri}")
    print("=" * 55)
    print("\nModel URI guardada en last_model_uri.txt")

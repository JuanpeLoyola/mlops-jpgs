"""
Experimento Completo — Paso 1: GridSearch + MLflow Tracking
============================================================
- Split 70% train / 20% val / 10% test (estratificado)
- GridSearchCV sobre RandomForest usando train+val
- Cada combinación de hiperparámetros queda registrada como un run en MLflow
- El script imprime al final el ranking de los mejores modelos

Uso:
    uv run python experimento_completo/1_train_gridsearch.py
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    make_scorer
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE     = Path(__file__).parent
DB_PATH  = HERE / "mlflow.db"
CSV_PATH = HERE.parent / "data" / "diabetes.csv"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("diabetes_completo")

# ---------------------------------------------------------------------------
# Carga y preprocesado
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Detección automática de columnas con ceros inválidos
feature_cols   = [c for c in df.columns if c not in ("Pregnancies", "Outcome")]
cols_con_ceros = [c for c in feature_cols if (df[c] == 0).any()]
for col in cols_con_ceros:
    median = df[col].replace(0, pd.NA).median()
    df[col] = df[col].replace(0, median)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# ---------------------------------------------------------------------------
# Split 70 / 20 / 10 (estratificado para preservar el ratio de clases)
# ---------------------------------------------------------------------------
# Primero separamos el 10% de test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)
# Del 90% restante, separamos 20/90 ≈ 22.2% → resulta en 20% del total
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.222, random_state=42, stratify=y_trainval
)

print(f"Train : {len(X_train):>4} muestras ({len(X_train)/len(X)*100:.0f}%)")
print(f"Val   : {len(X_val):>4} muestras ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test  : {len(X_test):>4} muestras ({len(X_test)/len(X)*100:.0f}%)")

# Guardamos el test set para los scripts posteriores
np.save(HERE / "X_test.npy", X_test.values)
np.save(HERE / "y_test.npy", y_test.values)

# ---------------------------------------------------------------------------
# GridSearchCV
# Criterio médico: priorizamos RECALL (minimizar falsos negativos).
# Un falso negativo = no detectar diabetes → más dañino que un falso positivo.
# Por eso el scoring principal es recall, aunque evaluamos todas las métricas.
# ---------------------------------------------------------------------------
param_grid = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [4, 6, 8, None],
    "min_samples_split": [2, 5, 10],
    "class_weight":    [None, "balanced"],  # "balanced" penaliza más los FN
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="recall",          # métrica de selección principal: recall
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=False,
)

# Entrenamos sobre train + val combinados (ya tenemos val separado para evaluar)
X_trainval_combined = pd.concat([X_train, X_val])
y_trainval_combined = pd.concat([y_train, y_val])

print(f"\nLanzando GridSearch ({len(grid_search.param_grid)} params)...")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

grid_search.fit(X_trainval_combined, y_trainval_combined)

# ---------------------------------------------------------------------------
# Registrar los mejores N candidatos en MLflow
# Criterio: top-5 por recall en CV, pero también evaluamos en val set
# ---------------------------------------------------------------------------
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)

TOP_N = 5
print(f"\nRegistrando top-{TOP_N} modelos en MLflow...")

results_summary = []

for rank, (_, row) in enumerate(cv_results.head(TOP_N).iterrows(), start=1):
    params = {
        "n_estimators":      int(row["param_n_estimators"]),
        "max_depth":         row["param_max_depth"],
        "min_samples_split": int(row["param_min_samples_split"]),
        "class_weight":      str(row["param_class_weight"]),
    }

    # Re-entrenamos con los params exactos sobre train+val
    model = RandomForestClassifier(random_state=42, **{
        k: (None if v == "None" else v) for k, v in params.items()
        if k != "class_weight"
    }, class_weight=None if params["class_weight"] == "None" else params["class_weight"])
    model.fit(X_trainval_combined, y_trainval_combined)

    # Evaluamos en validation set (separado, no visto en CV)
    y_val_pred = model.predict(X_val)
    acc  = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred)
    rec  = recall_score(y_val, y_val_pred)
    f1   = f1_score(y_val, y_val_pred)
    cv_recall = float(row["mean_test_score"])

    run_name = f"grid_rank{rank:02d}_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metric("cv_recall_mean", cv_recall)
        mlflow.log_metric("val_accuracy",   acc)
        mlflow.log_metric("val_precision",  prec)
        mlflow.log_metric("val_recall",     rec)
        mlflow.log_metric("val_f1",         f1)
        mlflow.log_metric("grid_rank",      rank)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_val.iloc[:2],
        )

        results_summary.append({
            "rank":       rank,
            "run_id":     run.info.run_id,
            "model_uri":  model_info.model_uri,
            "cv_recall":  cv_recall,
            "val_acc":    acc,
            "val_prec":   prec,
            "val_rec":    rec,
            "val_f1":     f1,
            **params,
        })

    print(f"  [rank {rank}] recall_cv={cv_recall:.3f} | val_recall={rec:.3f} | "
          f"val_acc={acc:.3f} | run_id={run.info.run_id[:8]}...")

# ---------------------------------------------------------------------------
# Guardar ranking para el siguiente script
# ---------------------------------------------------------------------------
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(HERE / "grid_results.csv", index=False)

print("\n" + "=" * 65)
print(f"{'Rank':<5} {'CV Recall':>10} {'Val Recall':>10} {'Val Acc':>9} {'Val F1':>8}")
print("-" * 65)
for r in results_summary:
    print(f"  {r['rank']:<4} {r['cv_recall']:>10.4f} {r['val_rec']:>10.4f} "
          f"{r['val_acc']:>9.4f} {r['val_f1']:>8.4f}")
print("=" * 65)
print(f"\nRanking guardado en grid_results.csv")
print(f"Visualiza los runs: uv run mlflow ui --backend-store-uri sqlite:///{DB_PATH}")

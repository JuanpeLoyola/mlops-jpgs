import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Iris LR Model")
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--alias", type=str, default=None)
    args = ap.parse_args()

    if args.alias:
        uri = f"models:/{args.model}@{args.alias}"
    else:
        v = args.version or 1
        uri = f"models:/{args.model}/{v}"

    model = mlflow.sklearn.load_model(uri)

    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    _, X_test, _, y_test = train_test_split(X, y, random_state=42)
    preds = model.predict(X_test[:5])

    print("=== PREDICCIONES VS. VALORES REALES ===")
    for idx, fila in enumerate(X_test[:5]):
        print(f"\n--- Fila {idx + 1} ---")
        for f_idx, feat in enumerate(fila):
            print(f"{feature_names[f_idx]}: {feat}")
        pred_code = preds[idx]
        real_code = y_test[idx]
        print(f"Predicción (cód.num): {pred_code} -> Especie predicha: {class_names[pred_code]}")
        print(f"Valor real (cód.num): {real_code} -> Especie real: {class_names[real_code]}")
    
import argparse
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def read_last_model_uri(path="last_model_uri.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Iris LR Model")
    ap.add_argument("--model_uri", type=str, default=None, help="URI devuelto por log_model()")
    args = ap.parse_args()

    model_uri = args.model_uri or read_last_model_uri()
    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    print(f"Modelo registrado: {result.name}, versión: {result.version}")

# ========================
#  COMPROBACIÓN DE ESTADO
# ========================
from mlflow.tracking import MlflowClient

# Indicar la versión que se asignó
print(f"Modelo registrado: {result.name}, versión: {result.version}")

# Comprobar el estado de la versión
client = MlflowClient()
model_version_info = client.get_model_version(name=result.name, version=result.version)
print("Estado del modelo registrado:", model_version_info.status)
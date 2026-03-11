import argparse
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Iris LR Model")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--alias", type=str, default="staging")  # p.ej. staging / champion / production
    args = ap.parse_args()

    client = MlflowClient()
    client.set_registered_model_alias(
        name=args.model,
        alias=args.alias,
        version=args.version,
    )
    print(f"Alias '{args.alias}' -> {args.model} v{args.version}")

'''
Algunos valores de `stage` comunes son:
- "None" (por defecto)
- "Staging"
- "Production"
- "Archived"

El parámetro `archive_existing_versions` (por defecto `False`) indica si quieres archivar otras versiones de ese modelo que se encuentren en el mismo stage.
'''
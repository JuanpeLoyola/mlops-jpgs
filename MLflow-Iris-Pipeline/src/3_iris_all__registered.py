import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configurar MLflow con SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Establece el nombre del experimento
# de los que queremos conocer las versiones de modelos
model_name = "Iris LR Model"


# 6. Revisar la versión del modelo recién creado usando MlflowClient
client = MlflowClient()
latest_versions = client.search_model_versions(f"name='{model_name}'")

for version_info in latest_versions:
    print(f" - name: {version_info.name}")
    print(f"   version: {version_info.version}")
    print(f"   current_stage: {version_info.current_stage}")
    print(f"   run_id: {version_info.run_id}")
    print("")

# Al terminar, deberías ver en consola TODAS las versiones existentes de "Iris LR Model"
'''
2025/03/27 14:34:31 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.
Successfully registered model 'Iris LR Model'.
Created version '1' of model 'Iris LR Model'.
Entrenamiento finalizado. Accuracy: 1.0
 - name: Iris LR Model
   version: 1
   current_stage: None
   run_id: e61c98059de041889b9fef92d4d97b68
'''
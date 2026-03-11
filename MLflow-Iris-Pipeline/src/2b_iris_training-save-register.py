import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

model_name = "Iris LR Model"

with mlflow.start_run(run_name="Iris_LR_Training") as run:
    mlflow.log_metric("accuracy", acc)

    input_example = X_test[:2]
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example,
        registered_model_name=model_name,
    )

print(f"Entrenamiento finalizado. Accuracy: {acc}")

client = MlflowClient()
latest_versions = client.search_model_versions(f"name='{model_name}'")
for v in latest_versions:
    print(f" - name: {v.name}\n   version: {v.version}\n   current_stage: {v.current_stage}\n   run_id: {v.run_id}\n")

# 6. Revisar la versión del modelo recién creado usando MlflowClient
client = MlflowClient()
latest_versions = client.search_model_versions(f"name='{model_name}'")

for version_info in latest_versions:
    print(f" - name: {version_info.name}")
    print(f"   version: {version_info.version}")
    print(f"   current_stage: {version_info.current_stage}")
    print(f"   run_id: {version_info.run_id}")
    print("")

# Al terminar, deberías ver en consola las versiones existentes de "Iris LR Model"
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
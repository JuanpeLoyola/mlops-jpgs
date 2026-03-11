import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from datetime import datetime
import numpy as np

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris")

run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

with mlflow.start_run(run_name=run_name) as run:
    max_iter = 10
    model = LogisticRegression(
        max_iter=4,
        warm_start=True,
        solver="saga",
        multi_class="multinomial",
    )

    mlflow.log_param("solver", "saga")
    mlflow.log_param("multi_class", "multinomial")
    mlflow.log_param("warm_start", True)
    mlflow.log_param("max_iter_total", max_iter)

    for epoch in range(1, max_iter + 1):
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred_proba)

        print(f"Iteración {epoch}: Accuracy={acc:.4f}, LogLoss={loss:.4f}")
        mlflow.log_metric("accuracy", acc, step=epoch)
        mlflow.log_metric("log_loss", loss, step=epoch)

    # ✅ MLflow 3: usa name= y aporta input_example
    input_example = X_test[:2]
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example,
    )

    print(f"\nRun ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Logged model URI (MLflow3): {model_info.model_uri}")

    # Guardar el model_uri para los siguientes scripts
    with open("last_model_uri.txt", "w", encoding="utf-8") as f:
        f.write(model_info.model_uri + "\n")
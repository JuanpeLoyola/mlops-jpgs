import requests
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1) Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
class_names = iris.target_names     # ['setosa' 'versicolor' 'virginica']

# 2) Split into train/test
_, X_test, _, y_test = train_test_split(X, y, random_state=42)

# 3) Prepare a JSON payload with the first 5 rows of X_test
payload = {
    "inputs": X_test[:5].tolist()  # Make sure to convert NumPy arrays to lists
}

# 4) Send POST request to the MLflow "invocations" endpoint
response = requests.post(
    url="http://localhost:1234/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

# 5) Get the predictions (should be a list of numeric class indices)
preds = response.json()

# 6) Print predictions in the same style as your local script
print("=== PREDICCIONES VS. VALORES REALES ===")

# Extraer la lista interna que contiene las predicciones
predictions_list = preds["predictions"]

for idx, features_row in enumerate(X_test[:5]):
    print(f"\n--- Fila {idx + 1} ---")
    
    # Imprimir cada feature name + value
    for f_idx, feat_value in enumerate(features_row):
        print(f"{feature_names[f_idx]}: {feat_value}")

    # Indice de predicción y real
    pred_code = predictions_list[idx]
    real_code = y_test[idx]

    print(f"Predicción (cód.num): {pred_code} -> Especie predicha: {class_names[pred_code]}")
    print(f"Valor real (cód.num): {real_code} -> Especie real: {class_names[real_code]}")

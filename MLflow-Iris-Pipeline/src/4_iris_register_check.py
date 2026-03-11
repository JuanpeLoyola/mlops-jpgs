import mlflow
from mlflow.tracking import MlflowClient

# 1. Configurar la URI de tracking de MLflow para usar la base de datos sqlite
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Crear un cliente para interactuar con el registro de MLflow
client = MlflowClient()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chequeo de modelo registrado")
    parser.add_argument("--model", type=str, default="Iris LR Model", help="Nombre del modelo")
    parser.add_argument("--version", type=int, default=1, help="Versión del modelo")

    args = parser.parse_args()
    model_name = args.model
    version_number = args.version
    
    '''
    # ESPECIFICA EL MODELO QUE QUIERES COMPROBAR (pasado a través de CLI)
    # Valores por defecto para el modelo de ejemplo
    model_name="Iris LR Model"
    version_number=1
    '''

    # 3. Verificar que el modelo "Iris LR Model" versión 1 está registrado
    try:
        model_version_details = client.get_model_version(name=model_name, version=version_number)
        print("El modelo ya existe en el registro:")
        print(f" - ID de versión: {model_version_details.version}")
        print(f" - Estado actual: {model_version_details.current_stage}")
    except Exception as e:
        print("El modelo no existe o no se puede obtener del registro.")
        print(f"Error: {e}")
        # Aquí podrías detener el proceso o manejar la excepción de otra forma
        raise SystemExit("Saliendo debido a que el modelo no se encontró.")

# Si llegaste aquí, significa que el modelo está disponible. Para servirlo:
# 4. Ejecuta el siguiente comando en tu terminal o script (no dentro de Python):
#
# mlflow models serve -m "models:/Iris LR Model/1" --port 5001 --host 0.0.0.0
#
# Esto levantará un servidor REST en el puerto 5001 que expone el modelo.


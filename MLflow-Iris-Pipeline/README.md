
# CASE STUDY: Iris Classification (MLflow 3)

Este laboratorio muestra el **ciclo completo de gestión del ciclo de vida de un modelo de Machine Learning con MLflow 3**, utilizando el dataset clásico **Iris**.

El flujo completo cubre:

1. Entrenamiento del modelo
2. Tracking de experimentos
3. Registro del modelo en el Model Registry
4. Gestión de versiones mediante **aliases**
5. Predicción local
6. Despliegue como API de inferencia
7. Consumo de la API

El backend de MLflow se configura usando **SQLite**.

---

# 1. MLflow Tracking

## Entrenamiento del modelo

Ejecutar:

```
python 1_iris_training.py
```

Este script:

• carga el dataset Iris
• entrena un modelo **LogisticRegression**
• registra métricas en cada epoch
• guarda el modelo en MLflow

La configuración del tracking es:

```
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris")
```

El entrenamiento se realiza de forma **iterativa** para poder registrar métricas durante el proceso:

```
solver="saga"
warm_start=True
max_iter=4
```

El entrenamiento se organiza en:

```
10 epochs
4 iteraciones internas por epoch
```

En cada epoch se registran:

```
accuracy
log_loss
```

Estas métricas pueden visualizarse posteriormente en la interfaz de MLflow.

El modelo se guarda mediante:

```
mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    input_example=X_test[:2]
)
```

MLflow devuelve un **model_uri**, que se guarda automáticamente en el archivo:

```
last_model_uri.txt
```

Este identificador se utilizará para registrar el modelo.

---

# 2. Acceder a la interfaz de MLflow

Para visualizar los experimentos:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir en el navegador:

```
http://localhost:5000
```

Desde esta interfaz se pueden consultar:

• runs
• parámetros
• métricas
• artefactos
• modelos registrados

---

# 3. Registrar el modelo en el Model Registry

Una vez entrenado el modelo:

```
python 2_iris_register.py
```

Este script:

1. Lee el `model_uri` generado durante el entrenamiento
2. Registra el modelo en el **Model Registry**

Ejemplo de resultado:

```
Modelo registrado: Iris LR Model
versión: 1
```

Cada nuevo registro crea automáticamente **una nueva versión del modelo**.

---

# 4. Consultar modelos registrados

Para listar todas las versiones:

```
python 3_iris_all__registered.py
```

La salida mostrará:

```
name: Iris LR Model
version: 1
run_id: ...
```

---

# 5. Comprobar si un modelo existe

Script:

```
python 4_iris_register_check.py
```

Este script verifica que una versión concreta existe en el registro.

Ejemplo:

```
python 4_iris_register_check.py --model "Iris LR Model" --version 1
```

---

# 6. Gestión de versiones mediante aliases (MLflow 3)

En MLflow 3 los **stages están en desuso**.

Se recomienda utilizar **aliases**, por ejemplo:

```
champion
staging
production
```

Asignar un alias a una versión:

```
python 5_set_model_stage.py --model "Iris LR Model" --version 1 --alias champion
```

Esto crea el alias:

```
models:/Iris LR Model@champion
```

De esta forma el sistema puede cambiar de modelo **sin modificar el código de inferencia**.

---

# 7. Predicciones cargando el modelo desde el registry

Script:

```
python 6_iris_predict.py
```

Este script puede cargar el modelo:

por **versión**

```
models:/Iris LR Model/1
```

o por **alias**

```
models:/Iris LR Model@champion
```

El script carga el modelo y realiza predicciones sobre el dataset de test.

Ejemplo de salida:

```
=== PREDICCIONES VS. VALORES REALES ===

Fila 1
sepal length: 6.1
...
Predicción: versicolor
Valor real: versicolor
```

---

# 8. Exponer el modelo como API de inferencia

MLflow permite servir modelos directamente sin escribir código de API.

Ejecutar:

```
$Env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

mlflow models serve \
-m "models:/Iris LR Model@champion" \
--host 0.0.0.0 \
--port 1234 \
--env-manager=local
```

Opciones utilizadas:

| Opción                 | Significado                  |
| ----------------------- | ---------------------------- |
| `-m`                  | modelo a servir              |
| `--port`              | puerto del servidor          |
| `--host`              | acepta conexiones externas   |
| `--env-manager=local` | usa el entorno Python actual |

El endpoint generado será:

```
POST http://localhost:1234/invocations
```

---

# 9. Realizar predicciones vía API

Script:

```
python 7_iris_API_prediction.py
```

Este script:

1. prepara un JSON con datos del dataset Iris
2. envía una petición POST
3. recibe las predicciones del servidor

Payload enviado:

```
{
  "inputs": [[...],[...]]
}
```

Respuesta típica:

```
{
  "predictions": [1,0,2,1,1]
}
```

El script traduce estos índices a las especies:

```
setosa
versicolor
virginica
```

---

# 10. Arquitectura del laboratorio

El flujo completo del laboratorio es:

```
Entrenamiento
     ↓
Experiment Tracking
     ↓
Model Logging
     ↓
Model Registry
     ↓
Alias Management
     ↓
Model Serving
     ↓
API Predictions
```

Este flujo representa un **pipeline básico de MLOps con MLflow**.

---

# 11. Diferencia entre MLflow Tracking Server y Model Serving

MLflow ofrece dos tipos de servicios distintos.

## MLflow Tracking Server

```
mlflow server
```

Proporciona:

• interfaz web
• almacenamiento de experimentos
• model registry
• REST API para logging

Ejemplo:

```
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns \
--host 0.0.0.0 \
--port 5000
```

---

## MLflow Model Serving

```
mlflow models serve
```

Sirve **un único modelo para inferencia**.

Proporciona el endpoint:

```
POST /invocations
```

Ejemplo:

```
mlflow models serve -m "models:/Iris LR Model@champion"
```

---

# Conclusión

Este laboratorio muestra un **pipeline mínimo de MLOps con MLflow 3** que cubre:

• experiment tracking
• model logging
• model registry
• version management mediante aliases
• model serving
• inferencia vía API

Con este flujo se puede gestionar el ciclo de vida completo de un modelo sin necesidad de implementar infraestructura adicional.

---

Si quieres, también puedo prepararte una **versión del README optimizada para GitHub (con diagramas de arquitectura y comandos para alumnos)** que queda muy bien como material de LAB de MLOps.

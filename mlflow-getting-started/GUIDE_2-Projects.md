### 2. MLFlow Projects

Ahora que tenemos nuestro código de entrenamiento, podemos crear un proyecto en un formato que sea reproducible en cualquier plataforma usando MLFlow Projects. Este tipo de proyecto es útil si deseas entrenar un modelo en la nube, por ejemplo, en Databricks.

#### 2.1. Preparando el MLproject
Para crear un nuevo proyecto, debemos agregar un archivo llamado `MLproject`, que contendrá las especificaciones del proyecto.

```python
name: MNIST Tutorial Project

entry_points:
  main:
    parameters:
      epochs: {type: int, default: 3}
      learning_rate: {type: float, default: 0.01}
      batch_size: {type: int, default: 64}
    command: "python scripts/train.py {epochs} {learning_rate} {batch_size}"
```
Para más información sobre las especificaciones del archivo `MLproject`, consulta [aquí](https://mlflow.org/docs/latest/projects.html).

#### 2.2. Ejecutando nuestro proyecto
Podemos ejecutar nuestro proyecto en el entorno local o dejar que MLFlow prepare un entorno para nuestro proyecto usando `pyenv` o `conda`.

```bash
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$run_name = "run_$timestamp"
mlflow run . --env-manager=local --experiment-name "MNIST_experiment" --run-name $run_name -P epochs=5 -P learning_rate=0.01 -P batch_size=64
```

> **Nota**: Los parámetros son opcionales. Para pasarlos, usa el argumento `-P` seguido del nombre del parámetro y el valor deseado.

# 1. MLflow Tracking
La carpeta `scripts` contiene los siguientes scripts:

* [dataset.py](scripts/dataset.py): Contiene la función que carga los dataloaders que se utilizarán para entrenar y probar el modelo.
* [model.py](scripts/model.py): Contiene la función que crea el modelo.
* [train.py](scripts/train.py): Script principal con la función de entrenamiento del modelo. Dentro de este script registramos las métricas y los parámetros con MLFlow tracking.


## 1.1 Setup del entorno

```bash
conda create -n mlflow python=3.11
conda activate mlflow
pip install -r requirements.txt
```

MLFlow tracking es un componente que permite al usuario crear y gestionar experimentos. Proporciona una API y una interfaz de usuario que permite guardar y visualizar métricas, parámetros, modelos y artefactos.

Los resultados de los experimentos realizados con MLFlow se almacenan localmente o en un servidor remoto. Por defecto, los resultados se almacenan localmente en archivos dentro del directorio `mlruns`.

Para más información, consulta la [documentación](https://mlflow.org/docs/latest/tracking.html).

## 1.2 Entrenamos una red neuronal simple con PyTorch
Usaremos el MNIST dataset.

Vamos a realizar el entrenamiento con un modelo en Pytorch formado por 2 capas densas de 256 by 128 neuronas, respectivamente.
```bash
python scripts/train.py --epochs <epochs> --learning_rate <learning_rate> --batch_size <batch_size>
```
Como ejemplo:
```
python scripts/train.py --epochs 5 --learning_rate 0.01 --batch_size 64
```

## 1.3 Visualizando los resultados
Después del entrenamiento, el script creará una carpeta llamada `mlruns` en el directorio actual. Dentro de esta carpeta se almacenarán los archivos de registro del experimento. MLFlow proporciona una interfaz de usuario para vis
ualizar los resultados del experimento. Para acceder a ella, abre el terminal y ejecuta:

```bash
$ mlflow ui
```

Ahora, en un navegador, puedes acceder a la interfaz de usuario de MLFlow en la dirección [http://127.0.0.1:5000](http://127.0.0.1:5000).

Verás una tabla con los resultados del experimento.

Al seleccionar el experimento deseado, tendrás acceso a los gráficos y artefactos generados durante el entrenamiento del modelo.

## ANNEX A: How data stream is gathered by MLflow
`./scripts/train.py`
```python
...
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("device", device)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test_acc, test_loss = test(test_dataloader, model, loss_fn)

            mlflow.log_metric("test_acc", test_acc, step=t)
            mlflow.log_metric("test_loss", test_loss, step=t)

            if test_acc > best_acc:
                best_acc = test_acc
                mlflow.log_metric("best_acc", best_acc, step=t)
                mlflow.pytorch.log_model(model.cpu(), "model", signature=signature, code_paths=["scripts/model.py"])
                model.to(device)

        # Save model to file
        model_path = "model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Modelo guardado en {model_path}")

    print("Done!")
...
```

It creates an **MLflow experiment run**, logs key **parameters**, **metrics**, and **the model** itself, and saves a local copy of the model after training.

```python
mlflow.set_experiment(experiment_name)
```
- Sets the **experiment** under which this run will be grouped in MLflow.
- If the experiment doesn't exist, MLflow creates it automatically.

```python
with mlflow.start_run(run_name=run_name):
```
- Starts a new **run** in MLflow, where all logging will be captured.
- `run_name` is a custom name for this run (e.g., includes a timestamp for uniqueness).

---

Inside the run context:

```python
    mlflow.log_param("device", device)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
```
Logs key **hyperparameters** for reproducibility:
- The device used (CPU or GPU)
- Number of training epochs
- Learning rate
- Batch size

These will show up in MLflow’s UI under the “Parameters” section.

---

```python
    for t in range(epochs):
        ...
        mlflow.log_metric("test_acc", test_acc, step=t)
        mlflow.log_metric("test_loss", test_loss, step=t)
```
Each epoch:
- Trains and tests the model
- Logs **metrics**: test accuracy and loss for that epoch (visible in MLflow charts)
- `step=t` allows MLflow to plot these metrics over epochs

---

```python
        if test_acc > best_acc:
            best_acc = test_acc
            mlflow.log_metric("best_acc", best_acc, step=t)
```
- Keeps track of the **best accuracy so far**.
- If the current model beats it, log it as `"best_acc"`.

---

```python
            mlflow.pytorch.log_model(
                model.cpu(), "model", signature=signature, code_paths=["scripts/model.py"]
            )
            model.to(device)
```
- Logs the current **best-performing model** to MLflow
- `model.cpu()` is used because MLflow expects models on CPU to serialize them
- `"model"` is the folder name in the MLflow UI where the model will be saved
- `signature` defines the input/output structure (helpful for deployment)
- `code_paths` includes the actual Python code used to build the model (great for reproducibility)

Then, the model is sent back to GPU (if needed) using `model.to(device)`.


## ANNEX B: Save and Load a Pytorch Model
Although MLflow stores the model, it will be useful to get it in the Pytorch format for using it outside MLflow.

- After training, saves the model weights to a local file using PyTorch:
```python
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")
```

- If you want to save the full model:
> `torch.save(model, "full_model.pth")`

To load the model in another environment (only weights):
```python
model = MyModelClass()           # Step 1: Define architecture
model.load_state_dict(torch.load("model.pth"))  # Step 2: Load weights
model.to(device)
model.eval()
```

And the full model:
```python
import torch

model = torch.load("full_model.pth")
model.to(device)
model.eval()  # Set to evaluation mode
```
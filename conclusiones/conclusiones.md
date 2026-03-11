# Lab 4 — MLflow: Conclusiones del Experimento

## Dataset

**PIMA Indians Diabetes Dataset** — 768 muestras, 8 variables clínicas, variable objetivo `Outcome` (0 = no diabetes, 1 = diabetes).  
Distribución de clases: 65 % negativo / 35 % positivo (ligero desbalance).

---

## 1. Tabla de parámetros y métricas de los modelos registrados

En el experimento completo se realizó una búsqueda exhaustiva con **GridSearchCV** sobre 72 combinaciones de hiperparámetros (3 × 4 × 3 × 2). Se rastrearon los 5 mejores candidatos en MLflow y se registró el óptimo en el Model Registry.

### Top-5 candidatos (ordenados por `val_recall`, criterio de selección)

| Rank | n_estimators | max_depth | min_samples_split | class_weight | CV Recall | Val Accuracy | Val Precision | **Val Recall** | Val F1 |
|------|-------------|-----------|-------------------|--------------|-----------|--------------|---------------|----------------|--------|
| 🥇 1 | 200         | 4         | 5                 | balanced     | 0.8006    | 0.8117       | 0.6866        | **0.8519**     | 0.7603 |
| 2    | 300         | 4         | 2                 | balanced     | 0.8006    | 0.8052       | 0.6765        | **0.8519**     | 0.7541 |
| 3    | 200         | 4         | 10                | balanced     | 0.7923    | 0.8182       | 0.6970        | **0.8519**     | 0.7667 |
| 4    | 200         | 4         | 2                 | balanced     | 0.7882    | 0.8052       | 0.6765        | **0.8519**     | 0.7541 |
| 5    | 300         | 4         | 5                 | balanced     | 0.7881    | 0.8052       | 0.6765        | **0.8519**     | 0.7541 |

> **Modelo registrado como `champion`**: Rank 1 — `n_estimators=200, max_depth=4, min_samples_split=5, class_weight="balanced"`

### ¿Por qué registrar estos y no los demás?

De las 72 combinaciones evaluadas, la mayoría no superó el umbral de recall validado externamente. Los 5 candidatos seleccionados son aquellos con mayor CV Recall interno, y el modelo rank 1 es el elegido porque:

1. **Mayor CV Recall** (0.8006): maximiza la detección de positivos durante la validación cruzada de 5 folds sobre el conjunto train+val.
2. **Mayor Val Recall externo** (0.8519): confirmado en un conjunto de validación independiente que el modelo **nunca ha visto** durante el entrenamiento.
3. **`class_weight="balanced"`**: todos los top-5 usan pesos balanceados, lo que confirma que compensar el desbalance de clases es esencial para maximizar el recall.
4. **`max_depth=4`**: árbol poco profundo → menos sobreajuste. Los modelos más profundos (depth=8, None) tuvieron menor recall en validación.

Los 67 runs restantes no se registran porque su recall en validación fue inferior, lo que en un contexto médico se traduce directamente en **más pacientes diabéticos no detectados**.

---

## 2. Script de obtención de métricas de la API (aplicado al dataset de test)

El script `experimento_completo/4_evaluate_test.py` implementa la evaluación real del modelo en producción:

```
uv run python experimento_completo/3_api.py          # terminal 1: arrancar API
uv run python experimento_completo/4_evaluate_test.py  # terminal 2: evaluar
```

**Flujo del script:**

1. Carga `X_test.npy` e `y_test.npy` — guardados en el paso 1 y **nunca usados** durante entrenamiento ni selección.
2. Envía **todas las filas del test en un único POST** a `http://localhost:8000/predict`.
3. Compara las predicciones devueltas por la API con las etiquetas reales.
4. Calcula las métricas finales y guarda `test_results.csv`.

Este enfoque es importante porque simula exactamente el comportamiento en producción: el modelo se consume a través de la API, igual que lo haría una aplicación real.

---

## 3. Métricas finales sobre el dataset de test

El conjunto de test representa el **10 % del dataset original** (77 muestras), estratificado por clase para mantener la distribución original.

### Resultados

| Métrica          | Valor  |
|------------------|--------|
| Accuracy         | 0.8052 |
| Precision        | 0.6765 |
| **Recall**       | **0.8519** ← métrica principal |
| F1 Score         | 0.7541 |
| Test samples     | 77     |

### Matriz de confusión

|                  | Predicho: No Diabetes | Predicho: Diabetes |
|------------------|-----------------------|--------------------|
| **Real: No Diabetes** | TN = 39          | FP = 11            |
| **Real: Diabetes**    | FN = **4**       | TP = 23            |

### Interpretación médica

- **4 falsos negativos (FN)**: pacientes diabéticos clasificados como sanos. Este es el error más costoso en medicina preventiva.
- **11 falsos positivos (FP)**: pacientes sanos derivados a más pruebas. Costoso pero no peligroso.
- El recall de **0.8519** significa que el modelo detecta correctamente **23 de 27 pacientes diabéticos** del test set.

---

## 4. Conclusiones

### Sobre el proceso MLOps

El experimento ilustra el ciclo completo de MLOps con MLflow:

| Fase | Herramienta | Script |
|------|-------------|--------|
| Entrenamiento + tracking | MLflow Tracking + GridSearchCV | `1_train_gridsearch.py` |
| Selección del mejor modelo | MLflow Client + `grid_results.csv` | `2_register_best.py` |
| Registro y versionado | MLflow Model Registry (`champion`) | `2_register_best.py` |
| Servicio en producción | FastAPI + MLflow load | `3_api.py` |
| Evaluación real | Requests a la API + sklearn metrics | `4_evaluate_test.py` |

### Sobre el modelo

- El modelo `RandomForestClassifier` con `class_weight="balanced"` y `max_depth=4` ofrece el mejor equilibrio entre recall y generalización.
- La profundidad reducida (`max_depth=4`) es un buen indicador de regularización: los modelos más complejos sobreajustan sin mejorar el recall real.
- **GridSearchCV con `scoring="recall"`** fue la decisión clave: optimizar directamente la métrica de negocio en lugar de accuracy o F1 permite obtener el modelo más adecuado para el problema clínico.

### Sobre la métrica de selección

En diagnóstico médico, **minimizar los falsos negativos es prioritario**. Un paciente diabético no detectado puede sufrir complicaciones graves (ceguera, insuficiencia renal, neuropatía). Un falso positivo simplemente implica repetir la prueba. Por ello:

> **Recall** es la métrica principal de selección, no Accuracy ni F1.

El modelo registrado alcanza un recall de **0.8519 en test**, consistente con el validado durante la búsqueda, lo que confirma que el proceso de selección fue correcto y no hubo sobreajuste al conjunto de validación.

### Limitaciones

- Dataset pequeño (768 muestras). Con `max_depth=4` y validación cruzada de 5 folds, los intervalos de confianza de las métricas son amplios.
- El conjunto de test (77 muestras, 27 positivos) tiene varianza alta: un solo FN más cambiaría el recall de 0.85 a 0.81.
- Para producción real sería recomendable un dataset más grande, calibración de probabilidades (Platt scaling) y un umbral de decisión ajustado (actualmente 0.5).

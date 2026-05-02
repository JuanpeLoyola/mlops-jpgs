# LAB A — Deriva del dato con Evidently AI (Titanic)

Análisis de deriva entre los conjuntos train/val/test del dataset Titanic bajo
12 condiciones experimentales (2 estratificación × 3 proporciones × 2 semillas).
Para cada condición se generan dos reportes Evidently AI (train vs val, train vs
test) — 24 reportes en total — y se imprime una tabla resumen con la fracción de
columnas con deriva.

---

## Estructura del proyecto

```
labA/
  data/
    titanic-dataset.csv          # dataset crudo (891 filas, 12 columnas)
  reports/                       # 24 reportes HTML generados (uno por par)
  generate_drift_reports.py      # script principal
  discusion_resultados.md        # tabla resumen + análisis + conclusiones
  README.md                      # este fichero
```

---

## Cómo ejecutar

Desde la **raíz del repositorio**:

```bash
uv sync                                        # instala dependencias (requiere uv)
uv run python labA/generate_drift_reports.py   # ejecuta el script
```

### Requisitos

Las dependencias están declaradas en `pyproject.toml` y pinadas en `uv.lock`:

| Paquete | Versión mínima | Para qué se usa |
| --- | --- | --- |
| `evidently` | `>=0.7.21` | generación de reportes de deriva |
| `pandas` | `>=3.0.2` | manipulación del DataFrame |
| `scikit-learn` | `>=1.6.1` | `train_test_split` para las particiones |

No se necesita ninguna variable de entorno ni credencial.

### Salida esperada en terminal

Al ejecutar el script se imprime por pantalla el progreso de cada una de las 12
condiciones y, al terminar, la tabla resumen en formato Markdown:

```
[run] strat_60-20-20_seed42
[run] strat_60-20-20_seed7
[run] strat_90-5-5_seed42
...
[run] nostrat_98-1-1_seed7

Filas totales del dataset: 891
Reportes guardados en:    .../labA/reports

Tabla resumen (copiar/pegar en discusion_resultados.md):

| stratify | ratio_train | ratio_val | ... | drifted_names_val | drifted_names_test |
| --- | --- | --- | ... | --- | --- |
| True | 0.6 | 0.2 | ... | - | - |
...
```

### Salida esperada en disco

Se crean (o sobreescriben) 24 ficheros HTML en `labA/reports/` con el nombre:

```
{estrat}_{ratios}_seed{semilla}__{train_vs_val|train_vs_test}.html
```

Ejemplos:

```
strat_60-20-20_seed42__train_vs_val.html
strat_60-20-20_seed42__train_vs_test.html
nostrat_98-1-1_seed7__train_vs_val.html
...
```

### Idempotencia y reproducibilidad

El script es **completamente determinista**: dado el mismo CSV de entrada, cada
ejecución produce exactamente los mismos HTMLs y la misma tabla. Esto se debe a
que los dos orígenes de aleatoriedad están fijados:

1. `train_test_split` usa `random_state=seed` (valores 42 y 7) — los splits son
   siempre los mismos.
2. Los tests estadísticos de Evidently (Kolmogorov-Smirnov, chi-cuadrado, Z-test)
   son deterministas dado el mismo input — no hay muestreo interno.

> **Nota git:** los ficheros HTML contienen un timestamp de generación en su
> interior, por lo que git los marcará como modificados en cada ejecución aunque
> el contenido analítico sea idéntico. Se recomienda añadir `labA/reports/` al
> `.gitignore`.

---

## Condiciones experimentales

El script itera sobre el producto cartesiano de tres parámetros:

| Parámetro | Valores | Total |
| --- | --- | --- |
| Estratificación por `Survived` | `True`, `False` | 2 |
| Proporciones train / val / test | 60/20/20, 90/5/5, 98/1/1 | 3 |
| Semilla aleatoria | 42, 7 | 2 |

**2 × 3 × 2 = 12 condiciones**, cada una con 2 reportes (val + test) = **24 HTMLs**.

---

## Cómo funciona el script

### Pipeline general

```
load_titanic()
    ↓
build_schema()
    ↓
for cada (stratify, ratios, seed):
    split_train_val_test()       → train_df, val_df, test_df
    to_dataset() × 3             → train_ds, val_ds, test_ds
    run_pair_report(train, val)  → HTML + métricas resumen
    run_pair_report(train, test) → HTML + métricas resumen
    ↓
tabla resumen Markdown
```

### Funciones auxiliares

#### `load_titanic(path)`

Lee el CSV y elimina las columnas que no aportan información analítica:
`PassengerId`, `Name`, `Ticket`, `Cabin`. El DataFrame resultante tiene 891 filas
y 8 columnas.

#### `split_train_val_test(df, target, ratios, seed, stratify)`

Divide el DataFrame en tres subconjuntos en **dos pasos** para respetar los ratios
exactos:

1. **Paso 1:** separa train del resto (`val + test`) usando `train_size = ratios[0]`.
2. **Paso 2:** del resto, separa val de test usando
   `train_size = val_size / (val_size + test_size)`.

Si `stratify=True`, se pasa `stratify=df[target]` a ambas llamadas de
`train_test_split`, garantizando que la proporción de `Survived=0/1` sea la misma
en los tres subconjuntos. Si `stratify=False`, el reparto es puramente aleatorio.

#### `build_schema() → DataDefinition`

Construye el esquema del dataset indicando a Evidently qué columnas son numéricas
y cuáles categóricas:

- **Numéricas:** `Age`, `SibSp`, `Parch`, `Fare`
- **Categóricas:** `Survived`, `Pclass`, `Sex`, `Embarked`

El tipo de columna determina qué test estadístico usará Evidently para calcular
la deriva (ver sección de tests más abajo).

#### `to_dataset(df, schema) → Dataset`

Envuelve un `pd.DataFrame` en un objeto `evidently.Dataset` asociándole el
schema. Es el tipo que Evidently espera recibir en `report.run(...)`.

#### `build_report(schema) → Report`

Construye el objeto `Report` con tres capas de métricas:

1. `DataDriftPreset()` — análisis global de deriva sobre todas las columnas.
2. `DataSummaryPreset()` — estadísticas descriptivas de cada columna.
3. `ValueDrift(column=col)` por cada columna del schema — deriva individual
   con p-value explícito.

Se pasa `include_tests=True` para que Evidently genere automáticamente tests
pass/fail sobre las métricas calculadas.

#### `run_pair_report(ref_ds, cur_ds, schema, out_path) → dict`

Orquesta la ejecución de un par (referencia, actual):

1. Llama a `build_report(schema)` para obtener un `Report` fresco.
2. Llama a `report.run(current_data=cur_ds, reference_data=ref_ds)` — aquí
   ocurre todo el cálculo estadístico.
3. Guarda el resultado como HTML en `out_path`.
4. Llama a `extract_drift_summary(result)` y devuelve el dict de métricas.

#### `extract_drift_summary(result) → dict`

Parsea el resultado serializado (`result.dict()`) para extraer dos cosas:

- **`DriftedColumnsCount`:** total de columnas con deriva detectada y su fracción
  sobre el total. Viene del `DataDriftPreset`.
- **`ValueDrift` con p-value < 0.05:** nombres de las columnas concretas cuya
  deriva individual es estadísticamente significativa.

Devuelve un dict con `drifted_columns` (int), `drift_share` (float) y
`drifted_column_names` (lista de strings).

#### `condition_label(stratify, ratios, seed) → str`

Genera la etiqueta legible que se usa para nombrar los ficheros HTML. Ejemplos:
`strat_60-20-20_seed42`, `nostrat_98-1-1_seed7`.

#### `_dataframe_to_markdown(df) → str`

Renderiza el DataFrame resumen como tabla Markdown sin dependencias externas
(no usa `tabulate`). Solo itera filas y columnas construyendo la cadena con
separadores `|`.

---

## Cómo funciona Evidently

### Conceptos fundamentales

Evidently trabaja con el concepto de **deriva del dato** (*data drift*): la idea
de que la distribución estadística de una variable puede cambiar entre dos
momentos o dos subconjuntos de datos. En producción esto suele ocurrir cuando el
comportamiento del mundo real cambia respecto a los datos con los que se entrenó
el modelo. En este lab lo provocamos artificialmente variando las proporciones de
los splits y la estratificación.

El flujo básico de Evidently es siempre el mismo:

```
DataDefinition (schema)
    ↓
Dataset.from_pandas(df, data_definition=schema)   ← referencia
Dataset.from_pandas(df, data_definition=schema)   ← actual
    ↓
Report(metrics=[...], include_tests=True)
    ↓
report.run(current_data=actual, reference_data=referencia)
    ↓
result.save_html(...)   /   result.dict()
```

**Referencia** es el dataset "esperado" (en este lab, el conjunto train).
**Actual** es el dataset que queremos comparar (val o test).

### `DataDefinition`

Es el objeto que describe el schema del dataset. Evidently lo necesita para saber
cómo tratar cada columna:

- Columnas **numéricas** → se comparan con tests para distribuciones continuas
  (Kolmogorov-Smirnov por defecto).
- Columnas **categóricas** → se comparan con tests para distribuciones discretas
  (chi-cuadrado o Z-test según el número de categorías).

Sin un `DataDefinition` correcto, Evidently podría inferir tipos erróneos (por
ejemplo, tratar `Pclass` como numérica en lugar de categórica).

### `Dataset`

Encapsula un `pd.DataFrame` junto con su `DataDefinition`. Internamente Evidently
valida que las columnas declaradas en el schema estén presentes y con los tipos
correctos. Es el objeto que se pasa a `report.run(...)`.

### `Report` y `include_tests=True`

Un `Report` es un contenedor de métricas. Al llamar a `report.run(...)` calcula
todas las métricas declaradas y devuelve un objeto resultado. Con
`include_tests=True`, cada métrica que lo soporte genera además un test
pass/fail automático con umbrales por defecto (por ejemplo, p-value < 0.05 para
deriva).

### Presets y métricas usados

#### `DataDriftPreset`

Calcula la deriva de **todas las columnas** del schema en una sola llamada.
Internamente aplica el test estadístico apropiado a cada columna según su tipo y
agrega los resultados en las métricas de nivel conjunto:

- **`DriftedColumnsCount`**: número y fracción de columnas con p-value < 0.05.
- **`DatasetDriftScore`**: fracción global de columnas con deriva, usada como
  indicador único del estado del dataset.
- Un resultado por columna indicando si hay o no hay deriva y el p-value.

#### `DataSummaryPreset`

Genera estadísticas descriptivas del dataset completo (sin comparación
estadística formal). Para cada columna calcula: recuento, valores nulos,
valores únicos y, según el tipo:

- **Numéricas:** media, desviación típica, mínimo, máximo, percentiles 25/50/75.
- **Categóricas:** frecuencias absolutas y relativas de cada categoría.

En el HTML aparece como la sección "Dataset Summary" y permite ver de un vistazo
si los dos datasets tienen distribuciones visualmente similares.

#### `ValueDrift(column=col)`

Calcula la deriva de **una columna concreta** y expone el p-value de forma
explícita. A diferencia de `DataDriftPreset`, que agrega los resultados,
`ValueDrift` permite acceder al valor numérico del p-value por columna para
hacer lógica programática (como en `extract_drift_summary`).

### Tests estadísticos que aplica Evidently

La elección del test depende del tipo de columna y, en el caso categórico,
del número de categorías únicas.

#### Kolmogorov-Smirnov (KS) — columnas numéricas

Compara las **funciones de distribución acumulada** (CDF) de la referencia y el
actual. El estadístico KS es la máxima diferencia vertical entre ambas CDFs:

```
D = max|F_ref(x) - F_cur(x)|
```

El p-value indica la probabilidad de observar esa diferencia (o mayor) si ambas
muestras provienen de la misma distribución. Un p-value < 0.05 rechaza la
hipótesis nula de igualdad de distribuciones.

- **Ventajas:** no asume ninguna forma de distribución (no paramétrico), sensible
  a diferencias en media, varianza y forma.
- **Limitación:** pierde potencia con muestras pequeñas (< 30 filas), lo que
  explica por qué los splits 98/1/1 (9 filas) raramente detectan deriva aunque
  exista.

Se aplica a: `Age`, `SibSp`, `Parch`, `Fare`.

#### Chi-cuadrado (χ²) — columnas categóricas con muchas categorías

Compara las **frecuencias observadas** de cada categoría en la referencia y el
actual con las frecuencias esperadas (las de la referencia). El estadístico es:

```
χ² = Σ (O_i - E_i)² / E_i
```

donde `O_i` es la frecuencia observada en el actual y `E_i` la esperada según la
referencia. Se distribuye como una chi-cuadrado con `k-1` grados de libertad
(`k` = número de categorías).

- **Ventajas:** directo, interpretable, bien conocido.
- **Limitación:** inestable si alguna categoría tiene frecuencia esperada < 5
  (regla de Cochran). Con splits pequeños esto ocurre con facilidad en columnas
  como `Embarked` (3 categorías con distribución desigual).

Se aplica típicamente a: `Sex`, `Embarked`.

#### Z-test de proporciones — columnas categóricas binarias

Para variables con exactamente 2 categorías (binarias), Evidently puede usar un
Z-test de proporciones en lugar de chi-cuadrado. Compara la proporción de una
categoría entre referencia y actual:

```
Z = (p_ref - p_cur) / sqrt(p(1-p)(1/n_ref + 1/n_cur))
```

donde `p` es la proporción pooled. El p-value se obtiene de la distribución
normal estándar.

Se aplica típicamente a: `Survived` (0/1), `Pclass` si se trata como binaria.

### Umbral de detección

En todos los tests el umbral por defecto es **p-value < 0.05**. Si el p-value
es menor, Evidently marca la columna como "drifted" y el test pasa a estado
**FAIL**. Si es mayor, el test pasa a **PASS**. Este umbral es configurable
pero no se modifica en este lab.

---

## El reporte HTML generado

Cada fichero HTML es un dashboard interactivo auto-contenido (sin servidor,
se abre directamente en el navegador). Está dividido en secciones que
corresponden a los presets y métricas declarados en el `Report`.

### Sección 1: Data Drift (DataDriftPreset)

Es la sección principal. Muestra:

- **Resumen global:** número de columnas analizadas, número con deriva detectada,
  fracción total (drift score).
- **Tabla por columna:** para cada columna, el nombre del test aplicado, el valor
  del estadístico, el p-value y el veredicto (drift / no drift).
- **Histogramas superpuestos:** para cada columna, el histograma de la referencia
  (azul) y del actual (naranja) superpuestos. Permite ver visualmente si las
  distribuciones difieren.

### Sección 2: Dataset Summary (DataSummaryPreset)

Muestra estadísticas descriptivas de ambos datasets en paralelo:

- Para columnas **numéricas:** media, std, min, max, percentiles, número de nulos.
- Para columnas **categóricas:** tabla de frecuencias con barras proporcionales.

Esta sección no genera tests pass/fail; es puramente informativa.

### Sección 3: Value Drift por columna (ValueDrift)

Una tarjeta por cada columna del schema con:

- El nombre del test estadístico aplicado.
- El valor del estadístico calculado.
- El **p-value** explícito.
- El veredicto: PASS (p ≥ 0.05, no deriva) o FAIL (p < 0.05, deriva detectada).
- El histograma de distribución de referencia vs actual para esa columna.

Esta sección repite parte de la información de `DataDriftPreset` pero de forma
individualizada y con el p-value accesible, que es lo que `extract_drift_summary`
parsea programáticamente.

### Tests automáticos (include_tests=True)

Con `include_tests=True`, cada métrica que lo soporte genera un test con
resultado **PASS** o **FAIL** visible en el HTML con color verde/rojo. Al inicio
del reporte aparece un resumen con el total de tests pasados y fallidos, lo que
permite de un vistazo saber si el dataset presenta problemas sin leer todas las
secciones.

---

## Resultados resumidos

Ver `discusion_resultados.md` para la tabla completa de las 12 condiciones, el
análisis detallado del efecto de la estratificación, las proporciones y la
semilla, y las conclusiones sobre buenas prácticas de splitting.

# LAB A — Deriva del dato con Evidently AI

**Dataset:** Titanic (`data/titanic-dataset.csv`, 891 filas, 12 columnas
originales). Tras eliminar identificadores y campos de texto libre
(`PassengerId`, `Name`, `Ticket`, `Cabin`) se conservan **8 columnas**:
4 numéricas (`Age`, `SibSp`, `Parch`, `Fare`) y 4 categóricas
(`Survived`, `Pclass`, `Sex`, `Embarked`). La variable objetivo es
`Survived`.

**Diseño experimental.** 12 condiciones = 2 (estratificación sí/no) ×
3 (proporciones 60/20/20, 90/5/5, 98/1/1) × 2 (semillas 42 y 7). Para
cada condición se generan dos reportes Evidently:

- `train_vs_val` — referencia = train, actual = val
- `train_vs_test` — referencia = train, actual = test

En total 24 reportes HTML en `reports/`. La fracción de columnas con
deriva se obtiene del métrica `DriftedColumnsCount` del
`DataDriftPreset` (umbral por defecto: K-S/χ²/Z-test con p < 0,05).

## Tabla resumen

| Estratif. | Split (tr/va/te) | Semilla | n train | n val | n test | Frac. deriva val | Frac. deriva test | Columnas en deriva (val / test) |
| :---: | :---: | :---: | ---: | ---: | ---: | :---: | :---: | :--- |
| sí  | 60/20/20 | 42 | 534 | 178 | 179 | 0/8 = 0,000 | 0/8 = 0,000 | — / — |
| sí  | 60/20/20 |  7 | 534 | 178 | 179 | 0/8 = 0,000 | **1/8 = 0,125** | — / Embarked |
| sí  | 90/5/5   | 42 | 801 |  45 |  45 | 0/8 = 0,000 | 0/8 = 0,000 | — / — |
| sí  | 90/5/5   |  7 | 801 |  45 |  45 | 0/8 = 0,000 | 0/8 = 0,000 | — / — |
| sí  | 98/1/1   | 42 | 873 |   9 |   9 | 0/8 = 0,000 | 0/8 = 0,000 | — / — |
| sí  | 98/1/1   |  7 | 873 |   9 |   9 | 0/8 = 0,000 | 0/8 = 0,000 | — / — |
| no  | 60/20/20 | 42 | 534 | 178 | 179 | 0/8 = 0,000 | 0/8 = 0,000 | — / — |
| no  | 60/20/20 |  7 | 534 | 178 | 179 | **1/8 = 0,125** | 0/8 = 0,000 | Fare / — |
| no  | 90/5/5   | 42 | 801 |  45 |  45 | **1/8 = 0,125** | 0/8 = 0,000 | Embarked / — |
| no  | 90/5/5   |  7 | 801 |  45 |  45 | 0/8 = 0,000 | 0/8 = 0,000 | — / — |
| no  | 98/1/1   | 42 | 873 |   9 |   9 | 0/8 = 0,000 | 0/8 = 0,000 | — / — |
| no  | 98/1/1   |  7 | 873 |   9 |   9 | **1/8 = 0,125** | **1/8 = 0,125** | Embarked / Pclass |

### Agregados rápidos

| Agrupación | Casos | Σ deriva val | Σ deriva test | Frac. media val | Frac. media test |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Estratificado    | 6 | 0 | 1 | 0,000 | 0,021 |
| No estratificado | 6 | 3 | 1 | 0,063 | 0,021 |
| 60/20/20 | 4 | 1 | 1 | 0,031 | 0,031 |
| 90/5/5   | 4 | 1 | 0 | 0,031 | 0,000 |
| 98/1/1   | 4 | 1 | 1 | 0,031 | 0,031 |
| Semilla 42 | 6 | 1 | 0 | 0,021 | 0,000 |
| Semilla  7 | 6 | 2 | 2 | 0,042 | 0,042 |

## Discusión

### Lectura general

En **8 de 12** condiciones no se detecta deriva entre train y los
subconjuntos de validación/test. El resto presenta a lo sumo **1 de 8
columnas** señalada (fracción 0,125), un nivel marginal coherente con
una partición aleatoria de un mismo dataset: con p-valor 0,05 y 8 tests,
es esperable algún falso positivo por puro azar incluso bajo idéntica
distribución. Las columnas señaladas son siempre `Embarked`, `Fare` o
`Pclass` — variables con desbalance fuerte (`Embarked` tiene ~72% S,
~19% C, ~9% Q y 2 nulos; `Pclass` está dominada por la clase 3) o
asimétricas (`Fare` con cola larga y outliers), justo las más sensibles
a la varianza muestral.

### Efecto de la **estratificación**

Es el parámetro con efecto más nítido sobre el conjunto de validación:

- Estratificando la variable objetivo: 0/6 condiciones con deriva en val.
- Sin estratificar: 3/6 condiciones con deriva en val (`Fare`, `Embarked`, `Embarked`).

La estratificación garantiza la misma proporción de `Survived` en los
tres subconjuntos, pero indirectamente también equilibra variables
**correlacionadas** con el target — en Titanic esto incluye `Sex`,
`Pclass` y `Fare`. Por eso la deriva detectada en condiciones no
estratificadas afecta precisamente a esas variables. Es el mecanismo
clásico: si el muestreo deja en validación una mezcla de clases distinta
a la de train, las distribuciones marginales de las features
correlacionadas también se desplazan.

### Efecto del **ratio de división**

A nivel de fracción de columnas con deriva los tres ratios producen
cifras similares (≈ 1/4 condiciones con alguna columna señalada). Pero
hay un matiz importante:

- Con **60/20/20** los conjuntos de val/test tienen 178–179 filas, suficiente
  para que los tests estadísticos detecten desplazamientos pequeños pero
  reales (caso `Fare` con seed 7). Aquí la deriva detectada **es más
  fiable**: poca probabilidad de falso positivo, pero también poca
  protección frente al sesgo de muestreo si no se estratifica.
- Con **90/5/5** los splits encogen a 45 filas: los tests pierden potencia
  estadística. Es más fácil que pasen desapercibidas diferencias
  reales, y los pocos casos de deriva detectada son más probablemente
  fluctuaciones genuinas en variables muy desbalanceadas (`Embarked`).
- Con **98/1/1** los splits son de **9 filas**. A ese tamaño, ni los
  resultados negativos (sin deriva) ni los positivos son
  estadísticamente concluyentes: los tests están funcionando casi al
  límite y los intervalos de confianza son muy anchos. El test de
  evaluación deja además de ser representativo del despliegue real.

Es decir, los tres ratios producen métricas de deriva similares **por
casualidad** del tamaño muestral: a partir del 90/5/5 la ausencia de
deriva detectada no debe interpretarse como "no hay diferencias", sino
como "el test no tiene potencia para detectarlas".

### Efecto de la **semilla aleatoria**

La semilla afecta a qué filas concretas terminan en cada subconjunto y,
por tanto, a si una columna marginalmente desplazada cruza o no el
umbral de p-valor. Con seed 42 sólo aparece deriva en 1/6 condiciones,
con seed 7 en 3/6. La condición que peor sale —no estratificada,
98/1/1, seed 7— resume todos los riesgos a la vez: muestreo aleatorio,
splits diminutos y elección desafortunada de filas; ahí Embarked y
Pclass aparecen en deriva tanto en val como en test. El mensaje práctico
es que **una sola ejecución no garantiza nada**: para conclusiones
robustas conviene repetir con varias semillas o usar validación
cruzada.

### Causas y consecuencias

- **Causas observadas** de las derivas detectadas:
  - Variables **fuertemente desbalanceadas** (`Embarked`, `Pclass`): pequeñas
    fluctuaciones del muestreo desplazan claramente las proporciones.
  - Variables con **distribución asimétrica/colas pesadas** (`Fare`): la
    presencia o ausencia de unos pocos pasajeros con tarifas extremas
    cambia la cola del histograma y dispara los tests K-S.
  - **Tamaño muestral pequeño**: amplifica la varianza del muestreo y
    reduce la potencia estadística de Evidently.
- **Consecuencias prácticas si no se controla**:
  - Métricas de validación inestables entre ejecuciones (modelos
    "ganadores" cambian con la semilla).
  - Test que sobreestima o subestima el rendimiento real del modelo,
    con consecuencias en la selección de hiperparámetros y en la
    decisión de pasar a producción.
  - Falso sentido de seguridad si se elige por defecto un split sin
    estratificación: el modelo puede mostrar buenas métricas en val por
    una mezcla de clases convenientemente similar a train, sin haberlo
    forzado.

## Conclusión general

1. En un dataset estático y pequeño como Titanic, la deriva entre
   train/val/test es siempre **artefacto del propio muestreo**, no
   deriva real del fenómeno. Aun así Evidently la detecta de forma
   ocasional sobre las variables más frágiles (`Embarked`, `Pclass`,
   `Fare`), lo que confirma su utilidad como sistema de aviso temprano.
2. La **estratificación de la variable objetivo es la palanca más
   eficaz**: elimina por completo la deriva en validación en este
   experimento y reduce a un cuarto la incidencia global (1 condición driftada frente a 3 sin estratificar). Debe ser la
   opción por defecto en problemas de clasificación con clases
   desbalanceadas.
3. Las **proporciones extremas (90/5/5, 98/1/1)** no son seguras: aunque
   numéricamente reportan poca deriva, lo hacen porque los tests pierden
   potencia. En la práctica conducen a métricas de validación/test poco
   informativas. Para Titanic, **60/20/20 estratificado** ofrece el mejor
   compromiso entre cantidad de datos para entrenar y splits suficientes
   para evaluar de forma estable.
4. La **semilla** es un origen no despreciable de variabilidad. Para
   resultados publicables conviene probar varias o reportar intervalos
   de confianza vía validación cruzada o bootstrapping, en lugar de
   confiar en una única división.
5. Como guía práctica de monitorización, conviene **fijar un protocolo
   reproducible (estratificación + ratio + lista de semillas)** y no
   basar la decisión de "hay/no hay deriva" en una única corrida; la
   deriva ocasional sobre 1 de 8 columnas observada aquí es típicamente
   ruido y debe interpretarse a la luz del tamaño muestral.

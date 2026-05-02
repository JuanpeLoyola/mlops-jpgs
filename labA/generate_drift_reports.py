"""LAB A — Deriva con Evidently AI sobre el dataset Titanic.

Genera 12 divisiones train/val/test combinando:
  - Estratificacion sobre la variable objetivo (Survived): si / no
  - Proporciones de division: 60/20/20, 90/5/5, 98/1/1
  - Dos semillas aleatorias: 42 y 7

Para cada una de las 12 condiciones se producen dos reportes Evidently
(train vs val, train vs test) y se acumula una tabla resumen con la
fraccion de columnas con deriva.
"""

from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset, DataSummaryPreset

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
LAB_DIR = Path(__file__).resolve().parent
DATA_PATH = LAB_DIR / "data" / "titanic-dataset.csv"
REPORTS_DIR = LAB_DIR / "reports"

TARGET = "Survived"
NUMERICAL = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Survived", "Pclass", "Sex", "Embarked"]
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]

RATIOS: list[tuple[float, float, float]] = [
    (0.60, 0.20, 0.20),
    (0.90, 0.05, 0.05),
    (0.98, 0.01, 0.01),
]
SEEDS: list[int] = [42, 7]
STRATIFY: list[bool] = [True, False]


# ---------------------------------------------------------------------------
# Funciones reutilizables
# ---------------------------------------------------------------------------
def load_titanic(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=DROP_COLUMNS, errors="ignore")


def split_train_val_test(
    df: pd.DataFrame,
    target: str,
    ratios: tuple[float, float, float],
    seed: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide en train/val/test respetando los ratios pedidos.

    Se hace en dos pasos: primero train vs (val+test) y despues val vs test.
    Si `stratify=True` se estratifica por la variable objetivo en ambos pasos.
    """
    train_size, val_size, test_size = ratios
    if not abs((train_size + val_size + test_size) - 1.0) < 1e-9:
        raise ValueError("Los ratios deben sumar 1.")

    strat_full = df[target] if stratify else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=strat_full,
    )

    val_relative = val_size / (val_size + test_size)
    strat_temp = temp_df[target] if stratify else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative,
        random_state=seed,
        stratify=strat_temp,
    )
    return train_df, val_df, test_df


def build_schema() -> DataDefinition:
    return DataDefinition(
        numerical_columns=NUMERICAL,
        categorical_columns=CATEGORICAL,
    )


def to_dataset(df: pd.DataFrame, schema: DataDefinition) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame(df), data_definition=schema)


def build_report(schema: DataDefinition) -> Report:
    """Reporte con DataDriftPreset + DataSummaryPreset + ValueDrift por columna."""
    metrics = [DataDriftPreset(), DataSummaryPreset()]
    for col in (schema.numerical_columns or []) + (schema.categorical_columns or []):
        metrics.append(ValueDrift(column=col))
    return Report(metrics, include_tests=True)


def extract_drift_summary(result) -> dict:
    """Extrae count/share del metric DriftedColumnsCount y los nombres de
    las columnas con deriva detectada (p-value < 0.05 segun ValueDrift)."""
    payload = result.dict()
    summary: dict = {
        "drifted_columns": -1,
        "drift_share": float("nan"),
        "drifted_column_names": [],
    }
    for m in payload["metrics"]:
        name = m["metric_name"]
        if "DriftedColumnsCount" in name:
            v = m["value"]
            summary["drifted_columns"] = int(v["count"])
            summary["drift_share"] = float(v["share"])
        elif name.startswith("ValueDrift("):
            try:
                p_value = float(m["value"])
            except (TypeError, ValueError):
                continue
            if p_value < 0.05:
                col = m["config"].get("column", "?")
                summary["drifted_column_names"].append(col)
    return summary


def condition_label(stratify: bool, ratios: tuple[float, float, float], seed: int) -> str:
    s = "strat" if stratify else "nostrat"
    r = f"{int(ratios[0] * 100)}-{int(ratios[1] * 100)}-{int(ratios[2] * 100)}"
    return f"{s}_{r}_seed{seed}"


def run_pair_report(
    ref_ds: Dataset,
    cur_ds: Dataset,
    schema: DataDefinition,
    out_path: Path,
) -> dict[str, float]:
    report = build_report(schema)
    result = report.run(current_data=cur_ds, reference_data=ref_ds)
    result.save_html(str(out_path))
    return extract_drift_summary(result)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render minimo de DataFrame -> tabla Markdown sin dependencias extra."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_titanic()
    schema = build_schema()
    n_total = len(df)

    rows: list[dict] = []
    n_total_cols = len(NUMERICAL) + len(CATEGORICAL)

    for stratify, ratios, seed in product(STRATIFY, RATIOS, SEEDS):
        label = condition_label(stratify, ratios, seed)
        print(f"[run] {label}")

        train_df, val_df, test_df = split_train_val_test(
            df, TARGET, ratios, seed, stratify
        )

        train_ds = to_dataset(train_df, schema)
        val_ds = to_dataset(val_df, schema)
        test_ds = to_dataset(test_df, schema)

        val_summary = run_pair_report(
            train_ds,
            val_ds,
            schema,
            REPORTS_DIR / f"{label}__train_vs_val.html",
        )
        test_summary = run_pair_report(
            train_ds,
            test_ds,
            schema,
            REPORTS_DIR / f"{label}__train_vs_test.html",
        )

        rows.append({
            "stratify": stratify,
            "ratio_train": ratios[0],
            "ratio_val": ratios[1],
            "ratio_test": ratios[2],
            "seed": seed,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_total_cols": n_total_cols,
            "drifted_cols_val": val_summary["drifted_columns"],
            "drifted_cols_test": test_summary["drifted_columns"],
            "drift_share_val": round(val_summary["drift_share"], 4),
            "drift_share_test": round(test_summary["drift_share"], 4),
            "drifted_names_val": ",".join(val_summary["drifted_column_names"]) or "-",
            "drifted_names_test": ",".join(test_summary["drifted_column_names"]) or "-",
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["stratify", "ratio_train", "seed"], ascending=[False, True, True]
    ).reset_index(drop=True)

    print()
    print(f"Filas totales del dataset: {n_total}")
    print(f"Reportes guardados en:    {REPORTS_DIR}")
    print()
    print("Tabla resumen (copiar/pegar en discusion_resultados.md):")
    print()
    print(_dataframe_to_markdown(summary))


if __name__ == "__main__":
    main()

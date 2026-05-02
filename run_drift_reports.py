from pathlib import Path

import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset, DataSummaryPreset

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
# Each entry maps a human-readable name to:
#   - reference CSV path
#   - list of (check_name, check_csv_path) tuples
#   - DataDefinition for the schema

KOLKATA_NUMERICAL = [
    "Minimum Temperature", "Maximum Temperature", "Temperature",
    "Dew Point", "Relative Humidity", "Heat Index",
    "Wind Speed", "Wind Gust", "Wind Direction",
    "Precipitation", "Precipitation Cover",
    "Visibility", "Cloud Cover", "Sea Level Pressure",
]

KOLKATA_CATEGORICAL = [
    "Weather Type", "Conditions",
]

# Columns with all-NaN values across reference / check files — excluded from schema
# kolkata-weather: Wind Chill, Snow Depth (numerical), Info (numerical)

kolkata_schema = DataDefinition(
    numerical_columns=KOLKATA_NUMERICAL,
    categorical_columns=KOLKATA_CATEGORICAL,
    datetime_columns=["Date time"],
)

TABULAR_NUMERICAL = ["num_sold"]
TABULAR_CATEGORICAL = ["country", "store", "product"]

tabular_schema = DataDefinition(
    numerical_columns=TABULAR_NUMERICAL,
    categorical_columns=TABULAR_CATEGORICAL,
    datetime_columns=["date"],
    id_column="row_id",
)

KOLKATA_DROP_COLUMNS = ["Latitude", "Longitude", "Address", "Resolved Address", "Name", "Wind Chill", "Snow Depth", "Info"]
TABULAR_DROP_COLUMNS: list[str] = []

DATASETS = [
    {
        "name": "kolkata-weather-ref1",
        "reference": "data/kolkata-weather/dataset-ref1.csv",
        "checks": [
            ("check1", "data/kolkata-weather/dataset-ref1-check1.csv"),
            ("check2", "data/kolkata-weather/dataset-ref1-check2.csv"),
        ],
        "schema": kolkata_schema,
        "drop_columns": KOLKATA_DROP_COLUMNS,
    },
    {
        "name": "kolkata-weather-ref2",
        "reference": "data/kolkata-weather/dataset-ref2.csv",
        "checks": [
            ("check1", "data/kolkata-weather/dataset-ref2-check1.csv"),
            ("check2", "data/kolkata-weather/dataset-ref2-check2.csv"),
        ],
        "schema": kolkata_schema,
        "drop_columns": KOLKATA_DROP_COLUMNS,
    },
    {
        "name": "tabular-playground",
        "reference": "data/tabular-playground/tabular-playground-reference.csv",
        "checks": [
            ("check1", "data/tabular-playground/tabular-playground-check1.csv"),
            ("check2", "data/tabular-playground/tabular-playground-check2.csv"),
        ],
        "schema": tabular_schema,
        "drop_columns": TABULAR_DROP_COLUMNS,
    },
]


def build_report(schema: DataDefinition) -> Report:
    """Build a Report that checks for data drift, concept drift proxies,
    and target/feature drift with auto-generated tests."""

    num_cols = schema.numerical_columns or []
    cat_cols = schema.categorical_columns or []

    metrics = [
        # Overall data drift (all columns)
        DataDriftPreset(),
        # Dataset-level summary with auto tests
        DataSummaryPreset(),
    ]

    # Per-column drift for numerical columns (concept / target drift signal)
    for col in num_cols:
        metrics.append(ValueDrift(column=col))

    # Per-column drift for categorical columns
    for col in cat_cols:
        metrics.append(ValueDrift(column=col))

    return Report(metrics, include_tests=True)


def run_all():
    for ds_config in DATASETS:
        ref_df = pd.read_csv(ds_config["reference"])
        schema = ds_config["schema"]
        drop_cols = ds_config["drop_columns"]

        ref_df = ref_df.drop(columns=drop_cols, errors="ignore")

        ref_dataset = Dataset.from_pandas(
            pd.DataFrame(ref_df),
            data_definition=schema,
        )

        for check_name, check_path in ds_config["checks"]:
            check_df = pd.read_csv(check_path)
            check_df = check_df.drop(columns=drop_cols, errors="ignore")

            check_dataset = Dataset.from_pandas(
                pd.DataFrame(check_df),
                data_definition=schema,
            )

            report = build_report(schema)
            result = report.run(
                current_data=check_dataset,
                reference_data=ref_dataset,
            )

            output_file = REPORTS_DIR / f"{ds_config['name']}_{check_name}.html"
            result.save_html(str(output_file))
            print(f"[OK] {output_file}")


if __name__ == "__main__":
    run_all()

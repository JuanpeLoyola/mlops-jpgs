# Drift and data summary reports with EvidentlyAI

The repository contains two datasets divided according to different criteria, detailed below.

In order to generate the repos, make sure to have uv installed, run `uv sync` to generate a virtual environment and execute `uv run run_drift_reports.py`.

## Datasets

### Kolkata Weather

Daily weather data for Kolkata, West Bengal, India. Each row contains 25 columns including temperature (min, max, avg), dew point, relative humidity, heat index, wind (speed, gust, direction, chill), precipitation, snow depth, visibility, cloud cover, sea level pressure, weather type, coordinates, and conditions.

| File | Rows | Date Range | Description |
| --- | ---: | --- | --- |
| `dataset-full.csv` | 2191 | 01/01/2017 – 12/31/2022 | Full dataset (6 years) |
| `dataset-ref1.csv` | 730 | 01/01/2017 – 12/31/2018 | Reference 1 (2 years) |
| `dataset-ref1-check1.csv` | 365 | 01/01/2019 – 12/31/2019 | Check 1 for reference 1 (1 year) |
| `dataset-ref1-check2.csv` | 365 | 01/01/2022 – 12/31/2022 | Check 2 for reference 1 (1 year) |
| `dataset-ref2.csv` | 365 | 01/01/2021 – 12/31/2021 | Reference 2 (1 year) |
| `dataset-ref2-check1.csv` | 365 | 01/01/2022 – 12/31/2022 | Check 1 for reference 2 (1 year) |
| `dataset-ref2-check2.csv` | 59 | 01/01/2022 – 02/28/2022 | Check 2 for reference 2 (2 months) |

### Tabular Playground

Product sales data across countries and stores. Each row contains 6 columns: `row_id`, `date`, `country`, `store`, `product`, and `num_sold`.

| File | Rows | Date Range | Description |
| --- | ---: | --- | --- |
| `tabular-playground-full.csv` | 70128 | 2017-01-01 – 2020-12-31 | Full dataset (4 years) |
| `tabular-playground-reference.csv` | 35040 | 2017-01-01 – 2018-12-31 | Reference dataset (2 years) |
| `tabular-playground-check1.csv` | 17520 | 2019-01-01 – 2019-12-31 | Check 1 (1 year) |
| `tabular-playground-check2.csv` | 17568 | 2020-01-01 – 2020-12-31 | Check 2 (1 year, leap year) |


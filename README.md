# Digital Twin

Predictive maintenance research for a master's project at **Universiti Teknologi Malaysia (UTM)**, using HYSYS simulation datasets for a heat exchanger and centrifugal pump.

This research prototype trains supervised Random Forest classifiers, evaluates simulated health states, plots degradation trends, and classifies new measurement snapshots. “Anomaly detection” here means classification into known fault states. There is no live HYSYS connection, sensor service, equipment control, remaining-useful-life model, or unseen-anomaly detection.

## Features

- Loads semicolon-delimited Latin-1 CSV exports and normalizes HYSYS column headers.
- Cleans numeric values and percentage strings; validates schemas, labels, asset names, and empty datasets.
- Drops missing, nonnumeric, and infinite readings with a logged count; rejects unknown nonempty labels.
- Converts raw heat duty from kJ/h to kW; already canonical `DUTY_kW` data remains in kW.
- Trains independent Random Forest models with 100 trees, balanced class weights, seed 42, and a default 80/20 split.
- Uses stratification when class counts and split sizes allow it; warns and falls back otherwise.
- Reports accuracy, confusion matrix, precision, recall, F1 score, and per-class support.
- Plots heat exchanger UA/duty and pump efficiency/duty/temperature trends with health-state regions.
- Returns health class, label, alert level, diagnosis, and confidence for supplied measurement vectors.
- Validates inference vector dimensions, finite values, and model availability.
- Supports custom input/output directories, configurable test size, headless plotting, and debug tracebacks.
- Preserves the original script and legacy in-process training/inference functions.

## Setup and execution

Use Python 3.10 or later:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m digital_twin
```

Additional commands:

```bash
python digital_twin_predictive_maintenance.py
python -m digital_twin --data-dir . --output-dir outputs --test-size 0.2
python -m digital_twin --debug
python -m unittest discover -s tests -v
```

The pipeline prints metrics and four demonstration predictions. It creates the output directory and saves `eda_heat_exchanger.png` and `eda_centrifugal_pump.png` there. Models remain in memory and are retrained on each run; model persistence is not implemented. Invalid data/configuration yields CLI exit code 2; `--debug` exposes full tracebacks.

## Datasets and units

| Asset | CSV lookup order | Health labels |
| --- | --- | --- |
| `heat_exchanger` | `heat_exchanger_final.csv`, then `heat XchangerOrg.csv` | Normal (0), Fouling (1), Failure (2) |
| `centrifugal_pump` | `pump_data.csv`, then `PUMP READINGSORG..csv` | Normal (0), Degradation (1), Failure (2) |

The Excel files `heat XchangerOrghh.xlsx` and `PUMP READINGSORG._hh.xlsx` are retained as research source artifacts; the loader reads CSV only. Inputs require `Time_Step`, `Target_Status`, and the asset features. Raw HYSYS headers are accepted as aliases; canonical files use the same separator and encoding.

Inference vectors must follow this order:

| Asset | Feature order | Units |
| --- | --- | --- |
| Heat exchanger | `Mass_Flow`, `ACTUAL_UA`, `DUTY_kW` | kg/h, kJ/(°C·h), kW |
| Pump | `Inlet_Mass_Flow`, `Pump_Efficiency`, `Pump_Duty_kW`, `Outlet_Temp_C` | kg/h, percentage points, kW, °C |

**Migration:** older code used raw kJ/h values under `DUTY_kW`. New heat exchanger inference must receive kW (`kJ/h ÷ 3600`), and models must be retrained with the corrected loader. Pump efficiency 75 means 75%, not 0.75.

## Python API

```python
from digital_twin import load_and_preprocess, train_random_forest, evaluate_asset_health

frame = load_and_preprocess("heat_exchanger")
model, x_train, x_test, y_train, y_test = train_random_forest(frame, "heat_exchanger")
models = {"heat_exchanger": model}
result = evaluate_asset_health("heat_exchanger", [2500, 5200, 518000 / 3600], models)
print(result)
```

Results contain `asset_type`, `predicted_class`, `predicted_label`, `alert_level` (INFO/WARNING/CRITICAL), `diagnosis`, and `confidence`. Confidence is the maximum class probability rounded to four decimals; it is not calibrated reliability. Predictions and diagnosis text are research advice requiring engineering validation before operational action.

## Architecture and debugging

```text
digital_twin/
  config.py          Asset schemas, labels, dataset paths, seed
  data.py            CSV adapter, cleaning, schema/unit validation
  training.py        Model training and evaluation
  inference.py       Predictions using explicitly supplied models
  visualization.py   Degradation plots
  cli.py             Application orchestration and arguments
  __main__.py        Package entry point
digital_twin_predictive_maintenance.py  Legacy compatibility adapter
tests/               Regression checks
```

Responsibilities are separated into small modules. The core inference function takes models explicitly; only the legacy adapter retains a global `TRAINED_MODELS` registry. Imports do not train models or read datasets. Training returns the model and split arrays; `run_pipeline()` returns model and metric dictionaries. Debug the responsible stage module and use `--debug` for tracebacks.

## Research limitations

A random split measures performance within these simulations. Adjacent time steps may occur in both training and testing; high accuracy does not demonstrate future-time or real-plant generalization. Thesis experiments should compare chronological and independent-scenario validation, report dataset provenance and class counts, and examine imbalance and uncertainty. Small-class fallback splits can omit classes from training or testing; inspect warnings and support values.

Health shading spans each class's earliest/latest time step. Pump duty and temperature share a secondary axis despite different units, so the figure is an exploratory trend view. No thesis results, university endorsements, operational validation, or predictive lead times are claimed.

## Repository identity

Display title: **Digital Twin**. Intended GitHub slug: **Digital-Twin**, since repository names use hyphens rather than spaces. A local folder/README rename does not rename the GitHub repository.

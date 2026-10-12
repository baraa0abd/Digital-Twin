"""Asset schemas, dataset paths, and reproducible training settings."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Canonical filenames from the project spec; fall back to workspace CSV names.
DATASET_PATHS = {
    "heat_exchanger": [
        BASE_DIR / "heat_exchanger_final.csv",
        BASE_DIR / "heat XchangerOrg.csv",
    ],
    "centrifugal_pump": [
        BASE_DIR / "pump_data.csv",
        BASE_DIR / "PUMP READINGSORG..csv",
    ],
}

CLASS_MAPS = {
    "heat_exchanger": {"Normal": 0, "Fouling": 1, "Failure": 2},
    "centrifugal_pump": {"Normal": 0, "Degradation": 1, "Failure": 2},
}

REVERSE_CLASS_MAPS = {
    asset: {code: label for label, code in mapping.items()}
    for asset, mapping in CLASS_MAPS.items()
}

FEATURE_COLUMNS = {
    "heat_exchanger": ["Mass_Flow", "ACTUAL_UA", "DUTY_kW"],
    "centrifugal_pump": [
        "Inlet_Mass_Flow",
        "Pump_Efficiency",
        "Pump_Duty_kW",
        "Outlet_Temp_C",
    ],
}

# Raw column names in the HYSYS export files -> canonical feature names.
COLUMN_RENAME = {
    "heat_exchanger": {
        "Time_Step": "Time_Step",
        "Mass_Flow_Kg/h": "Mass_Flow",
        "Actual_UA_KJ/C-h": "ACTUAL_UA",
        "Heat_Duty_KJ/h": "DUTY_kW",
        "Target_Status": "Target_Status",
    },
    "centrifugal_pump": {
        "Time_Step": "Time_Step",
        "Inlet_Mass_Flow_Kg/h": "Inlet_Mass_Flow",
        "Pump_Efficiency_%": "Pump_Efficiency",
        "Pump_Duty_kW": "Pump_Duty_kW",
        "Target_Status": "Target_Status",
    },
}

RANDOM_STATE = 42


def validate_asset(asset_type: str) -> None:
    if asset_type not in FEATURE_COLUMNS:
        raise ValueError(f"Unknown asset {asset_type!r}. Choose from {list(FEATURE_COLUMNS)}.")

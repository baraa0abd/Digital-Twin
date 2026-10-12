"""HYSYS CSV adapter with schema and unit validation."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .config import CLASS_MAPS, COLUMN_RENAME, DATASET_PATHS, FEATURE_COLUMNS, validate_asset

def _normalize_hysys_columns(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """Map HYSYS export headers (including special characters) to canonical names."""
    rename_map = dict(COLUMN_RENAME[asset_type])
    for col in df.columns:
        if col.startswith("Outlet_Temp"):
            rename_map[col] = "Outlet_Temp_C"
    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------------
# Step 1 - Data Preprocessing
# ---------------------------------------------------------------------------

def _resolve_dataset_path(asset_type: str) -> Path:
    """Return the first existing CSV path for the given asset."""
    for path in DATASET_PATHS[asset_type]:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No dataset found for '{asset_type}'. Expected one of: "
        f"{[str(p) for p in DATASET_PATHS[asset_type]]}"
    )


def load_and_preprocess(asset_type: str, data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load a HYSYS CSV export, normalize column names, clean dtypes,
    and encode Target_Status as integers.
    """
    validate_asset(asset_type)
    if data_dir is None:
        path = _resolve_dataset_path(asset_type)
    else:
        candidates = [Path(data_dir) / p.name for p in DATASET_PATHS[asset_type]]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No dataset for {asset_type} in {data_dir}")
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df.columns = df.columns.str.strip()
    duty_in_kj_per_hour = "Heat_Duty_KJ/h" in df.columns

    # Strip whitespace from string cells (semicolon exports often pad values).
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    df = _normalize_hysys_columns(df, asset_type)
    required = FEATURE_COLUMNS[asset_type] + ["Time_Step", "Target_Status"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {missing}")
    if df.columns.duplicated().any():
        raise ValueError(f"{path.name}: duplicate normalized column names")
    labels = df["Target_Status"]
    unknown = sorted(set(labels.dropna()) - set(CLASS_MAPS[asset_type]))
    if unknown:
        raise ValueError(f"{path.name}: unknown health labels: {unknown}")

    # Pump efficiency arrives as a percentage string (e.g. "75%").
    if asset_type == "centrifugal_pump":
        df["Pump_Efficiency"] = (
            df["Pump_Efficiency"]
            .astype(str)
            .str.replace("%", "", regex=False)
        )

    numeric_cols = FEATURE_COLUMNS[asset_type] + ["Time_Step"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # HYSYS heat duty is kJ/h; canonical DUTY_kW must actually be kW.
    if asset_type == "heat_exchanger" and duty_in_kj_per_hour:
        df["DUTY_kW"] /= 3600.0
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Encode textual health labels.
    class_map = CLASS_MAPS[asset_type]
    df["Target_Status"] = df["Target_Status"].map(class_map)

    # Handle missing values: drop rows that cannot be used for supervised learning.
    before = len(df)
    df = df.dropna(subset=numeric_cols + ["Target_Status"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logging.getLogger(__name__).warning("[%s] Dropped %d invalid rows", asset_type, dropped)

    if df.empty:
        raise ValueError(f"{path.name}: no valid training rows remain")
    df["Target_Status"] = df["Target_Status"].astype(int)
    return df

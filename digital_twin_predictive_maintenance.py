# -*- coding: utf-8 -*-
"""
Digital Twin - Predictive Maintenance and Anomaly Detection Engine
==================================================================
Machine-learning pipeline for Heat Exchanger and Centrifugal Pump health
classification using HYSYS-generated simulation datasets.

Steps:
  1. Data preprocessing and label encoding
  2. Exploratory data analysis (degradation visualizations)
  3. Random Forest model training (80/20 split)
  4. Model evaluation (accuracy, confusion matrix, classification report)
  5. Digital-twin inference via evaluate_asset_health()
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

# Non-interactive backend for headless / automated runs (override with MPLBACKEND).
if not os.environ.get("MPLBACKEND"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

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
TRAINED_MODELS: dict[str, RandomForestClassifier] = {}


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


def load_and_preprocess(asset_type: str) -> pd.DataFrame:
    """
    Load a HYSYS CSV export, normalize column names, clean dtypes,
    and encode Target_Status as integers.
    """
    path = _resolve_dataset_path(asset_type)
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df.columns = df.columns.str.strip()

    # Strip whitespace from string cells (semicolon exports often pad values).
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df = _normalize_hysys_columns(df, asset_type)

    # Pump efficiency arrives as a percentage string (e.g. "75%").
    if asset_type == "centrifugal_pump":
        df["Pump_Efficiency"] = (
            df["Pump_Efficiency"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

    numeric_cols = FEATURE_COLUMNS[asset_type]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encode textual health labels.
    class_map = CLASS_MAPS[asset_type]
    df["Target_Status"] = df["Target_Status"].map(class_map)

    # Handle missing values: drop rows that cannot be used for supervised learning.
    before = len(df)
    df = df.dropna(subset=numeric_cols + ["Target_Status"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[{asset_type}] Dropped {dropped} row(s) with missing or invalid data.")

    return df


# ---------------------------------------------------------------------------
# Step 2 - Exploratory Data Analysis
# ---------------------------------------------------------------------------

def plot_heat_exchanger_degradation(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Line chart of ACTUAL_UA and DUTY_kW over Time_Step (degradation trend)."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_ua = "#1f77b4"
    color_duty = "#ff7f0e"

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("ACTUAL_UA (kJ/C-h)", color=color_ua)
    ax1.plot(
        df["Time_Step"],
        df["ACTUAL_UA"],
        color=color_ua,
        marker="o",
        linewidth=2,
        label="ACTUAL_UA",
    )
    ax1.tick_params(axis="y", labelcolor=color_ua)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("DUTY_kW (kJ/h simulation units)", color=color_duty)
    ax2.plot(
        df["Time_Step"],
        df["DUTY_kW"],
        color=color_duty,
        marker="s",
        linewidth=2,
        linestyle="--",
        label="DUTY_kW",
    )
    ax2.tick_params(axis="y", labelcolor=color_duty)

    # Highlight health-state regions.
    status_labels = REVERSE_CLASS_MAPS["heat_exchanger"]
    for status_code in sorted(df["Target_Status"].unique()):
        mask = df["Target_Status"] == status_code
        if mask.any():
            ax1.axvspan(
                df.loc[mask, "Time_Step"].min(),
                df.loc[mask, "Time_Step"].max(),
                alpha=0.08,
                label=status_labels[status_code],
            )

    fig.suptitle(
        "Heat Exchanger - ACTUAL_UA and DUTY_kW Degradation Over Time",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved heat exchanger EDA plot -> {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_pump_surge_behavior(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """
    Dual-axis plot: Pump_Efficiency (declining) vs surging Pump_Duty_kW
    and Outlet_Temp_C.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_eff = "#2ca02c"
    color_duty = "#d62728"
    color_temp = "#9467bd"

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Pump Efficiency (%)", color=color_eff)
    ax1.plot(
        df["Time_Step"],
        df["Pump_Efficiency"],
        color=color_eff,
        marker="o",
        linewidth=2,
        label="Pump Efficiency",
    )
    ax1.invert_yaxis()  # Efficiency drops as fault progresses.
    ax1.tick_params(axis="y", labelcolor=color_eff)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Pump Duty (kW) / Outlet Temp (C)")
    ax2.plot(
        df["Time_Step"],
        df["Pump_Duty_kW"],
        color=color_duty,
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Pump_Duty_kW",
    )
    ax2.plot(
        df["Time_Step"],
        df["Outlet_Temp_C"],
        color=color_temp,
        marker="^",
        linewidth=2,
        linestyle=":",
        label="Outlet_Temp_C",
    )

    status_labels = REVERSE_CLASS_MAPS["centrifugal_pump"]
    for status_code in sorted(df["Target_Status"].unique()):
        mask = df["Target_Status"] == status_code
        if mask.any():
            ax1.axvspan(
                df.loc[mask, "Time_Step"].min(),
                df.loc[mask, "Time_Step"].max(),
                alpha=0.08,
                label=status_labels[status_code],
            )

    fig.suptitle(
        "Centrifugal Pump - Duty and Outlet Temp Surge as Efficiency Drops",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved pump EDA plot -> {save_path}")
        plt.close(fig)
    else:
        plt.show()


def run_exploratory_analysis(
    hx_df: pd.DataFrame, pump_df: pd.DataFrame, output_dir: Path | None = None
) -> None:
    """Generate and optionally persist EDA plots for both assets."""
    out = output_dir or BASE_DIR
    plot_heat_exchanger_degradation(hx_df, out / "eda_heat_exchanger.png")
    plot_pump_surge_behavior(pump_df, out / "eda_centrifugal_pump.png")


# ---------------------------------------------------------------------------
# Step 3 - Model Training
# ---------------------------------------------------------------------------

def train_random_forest(
    df: pd.DataFrame,
    asset_type: str,
    test_size: float = 0.2,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Random Forest classifier on 80% of the data (stratified when possible).
    Returns model, X_train, X_test, y_train, y_test.
    """
    features = FEATURE_COLUMNS[asset_type]
    X = df[features].values
    y = df["Target_Status"].values.astype(int)

    # Stratified split requires at least 2 samples per class.
    split_kwargs: dict[str, Any] = {
        "test_size": test_size,
        "random_state": RANDOM_STATE,
    }
    class_counts = pd.Series(y).value_counts()
    if (class_counts >= 2).all():
        split_kwargs["stratify"] = y
    else:
        warnings.warn(
            f"[{asset_type}] Stratified split skipped - at least one class has <2 samples. "
            "Using random_state=42 split instead.",
            UserWarning,
            stacklevel=2,
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    TRAINED_MODELS[asset_type] = model

    return model, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 4 - Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    asset_type: str,
) -> dict[str, Any]:
    """Print and return accuracy, confusion matrix, and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(REVERSE_CLASS_MAPS[asset_type]))
    target_names = [
        REVERSE_CLASS_MAPS[asset_type][code]
        for code in sorted(REVERSE_CLASS_MAPS[asset_type])
    ]
    report = classification_report(
        y_test, y_pred, labels=sorted(REVERSE_CLASS_MAPS[asset_type]), target_names=target_names
    )

    display_name = asset_type.replace("_", " ").title()
    print("\n" + "=" * 72)
    print(f"  EVALUATION - {display_name}")
    print("=" * 72)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nConfusion Matrix (rows = actual, cols = predicted):")
    print(f"Labels: {target_names}")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Step 5 - Digital Twin Inference
# ---------------------------------------------------------------------------

DIAGNOSIS_MESSAGES = {
    "heat_exchanger": {
        0: (
            "Heat Exchanger operating within design parameters. "
            "Heat transfer coefficient (UA) and duty are stable."
        ),
        1: (
            "FOULING DETECTED - Heat transfer degradation observed. "
            "ACTUAL_UA and/or DUTY_kW are declining. Schedule cleaning within "
            "the next maintenance window."
        ),
        2: (
            "CRITICAL FAILURE - Severe fouling or tube blockage suspected. "
            "IMMEDIATE SHUTDOWN RECOMMENDED. Isolate equipment and initiate "
            "emergency maintenance protocol."
        ),
    },
    "centrifugal_pump": {
        0: (
            "Centrifugal Pump operating normally. "
            "Efficiency, duty, and outlet temperature are within expected bounds."
        ),
        1: (
            "DEGRADATION ALERT - Pump efficiency is falling; duty and outlet "
            "temperature are rising. Inspect impeller wear, seals, and cavitation "
            "risk. Plan corrective maintenance."
        ),
        2: (
            "CRITICAL FAILURE - Pump efficiency critically low with excessive "
            "duty and outlet temperature. STOP PUMP IMMEDIATELY to prevent "
            "catastrophic damage and process upsets."
        ),
    },
}

SAFETY_ALERT_LEVELS = {0: "INFO", 1: "WARNING", 2: "CRITICAL"}


def evaluate_asset_health(
    asset_type: str,
    real_time_features: list[float] | np.ndarray,
) -> dict[str, str | int | float]:
    """
    Digital-twin inference endpoint: score live sensor readings and return
    a human-readable health diagnosis with safety alert level.

    Parameters
    ----------
    asset_type : str
        Either 'heat_exchanger' or 'centrifugal_pump'.
    real_time_features : array-like
        Feature vector matching FEATURE_COLUMNS[asset_type] order:
          - heat_exchanger:      [Mass_Flow, ACTUAL_UA, DUTY_kW]
          - centrifugal_pump:    [Inlet_Mass_Flow, Pump_Efficiency,
                                  Pump_Duty_kW, Outlet_Temp_C]

    Returns
    -------
    dict with keys: asset_type, predicted_class, predicted_label,
                    alert_level, diagnosis, confidence
    """
    if asset_type not in TRAINED_MODELS:
        raise ValueError(
            f"No trained model for '{asset_type}'. "
            "Run the training pipeline first (main())."
        )

    expected_len = len(FEATURE_COLUMNS[asset_type])
    features = np.asarray(real_time_features, dtype=float).reshape(1, -1)
    if features.shape[1] != expected_len:
        raise ValueError(
            f"Expected {expected_len} features for '{asset_type}', "
            f"received {features.shape[1]}."
        )

    model = TRAINED_MODELS[asset_type]
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities.max())

    label = REVERSE_CLASS_MAPS[asset_type][prediction]
    alert_level = SAFETY_ALERT_LEVELS[prediction]
    diagnosis = DIAGNOSIS_MESSAGES[asset_type][prediction]

    return {
        "asset_type": asset_type,
        "predicted_class": prediction,
        "predicted_label": label,
        "alert_level": alert_level,
        "diagnosis": diagnosis,
        "confidence": round(confidence, 4),
    }


def print_inference_demo() -> None:
    """Demonstrate live inference with representative sensor snapshots."""
    print("\n" + "=" * 72)
    print("  DIGITAL TWIN - Live Inference Demo")
    print("=" * 72)

    demo_readings = [
        (
            "heat_exchanger",
            [2500, 5200, 518000],
            "Normal operating snapshot",
        ),
        (
            "heat_exchanger",
            [2500, 4600, 483000],
            "Mid-fouling snapshot",
        ),
        (
            "centrifugal_pump",
            [2500, 75, 0.49, 25.14],
            "Healthy pump snapshot",
        ),
        (
            "centrifugal_pump",
            [2500, 45, 0.81, 25.5],
            "Degraded pump snapshot",
        ),
    ]

    for asset_type, features, scenario in demo_readings:
        result = evaluate_asset_health(asset_type, features)
        print(f"\n--- {scenario} ---")
        print(f"Asset:       {result['asset_type']}")
        print(f"Features:    {features}")
        print(f"Prediction:  {result['predicted_label']} (class {result['predicted_class']})")
        print(f"Confidence:  {result['confidence']:.2%}")
        print(f"Alert Level: {result['alert_level']}")
        print(f"Diagnosis:   {result['diagnosis']}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    sns.set_theme(style="whitegrid")

    print("=" * 72)
    print("  DIGITAL TWIN - Predictive Maintenance Pipeline")
    print("=" * 72)

    # Step 1 - Load and preprocess
    print("\n[Step 1] Loading and preprocessing datasets...")
    hx_df = load_and_preprocess("heat_exchanger")
    pump_df = load_and_preprocess("centrifugal_pump")
    print(
        f"  Heat Exchanger: {len(hx_df)} records | class distribution:\n"
        f"    {hx_df['Target_Status'].map(REVERSE_CLASS_MAPS['heat_exchanger']).value_counts().to_dict()}"
    )
    print(
        f"  Centrifugal Pump: {len(pump_df)} records | class distribution:\n"
        f"    {pump_df['Target_Status'].map(REVERSE_CLASS_MAPS['centrifugal_pump']).value_counts().to_dict()}"
    )

    # Step 2 - EDA
    print("\n[Step 2] Generating exploratory degradation plots...")
    run_exploratory_analysis(hx_df, pump_df, output_dir=BASE_DIR)

    # Step 3 - Train models
    print("\n[Step 3] Training Random Forest classifiers (80/20 split, random_state=42)...")
    hx_model, _, hx_X_test, _, hx_y_test = train_random_forest(hx_df, "heat_exchanger")
    pump_model, _, pump_X_test, _, pump_y_test = train_random_forest(pump_df, "centrifugal_pump")
    print("  Models trained and registered for digital-twin inference.")

    # Step 4 - Evaluate
    print("\n[Step 4] Evaluating model performance on held-out test sets...")
    evaluate_model(hx_model, hx_X_test, hx_y_test, "heat_exchanger")
    evaluate_model(pump_model, pump_X_test, pump_y_test, "centrifugal_pump")

    # Step 5 - Inference API demo
    print("\n[Step 5] Running digital-twin inference function demo...")
    print_inference_demo()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

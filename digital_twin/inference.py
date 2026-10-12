"""Health classification using explicitly supplied trained models."""
from typing import Mapping
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .config import FEATURE_COLUMNS, REVERSE_CLASS_MAPS, validate_asset

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
            "Engineering review required before operational action. Isolate equipment and initiate "
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
            "duty and outlet temperature. Urgent engineering review required to prevent "
            "catastrophic damage and process upsets."
        ),
    },
}

SAFETY_ALERT_LEVELS = {0: "INFO", 1: "WARNING", 2: "CRITICAL"}


def evaluate_asset_health(
    asset_type: str,
    real_time_features: list[float] | np.ndarray,
    models: Mapping[str, RandomForestClassifier],
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
    validate_asset(asset_type)
    if asset_type not in models:
        raise ValueError(
            f"No trained model for '{asset_type}'. "
            "Run the training pipeline first (main())."
        )

    expected_len = len(FEATURE_COLUMNS[asset_type])
    features = np.asarray(real_time_features, dtype=float)
    if features.ndim != 1 or features.size != expected_len:
        raise ValueError(
            f"Expected {expected_len} features for '{asset_type}', "
            f"received {features.size}."
        )

    if not np.isfinite(features).all():
        raise ValueError("Features must contain only finite numeric values")
    features = features.reshape(1, -1)
    model = models[asset_type]
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

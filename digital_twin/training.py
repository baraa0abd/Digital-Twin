"""Deterministic model training and held-out evaluation."""
import warnings
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from .config import FEATURE_COLUMNS, RANDOM_STATE, REVERSE_CLASS_MAPS, validate_asset

def train_random_forest(
    df: pd.DataFrame,
    asset_type: str,
    test_size: float = 0.2,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Random Forest classifier on 80% of the data (stratified when possible).
    Returns model, X_train, X_test, y_train, y_test.
    """
    validate_asset(asset_type)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(df) < 2:
        raise ValueError("Training requires at least two valid rows")
    features = FEATURE_COLUMNS[asset_type]
    X = df[features].values
    y = df["Target_Status"].values.astype(int)

    # Stratified split requires at least 2 samples per class.
    split_kwargs: dict[str, Any] = {
        "test_size": test_size,
        "random_state": RANDOM_STATE,
    }
    class_counts = pd.Series(y).value_counts()
    test_count = int(np.ceil(len(y) * test_size))
    train_count = len(y) - test_count
    if train_count < 1:
        raise ValueError("test_size leaves no training rows")
    if (class_counts >= 2).all() and min(test_count, train_count) >= len(class_counts):
        split_kwargs["stratify"] = y
    else:
        warnings.warn(
            f"[{asset_type}] Stratified split skipped - class counts or split size are too small. "
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
        y_test, y_pred, labels=sorted(REVERSE_CLASS_MAPS[asset_type]), target_names=target_names, zero_division=0
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

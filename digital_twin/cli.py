"""Application orchestration and command-line configuration."""
import argparse
import logging
from pathlib import Path
from .config import BASE_DIR, FEATURE_COLUMNS
from .data import load_and_preprocess
from .training import train_random_forest, evaluate_model
from .inference import evaluate_asset_health
from .visualization import run_exploratory_analysis

def print_inference_demo(models) -> None:
    """Demonstrate live inference with representative sensor snapshots."""
    print("\n" + "=" * 72)
    print("  DIGITAL TWIN - Live Inference Demo")
    print("=" * 72)

    demo_readings = [
        (
            "heat_exchanger",
            [2500, 5200, 518000 / 3600],
            "Normal operating snapshot",
        ),
        (
            "heat_exchanger",
            [2500, 4600, 483000 / 3600],
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
        result = evaluate_asset_health(asset_type, features, models)
        print(f"\n--- {scenario} ---")
        print(f"Asset:       {result['asset_type']}")
        print(f"Features:    {features}")
        print(f"Prediction:  {result['predicted_label']} (class {result['predicted_class']})")
        print(f"Confidence:  {result['confidence']:.2%}")
        print(f"Alert Level: {result['alert_level']}")
        print(f"Diagnosis:   {result['diagnosis']}")

def run_pipeline(data_dir: Path = BASE_DIR, output_dir: Path = BASE_DIR / "outputs", test_size: float = 0.2):
    """Load, plot, train, and evaluate each asset; return models and metrics."""
    frames = {asset: load_and_preprocess(asset, data_dir) for asset in FEATURE_COLUMNS}
    run_exploratory_analysis(frames["heat_exchanger"], frames["centrifugal_pump"], output_dir)
    models, metrics = {}, {}
    for asset, frame in frames.items():
        logging.info("%s: %d records; class counts %s", asset, len(frame), frame["Target_Status"].value_counts().to_dict())
        model, _, x_test, _, y_test = train_random_forest(frame, asset, test_size)
        models[asset] = model
        metrics[asset] = evaluate_model(model, x_test, y_test, asset)
    return models, metrics

def main():
    parser = argparse.ArgumentParser(description="Digital Twin: HYSYS asset health classification")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true", help="Show debug logging and full error tracebacks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        models, _ = run_pipeline(args.data_dir, args.output_dir, args.test_size)
        print_inference_demo(models)
    except (ValueError, FileNotFoundError) as error:
        if args.debug:
            raise
        parser.exit(2, f"Error: {error}\n")

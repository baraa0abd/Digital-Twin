"""Compatibility entry point; implementation lives in digital_twin modules."""
from digital_twin.config import BASE_DIR, CLASS_MAPS, DATASET_PATHS, FEATURE_COLUMNS, REVERSE_CLASS_MAPS
from digital_twin.data import load_and_preprocess
from digital_twin.training import evaluate_model, train_random_forest as _train
from digital_twin.inference import evaluate_asset_health as _evaluate
from digital_twin.visualization import plot_heat_exchanger_degradation, plot_pump_surge_behavior, run_exploratory_analysis
from digital_twin.cli import main as _main, run_pipeline

TRAINED_MODELS = {}

def train_random_forest(df, asset_type, test_size=0.2):
    result = _train(df, asset_type, test_size)
    TRAINED_MODELS[asset_type] = result[0]
    return result

def evaluate_asset_health(asset_type, real_time_features):
    return _evaluate(asset_type, real_time_features, TRAINED_MODELS)

def main():
    models, metrics = run_pipeline()
    TRAINED_MODELS.update(models)
    return models, metrics

if __name__ == "__main__":
    _main()

"""Regression checks for units, schema validation, splits, and inference."""
import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from digital_twin import load_and_preprocess, train_random_forest, evaluate_asset_health


class PipelineTests(unittest.TestCase):
    def test_raw_heat_duty_converted(self):
        frame = load_and_preprocess("heat_exchanger")
        self.assertAlmostEqual(frame.iloc[0]["DUTY_kW"], 519844.8045 / 3600)

    def test_canonical_units_and_invalid_rows(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            Path(directory, "heat_exchanger_final.csv").write_text(
                "Time_Step;Mass_Flow;ACTUAL_UA;DUTY_kW;Target_Status\n"
                "1;2500;5200;144;Normal\n2;inf;3;4;Normal\n3;2;3;4;\n4;bad;3;4;Normal\n"
            )
            frame = load_and_preprocess("heat_exchanger", Path(directory))
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["DUTY_kW"], 144)

    def test_invalid_schema_and_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory, "heat_exchanger_final.csv")
            path.write_text("Time_Step;Target_Status\n1;Normal\n")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_and_preprocess("heat_exchanger", Path(directory))
            path.write_text("Time_Step;Mass_Flow;ACTUAL_UA;DUTY_kW;Target_Status\n1;2;3;4;Typo\n")
            with self.assertRaisesRegex(ValueError, "unknown health labels"):
                load_and_preprocess("heat_exchanger", Path(directory))

    def test_small_split_and_inference(self):
        frame = pd.DataFrame({"Mass_Flow": range(6), "ACTUAL_UA": range(6), "DUTY_kW": range(6), "Target_Status": [0, 0, 1, 1, 2, 2]})
        with warnings.catch_warnings(record=True) as caught:
            model, _, x_test, _, _ = train_random_forest(frame, "heat_exchanger")
        self.assertEqual(len(x_test), 2)
        self.assertTrue(any("Stratified split skipped" in str(w.message) for w in caught))
        models = {"heat_exchanger": model}
        result = evaluate_asset_health("heat_exchanger", [1, 1, 1], models)
        self.assertIn(result["predicted_label"], ["Normal", "Fouling", "Failure"])
        for invalid in ([1, 2], [1, np.inf, 3], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                evaluate_asset_health("heat_exchanger", invalid, models)

    def test_unknown_asset_and_untrained_model(self):
        with self.assertRaisesRegex(ValueError, "Unknown asset"):
            load_and_preprocess("unknown")
        with self.assertRaisesRegex(ValueError, "No trained model"):
            evaluate_asset_health("heat_exchanger", [1, 2, 3], {})


if __name__ == "__main__":
    unittest.main()

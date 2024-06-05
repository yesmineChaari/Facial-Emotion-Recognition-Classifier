import unittest

import pandas as pd

from src.emotion_recognition.preprocessing import prepare_dataset, select_openface_features
from src.emotion_recognition.config import load_config


class PreprocessingTests(unittest.TestCase):
    def test_loads_project_configuration(self):
        config = load_config("configs/default.yaml")
        self.assertEqual(config["project"]["random_seed"], 42)

    def test_selects_supported_openface_columns(self):
        frame = pd.DataFrame({" x_0": [1.0], "pose_Rx": [2.0], " AU01_r": [0.5], "success": [1]})
        selected = select_openface_features(frame)
        self.assertEqual(list(selected.columns), ["x_0", "pose_Rx", "AU01_r"])

    def test_scaler_is_fit_on_training_data_only(self):
        frame = pd.DataFrame({"feature": range(16), "emotion": ["calm"] * 4 + ["happy"] * 4 + ["sad"] * 4 + ["angry"] * 4})
        prepared = prepare_dataset(frame, random_seed=42)
        self.assertAlmostEqual(float(prepared.x_train.mean().iloc[0]), 0.0, places=7)
        self.assertEqual(len(prepared.x_train), 12)
        self.assertEqual(len(prepared.x_test), 4)


if __name__ == "__main__":
    unittest.main()

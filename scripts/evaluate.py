"""Evaluate a trained classifier on a labeled OpenFace dataset."""

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.emotion_recognition.evaluation import calculate_metrics, classification_report_text
from src.emotion_recognition.preprocessing import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    from tensorflow.keras.models import load_model

    prepared = prepare_dataset(pd.read_csv(args.data))
    model = load_model(args.model)
    predictions = model.predict(prepared.x_test, verbose=0).argmax(axis=1)
    print(calculate_metrics(prepared.y_test, predictions))
    print(classification_report_text(prepared.y_test, predictions, prepared.label_encoder.classes_))


if __name__ == "__main__":
    main()

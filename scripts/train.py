"""Train the dense classifier on a labeled OpenFace dataset."""

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.emotion_recognition.models import build_dense_classifier
from src.emotion_recognition.preprocessing import prepare_dataset
from src.emotion_recognition.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepared = prepare_dataset(pd.read_csv(args.data), random_seed=args.seed)
    model = build_dense_classifier(prepared.x_train.shape[1], len(prepared.label_encoder.classes_), args.learning_rate)
    train_model(model, prepared.x_train, prepared.y_train, prepared.x_test, prepared.y_test, epochs=args.epochs, batch_size=args.batch_size, checkpoint=args.model)


if __name__ == "__main__":
    main()

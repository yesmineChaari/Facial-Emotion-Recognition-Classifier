"""Combine OpenFace CSV files into a labeled dataset."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.emotion_recognition.preprocessing import combine_openface_csvs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = combine_openface_csvs(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()

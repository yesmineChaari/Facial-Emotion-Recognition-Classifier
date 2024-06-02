"""Extract OpenFace features from RAVDESS videos."""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.emotion_recognition.openface import extract_features, find_feature_extractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--openface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    executable = find_feature_extractor(args.openface)
    outputs = extract_features(executable, args.videos, args.output, force=args.force)
    logging.info("Created or reused %d OpenFace outputs", len(outputs))


if __name__ == "__main__":
    main()

"""OpenFace feature extraction utilities."""

import logging
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}


def find_videos(directory: str | Path) -> list[Path]:
    """Return supported video files below a directory in stable order."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory not found: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def find_feature_extractor(openface_path: str | Path) -> Path:
    """Resolve an OpenFace directory or executable path."""
    candidate = Path(openface_path)
    if candidate.is_file():
        return candidate
    for name in ("FeatureExtraction.exe", "FeatureExtraction"):
        matches = list(candidate.rglob(name)) if candidate.is_dir() else []
        if matches:
            return matches[0]
    raise FileNotFoundError(f"OpenFace FeatureExtraction executable not found under {candidate}")


def expected_output(video: str | Path, output_directory: str | Path) -> Path:
    """Return the CSV path expected for one video."""
    return Path(output_directory) / f"{Path(video).stem}.csv"


def extract_video_features(
    executable: str | Path,
    video: str | Path,
    output_directory: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Run OpenFace on one video and return its CSV output path."""
    executable_path = find_feature_extractor(executable)
    video_path = Path(video)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = expected_output(video_path, output_dir)
    if output_csv.exists() and not force:
        LOGGER.info("Skipping existing output: %s", output_csv)
        return output_csv

    command = [str(executable_path), "-f", str(video_path), "-out_dir", str(output_dir),
               "-2Dfp", "false", "-pdmparams", "false", "-pose", "-aus", "-3Dfp", "-tracked"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"OpenFace failed for {video_path.name}: {result.stderr.strip()}")
    if not output_csv.exists():
        raise FileNotFoundError(f"OpenFace output was not created: {output_csv}")
    return output_csv


def extract_features(
    executable: str | Path,
    video_directory: str | Path,
    output_directory: str | Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Extract OpenFace features for every supported video in a directory."""
    return [extract_video_features(executable, video, output_directory, force=force)
            for video in find_videos(video_directory)]

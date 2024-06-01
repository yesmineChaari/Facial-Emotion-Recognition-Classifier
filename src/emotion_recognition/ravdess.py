"""RAVDESS filename parsing and video discovery."""

from dataclasses import dataclass
from pathlib import Path

EMOTION_LABELS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}
_VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}


@dataclass(frozen=True)
class RavdessMetadata:
    """Metadata encoded in a RAVDESS video filename."""

    path: Path
    modality: int
    vocal_channel: int
    emotion_code: int
    intensity: int
    statement: int
    repetition: int
    actor: int

    @property
    def emotion(self) -> str:
        """Return the human-readable emotion label."""
        return EMOTION_LABELS[self.emotion_code]


def parse_ravdess_filename(path: str | Path) -> RavdessMetadata:
    """Parse and validate the seven numeric fields in a RAVDESS filename."""
    source = Path(path)
    fields = source.stem.split("-")
    if len(fields) != 7 or any(not field.isdigit() for field in fields):
        raise ValueError(f"Invalid RAVDESS filename: {source.name}")

    values = [int(field) for field in fields]
    if values[2] not in EMOTION_LABELS:
        raise ValueError(f"Unsupported RAVDESS emotion code: {values[2]}")
    if not 1 <= values[6] <= 24:
        raise ValueError(f"Invalid RAVDESS actor number: {values[6]}")
    return RavdessMetadata(source, *values)


def discover_videos(directory: str | Path) -> list[RavdessMetadata]:
    """Discover supported video files below a RAVDESS directory."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"RAVDESS directory not found: {root}")

    videos = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS:
            videos.append(parse_ravdess_filename(path))
    return videos

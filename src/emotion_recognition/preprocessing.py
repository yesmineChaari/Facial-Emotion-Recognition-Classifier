"""Loading, cleaning, labeling, and splitting OpenFace data."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .ravdess import parse_ravdess_filename


@dataclass
class PreparedData:
    """Scaled train/test data and fitted preprocessing transformers."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    label_encoder: LabelEncoder


def select_openface_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric landmarks, pose, and Action Unit columns."""
    columns = []
    for column in frame.columns:
        name = column.strip()
        if len(name) > 1 and name[0].upper() in {"X", "Y", "Z"} and name[1] in {"_", "X", "Y", "Z"}:
            columns.append(column)
        elif name in {"pose_Rx", "pose_Ry", "pose_Rz", "pose_Tx", "pose_Ty", "pose_Tz"}:
            columns.append(column)
        elif name.startswith("AU") and (name.endswith("_r") or name.endswith("_c")):
            columns.append(column)
    if not columns:
        raise ValueError("No supported OpenFace feature columns were found")
    selected = frame[columns].copy()
    selected.columns = [column.strip() for column in selected.columns]
    return selected.apply(pd.to_numeric, errors="coerce")


def load_labeled_openface_csv(path: str | Path) -> pd.DataFrame:
    """Load one OpenFace CSV, select features, and attach its RAVDESS label."""
    source = Path(path)
    metadata = parse_ravdess_filename(source)
    selected = select_openface_features(pd.read_csv(source))
    selected["emotion"] = metadata.emotion
    selected["source_video"] = source.name
    return selected


def combine_openface_csvs(directory: str | Path) -> pd.DataFrame:
    """Combine all OpenFace CSV files below a directory."""
    files = sorted(Path(directory).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No OpenFace CSV files found in {directory}")
    return pd.concat((load_labeled_openface_csv(path) for path in files), ignore_index=True)


def prepare_dataset(frame: pd.DataFrame, *, test_size: float = 0.25, random_seed: int = 42) -> PreparedData:
    """Encode labels, split deterministically, and scale training data only."""
    if "emotion" not in frame.columns:
        raise ValueError("Dataset must contain an emotion column")
    features = frame.drop(columns=["emotion", "source_video"], errors="ignore")
    features = features.select_dtypes(include="number").replace([float("inf"), -float("inf")], pd.NA)
    features = features.fillna(features.median(numeric_only=True))
    labels = frame["emotion"].astype(str)
    label_encoder = LabelEncoder()
    encoded = pd.Series(label_encoder.fit_transform(labels), index=labels.index, name="emotion")
    x_train, x_test, y_train, y_test = train_test_split(
        features, encoded, test_size=test_size, random_state=random_seed, shuffle=True, stratify=encoded
    )
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=features.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=features.columns, index=x_test.index)
    return PreparedData(x_train_scaled, x_test_scaled, y_train, y_test, scaler, label_encoder)

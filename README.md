# Facial Emotion Recognition Using Convolutional Neural Networks

## Overview

This project classifies emotions from facial information extracted from RAVDESS videos with OpenFace. The primary representation is tabular facial landmarks, head pose, and Action Unit measurements.

## Project Objective

Build a reproducible pipeline from video files to an emotion classifier without committing the RAVDESS dataset, OpenFace binaries, generated features, or unsupported results.

## Pipeline

1. Discover and validate RAVDESS video filenames.
2. Extract facial features with OpenFace.
3. Select useful numeric OpenFace columns and attach emotion labels.
4. Shuffle and split data into 75% training and 25% test partitions.
5. Fit `StandardScaler` on training data only.
6. Train and evaluate a dense neural classifier.

## Dataset

Download the visual/video portion of RAVDESS separately. Place it in a local data directory and pass that directory to the extraction script. The full dataset must remain outside Git.

RAVDESS filenames encode seven fields. The third field is the emotion code:

| Code | Emotion   |
| ---- | --------- |
| 1    | neutral   |
| 2    | calm      |
| 3    | happy     |
| 4    | sad       |
| 5    | angry     |
| 6    | fearful   |
| 7    | disgust   |
| 8    | surprised |

## OpenFace Feature Extraction

Install OpenFace separately and provide either its installation directory or the path to `FeatureExtraction.exe`:

```powershell
python scripts/extract_openface.py --videos path\to\RAVDESS --openface path\to\OpenFace --output data\openface
```

Existing CSV outputs are skipped unless `--force` is supplied.

## Preprocessing

The preprocessing module keeps facial landmark coordinates, head pose, and Action Unit intensity/presence values. Non-numeric metadata is excluded from model features. Missing numeric values are median-filled, and the scaler is fitted only on the training partition to prevent leakage.

## Model Architecture

OpenFace produces tabular features, not image tensors. Therefore the implemented model is a dense neural classifier rather than a 2D image CNN. It uses two ReLU layers, batch normalization, dropout, and a softmax output layer.

## Training

```powershell
python scripts/prepare_dataset.py --input data\openface --output data\processed\dataset.csv
python scripts/train.py --data data\processed\dataset.csv --model results\models\emotion_classifier.keras
```

Training history and model outputs are generated locally and are intentionally ignored by Git.

## Evaluation

```powershell
python scripts/evaluate.py --data data\processed\dataset.csv --model results\models\emotion_classifier.keras
```

Metrics and figures should only be generated from an actual training run on real data. No unsupported results are included in this repository.

## Interpretability

`permutation_importance_table` supports compatible scikit-learn models. Interpretability output must be produced from a real fitted model and real feature data.

## Repository Structure

```text
configs/                 Configuration files
data/                    Local raw, OpenFace, and processed data
scripts/                 Command-line workflows
src/emotion_recognition/ Reusable project modules
results/                 Local models, metrics, and figures
tests/                   Unit tests and synthetic fixtures
```

## Installation

Use Python 3.11 or another version supported by the installed TensorFlow release:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Limitations

OpenFace must be installed separately. Training and evaluation require the real extracted dataset. Frame-level random splitting can overestimate generalization when frames from the same actor or video occur in both partitions; actor-aware evaluation is a useful future improvement.

## References

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
- Baltrusaitis, T., Zadeh, A., Lim, Y. C., & Morency, L.-P. (2018). OpenFace 2.0.

## Development Status

The repository contains the reproducible implementation and tests. Real training outputs are intentionally absent until the external RAVDESS videos and OpenFace executable are run through the pipeline.

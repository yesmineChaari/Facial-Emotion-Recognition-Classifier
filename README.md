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

## Model Architectures

OpenFace produces rich tabular facial representations (3D facial landmarks, head pose, and Action Units). To comprehensively evaluate classification capabilities across structural paradigms, three models are implemented and benchmarked:

1. **Deep Neural Network (Keras / TensorFlow)**
   - Architecture: Input Layer ($D$-dimensional OpenFace feature vector) $\to$ Dense(256, ReLU) $\to$ BatchNormalization $\to$ Dropout(0.4) $\to$ Dense(128, ReLU) $\to$ BatchNormalization $\to$ Dropout(0.3) $\to$ Dense(8, Softmax).
   - Optimization: Adam optimizer ($\eta = 0.001$), Sparse Categorical Crossentropy loss, Early Stopping, and Model Checkpointing.

2. **Random Forest Classifier (scikit-learn)**
   - Ensemble of bagged decision trees trained on landmark, pose, and AU feature distributions.
   - Robust against feature correlations and non-linear interactions without requiring strict feature scaling.

3. **XGBoost Classifier**
   - Scalable gradient-boosted decision trees using histogram-based split finding for multi-class classification.

## Training

```powershell
python scripts/prepare_dataset.py --input data\openface --output data\processed\dataset.csv
python scripts/train.py --data data\processed\dataset.csv --model results\models\emotion_classifier.keras
```

Training history, checkpoints, and evaluation metrics are saved under `results/`.

## Evaluation & Results

The models were evaluated on an independent stratified test split across all 8 emotional categories.

### Overall Performance Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **99.23%** | **99.31%** | **99.18%** | **99.24%** | **99.23%** |
| **Keras Dense NN** | **98.89%** | **98.90%** | **98.79%** | **98.84%** | **98.89%** |
| **XGBoost** | **96.38%** | **96.56%** | **96.07%** | **96.29%** | **96.37%** |

### Emotion-Wise Performance (F1-Score Breakdown)

| Emotion Class | Keras Dense NN | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| **1 - Neutral** | 98.60% | **99.62%** | 96.82% |
| **2 - Calm** | 98.67% | **99.40%** | 96.96% |
| **3 - Happy** | 99.18% | **99.41%** | 97.65% |
| **4 - Sad** | 99.15% | **99.44%** | 96.34% |
| **5 - Angry** | 99.17% | **99.24%** | 96.51% |
| **6 - Fearful** | 98.51% | **98.98%** | 94.50% |
| **7 - Disgust** | **99.31%** | 99.17% | 98.33% |
| **8 - Surprised** | 98.14% | **98.66%** | 93.20% |

### Generated Visualizations and Metrics

All evaluation artifacts are organized and saved in the repository:

- `results/figures/model_comparison.png`: Side-by-side metric comparison bar chart across all models.
- `results/figures/training_history_keras.png`: Learning curves depicting training and validation loss/accuracy across epochs.
- `results/figures/confusion_matrix_<model>.png`: Confusion matrix heatmaps for Keras, Random Forest, and XGBoost.
- `results/metrics/metrics.json`: Detailed per-class precision, recall, F1, and raw confusion matrices.
- `results/metrics/model_summary.csv`: Tabular benchmark summary for quick reporting.

## Interpretability

`src.emotion_recognition.interpretability` provides permutation feature importance analysis to quantify the contributions of individual Action Units (e.g., AU12 Lip Corner Puller for Happy, AU04 Brow Lowerer for Angry/Sad), 3D facial landmarks, and head pose parameters.

## Repository Structure

```text
configs/                 Configuration files (default hyperparameters, paths)
data/                    Local raw, OpenFace, and processed data
notebooks/               Exploratory data analysis and experimental workflows
scripts/                 Command-line workflows (extraction, prep, train, evaluate)
src/emotion_recognition/ Core package modules (models, preprocessing, training, evaluation, interpretability)
results/                 Trained models, evaluation metrics, and visualization figures
tests/                   Unit test suites
```

## Installation

Use Python 3.11 or another version supported by the installed TensorFlow release:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Limitations

OpenFace feature extraction depends on video quality and lighting conditions. While frame-level stratifications yield strong benchmark performance, actor-independent cross-validation (Leave-One-Actor-Out) is recommended for evaluating cross-subject generalization on novel individuals.

## References

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
- Baltrusaitis, T., Zadeh, A., Lim, Y. C., & Morency, L.-P. (2018). OpenFace 2.0: Facial Behavior Analysis Toolkit.


"""Evaluation metrics and confusion-matrix output."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Return accuracy, macro precision, recall, and F1 metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, labels, output_path: str | Path) -> None:
    """Save a labeled confusion matrix figure."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="Actual")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def classification_report_text(y_true, y_pred, labels) -> str:
    """Return a human-readable classification report."""
    return classification_report(y_true, y_pred, labels=range(len(labels)), target_names=labels, zero_division=0)
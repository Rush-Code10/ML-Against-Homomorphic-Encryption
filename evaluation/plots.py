from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(feature_importances: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = list(feature_importances.keys())
    values = list(feature_importances.values())

    plt.figure(figsize=(8, 5))
    plt.bar(features, values, color="#2b6cb0")
    plt.title("Random Forest Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(matrix)

    plt.figure(figsize=(8, 6))
    plt.imshow(array, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    threshold = array.max() / 2 if array.size else 0
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            color = "white" if array[row, col] > threshold else "black"
            plt.text(col, row, str(array[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_accuracy_comparison(results: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, accuracies, color=["#2f855a", "#d69e2e", "#c53030", "#2b6cb0"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    for index, accuracy in enumerate(accuracies):
        plt.text(index, accuracy + 0.01, f"{accuracy:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

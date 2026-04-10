from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, object]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "labels": list(label_names),
    }

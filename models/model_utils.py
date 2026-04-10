from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.config import ProjectConfig


def prepare_features(
    dataframe: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    features = dataframe[config.feature_columns].to_numpy(dtype=np.float32)
    labels = dataframe["label"].astype(str).to_numpy()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return features_scaled, encoded_labels, scaler, encoder


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    config: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = int(counts.min()) if counts.size else 0
    if min_count < 2:
        raise ValueError("Each class needs at least 2 samples for a train/test split.")

    test_size = min(config.test_size, max(1 / len(labels), 1 / min_count))
    if len(labels) * test_size < len(unique_labels):
        test_size = len(unique_labels) / len(labels)
    test_size = min(0.4, max(0.2, test_size))

    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=config.random_seed,
        stratify=labels,
    )


def train_dummy_baseline(x_train: np.ndarray, y_train: np.ndarray) -> DummyClassifier:
    model = DummyClassifier(strategy="most_frequent")
    model.fit(x_train, y_train)
    return model


def save_sklearn_artifact(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

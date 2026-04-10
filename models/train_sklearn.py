from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from evaluation.metrics import compute_classification_metrics
from models.model_utils import (
    prepare_features,
    save_sklearn_artifact,
    split_dataset,
    train_dummy_baseline,
)
from utils.config import ProjectConfig
from utils.logger import get_logger


@dataclass(slots=True)
class SklearnTrainingResult:
    metrics: dict[str, dict[str, object]]
    feature_importances: dict[str, float]


def train_sklearn_models(
    dataframe: pd.DataFrame,
    config: ProjectConfig,
    model_dir: Path | None = None,
) -> SklearnTrainingResult:
    logger = get_logger("train_sklearn")
    model_dir = model_dir or config.model_dir

    features, labels, scaler, encoder = prepare_features(dataframe, config)
    x_train, x_test, y_train, y_test = split_dataset(features, labels, config)

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=config.random_seed,
        class_weight="balanced",
    )
    svm = SVC(kernel="rbf", probability=True, random_state=config.random_seed)

    rf.fit(x_train, y_train)
    svm.fit(x_train, y_train)
    dummy = train_dummy_baseline(x_train, y_train)

    rf_predictions = rf.predict(x_test)
    svm_predictions = svm.predict(x_test)
    dummy_predictions = dummy.predict(x_test)

    rf_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=rf_predictions,
        label_names=encoder.classes_.tolist(),
    )
    svm_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=svm_predictions,
        label_names=encoder.classes_.tolist(),
    )
    dummy_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=dummy_predictions,
        label_names=encoder.classes_.tolist(),
    )

    save_sklearn_artifact(rf, model_dir / "random_forest.joblib")
    save_sklearn_artifact(svm, model_dir / "svm.joblib")
    save_sklearn_artifact(dummy, model_dir / "dummy_classifier.joblib")
    save_sklearn_artifact(scaler, model_dir / "scaler.joblib")
    save_sklearn_artifact(encoder, model_dir / "label_encoder.joblib")

    logger.info("RandomForest accuracy: %.4f", rf_metrics["accuracy"])
    logger.info("SVM accuracy: %.4f", svm_metrics["accuracy"])
    logger.info("Dummy accuracy: %.4f", dummy_metrics["accuracy"])

    return SklearnTrainingResult(
        metrics={
            "dummy": dummy_metrics,
            "random_forest": rf_metrics,
            "svm": svm_metrics,
        },
        feature_importances=dict(zip(config.feature_columns, rf.feature_importances_.tolist())),
    )

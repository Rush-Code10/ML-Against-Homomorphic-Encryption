from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.dataset_generator import DatasetGenerator
from evaluation.plots import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
)
from fhe.fhe_setup import init_context
from models.model_utils import save_json
from models.train_sklearn import train_sklearn_models
from models.train_torch import train_torch_model
from utils.config import ProjectConfig
from utils.logger import get_logger


@dataclass(slots=True)
class PipelineResult:
    backend: str
    config: dict[str, object]
    dataset_path: str
    total_rows: int
    baseline_rows: int
    defended_rows: int
    metrics: dict[str, dict[str, object]]
    accuracy_comparison: dict[str, float]
    plots: dict[str, str]


def build_config(overrides: dict[str, object] | None = None) -> ProjectConfig:
    config = ProjectConfig()
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    config.ensure_directories()
    return config


def run_pipeline(overrides: dict[str, object] | None = None) -> PipelineResult:
    config = build_config(overrides)
    logger = get_logger("main", config.artifacts_dir / "pipeline.log")
    logger.info("Starting metadata leakage pipeline")
    logger.info("Configuration: %s", config.as_dict())

    context = init_context(
        vector_size=config.vector_size,
        backend_preference=config.backend_preference,
    )
    logger.info("Initialized FHE backend: %s", context.backend)

    generator = DatasetGenerator(config=config, context=context)
    baseline_df = generator.generate(enable_defense=False)
    defended_df = generator.generate(enable_defense=True)
    combined_df = pd.concat([baseline_df, defended_df], ignore_index=True)
    generator.save(combined_df, config.dataset_path)

    baseline_results = _train_and_evaluate(baseline_df, config, "baseline")
    defended_results = _train_and_evaluate(defended_df, config, "defended")

    comparison = {
        "dummy_baseline": float(baseline_results["dummy"]["accuracy"]),
        "dummy_defended": float(defended_results["dummy"]["accuracy"]),
        "rf_baseline": float(baseline_results["random_forest"]["accuracy"]),
        "rf_defended": float(defended_results["random_forest"]["accuracy"]),
        "svm_baseline": float(baseline_results["svm"]["accuracy"]),
        "svm_defended": float(defended_results["svm"]["accuracy"]),
        "torch_baseline": float(baseline_results["torch"]["accuracy"]),
        "torch_defended": float(defended_results["torch"]["accuracy"]),
    }
    plot_accuracy_comparison(comparison, config.plots_dir / "accuracy_comparison.html")
    save_json(comparison, config.metrics_dir / "accuracy_comparison.json")

    logger.info("Baseline RF accuracy: %.4f", baseline_results["random_forest"]["accuracy"])
    logger.info("Defended RF accuracy: %.4f", defended_results["random_forest"]["accuracy"])
    logger.info("Baseline Torch accuracy: %.4f", baseline_results["torch"]["accuracy"])
    logger.info("Defended Torch accuracy: %.4f", defended_results["torch"]["accuracy"])
    logger.info("Pipeline complete")
    return PipelineResult(
        backend=context.backend,
        config=config.as_dict(),
        dataset_path=str(config.dataset_path),
        total_rows=int(len(combined_df)),
        baseline_rows=int(len(baseline_df)),
        defended_rows=int(len(defended_df)),
        metrics={
            "baseline": baseline_results,
            "defended": defended_results,
        },
        accuracy_comparison=comparison,
        plots={
            "accuracy_comparison": str(config.plots_dir / "accuracy_comparison.html"),
            "feature_importance_baseline": str(config.plots_dir / "feature_importance_baseline.html"),
            "feature_importance_defended": str(config.plots_dir / "feature_importance_defended.html"),
            "confusion_matrix_rf_baseline": str(config.plots_dir / "confusion_matrix_rf_baseline.html"),
            "confusion_matrix_rf_defended": str(config.plots_dir / "confusion_matrix_rf_defended.html"),
            "confusion_matrix_torch_baseline": str(config.plots_dir / "confusion_matrix_torch_baseline.html"),
            "confusion_matrix_torch_defended": str(config.plots_dir / "confusion_matrix_torch_defended.html"),
        },
    )


def _train_and_evaluate(
    dataframe: pd.DataFrame,
    config: ProjectConfig,
    run_name: str,
) -> dict[str, dict[str, object]]:
    sklearn_result = train_sklearn_models(
        dataframe=dataframe,
        config=config,
        model_dir=config.model_dir / run_name,
    )
    torch_result = train_torch_model(
        dataframe=dataframe,
        config=config,
        model_dir=config.model_dir / run_name,
    )

    plot_feature_importance(
        sklearn_result.feature_importances,
        config.plots_dir / f"feature_importance_{run_name}.png",
    )
    plot_confusion_matrix(
        sklearn_result.metrics["random_forest"]["confusion_matrix"],
        sklearn_result.metrics["random_forest"]["labels"],
        config.plots_dir / f"confusion_matrix_rf_{run_name}.png",
        title=f"Random Forest Confusion Matrix ({run_name})",
    )
    plot_confusion_matrix(
        torch_result.metrics["confusion_matrix"],
        torch_result.metrics["labels"],
        config.plots_dir / f"confusion_matrix_torch_{run_name}.png",
        title=f"Torch MLP Confusion Matrix ({run_name})",
    )

    payload = {
        "dummy": sklearn_result.metrics["dummy"],
        "random_forest": sklearn_result.metrics["random_forest"],
        "svm": sklearn_result.metrics["svm"],
        "torch": torch_result.metrics,
    }
    save_json(payload, config.metrics_dir / f"metrics_{run_name}.json")
    return payload


if __name__ == "__main__":
    run_pipeline()

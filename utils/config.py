from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(slots=True)
class ProjectConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    dataset_path: Path = field(init=False)
    random_seed: int = 42
    samples_per_operation: int = 1000
    test_size: float = 0.2
    val_size: float = 0.1
    vector_size: int = 16
    regression_features: int = 4
    defense_noise_scale: float = 0.08
    baseline_noise_scale: float = 0.07
    workload_jitter_scale: float = 0.2
    torch_hidden_dim: int = 32
    torch_epochs: int = 50
    torch_batch_size: int = 64
    torch_learning_rate: float = 1e-3
    backend_preference: str = "tenseal"
    operation_labels: List[str] = field(
        default_factory=lambda: [
            "mean",
            "variance",
            "dot_product",
            "linear_regression_inference",
            "logistic_regression_approx",
        ]
    )

    def __post_init__(self) -> None:
        self.random_seed = int(os.getenv("AML_FHE_RANDOM_SEED", self.random_seed))
        self.samples_per_operation = int(
            os.getenv("AML_FHE_SAMPLES_PER_OPERATION", self.samples_per_operation)
        )
        self.vector_size = int(os.getenv("AML_FHE_VECTOR_SIZE", self.vector_size))
        self.regression_features = int(
            os.getenv("AML_FHE_REGRESSION_FEATURES", self.regression_features)
        )
        self.defense_noise_scale = float(
            os.getenv("AML_FHE_DEFENSE_NOISE_SCALE", self.defense_noise_scale)
        )
        self.baseline_noise_scale = float(
            os.getenv("AML_FHE_BASELINE_NOISE_SCALE", self.baseline_noise_scale)
        )
        self.workload_jitter_scale = float(
            os.getenv("AML_FHE_WORKLOAD_JITTER_SCALE", self.workload_jitter_scale)
        )
        self.torch_hidden_dim = int(
            os.getenv("AML_FHE_TORCH_HIDDEN_DIM", self.torch_hidden_dim)
        )
        self.torch_epochs = int(os.getenv("AML_FHE_TORCH_EPOCHS", self.torch_epochs))
        self.torch_batch_size = int(
            os.getenv("AML_FHE_TORCH_BATCH_SIZE", self.torch_batch_size)
        )
        self.torch_learning_rate = float(
            os.getenv("AML_FHE_TORCH_LR", self.torch_learning_rate)
        )
        self.backend_preference = os.getenv(
            "AML_FHE_BACKEND_PREFERENCE",
            self.backend_preference,
        )
        self.data_dir = self.project_root / "data"
        self.artifacts_dir = self.project_root / "artifacts"
        self.model_dir = self.artifacts_dir / "models"
        self.plots_dir = self.artifacts_dir / "plots"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.dataset_path = self.data_dir / "dataset.csv"

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.artifacts_dir,
            self.model_dir,
            self.plots_dir,
            self.metrics_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def feature_columns(self) -> List[str]:
        return ["time", "size", "ops", "depth", "noise"]

    def metadata_columns(self, include_defense: bool = True) -> List[str]:
        columns = self.feature_columns.copy()
        if include_defense:
            columns.append("defense")
        columns.append("label")
        return columns

    def as_dict(self) -> Dict[str, object]:
        return {
            "random_seed": self.random_seed,
            "samples_per_operation": self.samples_per_operation,
            "vector_size": self.vector_size,
            "regression_features": self.regression_features,
            "defense_noise_scale": self.defense_noise_scale,
            "baseline_noise_scale": self.baseline_noise_scale,
            "workload_jitter_scale": self.workload_jitter_scale,
            "torch_hidden_dim": self.torch_hidden_dim,
            "torch_epochs": self.torch_epochs,
            "torch_batch_size": self.torch_batch_size,
            "torch_learning_rate": self.torch_learning_rate,
            "backend_preference": self.backend_preference,
        }

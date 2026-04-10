from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import compute_classification_metrics
from models.model_utils import prepare_features, split_dataset
from utils.config import ProjectConfig
from utils.logger import get_logger


class MetadataMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass(slots=True)
class TorchTrainingResult:
    metrics: dict[str, object]
    history: list[float]


def train_torch_model(
    dataframe: pd.DataFrame,
    config: ProjectConfig,
    model_dir: Path | None = None,
) -> TorchTrainingResult:
    logger = get_logger("train_torch")
    model_dir = model_dir or config.model_dir

    features, labels, _, encoder = prepare_features(dataframe, config)
    x_train, x_test, y_train, y_test = split_dataset(features, labels, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetadataMLP(
        input_dim=features.shape[1],
        hidden_dim=config.torch_hidden_dim,
        output_dim=len(encoder.classes_),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.torch_learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=min(config.torch_batch_size, max(8, len(x_train))),
        shuffle=True,
    )

    losses: list[float] = []
    model.train()
    for epoch in range(config.torch_epochs):
        epoch_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        mean_loss = epoch_loss / max(len(train_loader), 1)
        losses.append(mean_loss)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %s/%s - loss %.4f", epoch + 1, config.torch_epochs, mean_loss)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32, device=device))
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=np.asarray(predictions),
        label_names=encoder.classes_.tolist(),
    )

    target_path = model_dir / "metadata_mlp.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": features.shape[1],
            "hidden_dim": config.torch_hidden_dim,
            "output_dim": len(encoder.classes_),
            "classes": encoder.classes_.tolist(),
        },
        target_path,
    )
    logger.info("Torch MLP accuracy: %.4f", metrics["accuracy"])
    return TorchTrainingResult(metrics=metrics, history=losses)
